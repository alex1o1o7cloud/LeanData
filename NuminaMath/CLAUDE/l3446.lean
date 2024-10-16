import Mathlib

namespace NUMINAMATH_CALUDE_union_covers_reals_iff_a_leq_neg_two_l3446_344637

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x < -2 ∨ x ≥ 1}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≥ a}

-- State the theorem
theorem union_covers_reals_iff_a_leq_neg_two (a : ℝ) :
  A ∪ B a = Set.univ ↔ a ≤ -2 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_iff_a_leq_neg_two_l3446_344637


namespace NUMINAMATH_CALUDE_chinese_chess_tournament_l3446_344668

-- Define the winning relation
def Wins (n : ℕ) : (ℕ → ℕ → Prop) := sorry

-- Main theorem
theorem chinese_chess_tournament (n : ℕ) (h : n ≥ 2) :
  ∃ (P : ℕ → ℕ → ℕ),
    (∀ i j i' j', i ≤ n ∧ j ≤ n ∧ i' ≤ n ∧ j' ≤ n ∧ (i, j) ≠ (i', j') → P i j ≠ P i' j') ∧ 
    (∀ i j, i ≤ n ∧ j ≤ n → P i j ≤ 2*n^2) ∧
    (∀ i j i' j', i < i' ∧ i ≤ n ∧ j ≤ n ∧ i' ≤ n ∧ j' ≤ n → Wins n (P i j) (P i' j')) :=
by
  sorry

-- Transitive property of winning
axiom wins_trans (n : ℕ) : ∀ a b c, Wins n a b → Wins n b c → Wins n a c

-- Maximum number of draws
axiom max_draws (n : ℕ) : ∃ (draw_count : ℕ), draw_count ≤ n^3/16 ∧ 
  ∀ a b, a ≤ 2*n^2 ∧ b ≤ 2*n^2 ∧ a ≠ b → (Wins n a b ∨ Wins n b a ∨ (¬Wins n a b ∧ ¬Wins n b a))

end NUMINAMATH_CALUDE_chinese_chess_tournament_l3446_344668


namespace NUMINAMATH_CALUDE_wildflower_color_difference_l3446_344695

theorem wildflower_color_difference :
  let total_flowers : ℕ := 44
  let yellow_and_white : ℕ := 13
  let red_and_yellow : ℕ := 17
  let red_and_white : ℕ := 14
  let flowers_with_red : ℕ := red_and_yellow + red_and_white
  let flowers_with_white : ℕ := yellow_and_white + red_and_white
  flowers_with_red - flowers_with_white = 4 :=
by sorry

end NUMINAMATH_CALUDE_wildflower_color_difference_l3446_344695


namespace NUMINAMATH_CALUDE_solve_linear_equation_l3446_344683

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end NUMINAMATH_CALUDE_solve_linear_equation_l3446_344683


namespace NUMINAMATH_CALUDE_variance_mean_preserved_l3446_344642

def initial_set : List Int := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

def mean (xs : List Int) : ℚ := (xs.sum : ℚ) / xs.length

def variance (xs : List Int) : ℚ :=
  let m := mean xs
  (xs.map (fun x => ((x : ℚ) - m) ^ 2)).sum / xs.length

def replace_4_with_neg1_and_5 (xs : List Int) : List Int :=
  xs.filter (· ≠ 4) ++ [-1, 5]

def replace_neg4_with_1_and_neg5 (xs : List Int) : List Int :=
  xs.filter (· ≠ -4) ++ [1, -5]

theorem variance_mean_preserved :
  (mean initial_set = mean (replace_4_with_neg1_and_5 initial_set) ∧
   variance initial_set = variance (replace_4_with_neg1_and_5 initial_set)) ∨
  (mean initial_set = mean (replace_neg4_with_1_and_neg5 initial_set) ∧
   variance initial_set = variance (replace_neg4_with_1_and_neg5 initial_set)) :=
by sorry

end NUMINAMATH_CALUDE_variance_mean_preserved_l3446_344642


namespace NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l3446_344677

theorem ceiling_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_negative_three_point_seven_l3446_344677


namespace NUMINAMATH_CALUDE_mitch_weekday_hours_l3446_344696

/-- Represents the weekly work schedule and earnings of Mitch, a freelancer -/
structure MitchSchedule where
  weekdayHours : ℕ
  weekendHours : ℕ
  weekdayRate : ℕ
  weekendRate : ℕ
  totalEarnings : ℕ

/-- Theorem stating that Mitch works 25 hours from Monday to Friday -/
theorem mitch_weekday_hours (schedule : MitchSchedule) :
  schedule.weekendHours = 6 ∧
  schedule.weekdayRate = 3 ∧
  schedule.weekendRate = 6 ∧
  schedule.totalEarnings = 111 →
  schedule.weekdayHours = 25 := by
  sorry

end NUMINAMATH_CALUDE_mitch_weekday_hours_l3446_344696


namespace NUMINAMATH_CALUDE_solve_equation_l3446_344634

theorem solve_equation (x : ℝ) : 
  Real.sqrt ((2 / x) + 3) = 5 / 3 → x = -9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3446_344634


namespace NUMINAMATH_CALUDE_classroom_chairs_l3446_344604

theorem classroom_chairs (blue_chairs : ℕ) (green_chairs : ℕ) (white_chairs : ℕ) 
  (h1 : blue_chairs = 10)
  (h2 : green_chairs = 3 * blue_chairs)
  (h3 : white_chairs = blue_chairs + green_chairs - 13) :
  blue_chairs + green_chairs + white_chairs = 67 := by
  sorry

end NUMINAMATH_CALUDE_classroom_chairs_l3446_344604


namespace NUMINAMATH_CALUDE_range_of_b_l3446_344639

def solution_set (b : ℝ) : Set ℝ := {x : ℝ | |3*x - b| < 4}

theorem range_of_b :
  (∃ b : ℝ, solution_set b = {1, 2, 3}) →
  (∀ b : ℝ, solution_set b = {1, 2, 3} → b ∈ Set.Ioo 5 7) ∧
  (∀ b : ℝ, b ∈ Set.Ioo 5 7 → solution_set b = {1, 2, 3}) :=
by sorry

end NUMINAMATH_CALUDE_range_of_b_l3446_344639


namespace NUMINAMATH_CALUDE_root_in_interval_l3446_344686

def f (x : ℝ) := x^3 + 2*x - 1

theorem root_in_interval :
  (f 0 < 0) →
  (f 1 > 0) →
  ∃ x : ℝ, x ∈ (Set.Ioo 0 1) ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l3446_344686


namespace NUMINAMATH_CALUDE_parabola_properties_l3446_344665

-- Define the parabola function
def f (x : ℝ) : ℝ := (x - 2)^2 + 1

-- Theorem statement
theorem parabola_properties :
  -- 1. The parabola opens upwards
  (∀ x y : ℝ, f ((x + y) / 2) ≤ (f x + f y) / 2) ∧
  -- 2. The axis of symmetry is x = 2
  (∀ h : ℝ, f (2 + h) = f (2 - h)) ∧
  -- 3. The vertex is at (2, 1)
  (f 2 = 1 ∧ ∀ x : ℝ, f x ≥ 1) ∧
  -- 4. When x < 2, y decreases as x increases
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 2 → f x₁ > f x₂) :=
by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l3446_344665


namespace NUMINAMATH_CALUDE_village_population_problem_l3446_344651

theorem village_population_problem (P : ℕ) : 
  (P : ℝ) * 0.9 * 0.85 = 3213 → P = 4200 := by
  sorry

end NUMINAMATH_CALUDE_village_population_problem_l3446_344651


namespace NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3446_344673

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 5| = |x + 3| := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_unique_solution_l3446_344673


namespace NUMINAMATH_CALUDE_solve_exponential_equation_l3446_344613

theorem solve_exponential_equation :
  ∃ y : ℝ, (1000 : ℝ)^4 = 10^y ∧ y = 12 := by sorry

end NUMINAMATH_CALUDE_solve_exponential_equation_l3446_344613


namespace NUMINAMATH_CALUDE_line_parallel_plane_condition_l3446_344622

-- Define the types for our geometric objects
variable (Point Line Plane : Type)

-- Define the relationships between geometric objects
variable (parallel : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersects : Line → Plane → Prop)

-- The main theorem
theorem line_parallel_plane_condition :
  -- If a line is parallel to a plane, then it's not contained in the plane
  (∀ (l : Line) (p : Plane), parallel l p → ¬(contained_in l p)) ∧
  -- There exists a line and a plane such that the line is not contained in the plane
  -- but also not parallel to it (i.e., it intersects the plane)
  (∃ (l : Line) (p : Plane), ¬(contained_in l p) ∧ ¬(parallel l p) ∧ intersects l p) :=
sorry

end NUMINAMATH_CALUDE_line_parallel_plane_condition_l3446_344622


namespace NUMINAMATH_CALUDE_jim_bakes_two_loaves_l3446_344678

/-- The amount of flour Jim can bake into loaves -/
def jim_loaves (cupboard kitchen_counter pantry loaf_requirement : ℕ) : ℕ :=
  (cupboard + kitchen_counter + pantry) / loaf_requirement

/-- Theorem: Jim can bake 2 loaves of bread -/
theorem jim_bakes_two_loaves :
  jim_loaves 200 100 100 200 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jim_bakes_two_loaves_l3446_344678


namespace NUMINAMATH_CALUDE_point_in_region_l3446_344676

theorem point_in_region (m : ℝ) :
  (m^2 - 3*m + 2 > 0) ↔ (m < 1 ∨ m > 2) :=
by sorry

end NUMINAMATH_CALUDE_point_in_region_l3446_344676


namespace NUMINAMATH_CALUDE_vector_values_l3446_344698

-- Define the vectors
def OA (m : ℝ) : Fin 2 → ℝ := ![(-2 : ℝ), m]
def OB (n : ℝ) : Fin 2 → ℝ := ![n, (1 : ℝ)]
def OC : Fin 2 → ℝ := ![(5 : ℝ), (-1 : ℝ)]

-- Define collinearity
def collinear (A B C : Fin 2 → ℝ) : Prop :=
  ∃ (t : ℝ), B - A = t • (C - A)

-- Define perpendicularity
def perpendicular (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

theorem vector_values (m n : ℝ) :
  collinear (OA m) (OB n) OC ∧
  perpendicular (OA m) (OB n) →
  m = 3 ∧ n = 3/2 := by sorry

end NUMINAMATH_CALUDE_vector_values_l3446_344698


namespace NUMINAMATH_CALUDE_power_of_64_l3446_344661

theorem power_of_64 : (64 : ℝ) ^ (5/6) = 32 := by sorry

end NUMINAMATH_CALUDE_power_of_64_l3446_344661


namespace NUMINAMATH_CALUDE_sum_of_first_seven_primes_mod_eighth_prime_l3446_344616

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (List.sum (List.take 7 first_eight_primes)) % (List.get! first_eight_primes 7) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_seven_primes_mod_eighth_prime_l3446_344616


namespace NUMINAMATH_CALUDE_projectile_meeting_time_l3446_344620

theorem projectile_meeting_time (distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  distance = 1455 →
  speed1 = 470 →
  speed2 = 500 →
  (distance / (speed1 + speed2)) * 60 = 90 := by
sorry

end NUMINAMATH_CALUDE_projectile_meeting_time_l3446_344620


namespace NUMINAMATH_CALUDE_last_two_digits_product_l3446_344658

theorem last_two_digits_product (A B : Nat) : 
  (A ≤ 9) → 
  (B ≤ 9) → 
  (10 * A + B) % 5 = 0 → 
  A + B = 16 → 
  A * B = 30 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l3446_344658


namespace NUMINAMATH_CALUDE_gray_area_between_circles_l3446_344691

theorem gray_area_between_circles (r : ℝ) (R : ℝ) : 
  r > 0 → 
  R = 3 * r → 
  2 * r = 4 → 
  R^2 * π - r^2 * π = 32 * π := by
sorry

end NUMINAMATH_CALUDE_gray_area_between_circles_l3446_344691


namespace NUMINAMATH_CALUDE_division_by_240_property_l3446_344681

-- Define a function to check if a number has at least two digits
def hasAtLeastTwoDigits (n : ℕ) : Prop := n ≥ 10

-- Define the theorem
theorem division_by_240_property (a b : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (hta : hasAtLeastTwoDigits a) (htb : hasAtLeastTwoDigits b) 
  (hab : a > b) : 
  (∃ k : ℕ, a^4 - b^4 = 240 * k) ∧ 
  (∀ m : ℕ, m > 240 → ¬(∀ x y : ℕ, Nat.Prime x → Nat.Prime y → hasAtLeastTwoDigits x → hasAtLeastTwoDigits y → x > y → ∃ l : ℕ, x^4 - y^4 = m * l)) :=
by sorry


end NUMINAMATH_CALUDE_division_by_240_property_l3446_344681


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3446_344689

theorem partial_fraction_decomposition_product : 
  let f (x : ℝ) := (x^2 + 5*x - 14) / (x^3 + x^2 - 11*x - 13)
  let g (x : ℝ) (A B C : ℝ) := A / (x - 1) + B / (x + 1) + C / (x + 13)
  ∀ A B C : ℝ, (∀ x : ℝ, f x = g x A B C) → A * B * C = -360 / 343 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_product_l3446_344689


namespace NUMINAMATH_CALUDE_complement_of_60_18_l3446_344690

/-- Represents an angle in degrees and minutes -/
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

/-- Calculates the complement of an angle -/
def complement (α : Angle) : Angle :=
  let total_minutes := 90 * 60 - (α.degrees * 60 + α.minutes)
  ⟨total_minutes / 60, total_minutes % 60⟩

theorem complement_of_60_18 :
  let α : Angle := ⟨60, 18⟩
  complement α = ⟨29, 42⟩ := by
  sorry

end NUMINAMATH_CALUDE_complement_of_60_18_l3446_344690


namespace NUMINAMATH_CALUDE_m_range_l3446_344684

def p (m : ℝ) : Prop := ∀ x : ℝ, |x| + |x - 1| > m

def q (m : ℝ) : Prop := ∀ x y : ℝ, x < y → (-(7 - 3*m))^x > (-(7 - 3*m))^y

theorem m_range : 
  (∃ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m)) ∧ 
  (∀ m : ℝ, (p m ∧ ¬q m) ∨ (¬p m ∧ q m) → m ∈ Set.Icc 1 2) ∧
  (∀ m : ℝ, m ∈ Set.Icc 1 2 → (p m ∧ ¬q m) ∨ (¬p m ∧ q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l3446_344684


namespace NUMINAMATH_CALUDE_cubic_sum_given_sum_and_product_l3446_344638

theorem cubic_sum_given_sum_and_product (x y : ℝ) 
  (h1 : x + y = 10) (h2 : x * y = 16) : x^3 + y^3 = 520 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_given_sum_and_product_l3446_344638


namespace NUMINAMATH_CALUDE_point_on_linear_graph_l3446_344640

theorem point_on_linear_graph (a : ℝ) : (1 : ℝ) = 3 * a + 4 → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_point_on_linear_graph_l3446_344640


namespace NUMINAMATH_CALUDE_geometric_sequence_17th_term_l3446_344666

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_17th_term
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_5th : a 5 = 9)
  (h_13th : a 13 = 1152) :
  a 17 = 36864 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_17th_term_l3446_344666


namespace NUMINAMATH_CALUDE_sum_of_squares_zero_l3446_344611

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

/-- Given vectors a, b, c form a basis in space V, and real numbers x, y, z 
    satisfy the equation x*a + y*b + z*c = 0, then x^2 + y^2 + z^2 = 0 -/
theorem sum_of_squares_zero (a b c : V) (x y z : ℝ) 
  (h_basis : LinearIndependent ℝ ![a, b, c]) 
  (h_eq : x • a + y • b + z • c = 0) : 
  x^2 + y^2 + z^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_zero_l3446_344611


namespace NUMINAMATH_CALUDE_percentage_problem_l3446_344685

theorem percentage_problem (x : ℝ) : (35 / 100) * x = 126 → x = 360 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3446_344685


namespace NUMINAMATH_CALUDE_max_remainder_eleven_l3446_344617

theorem max_remainder_eleven (y : ℕ+) : ∃ (q r : ℕ), y = 11 * q + r ∧ r < 11 ∧ r ≤ 10 :=
sorry

end NUMINAMATH_CALUDE_max_remainder_eleven_l3446_344617


namespace NUMINAMATH_CALUDE_julies_landscaping_hours_l3446_344643

/-- Julie's landscaping business problem -/
theorem julies_landscaping_hours (mowing_rate pulling_rate pulling_hours total_earnings : ℕ) :
  mowing_rate = 4 →
  pulling_rate = 8 →
  pulling_hours = 3 →
  total_earnings = 248 →
  ∃ (mowing_hours : ℕ),
    2 * (mowing_rate * mowing_hours + pulling_rate * pulling_hours) = total_earnings ∧
    mowing_hours = 25 :=
by sorry

end NUMINAMATH_CALUDE_julies_landscaping_hours_l3446_344643


namespace NUMINAMATH_CALUDE_correct_missile_sampling_l3446_344648

/-- Represents a systematic sampling of missiles -/
structure MissileSampling where
  total : ℕ
  sample_size : ℕ
  first : ℕ
  interval : ℕ

/-- Generates the sequence of sampled missile numbers -/
def generate_sequence (ms : MissileSampling) : List ℕ :=
  List.range ms.sample_size |>.map (λ i => ms.first + i * ms.interval)

/-- Checks if all elements in the list are within the valid range -/
def valid_range (l : List ℕ) (max : ℕ) : Prop :=
  l.all (λ x => x > 0 ∧ x ≤ max)

theorem correct_missile_sampling :
  let ms : MissileSampling := {
    total := 60,
    sample_size := 6,
    first := 3,
    interval := 10
  }
  let sequence := generate_sequence ms
  (sequence = [3, 13, 23, 33, 43, 53]) ∧
  (ms.interval = ms.total / ms.sample_size) ∧
  (valid_range sequence ms.total) :=
by sorry

end NUMINAMATH_CALUDE_correct_missile_sampling_l3446_344648


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l3446_344600

/-- A circle passing through three given points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Check if a point lies on a circle -/
def Circle.contains (c : Circle) (p : ℝ × ℝ) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 = c.radius^2

/-- Check if a point lies on a line given by x - y + a = 0 -/
def on_line (p : ℝ × ℝ) (a : ℝ) : Prop :=
  p.1 - p.2 + a = 0

/-- Check if two points are perpendicular with respect to the origin -/
def perpendicular (p1 p2 : ℝ × ℝ) : Prop :=
  p1.1 * p2.1 + p1.2 * p2.2 = 0

theorem circle_intersection_theorem (c : Circle) (a : ℝ) :
  c.contains (0, 1) ∧ c.contains (3, 4) ∧ c.contains (6, 1) →
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    c.contains A ∧ c.contains B ∧ 
    on_line A a ∧ on_line B a ∧
    perpendicular A B) →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l3446_344600


namespace NUMINAMATH_CALUDE_reading_plan_theorem_l3446_344601

/-- Represents a book with a given number of pages --/
structure Book where
  pages : ℕ

/-- Represents Mrs. Hilt's reading plan --/
structure ReadingPlan where
  book1 : Book
  book2 : Book
  book3 : Book
  firstTwoDaysBook1Percent : ℝ
  firstTwoDaysBook2Percent : ℝ
  day3And4Book1Fraction : ℝ
  day3And4Book2Fraction : ℝ
  day3And4Book3Percent : ℝ
  readingRate : ℕ  -- pages per hour

def calculateRemainingPages (plan : ReadingPlan) : ℕ :=
  sorry

def calculateAverageSpeedFirstFourDays (plan : ReadingPlan) : ℝ :=
  sorry

def calculateTotalReadingHours (plan : ReadingPlan) : ℕ :=
  sorry

theorem reading_plan_theorem (plan : ReadingPlan) 
  (h1 : plan.book1.pages = 457)
  (h2 : plan.book2.pages = 336)
  (h3 : plan.book3.pages = 520)
  (h4 : plan.firstTwoDaysBook1Percent = 0.35)
  (h5 : plan.firstTwoDaysBook2Percent = 0.25)
  (h6 : plan.day3And4Book1Fraction = 1/3)
  (h7 : plan.day3And4Book2Fraction = 1/2)
  (h8 : plan.day3And4Book3Percent = 0.10)
  (h9 : plan.readingRate = 50) :
  calculateRemainingPages plan = 792 ∧
  calculateAverageSpeedFirstFourDays plan = 130.25 ∧
  calculateTotalReadingHours plan = 27 :=
sorry

end NUMINAMATH_CALUDE_reading_plan_theorem_l3446_344601


namespace NUMINAMATH_CALUDE_puzzle_pieces_count_l3446_344609

theorem puzzle_pieces_count :
  let border_pieces : ℕ := 75
  let trevor_pieces : ℕ := 105
  let joe_pieces : ℕ := 3 * trevor_pieces
  let missing_pieces : ℕ := 5
  let total_pieces : ℕ := border_pieces + trevor_pieces + joe_pieces + missing_pieces
  total_pieces = 500 := by
sorry

end NUMINAMATH_CALUDE_puzzle_pieces_count_l3446_344609


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l3446_344656

theorem unique_solution_quadratic (k : ℝ) : 
  (∃! x : ℝ, (x + 6) * (x + 3) = k + 3 * x) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l3446_344656


namespace NUMINAMATH_CALUDE_jack_buttons_theorem_l3446_344660

/-- The number of buttons Jack needs for all shirts -/
def total_buttons (shirts_per_kid : ℕ) (num_kids : ℕ) (buttons_per_shirt : ℕ) : ℕ :=
  shirts_per_kid * num_kids * buttons_per_shirt

/-- Theorem stating that Jack needs 63 buttons for all shirts -/
theorem jack_buttons_theorem :
  total_buttons 3 3 7 = 63 := by
  sorry

end NUMINAMATH_CALUDE_jack_buttons_theorem_l3446_344660


namespace NUMINAMATH_CALUDE_train_length_calculation_l3446_344608

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to pass the person completely. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (time_to_cross : ℝ) : 
  train_speed = 63 →
  man_speed = 3 →
  time_to_cross = 29.997600191984642 →
  (train_speed - man_speed) * time_to_cross * (1000 / 3600) = 500 := by
  sorry

#check train_length_calculation

end NUMINAMATH_CALUDE_train_length_calculation_l3446_344608


namespace NUMINAMATH_CALUDE_andrew_work_hours_l3446_344647

theorem andrew_work_hours : 
  let day1 : ℝ := 1.5
  let day2 : ℝ := 2.75
  let day3 : ℝ := 3.25
  day1 + day2 + day3 = 7.5 := by
sorry

end NUMINAMATH_CALUDE_andrew_work_hours_l3446_344647


namespace NUMINAMATH_CALUDE_stratified_sample_over_45_l3446_344602

/-- Represents the number of employees in a stratified sample from a given population -/
def stratified_sample (total_population : ℕ) (group_size : ℕ) (sample_size : ℕ) : ℕ :=
  (group_size * sample_size) / total_population

/-- Proves that the number of employees over 45 in the stratified sample is 10 -/
theorem stratified_sample_over_45 :
  stratified_sample 200 80 25 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sample_over_45_l3446_344602


namespace NUMINAMATH_CALUDE_stone_150_is_6_l3446_344623

/-- Represents the counting pattern described in the problem -/
def stone_count (n : ℕ) : ℕ := 
  if n ≤ 12 then n 
  else if n ≤ 23 then 24 - n 
  else stone_count ((n - 1) % 22 + 1)

/-- The total number of stones -/
def total_stones : ℕ := 12

/-- The number at which we want to find the corresponding stone -/
def target_count : ℕ := 150

/-- Theorem stating that the stone counted as 150 is originally stone number 6 -/
theorem stone_150_is_6 : 
  ∃ (k : ℕ), k ≤ total_stones ∧ stone_count target_count = stone_count k ∧ k = 6 := by
  sorry

end NUMINAMATH_CALUDE_stone_150_is_6_l3446_344623


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l3446_344632

theorem simplify_and_rationalize : 
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 9) * (Real.sqrt 6 / Real.sqrt 8) = Real.sqrt 35 / 14 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l3446_344632


namespace NUMINAMATH_CALUDE_dividend_calculation_l3446_344645

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 20)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 725 := by
  sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3446_344645


namespace NUMINAMATH_CALUDE_base9_multiplication_addition_l3446_344679

/-- Converts a base-9 number represented as a list of digits to a natural number. -/
def base9ToNat (digits : List Nat) : Nat :=
  digits.foldr (fun d acc => acc * 9 + d) 0

/-- Converts a natural number to its base-9 representation as a list of digits. -/
def natToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- The main theorem to be proved -/
theorem base9_multiplication_addition :
  (base9ToNat [3, 2, 4] * base9ToNat [4, 6, 7]) + base9ToNat [1, 2, 3] =
  base9ToNat [2, 3, 4, 4, 2] := by
  sorry

#eval natToBase9 ((base9ToNat [3, 2, 4] * base9ToNat [4, 6, 7]) + base9ToNat [1, 2, 3])

end NUMINAMATH_CALUDE_base9_multiplication_addition_l3446_344679


namespace NUMINAMATH_CALUDE_smallest_value_absolute_equation_l3446_344619

theorem smallest_value_absolute_equation :
  (∃ x : ℝ, |x - 8| = 15) ∧
  (∀ x : ℝ, |x - 8| = 15 → x ≥ -7) ∧
  |-7 - 8| = 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_value_absolute_equation_l3446_344619


namespace NUMINAMATH_CALUDE_felix_axe_sharpening_l3446_344646

def axe_sharpening_problem (sharpening_cost : ℕ) (total_spent : ℕ) (trees_chopped : ℕ) : Prop :=
  ∃ (trees_per_sharpening : ℕ),
    sharpening_cost > 0 ∧
    total_spent > 0 ∧
    trees_chopped ≥ 91 ∧
    total_spent / sharpening_cost * trees_per_sharpening ≥ trees_chopped ∧
    trees_per_sharpening = 13

theorem felix_axe_sharpening :
  axe_sharpening_problem 5 35 91 :=
sorry

end NUMINAMATH_CALUDE_felix_axe_sharpening_l3446_344646


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l3446_344614

theorem circle_diameter_from_area (r : ℝ) (h : π * r^2 = 4 * π) : 2 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l3446_344614


namespace NUMINAMATH_CALUDE_rico_justin_dog_difference_l3446_344621

theorem rico_justin_dog_difference (justin_dogs : ℕ) (camden_dog_legs : ℕ) (camden_rico_ratio : ℚ) :
  justin_dogs = 14 →
  camden_dog_legs = 72 →
  camden_rico_ratio = 3/4 →
  ∃ (rico_dogs : ℕ), rico_dogs - justin_dogs = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_rico_justin_dog_difference_l3446_344621


namespace NUMINAMATH_CALUDE_unique_factor_pair_l3446_344687

theorem unique_factor_pair : ∃! (x y : ℕ), 
  x > 0 ∧ y > 0 ∧ y ≥ x ∧ x + y ≤ 20 ∧ 
  (∃ (a b : ℕ), a ≠ x ∧ b ≠ y ∧ a * b = x * y) ∧
  (∀ (a b : ℕ), a > 0 → b > 0 → b ≥ a → a + b ≤ 20 → a * b ≠ x * y ∨ a + b ≠ 13) ∧
  x = 2 ∧ y = 11 := by
sorry

end NUMINAMATH_CALUDE_unique_factor_pair_l3446_344687


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3446_344633

theorem fraction_to_decimal : (3 : ℚ) / 40 = 0.075 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3446_344633


namespace NUMINAMATH_CALUDE_sticks_difference_l3446_344672

theorem sticks_difference (picked_up left : ℕ) 
  (h1 : picked_up = 14)
  (h2 : left = 4) :
  picked_up - left = 10 := by
  sorry

end NUMINAMATH_CALUDE_sticks_difference_l3446_344672


namespace NUMINAMATH_CALUDE_no_prime_solution_l3446_344605

theorem no_prime_solution : ¬∃ p : ℕ, Nat.Prime p ∧ p^3 + 6*p^2 + 4*p + 28 = 6*p^2 + 17*p + 6 := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3446_344605


namespace NUMINAMATH_CALUDE_journey_remaining_distance_l3446_344694

/-- Represents a journey with two stopovers and a final destination -/
structure Journey where
  total_distance : ℕ
  first_stopover : ℕ
  second_stopover : ℕ

/-- Calculates the remaining distance to the destination after the second stopover -/
def remaining_distance (j : Journey) : ℕ :=
  j.total_distance - (j.first_stopover + j.second_stopover)

/-- Theorem: For the given journey, the remaining distance is 68 miles -/
theorem journey_remaining_distance :
  let j : Journey := {
    total_distance := 436,
    first_stopover := 132,
    second_stopover := 236
  }
  remaining_distance j = 68 := by
  sorry

end NUMINAMATH_CALUDE_journey_remaining_distance_l3446_344694


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3446_344624

theorem rectangular_field_width (width : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = (7 / 5) * width →
  perimeter = 240 →
  2 * length + 2 * width = perimeter →
  width = 50 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3446_344624


namespace NUMINAMATH_CALUDE_bread_in_pond_l3446_344692

/-- Proves that the total number of bread pieces thrown in a pond is 100 given the specified conditions --/
theorem bread_in_pond (duck1_half : ℕ → ℕ) (duck2_pieces duck3_pieces left_in_water : ℕ) : 
  duck1_half = (λ x => x / 2) ∧ 
  duck2_pieces = 13 ∧ 
  duck3_pieces = 7 ∧ 
  left_in_water = 30 → 
  ∃ total : ℕ, 
    total = 100 ∧ 
    duck1_half total + duck2_pieces + duck3_pieces + left_in_water = total :=
by
  sorry

end NUMINAMATH_CALUDE_bread_in_pond_l3446_344692


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3446_344652

theorem basketball_free_throws (two_pointers three_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * two_pointers) →
  (free_throws = three_pointers) →
  (2 * two_pointers + 3 * three_pointers + free_throws = 73) →
  free_throws = 10 := by
sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3446_344652


namespace NUMINAMATH_CALUDE_sarah_initial_cupcakes_l3446_344667

theorem sarah_initial_cupcakes :
  ∀ (initial_cupcakes : ℕ),
    (initial_cupcakes / 3 : ℚ) + 5 = 11 - (2 * initial_cupcakes / 3 : ℚ) →
    initial_cupcakes = 9 := by
  sorry

end NUMINAMATH_CALUDE_sarah_initial_cupcakes_l3446_344667


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l3446_344655

/-- Proves that a man's rowing speed in still water is 15 kmph given the conditions of downstream rowing --/
theorem mans_rowing_speed (current_speed : ℝ) (distance : ℝ) (time : ℝ) : 
  current_speed = 3 →
  distance = 70 →
  time = 13.998880089592832 →
  (distance / time - current_speed * 1000 / 3600) * 3.6 = 15 := by
  sorry

end NUMINAMATH_CALUDE_mans_rowing_speed_l3446_344655


namespace NUMINAMATH_CALUDE_smallest_product_in_set_l3446_344612

def S : Set Int := {-10, -8, -3, 0, 4, 6}

theorem smallest_product_in_set (a b : Int) (ha : a ∈ S) (hb : b ∈ S) :
  ∃ (x y : Int), x ∈ S ∧ y ∈ S ∧ x * y ≤ a * b ∧ x * y = -60 := by
  sorry

end NUMINAMATH_CALUDE_smallest_product_in_set_l3446_344612


namespace NUMINAMATH_CALUDE_sine_intersection_theorem_l3446_344688

def M : Set ℝ := { y | ∃ x, y = Real.sin x }
def N : Set ℝ := {0, 1, 2}

theorem sine_intersection_theorem : M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_sine_intersection_theorem_l3446_344688


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3446_344607

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 9 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3446_344607


namespace NUMINAMATH_CALUDE_unique_k_for_quadratic_equation_l3446_344636

theorem unique_k_for_quadratic_equation : ∃! k : ℝ, k ≠ 0 ∧
  (∃! a : ℝ, a ≠ 0 ∧
    (∃! x : ℝ, x^2 - (a^3 + 1/a^3) * x + k = 0)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_k_for_quadratic_equation_l3446_344636


namespace NUMINAMATH_CALUDE_smallest_a1_l3446_344641

/-- Given a sequence of positive real numbers where aₙ = 8aₙ₋₁ - n² for all n > 1,
    the smallest possible value of a₁ is 2/7 -/
theorem smallest_a1 (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
    (h_rec : ∀ n > 1, a n = 8 * a (n - 1) - n^2) :
  ∀ a₁ > 0, (∀ n > 1, a n = 8 * a (n - 1) - n^2) → a₁ ≥ 2/7 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a1_l3446_344641


namespace NUMINAMATH_CALUDE_angle_value_l3446_344670

theorem angle_value (a : ℝ) : 
  (180 - a = 3 * (90 - a)) → a = 45 := by sorry

end NUMINAMATH_CALUDE_angle_value_l3446_344670


namespace NUMINAMATH_CALUDE_everton_college_order_l3446_344654

/-- The number of scientific calculators ordered by Everton college -/
def scientific_calculators : ℕ := 20

/-- The number of graphing calculators ordered by Everton college -/
def graphing_calculators : ℕ := 45 - scientific_calculators

/-- The cost of a single scientific calculator -/
def scientific_calculator_cost : ℕ := 10

/-- The cost of a single graphing calculator -/
def graphing_calculator_cost : ℕ := 57

/-- The total cost of the order -/
def total_cost : ℕ := 1625

/-- The total number of calculators ordered -/
def total_calculators : ℕ := 45

theorem everton_college_order :
  scientific_calculators * scientific_calculator_cost +
  graphing_calculators * graphing_calculator_cost = total_cost ∧
  scientific_calculators + graphing_calculators = total_calculators :=
sorry

end NUMINAMATH_CALUDE_everton_college_order_l3446_344654


namespace NUMINAMATH_CALUDE_first_super_monday_l3446_344653

/-- Represents a date with a month and day -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 3 => 31  -- March
  | 4 => 30  -- April
  | 5 => 31  -- May
  | _ => 30  -- Default (not used in this problem)

/-- Checks if a given date is a Monday -/
def isMonday (date : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Counts the number of Mondays in a given month -/
def countMondaysInMonth (month : Nat) (startDate : Date) (startDay : DayOfWeek) : Nat :=
  sorry

/-- Finds the date of the fifth Monday in a given month -/
def fifthMondayInMonth (month : Nat) (startDate : Date) (startDay : DayOfWeek) : Option Date :=
  sorry

/-- Theorem: The first Super Monday after school starts on Tuesday, March 1 is May 30 -/
theorem first_super_monday :
  let schoolStart : Date := ⟨3, 1⟩
  let firstSuperMonday := fifthMondayInMonth 5 schoolStart DayOfWeek.Tuesday
  firstSuperMonday = some ⟨5, 30⟩ :=
by
  sorry

#check first_super_monday

end NUMINAMATH_CALUDE_first_super_monday_l3446_344653


namespace NUMINAMATH_CALUDE_room_width_proof_l3446_344699

theorem room_width_proof (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) (width : ℝ) : 
  length = 5.5 → 
  cost_per_sqm = 750 → 
  total_cost = 16500 → 
  width * length * cost_per_sqm = total_cost → 
  width = 4 := by
sorry

end NUMINAMATH_CALUDE_room_width_proof_l3446_344699


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l3446_344606

theorem triangle_two_solutions (a b : ℝ) (A : ℝ) :
  a = 6 →
  b = 6 * Real.sqrt 3 →
  A = π / 6 →
  ∃! (c₁ c₂ B₁ B₂ C₁ C₂ : ℝ),
    (c₁ = 12 ∧ B₁ = π / 3 ∧ C₁ = π / 2) ∧
    (c₂ = 6 ∧ B₂ = 2 * π / 3 ∧ C₂ = π / 6) ∧
    (∀ c B C : ℝ,
      (c = c₁ ∧ B = B₁ ∧ C = C₁) ∨
      (c = c₂ ∧ B = B₂ ∧ C = C₂)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l3446_344606


namespace NUMINAMATH_CALUDE_michaels_book_purchase_l3446_344610

theorem michaels_book_purchase (m : ℚ) : 
  (∃ (n : ℚ), (1 / 3 : ℚ) * m = (1 / 2 : ℚ) * n * ((1 / 3 : ℚ) * m / ((1 / 2 : ℚ) * n))) →
  (5 : ℚ) = (1 / 15 : ℚ) * m →
  m - ((2 / 3 : ℚ) * m + (1 / 15 : ℚ) * m) = (4 / 15 : ℚ) * m :=
by sorry

end NUMINAMATH_CALUDE_michaels_book_purchase_l3446_344610


namespace NUMINAMATH_CALUDE_log_56342_between_consecutive_integers_l3446_344603

theorem log_56342_between_consecutive_integers :
  ∃ (c d : ℕ), c + 1 = d ∧ (c : ℝ) < Real.log 56342 / Real.log 10 ∧ Real.log 56342 / Real.log 10 < d ∧ c + d = 9 :=
by
  -- Assuming 10000 < 56342 < 100000
  have h1 : 10000 < 56342 := by sorry
  have h2 : 56342 < 100000 := by sorry
  sorry

end NUMINAMATH_CALUDE_log_56342_between_consecutive_integers_l3446_344603


namespace NUMINAMATH_CALUDE_bruno_books_l3446_344635

theorem bruno_books (initial_books : ℝ) : 
  initial_books - 4.5 + 10.25 = 39.75 → initial_books = 34 := by
sorry

end NUMINAMATH_CALUDE_bruno_books_l3446_344635


namespace NUMINAMATH_CALUDE_orange_cost_proof_l3446_344631

/-- Given that 5 dozen oranges cost $39.00, prove that 8 dozen oranges at the same rate cost $62.40 -/
theorem orange_cost_proof (cost_five_dozen : ℝ) (h1 : cost_five_dozen = 39) :
  let cost_per_dozen : ℝ := cost_five_dozen / 5
  let cost_eight_dozen : ℝ := 8 * cost_per_dozen
  cost_eight_dozen = 62.4 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_proof_l3446_344631


namespace NUMINAMATH_CALUDE_function_from_derivative_and_point_l3446_344628

open Real

theorem function_from_derivative_and_point (f : ℝ → ℝ) 
  (h1 : ∀ x, HasDerivAt f (4 * x^3) x) 
  (h2 : f 1 = -1) : 
  ∀ x, f x = x^4 - 2 := by
  sorry

end NUMINAMATH_CALUDE_function_from_derivative_and_point_l3446_344628


namespace NUMINAMATH_CALUDE_exist_six_numbers_l3446_344644

theorem exist_six_numbers : ∃ (a b c d e f : ℕ), 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ 
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ 
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ 
  d ≠ e ∧ d ≠ f ∧ 
  e ≠ f ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  (a + b + c + d + e + f : ℚ) / (1 / a + 1 / b + 1 / c + 1 / d + 1 / e + 1 / f) = 2012 := by
  sorry

end NUMINAMATH_CALUDE_exist_six_numbers_l3446_344644


namespace NUMINAMATH_CALUDE_least_positive_k_for_equation_l3446_344657

theorem least_positive_k_for_equation : ∃ (k : ℕ), 
  (k > 0) ∧ 
  (∃ (x : ℤ), x > 0 ∧ x + 6 + 8*k = k*(x + 8)) ∧
  (∀ (j : ℕ), 0 < j ∧ j < k → ¬∃ (y : ℤ), y > 0 ∧ y + 6 + 8*j = j*(y + 8)) ∧
  k = 2 := by
sorry

end NUMINAMATH_CALUDE_least_positive_k_for_equation_l3446_344657


namespace NUMINAMATH_CALUDE_percentage_increase_l3446_344625

theorem percentage_increase (original : ℝ) (new : ℝ) :
  original = 30 →
  new = 40 →
  (new - original) / original = 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_percentage_increase_l3446_344625


namespace NUMINAMATH_CALUDE_wheel_probability_l3446_344693

theorem wheel_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_D = 1/6 → p_A + p_B + p_C + p_D = 1 → p_C = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_wheel_probability_l3446_344693


namespace NUMINAMATH_CALUDE_ceiling_x_squared_values_l3446_344659

theorem ceiling_x_squared_values (x : ℝ) (h : ⌈x⌉ = 9) :
  ∃ (S : Finset ℕ), (∀ n ∈ S, ∃ y : ℝ, ⌈y⌉ = 9 ∧ ⌈y^2⌉ = n) ∧ S.card = 17 :=
sorry

end NUMINAMATH_CALUDE_ceiling_x_squared_values_l3446_344659


namespace NUMINAMATH_CALUDE_complement_intersection_empty_l3446_344671

def I : Set Char := {'a', 'b', 'c', 'd', 'e'}
def M : Set Char := {'a', 'c', 'd'}
def N : Set Char := {'b', 'd', 'e'}

theorem complement_intersection_empty :
  (I \ M) ∩ (I \ N) = ∅ :=
by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_empty_l3446_344671


namespace NUMINAMATH_CALUDE_no_real_solutions_iff_k_in_range_l3446_344663

theorem no_real_solutions_iff_k_in_range (k : ℝ) :
  (∀ x : ℝ, k * x^2 + Real.sqrt 2 * k * x + 2 ≥ 0) ↔ k ∈ Set.Icc 0 4 :=
by sorry

end NUMINAMATH_CALUDE_no_real_solutions_iff_k_in_range_l3446_344663


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l3446_344682

theorem max_value_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) : 
  Real.sqrt (49 + x) + Real.sqrt (49 - x) ≤ 14 ∧ 
  ∃ y, -49 ≤ y ∧ y ≤ 49 ∧ Real.sqrt (49 + y) + Real.sqrt (49 - y) = 14 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l3446_344682


namespace NUMINAMATH_CALUDE_earthwork_transport_theorem_prove_earthwork_transport_l3446_344626

/-- Represents the capacity of earthwork transport vehicles -/
structure VehicleCapacity where
  large : ℕ
  small : ℕ

/-- Represents a dispatch plan for earthwork transport vehicles -/
structure DispatchPlan where
  large : ℕ
  small : ℕ

/-- Theorem stating the correct vehicle capacities and possible dispatch plans -/
theorem earthwork_transport_theorem 
  (capacity : VehicleCapacity)
  (plans : List DispatchPlan) : Prop :=
  -- Conditions
  (3 * capacity.large + 4 * capacity.small = 44) ∧
  (4 * capacity.large + 6 * capacity.small = 62) ∧
  (∀ plan ∈ plans, 
    plan.large + plan.small = 12 ∧
    plan.small ≥ 4 ∧
    plan.large * capacity.large + plan.small * capacity.small ≥ 78) ∧
  -- Conclusions
  (capacity.large = 8 ∧ capacity.small = 5) ∧
  (plans = [
    DispatchPlan.mk 8 4,
    DispatchPlan.mk 7 5,
    DispatchPlan.mk 6 6
  ])

/-- Proof of the earthwork transport theorem -/
theorem prove_earthwork_transport : 
  ∃ (capacity : VehicleCapacity) (plans : List DispatchPlan),
    earthwork_transport_theorem capacity plans := by
  sorry

end NUMINAMATH_CALUDE_earthwork_transport_theorem_prove_earthwork_transport_l3446_344626


namespace NUMINAMATH_CALUDE_evaluate_expression_l3446_344664

theorem evaluate_expression (x y z w : ℚ) 
  (hx : x = 1/4) 
  (hy : y = 1/3) 
  (hz : z = 12) 
  (hw : w = -2) : 
  x^2 * y^3 * z * w = -1/18 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3446_344664


namespace NUMINAMATH_CALUDE_houses_with_pool_l3446_344675

theorem houses_with_pool (total : ℕ) (garage : ℕ) (neither : ℕ) (both : ℕ) 
  (h_total : total = 85)
  (h_garage : garage = 50)
  (h_neither : neither = 30)
  (h_both : both = 35) :
  ∃ pool : ℕ, pool = 40 ∧ total = (garage - both) + (pool - both) + both + neither :=
by sorry

end NUMINAMATH_CALUDE_houses_with_pool_l3446_344675


namespace NUMINAMATH_CALUDE_sphere_volume_from_tetrahedron_surface_l3446_344674

theorem sphere_volume_from_tetrahedron_surface (s : ℝ) (V : ℝ) : 
  s = 3 →
  (4 * π * (V / ((4/3) * π))^((1:ℝ)/3)^2) = (4 * s^2 * Real.sqrt 3) →
  V = (27 * Real.sqrt 2) / Real.sqrt π :=
by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_from_tetrahedron_surface_l3446_344674


namespace NUMINAMATH_CALUDE_min_students_l3446_344629

/-- Represents the number of students in each income group -/
structure IncomeGroups where
  low : ℕ
  middle : ℕ
  high : ℕ

/-- Represents the lowest salary in each income range -/
structure LowestSalaries where
  low : ℝ
  middle : ℝ
  high : ℝ

/-- Represents the median salary in each income range -/
def medianSalaries (lowest : LowestSalaries) : LowestSalaries :=
  { low := lowest.low + 50000
  , middle := lowest.middle + 40000
  , high := lowest.high + 30000 }

/-- The conditions of the problem -/
structure GraduatingClass where
  groups : IncomeGroups
  lowest : LowestSalaries
  salary_range : ℝ
  average_salary : ℝ
  median_salary : ℝ
  (high_twice_low : groups.high = 2 * groups.low)
  (middle_sum_others : groups.middle = groups.low + groups.high)
  (salary_range_constant : ∀ (r : LowestSalaries → ℝ), r (medianSalaries lowest) - r lowest = salary_range)
  (average_above_median : average_salary = median_salary + 20000)

/-- The theorem to prove -/
theorem min_students (c : GraduatingClass) : 
  c.groups.low + c.groups.middle + c.groups.high ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_students_l3446_344629


namespace NUMINAMATH_CALUDE_pants_and_belt_price_difference_l3446_344650

def price_difference (total_cost pants_price : ℝ) : ℝ :=
  total_cost - 2 * pants_price

theorem pants_and_belt_price_difference :
  let total_cost : ℝ := 70.93
  let pants_price : ℝ := 34
  pants_price < (total_cost - pants_price) →
  price_difference total_cost pants_price = 2.93 := by
sorry

end NUMINAMATH_CALUDE_pants_and_belt_price_difference_l3446_344650


namespace NUMINAMATH_CALUDE_max_leftover_grapes_l3446_344680

theorem max_leftover_grapes :
  ∀ n : ℕ, ∃ q r : ℕ, n = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_leftover_grapes_l3446_344680


namespace NUMINAMATH_CALUDE_fraction_multiplication_l3446_344649

theorem fraction_multiplication (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) :
  (5*x + 5*y) / ((5*x) * (5*y)) = (1 / 5) * ((x + y) / (x * y)) :=
by sorry

end NUMINAMATH_CALUDE_fraction_multiplication_l3446_344649


namespace NUMINAMATH_CALUDE_ralph_square_matchsticks_ralph_uses_eight_matchsticks_per_square_l3446_344615

theorem ralph_square_matchsticks (total_matchsticks : ℕ) (elvis_matchsticks_per_square : ℕ) 
  (elvis_squares : ℕ) (ralph_squares : ℕ) (matchsticks_left : ℕ) : ℕ :=
  let elvis_total_matchsticks := elvis_matchsticks_per_square * elvis_squares
  let total_used_matchsticks := total_matchsticks - matchsticks_left
  let ralph_total_matchsticks := total_used_matchsticks - elvis_total_matchsticks
  ralph_total_matchsticks / ralph_squares

theorem ralph_uses_eight_matchsticks_per_square :
  ralph_square_matchsticks 50 4 5 3 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ralph_square_matchsticks_ralph_uses_eight_matchsticks_per_square_l3446_344615


namespace NUMINAMATH_CALUDE_sequence_inequality_l3446_344630

theorem sequence_inequality (a : ℕ → ℝ) (h1 : ∀ n, a n ≥ 0) 
  (h2 : ∀ m n, a (m + n) ≤ a m + a n) :
  ∀ m n, m ≤ n → a n ≤ m * a 1 + (n / m - 1) * a m :=
by sorry

end NUMINAMATH_CALUDE_sequence_inequality_l3446_344630


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l3446_344697

theorem quadratic_root_condition (a b : ℝ) : 
  a > 0 → 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x - 1 = 0 ∧ a * y^2 + b * y - 1 = 0) →
  (∃ r : ℝ, 1 < r ∧ r < 2 ∧ a * r^2 + b * r - 1 = 0) →
  ∀ z : ℝ, z > -1 → ∃ a' b' : ℝ, a' - b' = z ∧ 
    a' > 0 ∧
    (∃ x y : ℝ, x ≠ y ∧ a' * x^2 + b' * x - 1 = 0 ∧ a' * y^2 + b' * y - 1 = 0) ∧
    (∃ r : ℝ, 1 < r ∧ r < 2 ∧ a' * r^2 + b' * r - 1 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l3446_344697


namespace NUMINAMATH_CALUDE_quadratic_function_proof_l3446_344662

/-- A quadratic function with specific properties -/
def f (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 11

theorem quadratic_function_proof :
  (∀ x : ℝ, f x ≤ 13) ∧  -- maximum value is 13
  f 3 = 5 ∧              -- f(3) = 5
  f (-1) = 5 ∧           -- f(-1) = 5
  (∀ x : ℝ, f x = -2 * x^2 + 4 * x + 11) -- explicit formula
  :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_proof_l3446_344662


namespace NUMINAMATH_CALUDE_int_endomorphisms_characterization_l3446_344627

/-- An endomorphism of the additive group of integers -/
def IntEndomorphism : Type := ℤ → ℤ

/-- The homomorphism property for integer endomorphisms -/
def IsHomomorphism (φ : IntEndomorphism) : Prop :=
  ∀ a b : ℤ, φ (a + b) = φ a + φ b

/-- The set of all endomorphisms of the additive group of integers -/
def IntEndomorphisms : Set IntEndomorphism :=
  {φ : IntEndomorphism | IsHomomorphism φ}

/-- A linear function with integer coefficient -/
def LinearIntFunction (d : ℤ) : IntEndomorphism :=
  fun x => d * x

theorem int_endomorphisms_characterization :
  ∀ φ : IntEndomorphism, φ ∈ IntEndomorphisms ↔ ∃ d : ℤ, φ = LinearIntFunction d :=
by sorry

end NUMINAMATH_CALUDE_int_endomorphisms_characterization_l3446_344627


namespace NUMINAMATH_CALUDE_xy_value_l3446_344669

theorem xy_value (x y : ℝ) (h : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) - 2) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l3446_344669


namespace NUMINAMATH_CALUDE_zoo_consumption_theorem_l3446_344618

/-- Represents the daily consumption of fish for an animal -/
structure DailyConsumption where
  trout : Float
  salmon : Float

/-- Calculates the total monthly consumption for all animals -/
def totalMonthlyConsumption (animals : List DailyConsumption) (days : Nat) : Float :=
  let dailyTotal := animals.foldl (fun acc x => acc + x.trout + x.salmon) 0
  dailyTotal * days.toFloat

/-- Theorem stating the total monthly consumption for the given animals -/
theorem zoo_consumption_theorem (pb1 pb2 pb3 sl1 sl2 : DailyConsumption)
    (h1 : pb1 = { trout := 0.2, salmon := 0.4 })
    (h2 : pb2 = { trout := 0.3, salmon := 0.5 })
    (h3 : pb3 = { trout := 0.25, salmon := 0.45 })
    (h4 : sl1 = { trout := 0.1, salmon := 0.15 })
    (h5 : sl2 = { trout := 0.2, salmon := 0.25 }) :
    totalMonthlyConsumption [pb1, pb2, pb3, sl1, sl2] 30 = 84 := by
  sorry


end NUMINAMATH_CALUDE_zoo_consumption_theorem_l3446_344618
