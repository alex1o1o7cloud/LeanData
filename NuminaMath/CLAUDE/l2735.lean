import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_square_of_binomial_l2735_273518

theorem quadratic_square_of_binomial (a : ℚ) : 
  (∃ r s : ℚ, ∀ x, a * x^2 + 22 * x + 9 = (r * x + s)^2) → 
  a = 121 / 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_square_of_binomial_l2735_273518


namespace NUMINAMATH_CALUDE_factor_theorem_quadratic_l2735_273570

theorem factor_theorem_quadratic (t : ℝ) : 
  (∃ (p : ℝ → ℝ), ∀ x, 4*x^2 - 8*x + 3 = (x - t) * p x) ↔ (t = 1.5 ∨ t = 0.5) :=
by sorry

end NUMINAMATH_CALUDE_factor_theorem_quadratic_l2735_273570


namespace NUMINAMATH_CALUDE_complex_parts_of_one_plus_sqrt_three_i_l2735_273513

theorem complex_parts_of_one_plus_sqrt_three_i :
  let z : ℂ := Complex.I * (1 + Real.sqrt 3)
  Complex.re z = 0 ∧ Complex.im z = 1 + Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_complex_parts_of_one_plus_sqrt_three_i_l2735_273513


namespace NUMINAMATH_CALUDE_sampling_suitable_for_large_events_sampling_suitable_for_beijing_olympics_l2735_273565

/-- Represents a survey method -/
inductive SurveyMethod
  | Sampling
  | Comprehensive

/-- Represents a large-scale event -/
structure LargeEvent where
  name : String
  potential_viewers : ℕ

/-- Defines when a survey method is suitable for an event -/
def is_suitable_survey_method (method : SurveyMethod) (event : LargeEvent) : Prop :=
  match method with
  | SurveyMethod.Sampling => 
      event.potential_viewers > 1000000 ∧ 
      (∀ n : ℕ, n < event.potential_viewers → ∃ sample : Finset ℕ, sample.card = n)
  | SurveyMethod.Comprehensive => 
      event.potential_viewers ≤ 1000000

/-- The main theorem stating that sampling survey is suitable for large events -/
theorem sampling_suitable_for_large_events (event : LargeEvent) 
  (h1 : event.potential_viewers > 1000000) 
  (h2 : ∀ n : ℕ, n < event.potential_viewers → ∃ sample : Finset ℕ, sample.card = n) :
  is_suitable_survey_method SurveyMethod.Sampling event :=
by sorry

/-- The Beijing Winter Olympics as an instance of LargeEvent -/
def beijing_winter_olympics : LargeEvent :=
  { name := "Beijing Winter Olympics"
  , potential_viewers := 2000000000 }  -- An example large number

/-- Theorem stating that sampling survey is suitable for the Beijing Winter Olympics -/
theorem sampling_suitable_for_beijing_olympics :
  is_suitable_survey_method SurveyMethod.Sampling beijing_winter_olympics :=
by sorry

end NUMINAMATH_CALUDE_sampling_suitable_for_large_events_sampling_suitable_for_beijing_olympics_l2735_273565


namespace NUMINAMATH_CALUDE_unique_valid_square_l2735_273512

/-- A perfect square less than 100 with ones digit 5, 6, or 7 -/
def ValidSquare (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = k^2 ∧ n < 100 ∧ (n % 10 = 5 ∨ n % 10 = 6 ∨ n % 10 = 7)

/-- There is exactly one perfect square less than 100 with ones digit 5, 6, or 7 -/
theorem unique_valid_square : ∃! (n : ℕ), ValidSquare n :=
sorry

end NUMINAMATH_CALUDE_unique_valid_square_l2735_273512


namespace NUMINAMATH_CALUDE_chessboard_coverage_l2735_273597

/-- An L-shaped tetromino covers exactly 4 squares. -/
def LTetromino : ℕ := 4

/-- Represents an m × n chessboard. -/
structure Chessboard where
  m : ℕ
  n : ℕ

/-- Predicate to check if a number is divisible by 8. -/
def divisible_by_eight (x : ℕ) : Prop := ∃ k, x = 8 * k

/-- Predicate to check if a chessboard can be covered by L-shaped tetrominoes. -/
def can_cover (board : Chessboard) : Prop :=
  divisible_by_eight (board.m * board.n) ∧ board.m ≠ 1 ∧ board.n ≠ 1

theorem chessboard_coverage (board : Chessboard) :
  (∃ (tiles : ℕ), board.m * board.n = tiles * LTetromino) ↔ can_cover board :=
sorry

end NUMINAMATH_CALUDE_chessboard_coverage_l2735_273597


namespace NUMINAMATH_CALUDE_simple_interest_rate_l2735_273578

/-- 
Given a principal amount P and a time period of 10 years, 
prove that the rate percent per annum R is 12% when the simple interest 
is 6/5 of the principal amount.
-/
theorem simple_interest_rate (P : ℝ) (P_pos : P > 0) : 
  let R := (6 / 5) * 100 / 10
  let simple_interest := (P * R * 10) / 100
  simple_interest = (6 / 5) * P → R = 12 := by
  sorry

#check simple_interest_rate

end NUMINAMATH_CALUDE_simple_interest_rate_l2735_273578


namespace NUMINAMATH_CALUDE_teachers_count_correct_teachers_count_l2735_273522

theorem teachers_count (num_students : ℕ) (ticket_cost : ℕ) (total_cost : ℕ) : ℕ :=
  let total_tickets := total_cost / ticket_cost
  let num_teachers := total_tickets - num_students
  num_teachers

theorem correct_teachers_count :
  teachers_count 20 5 115 = 3 := by
  sorry

end NUMINAMATH_CALUDE_teachers_count_correct_teachers_count_l2735_273522


namespace NUMINAMATH_CALUDE_function_value_l2735_273595

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.sqrt (a * x - 1) else -x^2 - 4*x

theorem function_value (a : ℝ) : f a (f a (-2)) = 3 → a = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_function_value_l2735_273595


namespace NUMINAMATH_CALUDE_centroid_of_concave_pentagon_l2735_273543

/-- A regular pentagon -/
structure RegularPentagon where
  vertices : Fin 5 → ℝ × ℝ
  is_regular : sorry

/-- A rhombus -/
structure Rhombus where
  vertices : Fin 4 → ℝ × ℝ
  is_rhombus : sorry

/-- The centroid of a planar figure -/
def centroid (figure : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

/-- Theorem: Centroid of concave pentagonal plate -/
theorem centroid_of_concave_pentagon
  (ABCDE : RegularPentagon)
  (ABFE : Rhombus)
  (hF : F = sorry) -- F is the intersection of diagonals EC and BD
  (hABFE : sorry) -- ABFE is cut out from ABCDE
  : centroid (sorry) = F := by sorry

end NUMINAMATH_CALUDE_centroid_of_concave_pentagon_l2735_273543


namespace NUMINAMATH_CALUDE_popsicle_stick_count_l2735_273573

/-- The number of popsicle sticks Steve has -/
def steve_sticks : ℕ := 12

/-- The number of popsicle sticks Sid has -/
def sid_sticks : ℕ := 2 * steve_sticks

/-- The number of popsicle sticks Sam has -/
def sam_sticks : ℕ := 3 * sid_sticks

/-- The total number of popsicle sticks -/
def total_sticks : ℕ := steve_sticks + sid_sticks + sam_sticks

theorem popsicle_stick_count : total_sticks = 108 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_stick_count_l2735_273573


namespace NUMINAMATH_CALUDE_blue_highlighters_count_l2735_273577

/-- Given the number of highlighters in a teacher's desk, calculate the number of blue highlighters. -/
theorem blue_highlighters_count 
  (total : ℕ) 
  (pink : ℕ) 
  (yellow : ℕ) 
  (h1 : total = 11) 
  (h2 : pink = 4) 
  (h3 : yellow = 2) : 
  total - pink - yellow = 5 := by
  sorry

#check blue_highlighters_count

end NUMINAMATH_CALUDE_blue_highlighters_count_l2735_273577


namespace NUMINAMATH_CALUDE_seat_3_9_description_l2735_273521

/-- Represents a seat in a movie theater -/
structure TheaterSeat where
  row : ℕ
  seat : ℕ

/-- Interprets a pair of natural numbers as a theater seat -/
def interpretSeat (p : ℕ × ℕ) : TheaterSeat :=
  { row := p.1, seat := p.2 }

/-- Describes a theater seat as a string -/
def describeSeat (s : TheaterSeat) : String :=
  s.row.repr ++ "th row, " ++ s.seat.repr ++ "th seat"

theorem seat_3_9_description :
  describeSeat (interpretSeat (3, 9)) = "3rd row, 9th seat" :=
sorry

end NUMINAMATH_CALUDE_seat_3_9_description_l2735_273521


namespace NUMINAMATH_CALUDE_circle_center_sum_l2735_273584

/-- Given a circle with equation x^2 + y^2 = 6x + 8y + 15, 
    prove that the sum of the x and y coordinates of its center is 7. -/
theorem circle_center_sum (x y : ℝ) : 
  x^2 + y^2 = 6*x + 8*y + 15 → 
  ∃ (h k : ℝ), (∀ (x y : ℝ), (x - h)^2 + (y - k)^2 = (x^2 + y^2 - 6*x - 8*y - 15)) ∧ 
               h + k = 7 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_sum_l2735_273584


namespace NUMINAMATH_CALUDE_weight_of_replaced_person_l2735_273586

theorem weight_of_replaced_person
  (n : ℕ) -- number of people in the group
  (w_new : ℝ) -- weight of the new person
  (w_avg_increase : ℝ) -- increase in average weight
  (h1 : n = 12) -- there are 12 people initially
  (h2 : w_new = 106) -- the new person weighs 106 kg
  (h3 : w_avg_increase = 4) -- average weight increases by 4 kg
  : ∃ w_old : ℝ, w_old = 58 ∧ n * w_avg_increase = w_new - w_old :=
by sorry

end NUMINAMATH_CALUDE_weight_of_replaced_person_l2735_273586


namespace NUMINAMATH_CALUDE_alex_cookies_l2735_273533

theorem alex_cookies (alex sam : ℕ) : 
  alex = sam + 8 → 
  sam = alex / 3 → 
  alex = 12 := by
sorry

end NUMINAMATH_CALUDE_alex_cookies_l2735_273533


namespace NUMINAMATH_CALUDE_square_digit_sequence_l2735_273587

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m ^ 2

def form_number (x y : ℕ) (n : ℕ) : ℕ :=
  x * (10^(2*n+1) - 10^(n+1)) / 9 + 6 * 10^n + y * (10^n - 1) / 9

theorem square_digit_sequence (x y : ℕ) : x ≠ 0 →
  (∀ n : ℕ, n ≥ 1 → is_perfect_square (form_number x y n)) →
  ((x = 4 ∧ y = 2) ∨ (x = 9 ∧ y = 0)) :=
sorry

end NUMINAMATH_CALUDE_square_digit_sequence_l2735_273587


namespace NUMINAMATH_CALUDE_most_marbles_l2735_273552

theorem most_marbles (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 24 →
  red = total / 4 →
  blue = red + 6 →
  yellow = total - red - blue →
  blue > red ∧ blue > yellow := by
sorry

end NUMINAMATH_CALUDE_most_marbles_l2735_273552


namespace NUMINAMATH_CALUDE_no_digit_reversal_double_l2735_273568

theorem no_digit_reversal_double :
  (∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → 10 * b + a ≠ 2 * (10 * a + b)) ∧
  (∀ a b c : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 →
    100 * c + 10 * b + a ≠ 2 * (100 * a + 10 * b + c)) := by
  sorry

end NUMINAMATH_CALUDE_no_digit_reversal_double_l2735_273568


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2735_273511

theorem polygon_sides_from_angle_sum :
  ∀ n : ℕ,
  (n - 2) * 180 = 720 →
  n = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l2735_273511


namespace NUMINAMATH_CALUDE_no_solution_inequality_l2735_273592

theorem no_solution_inequality (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 5| > m) ↔ m < 6 := by sorry

end NUMINAMATH_CALUDE_no_solution_inequality_l2735_273592


namespace NUMINAMATH_CALUDE_finley_age_l2735_273519

/-- Represents the ages of individuals in the problem -/
structure Ages where
  jill : ℕ
  roger : ℕ
  alex : ℕ
  finley : ℕ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.roger = 2 * ages.jill + 5 ∧
  ages.roger + 15 - (ages.jill + 15) = ages.finley - 30 ∧
  ages.jill = 20 ∧
  ages.roger = (ages.jill + ages.alex) / 2 ∧
  ages.alex = 3 * (ages.finley + 10) - 5

/-- The theorem to be proved -/
theorem finley_age (ages : Ages) : 
  problem_conditions ages → ages.finley = 15 := by
  sorry

end NUMINAMATH_CALUDE_finley_age_l2735_273519


namespace NUMINAMATH_CALUDE_geometric_to_arithmetic_progression_l2735_273545

theorem geometric_to_arithmetic_progression :
  ∀ (a q : ℝ),
    a > 0 → q > 0 →
    a + a * q + a * q^2 = 105 →
    ∃ d : ℝ, a * q - a = (a * q^2 - 15) - a * q →
    (a = 15 ∧ q = 2) ∨ (a = 60 ∧ q = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_to_arithmetic_progression_l2735_273545


namespace NUMINAMATH_CALUDE_chocolate_box_problem_l2735_273572

theorem chocolate_box_problem (total : ℕ) (p_peanut : ℚ) : 
  total = 50 → p_peanut = 64/100 → 
  ∃ (caramels nougats truffles peanuts : ℕ),
    nougats = 2 * caramels ∧
    truffles = caramels + 6 ∧
    caramels + nougats + truffles + peanuts = total ∧
    p_peanut = peanuts / total ∧
    caramels = 3 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_box_problem_l2735_273572


namespace NUMINAMATH_CALUDE_no_real_roots_l2735_273528

theorem no_real_roots : ¬∃ (x : ℝ), x^2 - 4*x + 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l2735_273528


namespace NUMINAMATH_CALUDE_quadratic_solution_symmetry_l2735_273574

/-- A quadratic function f(x) = ax^2 + bx + c where a ≠ 0 -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

theorem quadratic_solution_symmetry (a b c : ℝ) (h : a ≠ 0) :
  let f := QuadraticFunction a b c
  f (-5) = f 1 → f 2 = 0 → ∃ n : ℝ, f 3 = n ∧ (∀ x : ℝ, f x = n ↔ x = 3 ∨ x = -7) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_solution_symmetry_l2735_273574


namespace NUMINAMATH_CALUDE_pipe_fill_time_l2735_273551

/-- The time (in hours) it takes for Pipe A to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 8

/-- The time (in hours) it takes for Pipe A to fill the tank with the leak -/
def fill_time_with_leak : ℝ := 12

/-- The time (in hours) it takes for the leak to empty the full tank -/
def empty_time : ℝ := 24

theorem pipe_fill_time :
  (1 / fill_time_without_leak) - (1 / empty_time) = (1 / fill_time_with_leak) :=
sorry

end NUMINAMATH_CALUDE_pipe_fill_time_l2735_273551


namespace NUMINAMATH_CALUDE_smallest_n_for_g_with_large_digit_l2735_273591

/-- Sum of digits in base b representation of n -/
def digitSum (n : ℕ) (b : ℕ) : ℕ := sorry

/-- f(n) is the sum of digits in base-5 representation of n -/
def f (n : ℕ) : ℕ := digitSum n 5

/-- g(n) is the sum of digits in base-9 representation of f(n) -/
def g (n : ℕ) : ℕ := digitSum (f n) 9

/-- Checks if a number can be represented in base-17 using only digits 0-9 -/
def hasOnlyDigits0To9InBase17 (n : ℕ) : Prop := sorry

theorem smallest_n_for_g_with_large_digit : 
  (∀ m < 791, hasOnlyDigits0To9InBase17 (g m)) ∧ 
  ¬hasOnlyDigits0To9InBase17 (g 791) := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_g_with_large_digit_l2735_273591


namespace NUMINAMATH_CALUDE_amy_flash_drive_files_l2735_273588

/-- The number of files on Amy's flash drive -/
def total_files (music_files video_files picture_files : Float) : Float :=
  music_files + video_files + picture_files

/-- Theorem stating the total number of files on Amy's flash drive -/
theorem amy_flash_drive_files : 
  total_files 4.0 21.0 23.0 = 48.0 := by
  sorry

end NUMINAMATH_CALUDE_amy_flash_drive_files_l2735_273588


namespace NUMINAMATH_CALUDE_range_of_x_when_proposition_false_l2735_273526

theorem range_of_x_when_proposition_false :
  (∀ a : ℝ, -1 ≤ a ∧ a ≤ 3 → ∀ x : ℝ, a * x^2 - (2*a - 1) * x + (3 - a) ≥ 0) →
  ∀ x : ℝ, ((-1 ≤ x ∧ x ≤ 0) ∨ (5/3 ≤ x ∧ x ≤ 4)) :=
by sorry

end NUMINAMATH_CALUDE_range_of_x_when_proposition_false_l2735_273526


namespace NUMINAMATH_CALUDE_james_work_hours_l2735_273593

def minimum_wage : ℚ := 8
def meat_pounds : ℕ := 20
def meat_price : ℚ := 5
def fruit_veg_pounds : ℕ := 15
def fruit_veg_price : ℚ := 4
def bread_pounds : ℕ := 60
def bread_price : ℚ := 3/2
def janitor_hours : ℕ := 10
def janitor_wage : ℚ := 10

def total_cost : ℚ := 
  meat_pounds * meat_price + 
  fruit_veg_pounds * fruit_veg_price + 
  bread_pounds * bread_price + 
  janitor_hours * (janitor_wage * 3/2)

theorem james_work_hours : 
  total_cost / minimum_wage = 50 := by sorry

end NUMINAMATH_CALUDE_james_work_hours_l2735_273593


namespace NUMINAMATH_CALUDE_pen_cost_l2735_273589

/-- The cost of a pen, given the cost of a pencil and the total cost of a specific number of pencils and pens. -/
theorem pen_cost (pencil_cost : ℝ) (total_cost : ℝ) (num_pencils : ℕ) (num_pens : ℕ) : 
  pencil_cost = 2.5 →
  total_cost = 291 →
  num_pencils = 38 →
  num_pens = 56 →
  (num_pencils : ℝ) * pencil_cost + (num_pens : ℝ) * ((total_cost - (num_pencils : ℝ) * pencil_cost) / num_pens) = total_cost →
  (total_cost - (num_pencils : ℝ) * pencil_cost) / num_pens = 3.5 := by
sorry

end NUMINAMATH_CALUDE_pen_cost_l2735_273589


namespace NUMINAMATH_CALUDE_nba_player_age_distribution_l2735_273535

theorem nba_player_age_distribution (total_players : ℕ) 
  (h1 : total_players = 1000)
  (h2 : (2 : ℚ) / 5 * total_players = (players_25_to_35 : ℕ))
  (h3 : (3 : ℚ) / 8 * total_players = (players_over_35 : ℕ)) :
  total_players - (players_25_to_35 + players_over_35) = 225 :=
by sorry

end NUMINAMATH_CALUDE_nba_player_age_distribution_l2735_273535


namespace NUMINAMATH_CALUDE_cattle_breeder_milk_production_l2735_273520

/-- Calculates the weekly milk production for a given number of cows and daily milk production per cow. -/
def weekly_milk_production (num_cows : ℕ) (milk_per_cow_per_day : ℕ) : ℕ :=
  num_cows * milk_per_cow_per_day * 7

/-- Proves that 52 cows producing 1000 oz of milk per day will produce 364,000 oz of milk per week. -/
theorem cattle_breeder_milk_production :
  weekly_milk_production 52 1000 = 364000 := by
  sorry

#eval weekly_milk_production 52 1000

end NUMINAMATH_CALUDE_cattle_breeder_milk_production_l2735_273520


namespace NUMINAMATH_CALUDE_derivative_pos_implies_increasing_exists_increasing_not_always_pos_derivative_l2735_273532

open Function Real

-- Define a differentiable function f: ℝ → ℝ
variable (f : ℝ → ℝ)
variable (hf : Differentiable ℝ f)

-- Define what it means for f to be increasing
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- Statement 1: If f'(x) > 0 for all x, then f is increasing
theorem derivative_pos_implies_increasing :
  (∀ x, deriv f x > 0) → IsIncreasing f :=
sorry

-- Statement 2: There exists an increasing f where it's not true that f'(x) > 0 for all x
theorem exists_increasing_not_always_pos_derivative :
  ∃ f : ℝ → ℝ, Differentiable ℝ f ∧ IsIncreasing f ∧ ¬(∀ x, deriv f x > 0) :=
sorry

end NUMINAMATH_CALUDE_derivative_pos_implies_increasing_exists_increasing_not_always_pos_derivative_l2735_273532


namespace NUMINAMATH_CALUDE_border_material_length_l2735_273539

/-- Given a circular table top with an area of 616 square inches,
    calculate the length of border material needed to cover the circumference
    plus an additional 3 inches, using π ≈ 22/7. -/
theorem border_material_length : 
  let table_area : ℝ := 616
  let π_approx : ℝ := 22 / 7
  let radius : ℝ := Real.sqrt (table_area / π_approx)
  let circumference : ℝ := 2 * π_approx * radius
  let border_length : ℝ := circumference + 3
  border_length = 91 := by
  sorry

end NUMINAMATH_CALUDE_border_material_length_l2735_273539


namespace NUMINAMATH_CALUDE_symmetric_periodic_function_properties_l2735_273524

open Real

/-- A function satisfying specific symmetry and periodicity properties -/
structure SymmetricPeriodicFunction (a c d : ℝ) where
  f : ℝ → ℝ
  even_at_a : ∀ x, f (a + x) = f (a - x)
  sum_at_c : ∀ x, f (c + x) + f (c - x) = 2 * d
  a_neq_c : a ≠ c

theorem symmetric_periodic_function_properties
  {a c d : ℝ} (spf : SymmetricPeriodicFunction a c d) :
  (∀ x, (deriv spf.f) (c + x) = (deriv spf.f) (c - x)) ∧
  (∀ x, spf.f (x + 2 * |c - a|) = 2 * d - spf.f x) ∧
  (∀ x, spf.f (spf.f (a + x)) = spf.f (spf.f (a - x))) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_periodic_function_properties_l2735_273524


namespace NUMINAMATH_CALUDE_polynomial_expansion_l2735_273583

theorem polynomial_expansion (z : ℂ) : 
  (3 * z^3 + 4 * z^2 - 5 * z + 1) * (2 * z^2 - 3 * z + 4) * (z - 1) = 
  3 * z^6 - 3 * z^5 + 5 * z^4 + 2 * z^3 - 5 * z^2 + 4 * z - 16 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l2735_273583


namespace NUMINAMATH_CALUDE_seashells_sum_l2735_273538

/-- The number of seashells found by Mary and Keith -/
def total_seashells (mary_seashells keith_seashells : ℕ) : ℕ :=
  mary_seashells + keith_seashells

/-- Theorem stating that the total number of seashells is the sum of Mary's and Keith's seashells -/
theorem seashells_sum (mary_seashells keith_seashells : ℕ) :
  total_seashells mary_seashells keith_seashells = mary_seashells + keith_seashells :=
by sorry

end NUMINAMATH_CALUDE_seashells_sum_l2735_273538


namespace NUMINAMATH_CALUDE_extremum_implies_not_monotonic_l2735_273558

open Set
open Function

-- Define a real-valued function on R
variable (f : ℝ → ℝ)

-- Define differentiability
variable (h_diff : Differentiable ℝ f)

-- Define the existence of an extremum
variable (h_extremum : ∃ x₀ : ℝ, IsLocalExtremum ℝ f x₀)

-- Theorem statement
theorem extremum_implies_not_monotonic :
  ¬(StrictMono f ∨ StrictAnti f) :=
sorry

end NUMINAMATH_CALUDE_extremum_implies_not_monotonic_l2735_273558


namespace NUMINAMATH_CALUDE_yellow_given_popped_l2735_273529

/-- Represents the types of kernels in the bag -/
inductive KernelType
  | White
  | Yellow
  | Brown

/-- The probability of selecting a kernel of a given type -/
def selectionProb (k : KernelType) : ℚ :=
  match k with
  | KernelType.White => 3/5
  | KernelType.Yellow => 1/5
  | KernelType.Brown => 1/5

/-- The probability of a kernel popping given its type -/
def poppingProb (k : KernelType) : ℚ :=
  match k with
  | KernelType.White => 1/3
  | KernelType.Yellow => 3/4
  | KernelType.Brown => 1/2

/-- The probability of selecting and popping a kernel of a given type -/
def selectAndPopProb (k : KernelType) : ℚ :=
  selectionProb k * poppingProb k

/-- The total probability of selecting and popping any kernel -/
def totalPopProb : ℚ :=
  selectAndPopProb KernelType.White + selectAndPopProb KernelType.Yellow + selectAndPopProb KernelType.Brown

/-- The probability that a popped kernel is yellow -/
theorem yellow_given_popped :
  selectAndPopProb KernelType.Yellow / totalPopProb = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_yellow_given_popped_l2735_273529


namespace NUMINAMATH_CALUDE_dans_remaining_money_l2735_273550

/-- Calculates the remaining money after a purchase -/
def remaining_money (initial : ℕ) (cost : ℕ) : ℕ :=
  initial - cost

theorem dans_remaining_money :
  remaining_money 4 1 = 3 := by
  sorry

end NUMINAMATH_CALUDE_dans_remaining_money_l2735_273550


namespace NUMINAMATH_CALUDE_flight_speed_l2735_273585

/-- Given a flight distance of 256 miles and a flight time of 8 hours,
    prove that the speed is 32 miles per hour. -/
theorem flight_speed (distance : ℝ) (time : ℝ) (speed : ℝ) 
    (h1 : distance = 256) 
    (h2 : time = 8) 
    (h3 : speed = distance / time) : speed = 32 := by
  sorry

end NUMINAMATH_CALUDE_flight_speed_l2735_273585


namespace NUMINAMATH_CALUDE_simplify_expression_l2735_273506

theorem simplify_expression :
  (6 * 10^7) * (2 * 10^3)^2 / (4 * 10^4) = 6 * 10^9 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_l2735_273506


namespace NUMINAMATH_CALUDE_angle_theta_value_l2735_273525

theorem angle_theta_value (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) 
  (h2 : Real.sqrt 3 * Real.sin (20 * π / 180) = Real.cos θ - Real.sin θ) : 
  θ = 25 * π / 180 := by
sorry

end NUMINAMATH_CALUDE_angle_theta_value_l2735_273525


namespace NUMINAMATH_CALUDE_cafeteria_green_apples_l2735_273515

theorem cafeteria_green_apples :
  ∀ (red_apples students_wanting_fruit extra_apples green_apples : ℕ),
    red_apples = 25 →
    students_wanting_fruit = 10 →
    extra_apples = 32 →
    red_apples + green_apples - students_wanting_fruit = extra_apples →
    green_apples = 17 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_green_apples_l2735_273515


namespace NUMINAMATH_CALUDE_bicycle_ride_time_l2735_273559

/-- Proves the total time Hyeonil rode the bicycle given the conditions -/
theorem bicycle_ride_time (speed : ℝ) (initial_time : ℝ) (additional_distance : ℝ)
  (h1 : speed = 4.25)
  (h2 : initial_time = 60)
  (h3 : additional_distance = 29.75) :
  initial_time + additional_distance / speed = 67 := by
  sorry

#check bicycle_ride_time

end NUMINAMATH_CALUDE_bicycle_ride_time_l2735_273559


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l2735_273546

/-- RegularOctagon represents a regular octagon with center O and vertices A to H -/
structure RegularOctagon where
  O : Point
  A : Point
  B : Point
  C : Point
  D : Point
  E : Point
  F : Point
  G : Point
  H : Point

/-- Given a regular octagon, returns the area of the specified region -/
def shaded_area (octagon : RegularOctagon) : ℝ :=
  sorry

/-- The total area of the regular octagon -/
def total_area (octagon : RegularOctagon) : ℝ :=
  sorry

/-- Theorem stating that the shaded area is 5/8 of the total area -/
theorem shaded_area_fraction (octagon : RegularOctagon) :
  shaded_area octagon / total_area octagon = 5/8 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l2735_273546


namespace NUMINAMATH_CALUDE_factorization_2m_cubed_minus_8m_l2735_273505

theorem factorization_2m_cubed_minus_8m (m : ℝ) : 2*m^3 - 8*m = 2*m*(m+2)*(m-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2m_cubed_minus_8m_l2735_273505


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2735_273502

theorem necessary_but_not_sufficient_condition (a b : ℝ) :
  (∀ a b, a > b → a > b - 1) ∧ 
  (∃ a b, a > b - 1 ∧ ¬(a > b)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_condition_l2735_273502


namespace NUMINAMATH_CALUDE_orange_to_apple_ratio_l2735_273541

/-- Given the total weight of fruits and the weight of oranges, proves the ratio of oranges to apples -/
theorem orange_to_apple_ratio
  (total_weight : ℕ)
  (orange_weight : ℕ)
  (h1 : total_weight = 12)
  (h2 : orange_weight = 10) :
  orange_weight / (total_weight - orange_weight) = 5 := by
  sorry

end NUMINAMATH_CALUDE_orange_to_apple_ratio_l2735_273541


namespace NUMINAMATH_CALUDE_percent_problem_l2735_273594

theorem percent_problem (x : ℝ) (h : x / 100 * 60 = 12) : 15 / 100 * x = 3 := by
  sorry

end NUMINAMATH_CALUDE_percent_problem_l2735_273594


namespace NUMINAMATH_CALUDE_num_intersection_values_correct_l2735_273501

/-- The number of different possible values for the count of intersection points
    formed by 5 distinct lines on a plane. -/
def num_intersection_values : ℕ := 9

/-- The set of possible values for the count of intersection points
    formed by 5 distinct lines on a plane. -/
def possible_intersection_values : Finset ℕ :=
  {0, 1, 4, 5, 6, 7, 8, 9, 10}

/-- Theorem stating that the number of different possible values for the count
    of intersection points formed by 5 distinct lines on a plane is correct. -/
theorem num_intersection_values_correct :
    num_intersection_values = Finset.card possible_intersection_values := by
  sorry

end NUMINAMATH_CALUDE_num_intersection_values_correct_l2735_273501


namespace NUMINAMATH_CALUDE_medication_C_consumption_l2735_273599

def days_in_july : ℕ := 31

def doses_per_day_C : ℕ := 3

def missed_days_C : ℕ := 2

theorem medication_C_consumption :
  days_in_july * doses_per_day_C - missed_days_C * doses_per_day_C = 87 := by
  sorry

end NUMINAMATH_CALUDE_medication_C_consumption_l2735_273599


namespace NUMINAMATH_CALUDE_quadratic_roots_large_difference_l2735_273514

theorem quadratic_roots_large_difference :
  ∃ (p q p' q' u v u' v' : ℝ),
    (u > v) ∧ (u' > v') ∧
    (u^2 + p*u + q = 0) ∧
    (v^2 + p*v + q = 0) ∧
    (u'^2 + p'*u' + q' = 0) ∧
    (v'^2 + p'*v' + q' = 0) ∧
    (|p' - p| < 0.01) ∧
    (|q' - q| < 0.01) ∧
    (|u' - u| > 10000) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_large_difference_l2735_273514


namespace NUMINAMATH_CALUDE_two_digit_numbers_with_special_properties_l2735_273554

/-- Round a natural number to the nearest ten -/
def roundToNearestTen (n : ℕ) : ℕ :=
  10 * ((n + 5) / 10)

/-- Check if a natural number is a two-digit number -/
def isTwoDigit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem two_digit_numbers_with_special_properties (p q : ℕ) :
  isTwoDigit p ∧ isTwoDigit q ∧
  (roundToNearestTen p - roundToNearestTen q = p - q) ∧
  (roundToNearestTen p * roundToNearestTen q = p * q + 184) →
  ((p = 16 ∧ q = 26) ∨ (p = 26 ∧ q = 16)) := by
  sorry

end NUMINAMATH_CALUDE_two_digit_numbers_with_special_properties_l2735_273554


namespace NUMINAMATH_CALUDE_percentage_excess_l2735_273579

theorem percentage_excess (x y : ℝ) (h : x = 0.8 * y) : y = 1.25 * x := by
  sorry

end NUMINAMATH_CALUDE_percentage_excess_l2735_273579


namespace NUMINAMATH_CALUDE_scarf_final_price_l2735_273555

def original_price : ℝ := 15
def first_discount : ℝ := 0.20
def second_discount : ℝ := 0.25

theorem scarf_final_price :
  original_price * (1 - first_discount) * (1 - second_discount) = 9 := by
  sorry

end NUMINAMATH_CALUDE_scarf_final_price_l2735_273555


namespace NUMINAMATH_CALUDE_positive_real_inequalities_l2735_273562

/-- Given positive real numbers a and b, prove two inequalities based on given conditions -/
theorem positive_real_inequalities (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + b = 1 → (1 + 1/a) * (1 + 1/b) ≥ 9) ∧
  (2*a + b = a*b → a + b ≥ 3 + 2*Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_positive_real_inequalities_l2735_273562


namespace NUMINAMATH_CALUDE_sine_cosine_inequality_l2735_273566

theorem sine_cosine_inequality (x : ℝ) (h : Real.sin x + Real.cos x ≤ 0) :
  Real.sin x ^ 1993 + Real.cos x ^ 1993 ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sine_cosine_inequality_l2735_273566


namespace NUMINAMATH_CALUDE_problem_statement_l2735_273503

theorem problem_statement (x y z : ℝ) (h : (5 : ℝ) ^ x = (9 : ℝ) ^ y ∧ (9 : ℝ) ^ y = (225 : ℝ) ^ z) : 
  1 / z = 2 / x + 1 / y := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2735_273503


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l2735_273560

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between a line and a plane
variable (perpendicular_to_plane : Line → Plane → Prop)

-- Define the contained relation between a line and a plane
variable (contained_in_plane : Line → Plane → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular_to_line : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane_implies_perpendicular_to_contained_line 
  (l m : Line) (α : Plane) :
  perpendicular_to_plane l α → contained_in_plane m α → perpendicular_to_line l m :=
by sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_implies_perpendicular_to_contained_line_l2735_273560


namespace NUMINAMATH_CALUDE_product_ratio_simplification_l2735_273598

theorem product_ratio_simplification
  (a b c d e f g : ℝ)
  (h1 : a * b * c * d = 260)
  (h2 : b * c * d * e = 390)
  (h3 : c * d * e * f = 2000)
  (h4 : d * e * f * g = 500)
  (h5 : c ≠ 0)
  (h6 : e ≠ 0) :
  (a * g) / (c * e) = a / (4 * c) :=
by sorry

end NUMINAMATH_CALUDE_product_ratio_simplification_l2735_273598


namespace NUMINAMATH_CALUDE_system_solutions_l2735_273527

def is_solution (x y z : ℤ) : Prop :=
  x^2 + y^2 + 2*x + 6*y = -5 ∧
  x^2 + z^2 + 2*x - 4*z = 8 ∧
  y^2 + z^2 + 6*y - 4*z = -3

theorem system_solutions :
  is_solution 1 (-2) (-1) ∧
  is_solution 1 (-2) 5 ∧
  is_solution 1 (-4) (-1) ∧
  is_solution 1 (-4) 5 ∧
  is_solution (-3) (-2) (-1) ∧
  is_solution (-3) (-2) 5 ∧
  is_solution (-3) (-4) (-1) ∧
  is_solution (-3) (-4) 5 :=
by sorry


end NUMINAMATH_CALUDE_system_solutions_l2735_273527


namespace NUMINAMATH_CALUDE_smallest_positive_number_l2735_273549

theorem smallest_positive_number :
  let a := 8 - 3 * Real.sqrt 10
  let b := 3 * Real.sqrt 10 - 8
  let c := 23 - 6 * Real.sqrt 15
  let d := 58 - 12 * Real.sqrt 30
  let e := 12 * Real.sqrt 30 - 58
  (0 < b) ∧
  (a ≤ 0 ∨ b < a) ∧
  (c ≤ 0 ∨ b < c) ∧
  (d ≤ 0 ∨ b < d) ∧
  (e ≤ 0 ∨ b < e) :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_number_l2735_273549


namespace NUMINAMATH_CALUDE_milk_water_ratio_l2735_273561

theorem milk_water_ratio 
  (initial_volume : ℝ) 
  (initial_milk_ratio : ℝ) 
  (initial_water_ratio : ℝ) 
  (added_water : ℝ) : 
  initial_volume = 45 ∧ 
  initial_milk_ratio = 4 ∧ 
  initial_water_ratio = 1 ∧ 
  added_water = 21 → 
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let new_water := initial_water + added_water
  let new_ratio_milk := initial_milk / (initial_milk + new_water) * 11
  let new_ratio_water := new_water / (initial_milk + new_water) * 11
  new_ratio_milk = 6 ∧ new_ratio_water = 5 :=
by sorry


end NUMINAMATH_CALUDE_milk_water_ratio_l2735_273561


namespace NUMINAMATH_CALUDE_john_learning_time_l2735_273534

/-- The number of vowels in the English alphabet -/
def num_vowels : ℕ := 5

/-- The total number of days John needs to learn all vowels -/
def total_days : ℕ := 15

/-- The number of days John needs to learn one alphabet (vowel) -/
def days_per_alphabet : ℚ := total_days / num_vowels

theorem john_learning_time : days_per_alphabet = 3 := by
  sorry

end NUMINAMATH_CALUDE_john_learning_time_l2735_273534


namespace NUMINAMATH_CALUDE_inverse_composition_l2735_273575

-- Define the function f
def f : ℕ → ℕ
| 2 => 8
| 3 => 15
| 4 => 24
| 5 => 35
| 6 => 48
| _ => 0  -- For other inputs, we'll define it as 0

-- Define the inverse function f⁻¹
def f_inv : ℕ → ℕ
| 8 => 2
| 15 => 3
| 24 => 4
| 35 => 5
| 48 => 6
| _ => 0  -- For other inputs, we'll define it as 0

-- State the theorem
theorem inverse_composition :
  f_inv (f_inv 48 * f_inv 8 - f_inv 24) = 2 :=
by sorry

end NUMINAMATH_CALUDE_inverse_composition_l2735_273575


namespace NUMINAMATH_CALUDE_problem_statement_l2735_273567

theorem problem_statement (n m : ℕ) : 
  2 * 8^n * 16^n = 2^15 →
  (∀ x y : ℝ, (m*x + y) * (2*x - y) = 2*m*x^2 - y^2) →
  n - m = 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2735_273567


namespace NUMINAMATH_CALUDE_speed_of_light_scientific_notation_l2735_273582

def speed_of_light : ℝ := 300000000

theorem speed_of_light_scientific_notation : 
  speed_of_light = 3 * (10 : ℝ) ^ 8 := by sorry

end NUMINAMATH_CALUDE_speed_of_light_scientific_notation_l2735_273582


namespace NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l2735_273540

theorem largest_multiple_of_9_under_100 : ∃ n : ℕ, n = 99 ∧ 9 ∣ n ∧ n < 100 ∧ ∀ m : ℕ, 9 ∣ m → m < 100 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_of_9_under_100_l2735_273540


namespace NUMINAMATH_CALUDE_right_triangle_equation_roots_l2735_273581

theorem right_triangle_equation_roots (a b c : ℝ) (h_right_angle : a^2 + c^2 = b^2) :
  ∃ (x : ℝ), ¬ (∀ (y : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0 ↔ x = y) ∧
             ¬ (∀ (y z : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0 ∧
                             a * (z^2 - 1) - 2 * z + b * (z^2 + 1) = 0 → y ≠ z) ∧
             ¬ (¬ ∃ (y : ℝ), a * (y^2 - 1) - 2 * y + b * (y^2 + 1) = 0) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_equation_roots_l2735_273581


namespace NUMINAMATH_CALUDE_pages_read_difference_l2735_273556

theorem pages_read_difference (beatrix_pages : ℕ) (cristobal_pages : ℕ) : 
  beatrix_pages = 704 → 
  cristobal_pages = 15 + 3 * beatrix_pages → 
  cristobal_pages - beatrix_pages = 1423 := by
  sorry

end NUMINAMATH_CALUDE_pages_read_difference_l2735_273556


namespace NUMINAMATH_CALUDE_larry_wins_prob_l2735_273510

/-- The probability of Larry winning the game --/
def larry_wins : ℚ :=
  let larry_prob : ℚ := 3/4  -- Larry's probability of knocking the bottle off
  let julius_prob : ℚ := 1/4 -- Julius's probability of knocking the bottle off
  let max_rounds : ℕ := 5    -- Maximum number of rounds

  -- Probability of Larry winning in the 1st round
  let p1 : ℚ := larry_prob

  -- Probability of Larry winning in the 3rd round
  let p3 : ℚ := (1 - larry_prob) * julius_prob * larry_prob

  -- Probability of Larry winning in the 5th round
  let p5 : ℚ := ((1 - larry_prob) * julius_prob)^2 * larry_prob

  -- Total probability of Larry winning
  p1 + p3 + p5

/-- Theorem stating that the probability of Larry winning is 825/1024 --/
theorem larry_wins_prob : larry_wins = 825/1024 := by
  sorry

end NUMINAMATH_CALUDE_larry_wins_prob_l2735_273510


namespace NUMINAMATH_CALUDE_triangle_returns_after_six_rotations_l2735_273553

/-- Represents a triangle in 2D space -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Represents a rotation around a point by a given angle -/
def rotate (center : ℝ × ℝ) (angle : ℝ) (point : ℝ × ℝ) : ℝ × ℝ := sorry

/-- Performs a single rotation of the triangle around one of its vertices -/
def rotateTriangle (t : Triangle) (vertex : Fin 3) : Triangle := sorry

/-- Performs six successive rotations of the triangle -/
def sixRotations (t : Triangle) : Triangle := sorry

/-- Theorem stating that after six rotations, the triangle returns to its original position -/
theorem triangle_returns_after_six_rotations (t : Triangle) : 
  sixRotations t = t := by sorry

end NUMINAMATH_CALUDE_triangle_returns_after_six_rotations_l2735_273553


namespace NUMINAMATH_CALUDE_tan_45_degrees_l2735_273530

theorem tan_45_degrees : Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_l2735_273530


namespace NUMINAMATH_CALUDE_min_max_values_l2735_273580

theorem min_max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 4 * a + b = 1) :
  (1 / a + 1 / b ≥ 9) ∧ (a * b ≤ 1 / 16) := by sorry

end NUMINAMATH_CALUDE_min_max_values_l2735_273580


namespace NUMINAMATH_CALUDE_wall_bricks_l2735_273564

/-- Represents the time taken by Ben to build the wall alone -/
def ben_time : ℝ := 12

/-- Represents the time taken by Jerry to build the wall alone -/
def jerry_time : ℝ := 8

/-- Represents the decrease in combined output when working together -/
def output_decrease : ℝ := 15

/-- Represents the time taken to complete the job together -/
def combined_time : ℝ := 6

/-- Theorem stating that the number of bricks in the wall is 240 -/
theorem wall_bricks : ℝ := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_l2735_273564


namespace NUMINAMATH_CALUDE_copperfield_numbers_l2735_273596

theorem copperfield_numbers :
  ∃ (x₁ x₂ x₃ : ℕ) (k₁ k₂ k₃ : ℕ+),
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    x₁ * 3^(k₁.val) = x₁ + 2500 * k₁.val ∧
    x₂ * 3^(k₂.val) = x₂ + 2500 * k₂.val ∧
    x₃ * 3^(k₃.val) = x₃ + 2500 * k₃.val :=
by sorry

end NUMINAMATH_CALUDE_copperfield_numbers_l2735_273596


namespace NUMINAMATH_CALUDE_fraction_simplification_l2735_273547

theorem fraction_simplification (x : ℝ) : (x + 2) / 4 + (5 - 4 * x) / 3 = (-13 * x + 26) / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2735_273547


namespace NUMINAMATH_CALUDE_sin_45_degrees_l2735_273576

theorem sin_45_degrees : Real.sin (π / 4) = 1 / Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l2735_273576


namespace NUMINAMATH_CALUDE_max_value_theorem_l2735_273542

theorem max_value_theorem (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) 
  (h_eq : x^2 - x*y + 2*y^2 = 8) :
  x^2 + x*y + 2*y^2 ≤ (72 + 32*Real.sqrt 2) / 7 ∧ 
  ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀^2 - x₀*y₀ + 2*y₀^2 = 8 ∧
  x₀^2 + x₀*y₀ + 2*y₀^2 = (72 + 32*Real.sqrt 2) / 7 := by
  sorry

end NUMINAMATH_CALUDE_max_value_theorem_l2735_273542


namespace NUMINAMATH_CALUDE_train_meeting_distance_l2735_273571

/-- Proves that Train A travels 75 miles before meeting Train B -/
theorem train_meeting_distance (route_length : ℝ) (time_a : ℝ) (time_b : ℝ) 
  (h1 : route_length = 200)
  (h2 : time_a = 10)
  (h3 : time_b = 6)
  : (route_length / time_a) * (route_length / (route_length / time_a + route_length / time_b)) = 75 := by
  sorry

end NUMINAMATH_CALUDE_train_meeting_distance_l2735_273571


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l2735_273500

theorem product_of_three_numbers (a b c m : ℚ) : 
  a + b + c = 240 ∧ 
  6 * a = m ∧ 
  b - 12 = m ∧ 
  c + 12 = m ∧ 
  a ≤ c ∧ 
  c ≤ b → 
  a * b * c = 490108320 / 2197 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l2735_273500


namespace NUMINAMATH_CALUDE_mitchell_unchewed_gum_l2735_273590

theorem mitchell_unchewed_gum (packets : ℕ) (pieces_per_packet : ℕ) (chewed_pieces : ℕ) 
  (h1 : packets = 8) 
  (h2 : pieces_per_packet = 7) 
  (h3 : chewed_pieces = 54) : 
  packets * pieces_per_packet - chewed_pieces = 2 := by
  sorry

end NUMINAMATH_CALUDE_mitchell_unchewed_gum_l2735_273590


namespace NUMINAMATH_CALUDE_circle_trajectory_l2735_273523

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  center : Point
  radius : ℝ

/-- The fixed point A -/
def A : Point := ⟨2, 0⟩

/-- Checks if a circle passes through a given point -/
def passesThrough (c : Circle) (p : Point) : Prop :=
  (c.center.x - p.x)^2 + (c.center.y - p.y)^2 = c.radius^2

/-- Checks if a circle intersects the y-axis forming a chord of length 4 -/
def intersectsYAxis (c : Circle) : Prop :=
  c.radius^2 = c.center.x^2 + 4

/-- The trajectory of the center of the moving circle -/
def trajectory (p : Point) : Prop :=
  p.y^2 = 4 * p.x

/-- Theorem: The trajectory of the center of a circle that passes through (2,0) 
    and intersects the y-axis forming a chord of length 4 is y² = 4x -/
theorem circle_trajectory : 
  ∀ (c : Circle), 
    passesThrough c A → 
    intersectsYAxis c → 
    trajectory c.center :=
sorry

end NUMINAMATH_CALUDE_circle_trajectory_l2735_273523


namespace NUMINAMATH_CALUDE_ellipse_slope_theorem_l2735_273557

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 + y^2/4 = 1

-- Define points A and B as endpoints of minor axis
def A : ℝ × ℝ := (0, -1)
def B : ℝ × ℝ := (0, 1)

-- Define line l passing through (0,1)
def line_l (k : ℝ) (x y : ℝ) : Prop := y - 1 = k * x

-- Define points C and D on the ellipse and line l
def C (k : ℝ) : ℝ × ℝ := sorry
def D (k : ℝ) : ℝ × ℝ := sorry

-- Define slopes k₁ and k₂
def k₁ (k : ℝ) : ℝ := sorry
def k₂ (k : ℝ) : ℝ := sorry

theorem ellipse_slope_theorem (k : ℝ) :
  (∀ x y, ellipse x y → line_l k x y → (x, y) = C k ∨ (x, y) = D k) →
  k₁ k / k₂ k = 2 →
  k = 3 := by sorry

end NUMINAMATH_CALUDE_ellipse_slope_theorem_l2735_273557


namespace NUMINAMATH_CALUDE_cubic_two_roots_l2735_273504

/-- A cubic function with a parameter c -/
def f (c : ℝ) : ℝ → ℝ := fun x ↦ x^3 - 3*x + c

/-- The derivative of f -/
def f' (c : ℝ) : ℝ → ℝ := fun x ↦ 3*x^2 - 3

theorem cubic_two_roots (c : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f c x₁ = 0 ∧ f c x₂ = 0 ∧
    ∀ x, f c x = 0 → x = x₁ ∨ x = x₂) →
  c = 2 ∨ c = -2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_two_roots_l2735_273504


namespace NUMINAMATH_CALUDE_art_gallery_sculpture_fraction_l2735_273536

theorem art_gallery_sculpture_fraction 
  (total : ℕ) 
  (displayed : ℕ) 
  (sculptures_not_displayed : ℕ) 
  (h1 : displayed = total / 3)
  (h2 : sculptures_not_displayed = 800)
  (h3 : total = 1800)
  (h4 : (total - displayed) / 3 = total - displayed - sculptures_not_displayed) :
  3 * (sculptures_not_displayed + displayed - total + sculptures_not_displayed) = 2 * displayed := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_sculpture_fraction_l2735_273536


namespace NUMINAMATH_CALUDE_one_third_minus_0_3333_l2735_273509

-- Define 0.3333 as a rational number
def decimal_0_3333 : ℚ := 3333 / 10000

-- State the theorem
theorem one_third_minus_0_3333 : (1 : ℚ) / 3 - decimal_0_3333 = 1 / 10000 := by
  sorry

end NUMINAMATH_CALUDE_one_third_minus_0_3333_l2735_273509


namespace NUMINAMATH_CALUDE_gain_percent_calculation_l2735_273563

theorem gain_percent_calculation (cost_price selling_price : ℝ) 
  (h : 50 * cost_price = 32 * selling_price) : 
  (selling_price - cost_price) / cost_price * 100 = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_gain_percent_calculation_l2735_273563


namespace NUMINAMATH_CALUDE_order_of_operations_4_times_20_plus_30_l2735_273548

theorem order_of_operations_4_times_20_plus_30 : 
  let expression := 4 * (20 + 30)
  let correct_order := ["addition", "multiplication"]
  correct_order = ["addition", "multiplication"] := by sorry

end NUMINAMATH_CALUDE_order_of_operations_4_times_20_plus_30_l2735_273548


namespace NUMINAMATH_CALUDE_plane_division_l2735_273531

/-- The number of parts into which n lines divide a plane -/
def f (n : ℕ) : ℕ := 1 + n * (n + 1) / 2

/-- Theorem stating that f(n) correctly counts the number of parts for n ≥ 2 -/
theorem plane_division (n : ℕ) (h : n ≥ 2) : 
  f n = 1 + n * (n + 1) / 2 := by
  sorry

/-- Helper lemma for the induction step -/
lemma induction_step (k : ℕ) (h : k ≥ 2) :
  f (k + 1) = f k + (k + 1) := by
  sorry

end NUMINAMATH_CALUDE_plane_division_l2735_273531


namespace NUMINAMATH_CALUDE_arithmetic_mean_squares_first_four_odd_numbers_l2735_273537

theorem arithmetic_mean_squares_first_four_odd_numbers : 
  let odd_numbers := [1, 3, 5, 7]
  let squares := List.map (λ x => x^2) odd_numbers
  (List.sum squares) / (List.length squares) = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_squares_first_four_odd_numbers_l2735_273537


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l2735_273517

/-- Given that the solution set of ax^2 + (a-5)x - 2 > 0 is {x | -2 < x < -1/4},
    prove the following statements. -/
theorem quadratic_inequality_problem (a : ℝ) 
  (h : ∀ x, ax^2 + (a-5)*x - 2 > 0 ↔ -2 < x ∧ x < -1/4) :
  /- 1. a = -4 -/
  (a = -4) ∧ 
  /- 2. The solution set of 2x^2 + (2-a)x - a > 0 is (-∞, -2) ∪ (-1, ∞) -/
  (∀ x, 2*x^2 + (2-a)*x - a > 0 ↔ x < -2 ∨ x > -1) ∧
  /- 3. The range of b such that -ax^2 + bx + 3 ≥ 0 for all real x 
        is [-4√3, 4√3] -/
  (∀ b, (∀ x, -a*x^2 + b*x + 3 ≥ 0) ↔ -4*Real.sqrt 3 ≤ b ∧ b ≤ 4*Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l2735_273517


namespace NUMINAMATH_CALUDE_isabellas_house_number_l2735_273544

/-- A predicate that checks if a natural number is a two-digit number -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- A predicate that checks if a natural number has 9 as one of its digits -/
def has_digit_nine (n : ℕ) : Prop := ∃ d, d ∈ n.digits 10 ∧ d = 9

theorem isabellas_house_number :
  ∃! n : ℕ, is_two_digit n ∧
           ¬ Nat.Prime n ∧
           Even n ∧
           n % 7 = 0 ∧
           has_digit_nine n ∧
           n % 10 = 8 := by sorry

end NUMINAMATH_CALUDE_isabellas_house_number_l2735_273544


namespace NUMINAMATH_CALUDE_total_marbles_l2735_273569

/-- Represents the number of marbles of each color -/
structure MarbleCollection where
  yellow : ℕ
  purple : ℕ
  orange : ℕ

/-- The ratio of marbles (yellow:purple:orange) -/
def marble_ratio : MarbleCollection := ⟨2, 4, 6⟩

/-- The number of orange marbles -/
def orange_count : ℕ := 18

/-- Theorem stating the total number of marbles -/
theorem total_marbles (c : MarbleCollection) 
  (h1 : c.yellow * marble_ratio.purple = c.purple * marble_ratio.yellow)
  (h2 : c.yellow * marble_ratio.orange = c.orange * marble_ratio.yellow)
  (h3 : c.orange = orange_count) : 
  c.yellow + c.purple + c.orange = 36 := by
  sorry


end NUMINAMATH_CALUDE_total_marbles_l2735_273569


namespace NUMINAMATH_CALUDE_congruence_from_equal_sides_equal_sides_from_congruence_l2735_273507

/-- Triangle type -/
structure Triangle where
  vertices : Fin 3 → ℝ × ℝ

/-- Congruence relation between triangles -/
def congruent (t1 t2 : Triangle) : Prop := sorry

/-- Length of a side of a triangle -/
def side_length (t : Triangle) (i : Fin 3) : ℝ := sorry

/-- Two triangles have equal corresponding sides -/
def equal_sides (t1 t2 : Triangle) : Prop :=
  ∀ i : Fin 3, side_length t1 i = side_length t2 i

theorem congruence_from_equal_sides (t1 t2 : Triangle) :
  equal_sides t1 t2 → congruent t1 t2 := by sorry

theorem equal_sides_from_congruence (t1 t2 : Triangle) :
  congruent t1 t2 → equal_sides t1 t2 := by sorry

end NUMINAMATH_CALUDE_congruence_from_equal_sides_equal_sides_from_congruence_l2735_273507


namespace NUMINAMATH_CALUDE_additive_inverse_equation_l2735_273508

theorem additive_inverse_equation (x : ℝ) : (6 * x - 12 = -(4 + 2 * x)) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_additive_inverse_equation_l2735_273508


namespace NUMINAMATH_CALUDE_complement_of_union_MN_l2735_273516

def I : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 5}
def N : Set ℕ := {2, 3, 5}

theorem complement_of_union_MN :
  (M ∪ N)ᶜ = {4} :=
by
  sorry

end NUMINAMATH_CALUDE_complement_of_union_MN_l2735_273516
