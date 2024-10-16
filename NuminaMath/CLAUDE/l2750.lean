import Mathlib

namespace NUMINAMATH_CALUDE_circle_symmetry_ab_range_l2750_275058

/-- Given a circle x^2 + y^2 - 4x + 2y + 1 = 0 symmetric about the line ax - 2by - 1 = 0 (a, b ∈ ℝ),
    the range of ab is (-∞, 1/16]. -/
theorem circle_symmetry_ab_range (a b : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 - 4*x + 2*y + 1 = 0 → 
    (∃ x' y' : ℝ, x'^2 + y'^2 - 4*x' + 2*y' + 1 = 0 ∧ 
      a*x - 2*b*y - 1 = a*x' - 2*b*y' - 1 ∧ 
      (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2)) →
  a * b ≤ 1/16 := by
sorry

end NUMINAMATH_CALUDE_circle_symmetry_ab_range_l2750_275058


namespace NUMINAMATH_CALUDE_key_cleaning_time_l2750_275095

/-- The time it takes to clean one key -/
def clean_time : ℝ := 3

theorem key_cleaning_time :
  let assignment_time : ℝ := 10
  let remaining_keys : ℕ := 14
  let total_time : ℝ := 52
  clean_time * remaining_keys + assignment_time = total_time :=
by sorry

end NUMINAMATH_CALUDE_key_cleaning_time_l2750_275095


namespace NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2750_275011

theorem arithmetic_mean_after_removal (S : Finset ℝ) (x y : ℝ) :
  S.card = 60 →
  x ∈ S →
  y ∈ S →
  x = 60 →
  y = 75 →
  (S.sum id) / S.card = 45 →
  ((S.sum id - (x + y)) / (S.card - 2) : ℝ) = 465 / 106 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_after_removal_l2750_275011


namespace NUMINAMATH_CALUDE_square_difference_identity_l2750_275071

theorem square_difference_identity :
  287 * 287 + 269 * 269 - 2 * 287 * 269 = 324 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_identity_l2750_275071


namespace NUMINAMATH_CALUDE_min_a_value_l2750_275086

-- Define the functions f and g
def f (a x : ℝ) : ℝ := a * x^3
def g (x : ℝ) : ℝ := 9 * x^2 + 3 * x - 1

-- State the theorem
theorem min_a_value (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 2, f a x ≥ g x) → a ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_min_a_value_l2750_275086


namespace NUMINAMATH_CALUDE_third_group_first_student_l2750_275008

/-- Systematic sampling function that returns the number of the first student in a given group -/
def systematic_sample (total_students : ℕ) (sample_size : ℕ) (group : ℕ) : ℕ :=
  let interval := total_students / sample_size
  (group - 1) * interval

theorem third_group_first_student :
  systematic_sample 800 40 3 = 40 := by
  sorry

end NUMINAMATH_CALUDE_third_group_first_student_l2750_275008


namespace NUMINAMATH_CALUDE_patio_rearrangement_l2750_275067

theorem patio_rearrangement (total_tiles : ℕ) (initial_rows : ℕ) (column_reduction : ℕ) :
  total_tiles = 96 →
  initial_rows = 8 →
  column_reduction = 2 →
  let initial_columns := total_tiles / initial_rows
  let new_columns := initial_columns - column_reduction
  let new_rows := total_tiles / new_columns
  new_rows - initial_rows = 4 :=
by sorry

end NUMINAMATH_CALUDE_patio_rearrangement_l2750_275067


namespace NUMINAMATH_CALUDE_quadratic_root_proof_l2750_275063

theorem quadratic_root_proof : let x : ℝ := (-15 - Real.sqrt 181) / 8
  ∀ u : ℝ, u = 2.75 → 4 * x^2 + 15 * x + u = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_proof_l2750_275063


namespace NUMINAMATH_CALUDE_fourth_term_of_sequence_l2750_275084

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_sequence (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = (16 : ℝ) ^ (1/4) →
  a 2 = (16 : ℝ) ^ (1/6) →
  a 3 = (16 : ℝ) ^ (1/8) →
  a 4 = (2 : ℝ) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_fourth_term_of_sequence_l2750_275084


namespace NUMINAMATH_CALUDE_perfect_square_concatenation_l2750_275072

theorem perfect_square_concatenation (b m : ℕ) (h_b_odd : Odd b) :
  let A : ℕ := (5^b + 1) / 2
  let B : ℕ := 2^b * A * 100^m
  let AB : ℕ := 10^(Nat.digits 10 B).length * A + B
  ∃ (n : ℕ), AB = n^2 ∧ AB = 2 * A * B := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_concatenation_l2750_275072


namespace NUMINAMATH_CALUDE_largest_of_three_numbers_l2750_275003

theorem largest_of_three_numbers (d e f : ℝ) 
  (sum_eq : d + e + f = 3)
  (sum_prod_eq : d * e + d * f + e * f = -14)
  (prod_eq : d * e * f = 21) :
  max d (max e f) = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_largest_of_three_numbers_l2750_275003


namespace NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l2750_275089

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h_pos : 0 < n) (h_odd : Odd n) :
  n ∣ 2^(n.factorial) - 1 := by
  sorry

end NUMINAMATH_CALUDE_odd_divides_power_factorial_minus_one_l2750_275089


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l2750_275074

theorem greatest_multiple_of_5_and_6_less_than_1000 : ∃ n : ℕ,
  n < 1000 ∧ 
  5 ∣ n ∧ 
  6 ∣ n ∧ 
  ∀ m : ℕ, m < 1000 → 5 ∣ m → 6 ∣ m → m ≤ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_6_less_than_1000_l2750_275074


namespace NUMINAMATH_CALUDE_beth_marbles_l2750_275094

/-- The number of marbles Beth has initially -/
def initial_marbles : ℕ := 72

/-- The number of colors of marbles -/
def num_colors : ℕ := 3

/-- The number of red marbles Beth loses -/
def lost_red : ℕ := 5

/-- Calculates the number of marbles Beth has left after losing some -/
def marbles_left (initial : ℕ) (colors : ℕ) (lost_red : ℕ) : ℕ :=
  initial - (lost_red + 2 * lost_red + 3 * lost_red)

theorem beth_marbles :
  marbles_left initial_marbles num_colors lost_red = 42 := by
  sorry

end NUMINAMATH_CALUDE_beth_marbles_l2750_275094


namespace NUMINAMATH_CALUDE_sandy_token_difference_l2750_275077

/-- Represents the number of Safe Moon tokens Sandy bought -/
def total_tokens : ℕ := 1000000

/-- Represents the number of Sandy's siblings -/
def num_siblings : ℕ := 4

/-- Calculates the number of tokens Sandy keeps for herself -/
def sandy_tokens : ℕ := total_tokens / 2

/-- Calculates the number of tokens each sibling receives -/
def sibling_tokens : ℕ := (total_tokens - sandy_tokens) / num_siblings

/-- Proves that Sandy has 375,000 more tokens than any of her siblings -/
theorem sandy_token_difference : sandy_tokens - sibling_tokens = 375000 := by
  sorry

end NUMINAMATH_CALUDE_sandy_token_difference_l2750_275077


namespace NUMINAMATH_CALUDE_sally_picked_42_peaches_l2750_275022

/-- The number of peaches Sally picked up at the orchard -/
def peaches_picked (initial current : ℕ) : ℕ := current - initial

/-- Proof that Sally picked up 42 peaches -/
theorem sally_picked_42_peaches (initial current : ℕ) 
  (h_initial : initial = 13)
  (h_current : current = 55) :
  peaches_picked initial current = 42 := by
  sorry

end NUMINAMATH_CALUDE_sally_picked_42_peaches_l2750_275022


namespace NUMINAMATH_CALUDE_negation_of_symmetry_for_all_l2750_275078

-- Define a type for functions
variable {α : Type*} [LinearOrder α]

-- Define symmetry about y=x
def symmetric_about_y_eq_x (f : α → α) : Prop :=
  ∀ x y, f y = x ↔ f x = y

-- State the theorem
theorem negation_of_symmetry_for_all :
  (¬ ∀ f : α → α, symmetric_about_y_eq_x f) ↔
  (∃ f : α → α, ¬ symmetric_about_y_eq_x f) :=
sorry

end NUMINAMATH_CALUDE_negation_of_symmetry_for_all_l2750_275078


namespace NUMINAMATH_CALUDE_line_parallel_plane_condition_l2750_275059

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism relation for lines and planes
variable (parallelLine : Line → Line → Prop)
variable (parallelPlane : Plane → Plane → Prop)
variable (parallelLinePlane : Line → Plane → Prop)

-- Define the "lies in" relation for a line in a plane
variable (liesIn : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_plane_condition 
  (a : Line) (α β : Plane) :
  parallelPlane α β → liesIn a β → parallelLinePlane a α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_plane_condition_l2750_275059


namespace NUMINAMATH_CALUDE_gnome_ratio_is_half_l2750_275054

/-- Represents the ratio of gnomes with big noses to total gnomes -/
def gnome_ratio (total_gnomes red_hat_gnomes blue_hat_gnomes big_nose_blue_hat small_nose_red_hat : ℕ) : ℚ := 
  let big_nose_red_hat := red_hat_gnomes - small_nose_red_hat
  let total_big_nose := big_nose_blue_hat + big_nose_red_hat
  (total_big_nose : ℚ) / total_gnomes

theorem gnome_ratio_is_half :
  let total_gnomes : ℕ := 28
  let red_hat_gnomes : ℕ := (3 * total_gnomes) / 4
  let blue_hat_gnomes : ℕ := total_gnomes - red_hat_gnomes
  let big_nose_blue_hat : ℕ := 6
  let small_nose_red_hat : ℕ := 13
  gnome_ratio total_gnomes red_hat_gnomes blue_hat_gnomes big_nose_blue_hat small_nose_red_hat = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_gnome_ratio_is_half_l2750_275054


namespace NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l2750_275097

theorem factorization_of_2m_squared_minus_2 (m : ℝ) : 2 * m^2 - 2 = 2 * (m + 1) * (m - 1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_of_2m_squared_minus_2_l2750_275097


namespace NUMINAMATH_CALUDE_sandro_children_l2750_275029

/-- Calculates the total number of children for a person with a given number of sons
    and a ratio of daughters to sons. -/
def totalChildren (numSons : ℕ) (daughterToSonRatio : ℕ) : ℕ :=
  numSons + numSons * daughterToSonRatio

/-- Theorem stating that for a person with 3 sons and 6 times as many daughters as sons,
    the total number of children is 21. -/
theorem sandro_children :
  totalChildren 3 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sandro_children_l2750_275029


namespace NUMINAMATH_CALUDE_incorrect_height_calculation_l2750_275052

/-- Proves that the incorrect height of a student is 151 cm given the conditions of the problem -/
theorem incorrect_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  n = 30 ∧ 
  initial_avg = 175 ∧ 
  actual_height = 136 ∧ 
  actual_avg = 174.5 → 
  ∃ (incorrect_height : ℝ), 
    incorrect_height = 151 ∧ 
    n * initial_avg = (n - 1) * actual_avg + incorrect_height ∧
    n * actual_avg = (n - 1) * actual_avg + actual_height :=
by sorry

end NUMINAMATH_CALUDE_incorrect_height_calculation_l2750_275052


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_common_chord_l2750_275039

-- Define the curves C1 and C2
def C1 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 4

def C2 (x y : ℝ) : Prop := x^2 + y^2 = 4*y

-- Define the polar equation of the perpendicular bisector
def perpendicular_bisector (ρ θ : ℝ) : Prop :=
  ρ * Real.cos (θ - Real.pi/4) = Real.sqrt 2

-- Theorem statement
theorem perpendicular_bisector_of_common_chord :
  ∀ (x y ρ θ : ℝ), C1 x y → C2 x y →
  x = ρ * Real.cos θ → y = ρ * Real.sin θ →
  perpendicular_bisector ρ θ :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_common_chord_l2750_275039


namespace NUMINAMATH_CALUDE_roots_of_p_l2750_275019

def p (x : ℝ) : ℝ := x * (x + 3)^2 * (5 - x)

theorem roots_of_p :
  ∀ x : ℝ, p x = 0 ↔ x = 0 ∨ x = -3 ∨ x = 5 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_p_l2750_275019


namespace NUMINAMATH_CALUDE_brick_width_is_11_l2750_275024

-- Define the dimensions and quantities
def wall_length : ℝ := 200 -- in cm
def wall_width : ℝ := 300  -- in cm
def wall_height : ℝ := 2   -- in cm
def brick_length : ℝ := 25 -- in cm
def brick_height : ℝ := 6  -- in cm
def num_bricks : ℝ := 72.72727272727273

-- Define the theorem
theorem brick_width_is_11 :
  ∃ (brick_width : ℝ),
    brick_width = 11 ∧
    wall_length * wall_width * wall_height = num_bricks * brick_length * brick_width * brick_height :=
by sorry

end NUMINAMATH_CALUDE_brick_width_is_11_l2750_275024


namespace NUMINAMATH_CALUDE_income_percentage_l2750_275016

/-- Given that Mart's income is 60% more than Tim's income, and Tim's income is 60% less than Juan's income, 
    prove that Mart's income is 64% of Juan's income. -/
theorem income_percentage (juan tim mart : ℝ) 
  (h1 : tim = 0.4 * juan)  -- Tim's income is 60% less than Juan's
  (h2 : mart = 1.6 * tim)  -- Mart's income is 60% more than Tim's
  : mart = 0.64 * juan := by
  sorry


end NUMINAMATH_CALUDE_income_percentage_l2750_275016


namespace NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l2750_275092

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (containedIn : Line → Plane → Prop)
variable (parallelTo : Line → Plane → Prop)
variable (planeparallel : Plane → Plane → Prop)

-- State the theorem
theorem line_parallel_to_parallel_plane 
  (m : Line) (α β : Plane) :
  containedIn m α → planeparallel α β → parallelTo m β := by
  sorry

end NUMINAMATH_CALUDE_line_parallel_to_parallel_plane_l2750_275092


namespace NUMINAMATH_CALUDE_sarah_tic_tac_toe_wins_l2750_275013

/-- Represents the outcome of Sarah's tic-tac-toe games -/
structure TicTacToeOutcome where
  wins : ℕ
  ties : ℕ
  losses : ℕ
  total_games : ℕ
  net_earnings : ℤ

/-- Calculates the net earnings based on game outcomes -/
def calculate_earnings (outcome : TicTacToeOutcome) : ℤ :=
  4 * outcome.wins + outcome.ties - 3 * outcome.losses

theorem sarah_tic_tac_toe_wins : 
  ∀ (outcome : TicTacToeOutcome),
    outcome.total_games = 200 →
    outcome.ties = 60 →
    outcome.net_earnings = -84 →
    calculate_earnings outcome = outcome.net_earnings →
    outcome.wins + outcome.ties + outcome.losses = outcome.total_games →
    outcome.wins = 39 := by
  sorry


end NUMINAMATH_CALUDE_sarah_tic_tac_toe_wins_l2750_275013


namespace NUMINAMATH_CALUDE_h_range_l2750_275098

-- Define the function h
def h (x : ℝ) : ℝ := 3 * (x - 5)

-- State the theorem
theorem h_range :
  {y : ℝ | ∃ x : ℝ, x ≠ -9 ∧ h x = y} = {y : ℝ | y < -42 ∨ y > -42} :=
by sorry

end NUMINAMATH_CALUDE_h_range_l2750_275098


namespace NUMINAMATH_CALUDE_population_ratio_l2750_275088

/-- Given three cities X, Y, and Z, where the population of X is 5 times that of Y,
    and the population of Y is twice that of Z, prove that the ratio of the
    population of X to Z is 10:1 -/
theorem population_ratio (x y z : ℕ) (hxy : x = 5 * y) (hyz : y = 2 * z) :
  x / z = 10 := by
  sorry

end NUMINAMATH_CALUDE_population_ratio_l2750_275088


namespace NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l2750_275026

/-- Given a quadratic function with vertex (4, 10) and one x-intercept at (-1, 0),
    the x-coordinate of the other x-intercept is 9. -/
theorem other_x_intercept_of_quadratic (a b c : ℝ) :
  (∀ x, a * x^2 + b * x + c = 10 + a * (x - 4)^2) →  -- vertex form of quadratic
  a * (-1)^2 + b * (-1) + c = 0 →                    -- x-intercept at (-1, 0)
  ∃ x, x ≠ -1 ∧ a * x^2 + b * x + c = 0 ∧ x = 9      -- other x-intercept at 9
  := by sorry

end NUMINAMATH_CALUDE_other_x_intercept_of_quadratic_l2750_275026


namespace NUMINAMATH_CALUDE_outfits_count_l2750_275062

/-- The number of different outfits that can be created -/
def number_of_outfits (shirts : ℕ) (ties : ℕ) (pants : ℕ) (belts : ℕ) : ℕ :=
  shirts * pants * (ties + 1) * (belts + 1)

/-- Theorem stating the number of outfits given specific quantities of clothing items -/
theorem outfits_count :
  number_of_outfits 8 4 3 2 = 360 :=
by sorry

end NUMINAMATH_CALUDE_outfits_count_l2750_275062


namespace NUMINAMATH_CALUDE_function_ordering_l2750_275035

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Define the property of being monotonically decreasing on an interval
def is_monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

-- State the theorem
theorem function_ordering (h1 : is_even f) (h2 : is_monotone_decreasing_on (fun x => f (x - 2)) 0 2) :
  f 0 < f (-1) ∧ f (-1) < f 2 :=
sorry

end NUMINAMATH_CALUDE_function_ordering_l2750_275035


namespace NUMINAMATH_CALUDE_car_trip_duration_l2750_275096

/-- Represents the duration of a car trip with varying speeds. -/
def CarTrip (initial_speed initial_time additional_speed average_speed : ℝ) : Prop :=
  ∃ (total_time : ℝ),
    total_time > initial_time ∧
    (initial_speed * initial_time + additional_speed * (total_time - initial_time)) / total_time = average_speed

/-- The car trip lasts 12 hours given the specified conditions. -/
theorem car_trip_duration :
  CarTrip 45 4 75 65 → ∃ (total_time : ℝ), total_time = 12 :=
by sorry

end NUMINAMATH_CALUDE_car_trip_duration_l2750_275096


namespace NUMINAMATH_CALUDE_vector_parallel_proof_l2750_275082

def a (m : ℝ) : ℝ × ℝ := (m, 3)
def b : ℝ × ℝ := (-2, 2)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_parallel_proof (m : ℝ) :
  parallel ((a m).1 - b.1, (a m).2 - b.2) b → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_proof_l2750_275082


namespace NUMINAMATH_CALUDE_line_circle_separation_l2750_275043

theorem line_circle_separation (α β : ℝ) : 
  let m : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)
  let n : ℝ × ℝ := (3 * Real.cos β, 3 * Real.sin β)
  let angle_between := Real.arccos ((m.1 * n.1 + m.2 * n.2) / (Real.sqrt (m.1^2 + m.2^2) * Real.sqrt (n.1^2 + n.2^2)))
  let line_eq (x y : ℝ) := x * Real.cos α - y * Real.sin α + 1/2
  let circle_center : ℝ × ℝ := (Real.cos β, -Real.sin β)
  let circle_radius : ℝ := Real.sqrt 2 / 2
  let distance_to_line := |line_eq circle_center.1 circle_center.2|
  angle_between = π/3 → distance_to_line > circle_radius :=
by sorry

end NUMINAMATH_CALUDE_line_circle_separation_l2750_275043


namespace NUMINAMATH_CALUDE_fifteen_factorial_trailing_zeros_l2750_275083

/-- The factorial function -/
def factorial (n : ℕ) : ℕ := sorry

/-- The number of trailing zeros in n when expressed in base b -/
def trailingZeros (n : ℕ) (b : ℕ) : ℕ := sorry

/-- Base 18 expressed as 2 · 3² -/
def base18 : ℕ := 2 * 3^2

/-- The main theorem -/
theorem fifteen_factorial_trailing_zeros :
  trailingZeros (factorial 15) base18 = 3 := by sorry

end NUMINAMATH_CALUDE_fifteen_factorial_trailing_zeros_l2750_275083


namespace NUMINAMATH_CALUDE_dividend_divisor_change_l2750_275012

theorem dividend_divisor_change (a b : ℝ) (h : b ≠ 0) :
  (11 * a) / (10 * b) ≠ a / b :=
sorry

end NUMINAMATH_CALUDE_dividend_divisor_change_l2750_275012


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2750_275099

-- Define the propositions
variable (f : ℝ → ℝ)
def p (f : ℝ → ℝ) : Prop := ∃ c : ℝ, ∀ x : ℝ, deriv f x = c
def q (f : ℝ → ℝ) : Prop := ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b

-- Theorem statement
theorem p_necessary_not_sufficient_for_q :
  (∀ f : ℝ → ℝ, q f → p f) ∧ (∃ f : ℝ → ℝ, p f ∧ ¬q f) :=
sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l2750_275099


namespace NUMINAMATH_CALUDE_trig_identity_l2750_275061

theorem trig_identity (α β : Real) 
  (h : (Real.cos α)^6 / (Real.cos β)^3 + (Real.sin α)^6 / (Real.sin β)^3 = 1) :
  (Real.sin β)^6 / (Real.sin α)^3 + (Real.cos β)^6 / (Real.cos α)^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l2750_275061


namespace NUMINAMATH_CALUDE_correct_average_l2750_275030

theorem correct_average (numbers : Finset ℕ) (incorrect_sum : ℕ) (incorrect_number correct_number : ℕ) :
  numbers.card = 10 →
  incorrect_sum = 17 * 10 →
  incorrect_number = 26 →
  correct_number = 56 →
  (incorrect_sum - incorrect_number + correct_number) / numbers.card = 20 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l2750_275030


namespace NUMINAMATH_CALUDE_final_mixture_percentage_l2750_275007

/-- Percentage of material A in solution X -/
def x_percentage : ℝ := 0.20

/-- Percentage of material A in solution Y -/
def y_percentage : ℝ := 0.30

/-- Percentage of solution X in the final mixture -/
def x_mixture_percentage : ℝ := 0.80

/-- Calculate the percentage of material A in the final mixture -/
def final_percentage : ℝ := x_percentage * x_mixture_percentage + y_percentage * (1 - x_mixture_percentage)

/-- Theorem stating that the percentage of material A in the final mixture is 22% -/
theorem final_mixture_percentage : final_percentage = 0.22 := by
  sorry

end NUMINAMATH_CALUDE_final_mixture_percentage_l2750_275007


namespace NUMINAMATH_CALUDE_mythical_zoo_count_l2750_275079

theorem mythical_zoo_count (total_heads : ℕ) (total_legs : ℕ) : 
  total_heads = 300 → 
  total_legs = 798 → 
  ∃ (two_legged three_legged : ℕ), 
    two_legged + three_legged = total_heads ∧ 
    2 * two_legged + 3 * three_legged = total_legs ∧ 
    two_legged = 102 := by
sorry

end NUMINAMATH_CALUDE_mythical_zoo_count_l2750_275079


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2750_275009

-- Define the inequality
def inequality (x : ℝ) : Prop := |x^2 - 5*x + 6| < x^2 - 4

-- Define the solution set
def solution_set : Set ℝ := {x | x > 2}

-- Theorem statement
theorem inequality_solution_set :
  {x : ℝ | inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2750_275009


namespace NUMINAMATH_CALUDE_total_fans_l2750_275025

/-- The number of students who like basketball -/
def basketball_fans : ℕ := 7

/-- The number of students who like cricket -/
def cricket_fans : ℕ := 5

/-- The number of students who like both basketball and cricket -/
def both_fans : ℕ := 3

/-- Theorem: The number of students who like basketball or cricket or both is 9 -/
theorem total_fans : basketball_fans + cricket_fans - both_fans = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_fans_l2750_275025


namespace NUMINAMATH_CALUDE_complex_power_result_l2750_275060

theorem complex_power_result : (3 * Complex.cos (π / 4) + 3 * Complex.I * Complex.sin (π / 4)) ^ 4 = (-81 : ℂ) := by
  sorry

end NUMINAMATH_CALUDE_complex_power_result_l2750_275060


namespace NUMINAMATH_CALUDE_square_difference_l2750_275081

theorem square_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x - y = 4) : x^2 - y^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_l2750_275081


namespace NUMINAMATH_CALUDE_xy_squared_sum_l2750_275056

theorem xy_squared_sum (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 3) :
  x^2 * y + x * y^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_xy_squared_sum_l2750_275056


namespace NUMINAMATH_CALUDE_brown_family_seating_l2750_275001

/-- The number of ways to arrange boys and girls in a row --/
def seating_arrangements (num_boys : ℕ) (num_girls : ℕ) : ℕ :=
  Nat.factorial (num_boys + num_girls) - (Nat.factorial num_boys * Nat.factorial num_girls)

/-- Theorem stating the number of valid seating arrangements for 6 boys and 5 girls --/
theorem brown_family_seating :
  seating_arrangements 6 5 = 39830400 := by
  sorry

end NUMINAMATH_CALUDE_brown_family_seating_l2750_275001


namespace NUMINAMATH_CALUDE_angle_ABC_measure_l2750_275053

/-- A regular octagon with a square inscribed such that one side of the square
    coincides with one side of the octagon. -/
structure OctagonWithSquare where
  /-- The measure of an interior angle of the regular octagon -/
  octagon_interior_angle : ℝ
  /-- The measure of an interior angle of the square -/
  square_interior_angle : ℝ
  /-- A is a vertex of the octagon -/
  A : Point
  /-- B is the next vertex of the octagon after A -/
  B : Point
  /-- C is a vertex of the inscribed square on the line extended from side AB -/
  C : Point
  /-- The measure of angle ABC -/
  angle_ABC : ℝ
  /-- The octagon is regular -/
  octagon_regular : octagon_interior_angle = 135
  /-- The square has right angles -/
  square_right_angle : square_interior_angle = 90

/-- The measure of angle ABC in the described configuration is 67.5 degrees -/
theorem angle_ABC_measure (config : OctagonWithSquare) : config.angle_ABC = 67.5 := by
  sorry

end NUMINAMATH_CALUDE_angle_ABC_measure_l2750_275053


namespace NUMINAMATH_CALUDE_brad_lemonade_profit_l2750_275090

/-- Calculates the net profit from a lemonade stand given the specified conditions. -/
def lemonade_stand_profit (
  glasses_per_gallon : ℕ)
  (cost_per_gallon : ℚ)
  (gallons_made : ℕ)
  (price_per_glass : ℚ)
  (glasses_drunk : ℕ)
  (glasses_unsold : ℕ) : ℚ :=
  let total_glasses := glasses_per_gallon * gallons_made
  let glasses_sold := total_glasses - (glasses_drunk + glasses_unsold)
  let total_cost := cost_per_gallon * gallons_made
  let total_revenue := price_per_glass * glasses_sold
  total_revenue - total_cost

/-- Theorem stating that Brad's net profit is $14.00 given the specified conditions. -/
theorem brad_lemonade_profit :
  lemonade_stand_profit 16 3.5 2 1 5 6 = 14 := by
  sorry

end NUMINAMATH_CALUDE_brad_lemonade_profit_l2750_275090


namespace NUMINAMATH_CALUDE_min_delivery_time_75_minutes_l2750_275066

/-- Represents the train's cargo and delivery constraints -/
structure TrainDelivery where
  coal_cars : ℕ
  iron_cars : ℕ
  wood_cars : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  travel_time : ℕ

/-- Calculates the minimum number of stops required to deliver all cars -/
def min_stops (td : TrainDelivery) : ℕ :=
  max (td.coal_cars / td.max_coal_deposit)
      (max (td.iron_cars / td.max_iron_deposit)
           (td.wood_cars / td.max_wood_deposit))

/-- Calculates the total delivery time based on the number of stops -/
def total_delivery_time (td : TrainDelivery) : ℕ :=
  (min_stops td - 1) * td.travel_time

/-- The main theorem stating the minimum time required for delivery -/
theorem min_delivery_time_75_minutes (td : TrainDelivery) 
  (h1 : td.coal_cars = 6)
  (h2 : td.iron_cars = 12)
  (h3 : td.wood_cars = 2)
  (h4 : td.max_coal_deposit = 2)
  (h5 : td.max_iron_deposit = 3)
  (h6 : td.max_wood_deposit = 1)
  (h7 : td.travel_time = 25) :
  total_delivery_time td = 75 := by
  sorry

#eval total_delivery_time {
  coal_cars := 6,
  iron_cars := 12,
  wood_cars := 2,
  max_coal_deposit := 2,
  max_iron_deposit := 3,
  max_wood_deposit := 1,
  travel_time := 25
}

end NUMINAMATH_CALUDE_min_delivery_time_75_minutes_l2750_275066


namespace NUMINAMATH_CALUDE_extraneous_roots_no_solution_l2750_275080

-- Define the fractional equation
def fractional_equation (x a : ℝ) : Prop :=
  (x - a) / (x - 2) - 5 / x = 1

-- Theorem for extraneous roots
theorem extraneous_roots : 
  ∃ x, fractional_equation x 2 ∧ (x = 0 ∨ x = 2) :=
sorry

-- Theorem for no solution
theorem no_solution :
  (∀ x, ¬fractional_equation x (-3)) ∧
  (∀ x, ¬fractional_equation x 2) :=
sorry

end NUMINAMATH_CALUDE_extraneous_roots_no_solution_l2750_275080


namespace NUMINAMATH_CALUDE_binomial_expansion_ratio_l2750_275085

theorem binomial_expansion_ratio : 
  let n : ℕ := 10
  let k : ℕ := 5
  let a : ℕ := Nat.choose n k
  let b : ℤ := -Nat.choose n 3 * (-2)^3
  (b : ℚ) / a = -80 / 21 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_ratio_l2750_275085


namespace NUMINAMATH_CALUDE_point_A_on_curve_l2750_275046

/-- The equation of the curve C is x^2 - xy + y - 5 = 0 -/
def curve_equation (x y : ℝ) : Prop := x^2 - x*y + y - 5 = 0

/-- Point A lies on curve C -/
theorem point_A_on_curve : curve_equation (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_point_A_on_curve_l2750_275046


namespace NUMINAMATH_CALUDE_cells_after_three_divisions_l2750_275042

/-- The number of cells after n divisions, starting with 1 cell -/
def cells_after_divisions (n : ℕ) : ℕ := 2^n

/-- Theorem: After 3 divisions, the number of cells is 8 -/
theorem cells_after_three_divisions : cells_after_divisions 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_cells_after_three_divisions_l2750_275042


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2750_275075

theorem quadratic_transformation (x : ℝ) :
  x^2 - 6*x - 1 = 0 ↔ (x + 3)^2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2750_275075


namespace NUMINAMATH_CALUDE_intersection_sum_l2750_275004

/-- Given two functions f and g where
    f(x) = -|x - a| + b
    g(x) = |x - c| + d
    If f and g intersect at points (2, 5) and (8, 3), then a + c = 10 -/
theorem intersection_sum (a b c d : ℝ) : 
  (∀ x, -|x - a| + b = |x - c| + d → x = 2 ∨ x = 8) →
  -|2 - a| + b = 5 →
  -|8 - a| + b = 3 →
  |2 - c| + d = 5 →
  |8 - c| + d = 3 →
  a + c = 10 := by
  sorry

end NUMINAMATH_CALUDE_intersection_sum_l2750_275004


namespace NUMINAMATH_CALUDE_two_digit_number_swap_l2750_275031

theorem two_digit_number_swap : 
  ∃! (n : Finset ℕ), 
    (∀ x ∈ n, 10 ≤ x ∧ x < 100) ∧ 
    (∀ x ∈ n, let a := x / 10
              let b := x % 10
              (10 * b + a : ℚ) = (7 / 4) * x) ∧
    n.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_swap_l2750_275031


namespace NUMINAMATH_CALUDE_ratio_problem_l2750_275028

theorem ratio_problem (x : ℚ) : x / 8 = 6 / (4 * 60) ↔ x = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2750_275028


namespace NUMINAMATH_CALUDE_root_difference_ratio_l2750_275076

theorem root_difference_ratio (a b : ℝ) : 
  a^4 - 7*a - 3 = 0 → 
  b^4 - 7*b - 3 = 0 → 
  a > b → 
  (a - b) / (a^4 - b^4) = 1/7 := by
sorry

end NUMINAMATH_CALUDE_root_difference_ratio_l2750_275076


namespace NUMINAMATH_CALUDE_unique_integer_solution_l2750_275091

theorem unique_integer_solution : ∃! (x y : ℤ), x^4 + y^2 - 4*y + 4 = 4 := by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l2750_275091


namespace NUMINAMATH_CALUDE_distance_between_trees_441_22_l2750_275057

/-- The distance between consecutive trees in a yard -/
def distance_between_trees (yard_length : ℕ) (num_trees : ℕ) : ℚ :=
  (yard_length : ℚ) / (num_trees - 1 : ℚ)

/-- Theorem: The distance between consecutive trees in a 441-metre yard with 22 trees is 21 metres -/
theorem distance_between_trees_441_22 :
  distance_between_trees 441 22 = 21 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_441_22_l2750_275057


namespace NUMINAMATH_CALUDE_unique_odd_k_for_sum_1372_l2750_275070

theorem unique_odd_k_for_sum_1372 :
  ∃! (k : ℤ), ∃ (m : ℕ), 
    (k % 2 = 1) ∧ 
    (m > 0) ∧ 
    (k * m + 5 * (m * (m - 1) / 2) = 1372) ∧ 
    (k = 211) := by
  sorry

end NUMINAMATH_CALUDE_unique_odd_k_for_sum_1372_l2750_275070


namespace NUMINAMATH_CALUDE_krishans_money_l2750_275041

/-- Proves that Krishan has Rs. 4335 given the conditions of the problem -/
theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 735 →
  krishan = 4335 := by
  sorry

end NUMINAMATH_CALUDE_krishans_money_l2750_275041


namespace NUMINAMATH_CALUDE_play_dough_quantity_l2750_275018

def lego_price : ℕ := 250
def sword_price : ℕ := 120
def dough_price : ℕ := 35
def lego_quantity : ℕ := 3
def sword_quantity : ℕ := 7
def total_paid : ℕ := 1940

theorem play_dough_quantity :
  (total_paid - (lego_price * lego_quantity + sword_price * sword_quantity)) / dough_price = 10 := by
  sorry

end NUMINAMATH_CALUDE_play_dough_quantity_l2750_275018


namespace NUMINAMATH_CALUDE_square_configuration_counts_l2750_275044

/-- A configuration of points and line segments in a square -/
structure SquareConfiguration where
  /-- The number of interior points in the square -/
  interior_points : Nat
  /-- The total number of line segments -/
  line_segments : Nat
  /-- The total number of triangles formed -/
  triangles : Nat
  /-- No three points (including square vertices) are collinear -/
  no_collinear_triple : Prop
  /-- No two segments (except at endpoints) share common points -/
  no_intersecting_segments : Prop

/-- Theorem about the number of line segments and triangles in a specific square configuration -/
theorem square_configuration_counts (config : SquareConfiguration) :
  config.interior_points = 1000 →
  config.no_collinear_triple →
  config.no_intersecting_segments →
  config.line_segments = 3001 ∧ config.triangles = 2002 := by
  sorry


end NUMINAMATH_CALUDE_square_configuration_counts_l2750_275044


namespace NUMINAMATH_CALUDE_inequality_proof_l2750_275087

theorem inequality_proof (a b c d : ℝ) (ha : a ≥ -1) (hb : b ≥ -1) (hc : c ≥ -1) (hd : d ≥ -1) :
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3/4) * (a + b + c + d) ∧
  ∀ k > 3/4, ∃ a' b' c' d' : ℝ, a' ≥ -1 ∧ b' ≥ -1 ∧ c' ≥ -1 ∧ d' ≥ -1 ∧
    a'^3 + b'^3 + c'^3 + d'^3 + 1 < k * (a' + b' + c' + d') :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2750_275087


namespace NUMINAMATH_CALUDE_team_not_losing_probability_l2750_275000

/-- Represents the positions Player A can play -/
inductive Position
| CenterForward
| Winger
| AttackingMidfielder

/-- The appearance rate for each position -/
def appearanceRate (pos : Position) : ℝ :=
  match pos with
  | .CenterForward => 0.3
  | .Winger => 0.5
  | .AttackingMidfielder => 0.2

/-- The probability of the team losing when Player A plays in each position -/
def losingProbability (pos : Position) : ℝ :=
  match pos with
  | .CenterForward => 0.3
  | .Winger => 0.2
  | .AttackingMidfielder => 0.2

/-- The probability of the team not losing when Player A participates -/
def teamNotLosingProbability : ℝ :=
  (appearanceRate Position.CenterForward * (1 - losingProbability Position.CenterForward)) +
  (appearanceRate Position.Winger * (1 - losingProbability Position.Winger)) +
  (appearanceRate Position.AttackingMidfielder * (1 - losingProbability Position.AttackingMidfielder))

theorem team_not_losing_probability :
  teamNotLosingProbability = 0.77 := by
  sorry

end NUMINAMATH_CALUDE_team_not_losing_probability_l2750_275000


namespace NUMINAMATH_CALUDE_circular_arrangement_theorem_l2750_275064

theorem circular_arrangement_theorem (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ (a b : ℕ), a = 6 ∧ b = 16 ∧ a < n ∧ b < n ∧ (b - a) * 2 = n) :
  n = 20 := by
sorry

end NUMINAMATH_CALUDE_circular_arrangement_theorem_l2750_275064


namespace NUMINAMATH_CALUDE_max_non_managers_proof_l2750_275023

/-- Represents the number of managers in department A -/
def managers : ℕ := 9

/-- Represents the ratio of managers to non-managers in department A -/
def manager_ratio : ℚ := 7 / 37

/-- Represents the ratio of specialists to generalists in department A -/
def specialist_ratio : ℚ := 2 / 1

/-- Calculates the maximum number of non-managers in department A -/
def max_non_managers : ℕ := 39

/-- Theorem stating that 39 is the maximum number of non-managers in department A -/
theorem max_non_managers_proof :
  ∀ n : ℕ, 
    (n : ℚ) / managers > manager_ratio ∧ 
    n % 3 = 0 ∧
    (2 * n / 3 : ℚ) / (n / 3 : ℚ) = specialist_ratio →
    n ≤ max_non_managers :=
by sorry

end NUMINAMATH_CALUDE_max_non_managers_proof_l2750_275023


namespace NUMINAMATH_CALUDE_yanna_afternoon_butter_cookies_l2750_275055

/-- Represents the number of butter cookies Yanna baked in the afternoon -/
def afternoon_butter_cookies : ℕ := sorry

/-- Represents the number of butter cookies Yanna baked in the morning -/
def morning_butter_cookies : ℕ := 20

/-- Represents the number of biscuits Yanna baked in the morning -/
def morning_biscuits : ℕ := 40

/-- Represents the number of biscuits Yanna baked in the afternoon -/
def afternoon_biscuits : ℕ := 20

theorem yanna_afternoon_butter_cookies :
  afternoon_butter_cookies = 20 ∧
  morning_butter_cookies + afternoon_butter_cookies + 30 =
  morning_biscuits + afternoon_biscuits :=
sorry

end NUMINAMATH_CALUDE_yanna_afternoon_butter_cookies_l2750_275055


namespace NUMINAMATH_CALUDE_final_balance_is_correct_l2750_275040

/-- Represents a bank account with transactions and interest --/
structure BankAccount where
  initialBalance : ℝ
  annualInterestRate : ℝ
  monthlyInterestRate : ℝ
  shoeWithdrawalPercent : ℝ
  shoeDepositPercent : ℝ
  paycheckDepositPercent : ℝ
  giftWithdrawalPercent : ℝ

/-- Calculates the final balance after all transactions and interest --/
def finalBalance (account : BankAccount) : ℝ :=
  let shoeWithdrawal := account.initialBalance * account.shoeWithdrawalPercent
  let balanceAfterShoes := account.initialBalance - shoeWithdrawal
  let shoeDeposit := shoeWithdrawal * account.shoeDepositPercent
  let balanceAfterShoeDeposit := balanceAfterShoes + shoeDeposit
  let januaryInterest := balanceAfterShoeDeposit * account.monthlyInterestRate
  let balanceAfterJanuary := balanceAfterShoeDeposit + januaryInterest
  let paycheckDeposit := shoeWithdrawal * account.paycheckDepositPercent
  let balanceAfterPaycheck := balanceAfterJanuary + paycheckDeposit
  let februaryInterest := balanceAfterPaycheck * account.monthlyInterestRate
  let balanceAfterFebruary := balanceAfterPaycheck + februaryInterest
  let giftWithdrawal := balanceAfterFebruary * account.giftWithdrawalPercent
  let balanceAfterGift := balanceAfterFebruary - giftWithdrawal
  let marchInterest := balanceAfterGift * account.monthlyInterestRate
  balanceAfterGift + marchInterest

/-- Theorem stating that the final balance is correct --/
theorem final_balance_is_correct (account : BankAccount) : 
  account.initialBalance = 1200 ∧
  account.annualInterestRate = 0.03 ∧
  account.monthlyInterestRate = account.annualInterestRate / 12 ∧
  account.shoeWithdrawalPercent = 0.08 ∧
  account.shoeDepositPercent = 0.25 ∧
  account.paycheckDepositPercent = 1.5 ∧
  account.giftWithdrawalPercent = 0.05 →
  finalBalance account = 1217.15 := by
  sorry


end NUMINAMATH_CALUDE_final_balance_is_correct_l2750_275040


namespace NUMINAMATH_CALUDE_total_worth_is_14000_l2750_275068

/-- The cost of the ring John gave to his fiancee -/
def ring_cost : ℕ := 4000

/-- The cost of the car John gave to his fiancee -/
def car_cost : ℕ := 2000

/-- The cost of the diamond brace John gave to his fiancee -/
def brace_cost : ℕ := 2 * ring_cost

/-- The total worth of the presents John gave to his fiancee -/
def total_worth : ℕ := ring_cost + car_cost + brace_cost

theorem total_worth_is_14000 : total_worth = 14000 := by
  sorry

end NUMINAMATH_CALUDE_total_worth_is_14000_l2750_275068


namespace NUMINAMATH_CALUDE_mom_has_enough_money_l2750_275036

/-- Proves that the amount of money mom brought is sufficient to buy the discounted clothing item -/
theorem mom_has_enough_money (mom_money : ℝ) (original_price : ℝ) 
  (h1 : mom_money = 230)
  (h2 : original_price = 268)
  : mom_money ≥ 0.8 * original_price := by
  sorry

end NUMINAMATH_CALUDE_mom_has_enough_money_l2750_275036


namespace NUMINAMATH_CALUDE_probability_cos_geq_half_is_two_thirds_l2750_275005

noncomputable def probability_cos_geq_half : ℝ := by sorry

theorem probability_cos_geq_half_is_two_thirds :
  probability_cos_geq_half = 2/3 := by sorry

end NUMINAMATH_CALUDE_probability_cos_geq_half_is_two_thirds_l2750_275005


namespace NUMINAMATH_CALUDE_tan_8100_degrees_l2750_275015

theorem tan_8100_degrees : Real.tan (8100 * π / 180) = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_8100_degrees_l2750_275015


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_ratio_l2750_275049

theorem geometric_sequence_minimum_ratio :
  ∀ (a : ℕ → ℕ) (q : ℚ),
  (∀ n : ℕ, 1 ≤ n → n < 2016 → a (n + 1) = a n * q) →
  (1 < q ∧ q < 2) →
  (∀ r : ℚ, 1 < r ∧ r < 2 → a 2016 ≤ a 1 * r^2015) →
  q = 6/5 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_ratio_l2750_275049


namespace NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l2750_275093

-- Define propositions p and q
def p (x : ℝ) : Prop := abs (x + 2) > 2
def q (x : ℝ) : Prop := 1 / (3 - x) > 1

-- Define the negations of p and q
def not_p (x : ℝ) : Prop := ¬(p x)
def not_q (x : ℝ) : Prop := ¬(q x)

-- Theorem stating that ¬q is necessary but not sufficient for ¬p
theorem not_q_necessary_not_sufficient_for_not_p :
  (∀ x : ℝ, not_p x → not_q x) ∧ 
  ¬(∀ x : ℝ, not_q x → not_p x) :=
sorry

end NUMINAMATH_CALUDE_not_q_necessary_not_sufficient_for_not_p_l2750_275093


namespace NUMINAMATH_CALUDE_mean_of_scores_l2750_275047

def scores : List ℝ := [69, 68, 70, 61, 74, 62, 65, 74]

theorem mean_of_scores :
  (scores.sum / scores.length : ℝ) = 67.875 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_scores_l2750_275047


namespace NUMINAMATH_CALUDE_juan_oranges_picked_l2750_275050

theorem juan_oranges_picked (total : ℕ) (del_per_day : ℕ) (del_days : ℕ) : 
  total = 107 → del_per_day = 23 → del_days = 2 → 
  total - (del_per_day * del_days) = 61 := by
  sorry

end NUMINAMATH_CALUDE_juan_oranges_picked_l2750_275050


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2750_275033

theorem quadratic_roots_sum (a b : ℝ) : 
  (∀ x, ax^2 + bx + 2 = 0 ↔ x = -1/2 ∨ x = 1/3) → 
  a + b = -14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2750_275033


namespace NUMINAMATH_CALUDE_steve_bakes_more_apple_pies_l2750_275032

/-- The number of days Steve bakes apple pies in a week -/
def apple_pie_days : ℕ := 3

/-- The number of days Steve bakes cherry pies in a week -/
def cherry_pie_days : ℕ := 2

/-- The number of pies Steve bakes per day -/
def pies_per_day : ℕ := 12

/-- The difference between apple pies and cherry pies baked in a week -/
def pie_difference : ℕ := apple_pie_days * pies_per_day - cherry_pie_days * pies_per_day

theorem steve_bakes_more_apple_pies : pie_difference = 12 := by
  sorry

end NUMINAMATH_CALUDE_steve_bakes_more_apple_pies_l2750_275032


namespace NUMINAMATH_CALUDE_football_count_white_patch_count_l2750_275034

/- Define the number of students -/
def num_students : ℕ := 36

/- Define the number of footballs -/
def num_footballs : ℕ := 27

/- Define the number of black patches -/
def num_black_patches : ℕ := 12

/- Define the number of white patches -/
def num_white_patches : ℕ := 20

/- Theorem for the number of footballs -/
theorem football_count : 
  (num_students - 9 = num_footballs) ∧ 
  (num_students / 2 + 9 = num_footballs) := by
  sorry

/- Theorem for the number of white patches -/
theorem white_patch_count :
  2 * num_black_patches * 5 = 6 * num_white_patches := by
  sorry

end NUMINAMATH_CALUDE_football_count_white_patch_count_l2750_275034


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2750_275027

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

-- Define set A
def A : Set ℝ := {x | x^2 - (floor x : ℝ) = 2}

-- Define set B
def B : Set ℝ := {x | -2 < x ∧ x < 2}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {-1, Real.sqrt 3} :=
by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2750_275027


namespace NUMINAMATH_CALUDE_room_population_change_l2750_275021

theorem room_population_change (initial_men initial_women : ℕ) : 
  initial_men / initial_women = 4 / 5 →
  ∃ (current_women : ℕ),
    initial_men + 2 = 14 ∧
    current_women = 2 * (initial_women - 3) ∧
    current_women = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_room_population_change_l2750_275021


namespace NUMINAMATH_CALUDE_triangle_rational_area_l2750_275010

/-- Triangle with rational side lengths and angle bisectors has rational area -/
theorem triangle_rational_area (a b c fa fb fc : ℚ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- positive side lengths
  fa > 0 ∧ fb > 0 ∧ fc > 0 →  -- positive angle bisector lengths
  ∃ (area : ℚ), area > 0 ∧ area^2 = (a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c) / 16 :=
sorry

end NUMINAMATH_CALUDE_triangle_rational_area_l2750_275010


namespace NUMINAMATH_CALUDE_rectangle_area_increase_l2750_275002

theorem rectangle_area_increase (p a : ℝ) (h_a : a > 0) :
  let perimeter := 2 * p
  let increase := a
  let area_increase := 
    fun (x y : ℝ) => 
      ((x + increase) * (y + increase)) - (x * y)
  ∀ x y, x + y = p → area_increase x y = a * (p + a) := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_increase_l2750_275002


namespace NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l2750_275045

/-- If 46 cows eat 46 bags of husk in 46 days, then one cow will eat one bag of husk in 46 days. -/
theorem dairy_farm_husk_consumption 
  (cows : ℕ) (bags : ℕ) (days : ℕ) (one_cow_days : ℕ) :
  cows = 46 → bags = 46 → days = 46 → 
  (cows * bags = cows * days) →
  one_cow_days = 46 :=
by sorry

end NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l2750_275045


namespace NUMINAMATH_CALUDE_max_profit_at_25_yuan_manager_decision_suboptimal_l2750_275017

/-- Profit function based on price reduction -/
def profit (x : ℝ) : ℝ := (2 * x - 20) * (40 - x)

/-- Initial daily sales -/
def initial_sales : ℝ := 20

/-- Initial profit per piece -/
def initial_profit_per_piece : ℝ := 40

/-- Sales increase per yuan of price reduction -/
def sales_increase_rate : ℝ := 2

/-- Theorem stating the maximum profit and corresponding price reduction -/
theorem max_profit_at_25_yuan :
  ∃ (max_reduction : ℝ) (max_profit : ℝ),
    max_reduction = 25 ∧
    max_profit = 1250 ∧
    ∀ x, 0 ≤ x ∧ x ≤ 40 → profit x ≤ max_profit :=
sorry

/-- Theorem comparing manager's decision to optimal decision -/
theorem manager_decision_suboptimal (manager_reduction : ℝ) (h : manager_reduction = 15) :
  ∃ (optimal_reduction : ℝ) (optimal_profit : ℝ),
    optimal_reduction ≠ manager_reduction ∧
    optimal_profit > profit manager_reduction :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_25_yuan_manager_decision_suboptimal_l2750_275017


namespace NUMINAMATH_CALUDE_base_conversion_subtraction_l2750_275037

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- The problem statement -/
theorem base_conversion_subtraction :
  let base7_number := [3, 0, 4, 2, 5]  -- 52403 in base 7 (least significant digit first)
  let base5_number := [5, 4, 3, 0, 2]  -- 20345 in base 5 (least significant digit first)
  toBase10 base7_number 7 - toBase10 base5_number 5 = 11540 := by
sorry

end NUMINAMATH_CALUDE_base_conversion_subtraction_l2750_275037


namespace NUMINAMATH_CALUDE_negative_square_cubed_l2750_275069

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l2750_275069


namespace NUMINAMATH_CALUDE_coin_weighing_strategy_exists_l2750_275048

/-- Represents the possible weights of a coin type -/
inductive CoinWeight
  | Five
  | Six
  | Seven
  | Eight

/-- Represents the result of a weighing -/
inductive WeighingResult
  | LeftHeavier
  | RightHeavier
  | Equal

/-- Represents a weighing strategy -/
structure WeighingStrategy :=
  (firstWeighing : WeighingResult → Option WeighingResult)

/-- Represents the coin set -/
structure CoinSet :=
  (doubloonWeight : CoinWeight)
  (crownWeight : CoinWeight)

/-- Determines if a weighing strategy can identify the exact weights -/
def canIdentifyWeights (strategy : WeighingStrategy) (coins : CoinSet) : Prop :=
  ∃ (result1 : WeighingResult) (result2 : Option WeighingResult),
    (result2 = strategy.firstWeighing result1) ∧
    (∀ (otherCoins : CoinSet),
      (otherCoins ≠ coins) →
      (∃ (otherResult1 : WeighingResult) (otherResult2 : Option WeighingResult),
        (otherResult2 = strategy.firstWeighing otherResult1) ∧
        ((otherResult1 ≠ result1) ∨ (otherResult2 ≠ result2))))

theorem coin_weighing_strategy_exists :
  ∃ (strategy : WeighingStrategy),
    ∀ (coins : CoinSet),
      (coins.doubloonWeight = CoinWeight.Five ∨ coins.doubloonWeight = CoinWeight.Six) →
      (coins.crownWeight = CoinWeight.Seven ∨ coins.crownWeight = CoinWeight.Eight) →
      canIdentifyWeights strategy coins :=
by sorry


end NUMINAMATH_CALUDE_coin_weighing_strategy_exists_l2750_275048


namespace NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2750_275014

theorem quadratic_coefficient_sum (p q a b : ℝ) : 
  (∀ x, -x^2 + p*x + q = 0 ↔ x = a ∨ x = b) →
  b < 1 →
  1 < a →
  p + q > 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_sum_l2750_275014


namespace NUMINAMATH_CALUDE_blue_balls_count_l2750_275073

theorem blue_balls_count (total : ℕ) (removed : ℕ) (prob : ℚ) : 
  total = 35 → 
  removed = 5 → 
  prob = 5 / 21 → 
  (∃ initial : ℕ, 
    initial ≤ total ∧ 
    (initial - removed : ℚ) / (total - removed : ℚ) = prob ∧ 
    initial = 12) :=
by sorry

end NUMINAMATH_CALUDE_blue_balls_count_l2750_275073


namespace NUMINAMATH_CALUDE_giyun_distance_to_school_l2750_275020

/-- The distance between Giyun's house and school -/
def distance_to_school (step_length : ℝ) (steps_per_minute : ℕ) (time_taken : ℕ) : ℝ :=
  step_length * (steps_per_minute : ℝ) * time_taken

/-- Theorem stating the distance between Giyun's house and school -/
theorem giyun_distance_to_school :
  distance_to_school 0.75 70 13 = 682.5 := by
  sorry

end NUMINAMATH_CALUDE_giyun_distance_to_school_l2750_275020


namespace NUMINAMATH_CALUDE_fraction_addition_l2750_275051

theorem fraction_addition (x : ℝ) (h : x ≠ 1) : 
  (1 : ℝ) / (x - 1) + (3 : ℝ) / (x - 1) = (4 : ℝ) / (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l2750_275051


namespace NUMINAMATH_CALUDE_hushpuppy_cooking_time_l2750_275065

/-- Given the conditions of Walter's hushpuppy cooking scenario, 
    prove that it takes 80 minutes to cook all the hushpuppies. -/
theorem hushpuppy_cooking_time :
  let guests : ℕ := 20
  let hushpuppies_per_guest : ℕ := 5
  let hushpuppies_per_batch : ℕ := 10
  let minutes_per_batch : ℕ := 8

  let total_hushpuppies : ℕ := guests * hushpuppies_per_guest
  let total_batches : ℕ := total_hushpuppies / hushpuppies_per_batch
  let total_minutes : ℕ := total_batches * minutes_per_batch

  total_minutes = 80 := by sorry

end NUMINAMATH_CALUDE_hushpuppy_cooking_time_l2750_275065


namespace NUMINAMATH_CALUDE_min_value_theorem_l2750_275038

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  (1 / a + 3 / b) ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2750_275038


namespace NUMINAMATH_CALUDE_problem_statement_l2750_275006

theorem problem_statement (w x y : ℝ) 
  (h1 : 7 / w + 7 / x = 7 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  x = 0.5 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l2750_275006
