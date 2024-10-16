import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_properties_l1373_137329

/-- A quadratic function passing through two given points -/
def QuadraticFunction (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 3

theorem quadratic_function_properties (a b : ℝ) :
  (QuadraticFunction a b (-3) = 0) ∧
  (QuadraticFunction a b 2 = -5) →
  (∀ x, QuadraticFunction a b x = -x^2 - 2*x + 3) ∧
  (∃ x y, x = -1 ∧ y = 4 ∧ ∀ t, QuadraticFunction a b t ≤ QuadraticFunction a b x) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l1373_137329


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_30_and_18_l1373_137363

theorem arithmetic_mean_of_30_and_18 : (30 + 18) / 2 = 24 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_30_and_18_l1373_137363


namespace NUMINAMATH_CALUDE_unique_divisible_by_18_l1373_137358

/-- The function that constructs the four-digit number x47x from a single digit x -/
def construct_number (x : ℕ) : ℕ := 1000 * x + 470 + x

/-- Predicate that checks if a number is a single digit -/
def is_single_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 9

theorem unique_divisible_by_18 :
  ∃! x : ℕ, is_single_digit x ∧ (construct_number x) % 18 = 0 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_unique_divisible_by_18_l1373_137358


namespace NUMINAMATH_CALUDE_modular_arithmetic_problem_l1373_137378

theorem modular_arithmetic_problem (m : ℕ) : 
  13^5 % 7 = m → 0 ≤ m → m < 7 → m = 0 := by
  sorry

end NUMINAMATH_CALUDE_modular_arithmetic_problem_l1373_137378


namespace NUMINAMATH_CALUDE_fourth_child_age_is_eight_l1373_137348

/-- The age of the first child -/
def first_child_age : ℕ := 15

/-- The age difference between the first and second child -/
def age_diff_first_second : ℕ := 1

/-- The age of the second child when the third child was born -/
def second_child_age_at_third_birth : ℕ := 4

/-- The age difference between the third and fourth child -/
def age_diff_third_fourth : ℕ := 2

/-- The age of the fourth child -/
def fourth_child_age : ℕ := first_child_age - age_diff_first_second - second_child_age_at_third_birth - age_diff_third_fourth

theorem fourth_child_age_is_eight : fourth_child_age = 8 := by
  sorry

end NUMINAMATH_CALUDE_fourth_child_age_is_eight_l1373_137348


namespace NUMINAMATH_CALUDE_web_pages_scientific_notation_l1373_137317

/-- The number of web pages found when searching for "Mount Fanjing" in "Sogou" -/
def web_pages : ℕ := 1630000

/-- The scientific notation representation of the number of web pages -/
def scientific_notation : ℝ := 1.63 * (10 : ℝ) ^ 6

/-- Theorem stating that the number of web pages is equal to its scientific notation representation -/
theorem web_pages_scientific_notation : (web_pages : ℝ) = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_web_pages_scientific_notation_l1373_137317


namespace NUMINAMATH_CALUDE_expression_simplification_l1373_137307

theorem expression_simplification (x y m : ℝ) 
  (h1 : (x - 5)^2 + |m - 1| = 0)
  (h2 : y + 1 = 5) :
  (2*x^2 - 3*x*y - 4*y^2) - m*(3*x^2 - x*y + 9*y^2) = -273 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1373_137307


namespace NUMINAMATH_CALUDE_gyeong_hun_climb_l1373_137359

/-- Gyeong-hun's mountain climbing problem -/
theorem gyeong_hun_climb (uphill_speed downhill_speed : ℝ)
                         (downhill_extra_distance total_time : ℝ)
                         (h1 : uphill_speed = 3)
                         (h2 : downhill_speed = 4)
                         (h3 : downhill_extra_distance = 2)
                         (h4 : total_time = 4) :
  ∃ (distance : ℝ),
    distance / uphill_speed + (distance + downhill_extra_distance) / downhill_speed = total_time ∧
    distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_gyeong_hun_climb_l1373_137359


namespace NUMINAMATH_CALUDE_calculate_expression_l1373_137397

theorem calculate_expression : 
  Real.sqrt 12 - 3 - ((1/3) * Real.sqrt 27 - Real.sqrt 9) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l1373_137397


namespace NUMINAMATH_CALUDE_wash_time_is_three_hours_l1373_137319

/-- The number of hours required to wash all clothes given the number of items and washing machine capacity -/
def wash_time (shirts pants sweaters jeans : ℕ) (max_items_per_cycle : ℕ) (minutes_per_cycle : ℕ) : ℚ :=
  let total_items := shirts + pants + sweaters + jeans
  let num_cycles := (total_items + max_items_per_cycle - 1) / max_items_per_cycle
  (num_cycles * minutes_per_cycle : ℚ) / 60

/-- Theorem stating that it takes 3 hours to wash all the clothes under given conditions -/
theorem wash_time_is_three_hours :
  wash_time 18 12 17 13 15 45 = 3 := by
  sorry

end NUMINAMATH_CALUDE_wash_time_is_three_hours_l1373_137319


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1373_137347

def set_A : Set ℝ := {x | x^2 = 1}
def set_B : Set ℝ := {x | x^2 - 2*x - 3 = 0}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {-1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1373_137347


namespace NUMINAMATH_CALUDE_stratified_sampling_l1373_137371

theorem stratified_sampling (total_employees : ℕ) (male_employees : ℕ) (sample_size : ℕ) :
  total_employees = 750 →
  male_employees = 300 →
  sample_size = 45 →
  (sample_size - (male_employees * sample_size / total_employees) : ℕ) = 27 := by sorry

end NUMINAMATH_CALUDE_stratified_sampling_l1373_137371


namespace NUMINAMATH_CALUDE_largest_n_for_equation_l1373_137392

theorem largest_n_for_equation : 
  (∃ (n : ℕ), n > 0 ∧ 
    (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
      n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12) ∧
    (∀ (m : ℕ), m > n → 
      ¬(∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
        m^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12))) ∧
  (∀ (n : ℕ), (∃ (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    n^2 = x^2 + y^2 + z^2 + 2*x*y + 2*y*z + 2*z*x + 2*x + 2*y + 2*z - 12) →
    n ≤ 6) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_for_equation_l1373_137392


namespace NUMINAMATH_CALUDE_washer_cost_l1373_137373

/-- Given a washer-dryer combination costing 1200 dollars, where the washer costs 220 dollars more than the dryer, prove that the washer costs 710 dollars. -/
theorem washer_cost (total : ℝ) (difference : ℝ) (washer : ℝ) (dryer : ℝ) : 
  total = 1200 →
  difference = 220 →
  washer = dryer + difference →
  total = washer + dryer →
  washer = 710 := by
sorry

end NUMINAMATH_CALUDE_washer_cost_l1373_137373


namespace NUMINAMATH_CALUDE_range_of_f_l1373_137330

def f (x : ℕ) : ℤ := x^2 - 2*x

def domain : Set ℕ := {0, 1, 2, 3}

theorem range_of_f :
  {y : ℤ | ∃ x ∈ domain, f x = y} = {-1, 0, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1373_137330


namespace NUMINAMATH_CALUDE_total_games_in_season_l1373_137369

theorem total_games_in_season (total_teams : ℕ) (num_divisions : ℕ) (teams_per_division : ℕ)
  (intra_division_games : ℕ) (inter_division_games : ℕ)
  (h1 : total_teams = 24)
  (h2 : num_divisions = 3)
  (h3 : teams_per_division = 8)
  (h4 : total_teams = num_divisions * teams_per_division)
  (h5 : intra_division_games = 3)
  (h6 : inter_division_games = 2) :
  (total_teams * (((teams_per_division - 1) * intra_division_games) +
   ((total_teams - teams_per_division) * inter_division_games))) / 2 = 636 := by
  sorry

end NUMINAMATH_CALUDE_total_games_in_season_l1373_137369


namespace NUMINAMATH_CALUDE_existence_of_pairs_l1373_137338

theorem existence_of_pairs (N : ℕ) : ∃ (a b c d : ℕ),
  (a + b = c + d) ∧ 
  (c * d = N * (a * b)) ∧
  (a = 4 * N - 2) ∧ 
  (b = 1) ∧ 
  (c = 2 * N) ∧ 
  (d = 2 * N - 1) := by
sorry

end NUMINAMATH_CALUDE_existence_of_pairs_l1373_137338


namespace NUMINAMATH_CALUDE_library_books_checkout_l1373_137383

theorem library_books_checkout (total : ℕ) (ratio_nf : ℕ) (ratio_f : ℕ) (h1 : total = 52) (h2 : ratio_nf = 7) (h3 : ratio_f = 6) : 
  (total * ratio_f) / (ratio_nf + ratio_f) = 24 := by
  sorry

end NUMINAMATH_CALUDE_library_books_checkout_l1373_137383


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1373_137361

theorem sum_of_repeating_decimals : 
  (2 : ℚ) / 9 + (2 : ℚ) / 99 + (2 : ℚ) / 9999 = 224422 / 9999 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l1373_137361


namespace NUMINAMATH_CALUDE_root_relationship_l1373_137351

theorem root_relationship (p q : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x + q = 0 ∧ y^2 + p*y + q = 0 ∧ y = 2*x) → 
  2*p^2 = 9*q := by
sorry

end NUMINAMATH_CALUDE_root_relationship_l1373_137351


namespace NUMINAMATH_CALUDE_polynomial_roots_l1373_137332

theorem polynomial_roots : ∃ (p : ℝ → ℝ), 
  (∀ x, p x = 8*x^4 + 14*x^3 - 66*x^2 + 40*x) ∧ 
  (p 0 = 0) ∧ (p (1/2) = 0) ∧ (p 2 = 0) ∧ (p (-5) = 0) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1373_137332


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1373_137304

/-- The constant term in the expansion of (3x + 2/x)^8 is 90720 -/
theorem constant_term_binomial_expansion : 
  (Finset.sum (Finset.range 9) (fun k => Nat.choose 8 k * 3^(8-k) * 2^k * (if k = 4 then 1 else 0))) = 90720 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l1373_137304


namespace NUMINAMATH_CALUDE_min_cut_area_for_given_board_l1373_137360

/-- Represents a rectangular board with a damaged corner -/
structure Board :=
  (length : ℝ)
  (width : ℝ)
  (damaged_length : ℝ)
  (damaged_width : ℝ)

/-- Calculates the minimum area that needs to be cut off -/
def min_cut_area (b : Board) : ℝ :=
  2 + b.damaged_length * b.damaged_width

/-- Theorem stating the minimum area to be cut off for the given board -/
theorem min_cut_area_for_given_board :
  let b : Board := ⟨7, 5, 2, 1⟩
  min_cut_area b = 4 := by sorry

end NUMINAMATH_CALUDE_min_cut_area_for_given_board_l1373_137360


namespace NUMINAMATH_CALUDE_otimes_composition_l1373_137311

-- Define the binary operation ⊗
def otimes (x y : ℝ) : ℝ := x^2 + y

-- State the theorem
theorem otimes_composition (h : ℝ) : otimes h (otimes h h) = 2 * h^2 + h := by
  sorry

end NUMINAMATH_CALUDE_otimes_composition_l1373_137311


namespace NUMINAMATH_CALUDE_m_range_l1373_137350

-- Define proposition p
def p (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

-- Define proposition q
def q (x m : ℝ) : Prop := (x - 1 - m) * (x - 1 + m) ≤ 0

-- Define the theorem
theorem m_range (m : ℝ) :
  (m > 0) →
  (∀ x, q x m → p x) →
  (∃ x, p x ∧ ¬q x m) →
  m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_m_range_l1373_137350


namespace NUMINAMATH_CALUDE_quadrilateral_angles_not_always_form_triangle_l1373_137334

theorem quadrilateral_angles_not_always_form_triangle : ∃ (α β γ δ : ℝ),
  α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0 ∧
  α + β + γ + δ = 360 ∧
  ¬(α + β > γ ∧ β + γ > α ∧ γ + α > β) ∧
  ¬(α + β > δ ∧ β + δ > α ∧ δ + α > β) ∧
  ¬(α + γ > δ ∧ γ + δ > α ∧ δ + α > γ) ∧
  ¬(β + γ > δ ∧ γ + δ > β ∧ δ + β > γ) :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_angles_not_always_form_triangle_l1373_137334


namespace NUMINAMATH_CALUDE_function_composition_value_l1373_137335

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := 4 * x - 6
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_value (a b : ℝ) :
  (∀ x : ℝ, h a b x = (x - 14) / 2) → a - 2 * b = 101 / 8 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_value_l1373_137335


namespace NUMINAMATH_CALUDE_arctan_sum_special_case_l1373_137336

theorem arctan_sum_special_case : Real.arctan (3/7) + Real.arctan (7/3) = π/2 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_special_case_l1373_137336


namespace NUMINAMATH_CALUDE_faster_bike_speed_l1373_137354

/-- Proves that given two motorbikes traveling the same distance,
    where one bike is faster and takes 1 hour less than the other bike,
    the speed of the faster bike is 60 kmph. -/
theorem faster_bike_speed
  (distance : ℝ)
  (speed_fast : ℝ)
  (time_diff : ℝ)
  (h1 : distance = 960)
  (h2 : speed_fast = 60)
  (h3 : time_diff = 1)
  (h4 : distance / speed_fast + time_diff = distance / (distance / (distance / speed_fast + time_diff))) :
  speed_fast = 60 := by
  sorry

end NUMINAMATH_CALUDE_faster_bike_speed_l1373_137354


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_nineteen_twelfths_l1373_137310

theorem sum_of_roots_eq_nineteen_twelfths :
  let f : ℝ → ℝ := λ x ↦ (4*x + 7) * (3*x - 10)
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 19/12 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_nineteen_twelfths_l1373_137310


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l1373_137300

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Define the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 2 = 40 →
  a 3 + a 4 = 60 →
  a 5 + a 6 = 90 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l1373_137300


namespace NUMINAMATH_CALUDE_sheet_to_box_volume_l1373_137342

/-- Represents the dimensions of a rectangular sheet. -/
structure SheetDimensions where
  length : ℝ
  width : ℝ

/-- Represents the sizes of squares cut from corners. -/
structure CornerCuts where
  cut1 : ℝ
  cut2 : ℝ
  cut3 : ℝ
  cut4 : ℝ

/-- Represents the dimensions of the resulting box. -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions. -/
def boxVolume (box : BoxDimensions) : ℝ :=
  box.length * box.width * box.height

/-- Theorem stating the relationship between the original sheet, corner cuts, and resulting box. -/
theorem sheet_to_box_volume 
  (sheet : SheetDimensions) 
  (cuts : CornerCuts) 
  (box : BoxDimensions) : 
  sheet.length = 48 ∧ 
  sheet.width = 36 ∧
  cuts.cut1 = 7 ∧ 
  cuts.cut2 = 5 ∧ 
  cuts.cut3 = 6 ∧ 
  cuts.cut4 = 4 ∧
  box.length = sheet.length - (cuts.cut1 + cuts.cut4) ∧
  box.width = sheet.width - (cuts.cut2 + cuts.cut3) ∧
  box.height = min cuts.cut1 (min cuts.cut2 (min cuts.cut3 cuts.cut4)) →
  boxVolume box = 3700 ∧ 
  box.length = 37 ∧ 
  box.width = 25 := by
  sorry

end NUMINAMATH_CALUDE_sheet_to_box_volume_l1373_137342


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l1373_137384

theorem equation_has_real_roots (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l1373_137384


namespace NUMINAMATH_CALUDE_vector_magnitude_l1373_137372

/-- Given vectors a and b, if a is collinear with a + b, then |a - b| = 2√5 -/
theorem vector_magnitude (x : ℝ) : 
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![3, x]
  (∃ (k : ℝ), a = k • (a + b)) →
  ‖a - b‖ = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_l1373_137372


namespace NUMINAMATH_CALUDE_lateral_edge_length_l1373_137341

/-- A regular pyramid with a square base -/
structure RegularPyramid where
  -- The side length of the square base
  base_side : ℝ
  -- The volume of the pyramid
  volume : ℝ
  -- The length of a lateral edge
  lateral_edge : ℝ

/-- Theorem: In a regular pyramid with square base, if the volume is 4/3 and the base side length is 2, 
    then the lateral edge length is √3 -/
theorem lateral_edge_length (p : RegularPyramid) 
  (h1 : p.volume = 4/3) 
  (h2 : p.base_side = 2) : 
  p.lateral_edge = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_lateral_edge_length_l1373_137341


namespace NUMINAMATH_CALUDE_polynomial_equality_l1373_137364

theorem polynomial_equality (x : ℝ) : 
  (x - 1)^4 + 4*(x - 1)^3 + 6*(x - 1)^2 + 4*x - 3 = x^4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l1373_137364


namespace NUMINAMATH_CALUDE_car_journey_digit_squares_sum_l1373_137352

/-- Represents a car journey with specific odometer conditions -/
structure CarJourney where
  a : ℕ
  b : ℕ
  c : ℕ
  hours : ℕ
  initialReading : ℕ
  finalReading : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem car_journey_digit_squares_sum
  (journey : CarJourney)
  (h1 : journey.a ≥ 1)
  (h2 : journey.a + journey.b + journey.c = 9)
  (h3 : journey.initialReading = 100 * journey.a + 10 * journey.b + journey.c)
  (h4 : journey.finalReading = 100 * journey.c + 10 * journey.b + journey.a)
  (h5 : journey.finalReading - journey.initialReading = 65 * journey.hours) :
  journey.a^2 + journey.b^2 + journey.c^2 = 53 :=
sorry

end NUMINAMATH_CALUDE_car_journey_digit_squares_sum_l1373_137352


namespace NUMINAMATH_CALUDE_smallest_n_for_pencil_paradox_l1373_137303

theorem smallest_n_for_pencil_paradox : ∃ n : ℕ, n = 100 ∧ 
  (∀ m : ℕ, m < n → 
    ¬(∃ a b c d : ℕ, 
      6 * a + 10 * b = m ∧ 
      6 * c + 10 * d = m + 2 ∧ 
      7 * a + 12 * b > 7 * c + 12 * d)) ∧
  (∃ a b c d : ℕ, 
    6 * a + 10 * b = n ∧ 
    6 * c + 10 * d = n + 2 ∧ 
    7 * a + 12 * b > 7 * c + 12 * d) := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_pencil_paradox_l1373_137303


namespace NUMINAMATH_CALUDE_opposite_of_negative_sqrt_three_l1373_137367

theorem opposite_of_negative_sqrt_three : -(-(Real.sqrt 3)) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_sqrt_three_l1373_137367


namespace NUMINAMATH_CALUDE_hyperbola_range_l1373_137323

theorem hyperbola_range (m : ℝ) : 
  (∃ x y : ℝ, x^2 / (m - 2) + y^2 / (m + 3) = 1) ↔ -3 < m ∧ m < 2 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_range_l1373_137323


namespace NUMINAMATH_CALUDE_initial_liquid_a_amount_l1373_137355

/-- Represents the amount of liquid in liters -/
@[ext] structure Mixture where
  a : ℝ  -- Amount of liquid A
  b : ℝ  -- Amount of liquid B

/-- The initial ratio of liquid A to B is 4:1 -/
def initial_ratio (m : Mixture) : Prop :=
  m.a / m.b = 4

/-- The new ratio after replacement is 2:3 -/
def new_ratio (m : Mixture) : Prop :=
  m.a / (m.b + 20) = 2/3

/-- Theorem stating the initial amount of liquid A -/
theorem initial_liquid_a_amount
  (m : Mixture)
  (h1 : initial_ratio m)
  (h2 : new_ratio m) :
  m.a = 16 := by
  sorry

end NUMINAMATH_CALUDE_initial_liquid_a_amount_l1373_137355


namespace NUMINAMATH_CALUDE_impossible_to_tile_with_sphinx_l1373_137398

/-- Represents a sphinx tile -/
structure SphinxTile :=
  (upward_triangles : Nat)
  (downward_triangles : Nat)

/-- Represents an equilateral triangle -/
structure EquilateralTriangle :=
  (side_length : Nat)

/-- Defines the properties of a sphinx tile -/
def is_valid_sphinx_tile (tile : SphinxTile) : Prop :=
  tile.upward_triangles + tile.downward_triangles = 6 ∧
  (tile.upward_triangles = 4 ∧ tile.downward_triangles = 2) ∨
  (tile.upward_triangles = 2 ∧ tile.downward_triangles = 4)

/-- Calculates the number of unit triangles in an equilateral triangle -/
def num_unit_triangles (triangle : EquilateralTriangle) : Nat :=
  triangle.side_length * (triangle.side_length + 1)

/-- Calculates the number of upward-pointing unit triangles -/
def num_upward_triangles (triangle : EquilateralTriangle) : Nat :=
  (triangle.side_length * (triangle.side_length - 1)) / 2

/-- Calculates the number of downward-pointing unit triangles -/
def num_downward_triangles (triangle : EquilateralTriangle) : Nat :=
  (triangle.side_length * (triangle.side_length + 1)) / 2

/-- Theorem stating the impossibility of tiling the triangle with sphinx tiles -/
theorem impossible_to_tile_with_sphinx (triangle : EquilateralTriangle) 
  (h1 : triangle.side_length = 6) : 
  ¬ ∃ (tiling : List SphinxTile), 
    (∀ tile ∈ tiling, is_valid_sphinx_tile tile) ∧ 
    (List.sum (tiling.map (λ tile => tile.upward_triangles)) = num_upward_triangles triangle) ∧
    (List.sum (tiling.map (λ tile => tile.downward_triangles)) = num_downward_triangles triangle) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_tile_with_sphinx_l1373_137398


namespace NUMINAMATH_CALUDE_simplified_and_rationalized_l1373_137316

theorem simplified_and_rationalized (x : ℝ) : 
  1 / (2 + 1 / (Real.sqrt 5 + 2)) = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplified_and_rationalized_l1373_137316


namespace NUMINAMATH_CALUDE_four_false_statements_l1373_137377

/-- Represents a statement on the card -/
inductive Statement
| one
| two
| three
| four
| all

/-- The truth value of a statement -/
def isFalse : Statement → Bool
| Statement.one => true
| Statement.two => true
| Statement.three => true
| Statement.four => false
| Statement.all => true

/-- The claim made by each statement -/
def claim : Statement → Nat
| Statement.one => 1
| Statement.two => 2
| Statement.three => 3
| Statement.four => 4
| Statement.all => 5

/-- The total number of false statements -/
def totalFalse : Nat := 
  (Statement.one :: Statement.two :: Statement.three :: Statement.four :: Statement.all :: []).filter isFalse |>.length

/-- Theorem stating that exactly 4 statements are false -/
theorem four_false_statements : totalFalse = 4 ∧ 
  ∀ s : Statement, isFalse s = true ↔ claim s ≠ totalFalse :=
  sorry


end NUMINAMATH_CALUDE_four_false_statements_l1373_137377


namespace NUMINAMATH_CALUDE_toms_video_game_spending_l1373_137340

/-- The cost of the Batman game in dollars -/
def batman_cost : ℚ := 13.60

/-- The cost of the Superman game in dollars -/
def superman_cost : ℚ := 5.06

/-- The number of games Tom already owns -/
def existing_games : ℕ := 2

/-- The total amount Tom spent on video games -/
def total_spent : ℚ := batman_cost + superman_cost

theorem toms_video_game_spending :
  total_spent = 18.66 := by sorry

end NUMINAMATH_CALUDE_toms_video_game_spending_l1373_137340


namespace NUMINAMATH_CALUDE_mary_lambs_count_l1373_137305

def lambs_problem (initial_lambs : ℕ) (mother_lambs : ℕ) (babies_per_lamb : ℕ) 
                  (traded_lambs : ℕ) (extra_lambs : ℕ) : Prop :=
  let new_babies := mother_lambs * babies_per_lamb
  let after_births := initial_lambs + new_babies
  let after_trade := after_births - traded_lambs
  let final_count := after_trade + extra_lambs
  final_count = 34

theorem mary_lambs_count : 
  lambs_problem 12 4 3 5 15 := by
  sorry

end NUMINAMATH_CALUDE_mary_lambs_count_l1373_137305


namespace NUMINAMATH_CALUDE_point_coordinate_sum_l1373_137349

theorem point_coordinate_sum (a b : ℝ) : 
  (2 = b - 1 ∧ -1 = a + 3) → a + b = -1 := by
sorry

end NUMINAMATH_CALUDE_point_coordinate_sum_l1373_137349


namespace NUMINAMATH_CALUDE_farm_tree_sub_branches_l1373_137301

/-- Proves that the number of sub-branches per branch is 40, given the conditions from the farm tree problem -/
theorem farm_tree_sub_branches :
  let branches_per_tree : ℕ := 10
  let leaves_per_sub_branch : ℕ := 60
  let total_trees : ℕ := 4
  let total_leaves : ℕ := 96000
  ∃ (sub_branches_per_branch : ℕ),
    sub_branches_per_branch = 40 ∧
    total_leaves = total_trees * branches_per_tree * leaves_per_sub_branch * sub_branches_per_branch :=
by sorry

end NUMINAMATH_CALUDE_farm_tree_sub_branches_l1373_137301


namespace NUMINAMATH_CALUDE_age_problem_l1373_137382

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 3 * d →
  a + b + c + d = 87 →
  b = 30 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l1373_137382


namespace NUMINAMATH_CALUDE_vans_needed_l1373_137321

theorem vans_needed (total_people : ℕ) (van_capacity : ℕ) (h1 : total_people = 35) (h2 : van_capacity = 4) :
  ↑⌈(total_people : ℚ) / van_capacity⌉ = 9 := by
  sorry

end NUMINAMATH_CALUDE_vans_needed_l1373_137321


namespace NUMINAMATH_CALUDE_percentage_problem_l1373_137370

/-- Prove that the percentage is 50% given the conditions -/
theorem percentage_problem (x : ℝ) (a : ℝ) : 
  (x / 100) * a = 95 → a = 190 → x = 50 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1373_137370


namespace NUMINAMATH_CALUDE_expression_simplification_l1373_137318

theorem expression_simplification (x y : ℝ) :
  8 * x + 3 * y - 2 * x + y + 20 + 15 = 6 * x + 4 * y + 35 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1373_137318


namespace NUMINAMATH_CALUDE_circle_area_ratio_l1373_137326

theorem circle_area_ratio (r : ℝ) (h : r > 0) : 
  (π * r^2) / (π * (3*r)^2) = 1/9 := by sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l1373_137326


namespace NUMINAMATH_CALUDE_trapezoid_perimeter_is_34_l1373_137314

/-- A trapezoid with specific side lengths -/
structure Trapezoid :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DA : ℝ)
  (h_AB_eq_CD : AB = CD)
  (h_AB : AB = 8)
  (h_CD : CD = 16)
  (h_BC_eq_DA : BC = DA)
  (h_BC : BC = 5)

/-- The perimeter of a trapezoid is the sum of its sides -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + t.DA

/-- Theorem: The perimeter of the specified trapezoid is 34 -/
theorem trapezoid_perimeter_is_34 (t : Trapezoid) : perimeter t = 34 := by
  sorry

end NUMINAMATH_CALUDE_trapezoid_perimeter_is_34_l1373_137314


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1373_137362

/-- An arithmetic sequence. -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence, if a₂ + a₈ = 12, then a₅ = 6. -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) (h_sum : a 2 + a 8 = 12) : a 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1373_137362


namespace NUMINAMATH_CALUDE_floor_sum_inequality_l1373_137396

theorem floor_sum_inequality (x y : ℝ) : ⌊x + y⌋ ≤ ⌊x⌋ + ⌊y⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_inequality_l1373_137396


namespace NUMINAMATH_CALUDE_wrong_to_right_ratio_l1373_137312

theorem wrong_to_right_ratio (total : ℕ) (correct : ℕ) (h1 : total = 24) (h2 : correct = 8) :
  (total - correct) / correct = 2 := by
  sorry

end NUMINAMATH_CALUDE_wrong_to_right_ratio_l1373_137312


namespace NUMINAMATH_CALUDE_commission_calculation_l1373_137393

/-- The commission calculation problem -/
theorem commission_calculation
  (commission_rate : ℝ)
  (total_sales : ℝ)
  (h1 : commission_rate = 0.04)
  (h2 : total_sales = 312.5) :
  commission_rate * total_sales = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_commission_calculation_l1373_137393


namespace NUMINAMATH_CALUDE_sara_joe_height_difference_l1373_137331

/-- Given the heights of Sara, Joe, and Roy, prove that Sara is 6 inches taller than Joe. -/
theorem sara_joe_height_difference :
  ∀ (sara_height joe_height roy_height : ℕ),
    sara_height = 45 →
    joe_height = roy_height + 3 →
    roy_height = 36 →
    sara_height - joe_height = 6 := by
sorry

end NUMINAMATH_CALUDE_sara_joe_height_difference_l1373_137331


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l1373_137313

/-- An isosceles right triangle with hypotenuse 68 and leg 48 -/
structure IsoscelesRightTriangle where
  hypotenuse : ℝ
  leg : ℝ
  hypotenuse_eq : hypotenuse = 68
  leg_eq : leg = 48

/-- A circle inscribed in the right angle of the triangle -/
structure InscribedCircle where
  radius : ℝ
  radius_eq : radius = 12

/-- A circle externally tangent to the inscribed circle and inscribed in the remaining space -/
structure TangentCircle where
  radius : ℝ

/-- The main theorem stating that the radius of the tangent circle is 8 -/
theorem tangent_circle_radius 
  (triangle : IsoscelesRightTriangle) 
  (inscribed : InscribedCircle) 
  (tangent : TangentCircle) : tangent.radius = 8 := by
  sorry


end NUMINAMATH_CALUDE_tangent_circle_radius_l1373_137313


namespace NUMINAMATH_CALUDE_michaels_fish_count_l1373_137325

theorem michaels_fish_count 
  (initial_fish : ℝ) 
  (fish_from_ben : ℝ) 
  (fish_from_maria : ℝ) 
  (h1 : initial_fish = 49.5)
  (h2 : fish_from_ben = 18.25)
  (h3 : fish_from_maria = 23.75) :
  initial_fish + fish_from_ben + fish_from_maria = 91.5 := by
  sorry

end NUMINAMATH_CALUDE_michaels_fish_count_l1373_137325


namespace NUMINAMATH_CALUDE_complement_union_theorem_l1373_137337

-- Define the sets
def U : Set ℝ := Set.univ
def M : Set ℝ := {x | |x| < 2}
def N : Set ℝ := {y | ∃ x, y = 2^x - 1}

-- State the theorem
theorem complement_union_theorem :
  (Set.compl M ∪ Set.compl N) = {x | x ≤ -1 ∨ x ≥ 2} := by sorry

end NUMINAMATH_CALUDE_complement_union_theorem_l1373_137337


namespace NUMINAMATH_CALUDE_debate_team_groups_l1373_137320

theorem debate_team_groups (boys : ℕ) (girls : ℕ) (group_size : ℕ) : 
  boys = 31 → girls = 32 → group_size = 9 → 
  (boys + girls) / group_size = 7 ∧ (boys + girls) % group_size = 0 := by
  sorry

end NUMINAMATH_CALUDE_debate_team_groups_l1373_137320


namespace NUMINAMATH_CALUDE_favorite_fruit_strawberries_l1373_137333

theorem favorite_fruit_strawberries (total : ℕ) (oranges pears apples : ℕ)
  (h1 : total = 450)
  (h2 : oranges = 70)
  (h3 : pears = 120)
  (h4 : apples = 147) :
  total - (oranges + pears + apples) = 113 :=
by sorry

end NUMINAMATH_CALUDE_favorite_fruit_strawberries_l1373_137333


namespace NUMINAMATH_CALUDE_part_one_part_two_part_three_l1373_137380

/-- A function f(x) = ax - bx² where a > 0 and b > 0 -/
def f (a b x : ℝ) : ℝ := a * x - b * x^2

/-- Theorem for part I -/
theorem part_one (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x : ℝ, f a b x ≤ 1) → a ≤ 2 * Real.sqrt b :=
sorry

/-- Theorem for part II -/
theorem part_two (a b : ℝ) (ha : a > 0) (hb : b > 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ (b - 1 ≤ a ∧ a ≤ 2 * Real.sqrt b) :=
sorry

/-- Theorem for part III -/
theorem part_three (a b : ℝ) (ha : a > 0) (hb : 0 < b) (hb' : b ≤ 1) :
  (∀ x ∈ Set.Icc 0 1, |f a b x| ≤ 1) ↔ a ≤ b + 1 :=
sorry

end NUMINAMATH_CALUDE_part_one_part_two_part_three_l1373_137380


namespace NUMINAMATH_CALUDE_convex_pentagon_with_equal_diagonals_and_sides_l1373_137345

-- Define a pentagon as a set of 5 points in 2D space
def Pentagon := Fin 5 → ℝ × ℝ

-- Define a function to check if a pentagon is convex
def is_convex (p : Pentagon) : Prop := sorry

-- Define a function to calculate the length of a line segment
def length (a b : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a line segment is a diagonal of the pentagon
def is_diagonal (p : Pentagon) (i j : Fin 5) : Prop :=
  (i.val + 2) % 5 ≤ j.val ∨ (j.val + 2) % 5 ≤ i.val

-- Define a function to check if a line segment is a side of the pentagon
def is_side (p : Pentagon) (i j : Fin 5) : Prop :=
  (i.val + 1) % 5 = j.val ∨ (j.val + 1) % 5 = i.val

-- Theorem: There exists a convex pentagon where each diagonal is equal to some side
theorem convex_pentagon_with_equal_diagonals_and_sides :
  ∃ (p : Pentagon), is_convex p ∧
    ∀ (i j : Fin 5), is_diagonal p i j →
      ∃ (k l : Fin 5), is_side p k l ∧ length (p i) (p j) = length (p k) (p l) :=
sorry

end NUMINAMATH_CALUDE_convex_pentagon_with_equal_diagonals_and_sides_l1373_137345


namespace NUMINAMATH_CALUDE_superhero_movie_count_l1373_137368

theorem superhero_movie_count (total_movies : ℕ) (dalton_movies : ℕ) (alex_movies : ℕ) (shared_movies : ℕ) :
  total_movies = 30 →
  dalton_movies = 7 →
  alex_movies = 15 →
  shared_movies = 2 →
  ∃ (hunter_movies : ℕ), hunter_movies = total_movies - dalton_movies - alex_movies + shared_movies :=
by
  sorry

end NUMINAMATH_CALUDE_superhero_movie_count_l1373_137368


namespace NUMINAMATH_CALUDE_triangle_square_side_ratio_l1373_137394

theorem triangle_square_side_ratio (t s : ℝ) : 
  t > 0 → s > 0 → 3 * t = 12 → 4 * s = 12 → t / s = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_square_side_ratio_l1373_137394


namespace NUMINAMATH_CALUDE_joeys_swimming_time_l1373_137399

theorem joeys_swimming_time (ethan_time : ℝ) 
  (h1 : ethan_time > 0)
  (h2 : 3/4 * ethan_time = 9)
  (h3 : ethan_time = 12) :
  1/2 * ethan_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_joeys_swimming_time_l1373_137399


namespace NUMINAMATH_CALUDE_geric_initial_bills_l1373_137308

/-- The number of bills Jessa had initially -/
def jessa_initial : ℕ := 7 + 3

/-- The number of bills Kyla had -/
def kyla : ℕ := jessa_initial - 2

/-- The number of bills Geric had initially -/
def geric_initial : ℕ := 2 * kyla

theorem geric_initial_bills : geric_initial = 16 := by
  sorry

end NUMINAMATH_CALUDE_geric_initial_bills_l1373_137308


namespace NUMINAMATH_CALUDE_log_one_fourth_sixteen_l1373_137379

-- Define the logarithm function for an arbitrary base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_fourth_sixteen : log (1/4) 16 = -2 := by sorry

end NUMINAMATH_CALUDE_log_one_fourth_sixteen_l1373_137379


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l1373_137324

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  GeometricSequence a →
  (a 1 * a 19 = 16) →
  (a 1 + a 19 = 10) →
  a 8 * a 10 * a 12 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l1373_137324


namespace NUMINAMATH_CALUDE_first_complete_column_coverage_l1373_137343

theorem first_complete_column_coverage : 
  let triangular (n : ℕ) := n * (n + 1) / 2
  ∃ (k : ℕ), k > 0 ∧ 
    (∀ (r : ℕ), r < 8 → ∃ (n : ℕ), n ≤ k ∧ triangular n % 8 = r) ∧
    (∀ (m : ℕ), m < k → ¬(∀ (r : ℕ), r < 8 → ∃ (n : ℕ), n ≤ m ∧ triangular n % 8 = r)) ∧
  k = 15 :=
by sorry

end NUMINAMATH_CALUDE_first_complete_column_coverage_l1373_137343


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1373_137328

theorem quadratic_equation_roots (m : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*m*x + 3*m^2
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (m > 0 → (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ |x₁ - x₂| = 2) → m = 1) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1373_137328


namespace NUMINAMATH_CALUDE_altitude_angle_relation_l1373_137309

/-- For an acute-angled triangle with circumradius R and altitude h from a vertex,
    the angle α at that vertex satisfies the given conditions. -/
theorem altitude_angle_relation (α : Real) (R h : ℝ) : 
  (α < Real.pi / 3 ↔ h < R) ∧ 
  (α = Real.pi / 3 ↔ h = R) ∧ 
  (α > Real.pi / 3 ↔ h > R) :=
by sorry

end NUMINAMATH_CALUDE_altitude_angle_relation_l1373_137309


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l1373_137339

theorem area_between_concentric_circles (R : ℝ) (chord_length : ℝ) : 
  R = 13 → 
  chord_length = 24 → 
  (π * R^2) - (π * (R^2 - (chord_length/2)^2)) = 144 * π :=
by
  sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l1373_137339


namespace NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l1373_137386

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_15th_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_3 : a 3 = 4) 
  (h_9 : a 9 = 10) : 
  a 15 = 16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_15th_term_l1373_137386


namespace NUMINAMATH_CALUDE_circle_diameter_from_area_l1373_137365

theorem circle_diameter_from_area (A : ℝ) (r : ℝ) (d : ℝ) : A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_from_area_l1373_137365


namespace NUMINAMATH_CALUDE_first_month_sale_l1373_137395

/-- Given the average sale and sales for 4 out of 5 months, calculate the sale for the first month -/
theorem first_month_sale (average : ℕ) (sale2 sale3 sale4 sale5 : ℕ) 
  (h_average : average = 6000)
  (h_sale2 : sale2 = 5660)
  (h_sale3 : sale3 = 6200)
  (h_sale4 : sale4 = 6350)
  (h_sale5 : sale5 = 6500) :
  ∃ (sale1 : ℕ), sale1 + sale2 + sale3 + sale4 + sale5 = 5 * average ∧ sale1 = 5290 := by
  sorry

end NUMINAMATH_CALUDE_first_month_sale_l1373_137395


namespace NUMINAMATH_CALUDE_drawing_set_prices_and_quantity_l1373_137390

/-- Represents the cost and selling prices of drawing tool sets from two brands -/
structure DrawingSetPrices where
  costA : ℝ
  costB : ℝ
  sellA : ℝ
  sellB : ℝ

/-- Theorem stating the properties of the drawing set prices and minimum purchase quantity -/
theorem drawing_set_prices_and_quantity (p : DrawingSetPrices)
  (h1 : p.costA = p.costB + 2.5)
  (h2 : 200 / p.costA = 2 * (75 / p.costB))
  (h3 : p.sellA = 13)
  (h4 : p.sellB = 9.5) :
  p.costA = 10 ∧ p.costB = 7.5 ∧
  (∀ a : ℕ, (p.sellA - p.costA) * a + (p.sellB - p.costB) * (2 * a + 4) > 120 → a ≥ 17) :=
sorry

end NUMINAMATH_CALUDE_drawing_set_prices_and_quantity_l1373_137390


namespace NUMINAMATH_CALUDE_golf_course_distance_l1373_137302

/-- Represents a golf shot with distance and wind conditions -/
structure GolfShot where
  distance : ℝ
  windSpeed : ℝ
  windDirection : String

/-- Calculates the total distance to the hole given three golf shots -/
def distanceToHole (shot1 shot2 shot3 : GolfShot) (slopeEffect : ℝ) : ℝ :=
  shot1.distance + (shot2.distance - slopeEffect)

theorem golf_course_distance :
  let shot1 : GolfShot := { distance := 180, windSpeed := 10, windDirection := "tailwind" }
  let shot2 : GolfShot := { distance := 90, windSpeed := 7, windDirection := "crosswind" }
  let shot3 : GolfShot := { distance := 0, windSpeed := 5, windDirection := "headwind" }
  let slopeEffect : ℝ := 20
  distanceToHole shot1 shot2 shot3 slopeEffect = 270 := by
  sorry

end NUMINAMATH_CALUDE_golf_course_distance_l1373_137302


namespace NUMINAMATH_CALUDE_boy_position_in_line_l1373_137366

/-- The position of a boy in a line of boys, where he is equidistant from both ends -/
def midPosition (n : ℕ) : ℕ :=
  (n + 1) / 2

/-- Theorem: In a line of 37 boys, the boy who is equidistant from both ends is in position 19 -/
theorem boy_position_in_line :
  midPosition 37 = 19 := by
  sorry

end NUMINAMATH_CALUDE_boy_position_in_line_l1373_137366


namespace NUMINAMATH_CALUDE_discount_comparison_l1373_137388

def initial_amount : ℝ := 20000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.25, 0.15, 0.05]
def option2_discounts : List ℝ := [0.35, 0.10, 0.05]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

def final_price_option1 : ℝ :=
  apply_successive_discounts initial_amount option1_discounts

def final_price_option2 : ℝ :=
  apply_successive_discounts initial_amount option2_discounts

theorem discount_comparison :
  final_price_option1 - final_price_option2 = 997.50 ∧
  final_price_option2 < final_price_option1 :=
sorry

end NUMINAMATH_CALUDE_discount_comparison_l1373_137388


namespace NUMINAMATH_CALUDE_perpendicular_radii_intercept_l1373_137376

/-- Given a circle and a line intersecting it, if the radii to the intersection points are perpendicular, then the y-intercept of the line has specific values. -/
theorem perpendicular_radii_intercept (b : ℝ) : 
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 - 4*x = 0}
  let line := {(x, y) : ℝ × ℝ | y = x + b}
  let C := (2, 0)
  ∃ (M N : ℝ × ℝ), 
    M ∈ circle ∧ M ∈ line ∧
    N ∈ circle ∧ N ∈ line ∧
    M ≠ N ∧
    (M.1 - C.1) * (N.1 - C.1) + (M.2 - C.2) * (N.2 - C.2) = 0 →
    b = 0 ∨ b = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_radii_intercept_l1373_137376


namespace NUMINAMATH_CALUDE_castle_provisions_theorem_l1373_137387

/-- Represents the number of days provisions last given initial conditions and a change in population -/
def days_until_food_runs_out (initial_people : ℕ) (initial_days : ℕ) (days_passed : ℕ) (people_left : ℕ) : ℕ :=
  let remaining_days := initial_days - days_passed
  let new_duration := (remaining_days * initial_people) / people_left
  new_duration

/-- Theorem stating that under given conditions, food lasts for 90 more days after population change -/
theorem castle_provisions_theorem (initial_people : ℕ) (initial_days : ℕ) 
  (days_passed : ℕ) (people_left : ℕ) :
  initial_people = 300 ∧ initial_days = 90 ∧ days_passed = 30 ∧ people_left = 200 →
  days_until_food_runs_out initial_people initial_days days_passed people_left = 90 :=
by
  sorry

#eval days_until_food_runs_out 300 90 30 200

end NUMINAMATH_CALUDE_castle_provisions_theorem_l1373_137387


namespace NUMINAMATH_CALUDE_max_value_of_g_l1373_137315

def S : Set Int := {-3, -2, 1, 2, 3, 4}

def g (a b : Int) : ℚ := -((a - b)^2 : ℚ) / 4

theorem max_value_of_g :
  ∃ (max : ℚ), max = -1/4 ∧
  ∀ (a b : Int), a ∈ S → b ∈ S → a ≠ b → g a b ≤ max ∧
  ∃ (a₀ b₀ : Int), a₀ ∈ S ∧ b₀ ∈ S ∧ a₀ ≠ b₀ ∧ g a₀ b₀ = max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_g_l1373_137315


namespace NUMINAMATH_CALUDE_infinite_common_elements_l1373_137381

def sequence_a : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 14 * sequence_a (n + 1) + sequence_a n

def sequence_b : ℕ → ℤ
| 0 => 2
| 1 => 14
| (n + 2) => 6 * sequence_b (n + 1) - sequence_b n

def subsequence_a (k : ℕ) : ℤ := sequence_a (2 * k + 1)

def subsequence_b (k : ℕ) : ℤ := sequence_b (3 * k + 1)

theorem infinite_common_elements : 
  ∀ k : ℕ, subsequence_a k = subsequence_b k :=
sorry

end NUMINAMATH_CALUDE_infinite_common_elements_l1373_137381


namespace NUMINAMATH_CALUDE_painted_cube_theorem_l1373_137385

theorem painted_cube_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1/3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_painted_cube_theorem_l1373_137385


namespace NUMINAMATH_CALUDE_train_passing_pole_time_l1373_137346

/-- Proves that a train 100 meters long, traveling at 72 km/hr, takes 5 seconds to pass a pole -/
theorem train_passing_pole_time (train_length : Real) (train_speed_kmh : Real) :
  train_length = 100 ∧ train_speed_kmh = 72 →
  (train_length / (train_speed_kmh * 1000 / 3600)) = 5 := by
sorry

end NUMINAMATH_CALUDE_train_passing_pole_time_l1373_137346


namespace NUMINAMATH_CALUDE_min_value_a_plus_2b_l1373_137356

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a * b = a + b + 1) :
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ x * y = x + y + 1 → a + 2 * b ≤ x + 2 * y ∧ ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ a₀ * b₀ = a₀ + b₀ + 1 ∧ a₀ + 2 * b₀ = 7 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_plus_2b_l1373_137356


namespace NUMINAMATH_CALUDE_fifteenth_term_ratio_l1373_137306

-- Define the sums of arithmetic sequences
def S (a d n : ℚ) : ℚ := n * (2 * a + (n - 1) * d) / 2
def T (b e n : ℚ) : ℚ := n * (2 * b + (n - 1) * e) / 2

-- Define the ratio condition
def ratio_condition (a d b e n : ℚ) : Prop :=
  S a d n / T b e n = (5 * n + 3) / (3 * n + 17)

-- Define the 15th term of each sequence
def term_15 (a d : ℚ) : ℚ := a + 14 * d

-- Theorem statement
theorem fifteenth_term_ratio 
  (a d b e : ℚ) 
  (h : ∀ n : ℚ, ratio_condition a d b e n) : 
  term_15 a d / term_15 b e = 44 / 95 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_ratio_l1373_137306


namespace NUMINAMATH_CALUDE_ducks_at_north_pond_l1373_137353

/-- The number of ducks at North Pond given the specified conditions -/
theorem ducks_at_north_pond :
  let mallard_lake_michigan : ℕ := 100
  let pintail_lake_michigan : ℕ := 75
  let mallard_north_pond : ℕ := 2 * mallard_lake_michigan + 6
  let pintail_north_pond : ℕ := 4 * mallard_lake_michigan
  mallard_north_pond + pintail_north_pond = 606 :=
by sorry


end NUMINAMATH_CALUDE_ducks_at_north_pond_l1373_137353


namespace NUMINAMATH_CALUDE_choose_materials_eq_120_l1373_137375

/-- The number of ways two students can choose 2 out of 6 materials each, 
    such that they have exactly 1 material in common -/
def choose_materials : ℕ :=
  let total_materials : ℕ := 6
  let materials_per_student : ℕ := 2
  let common_materials : ℕ := 1
  Nat.choose total_materials common_materials *
  (total_materials - common_materials) * (total_materials - common_materials - 1)

theorem choose_materials_eq_120 : choose_materials = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_materials_eq_120_l1373_137375


namespace NUMINAMATH_CALUDE_sequence_general_term_l1373_137357

theorem sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 1) 
    (h2 : ∀ n : ℕ, n ≥ 2 → a n - a (n-1) = 2) : 
    ∀ n : ℕ, a n = 2 * n - 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1373_137357


namespace NUMINAMATH_CALUDE_unique_cyclic_number_l1373_137322

def is_permutation (a b : Nat) : Prop := sorry

def has_distinct_digits (n : Nat) : Prop := sorry

theorem unique_cyclic_number : ∃! n : Nat, 
  100000 ≤ n ∧ n < 1000000 ∧ 
  has_distinct_digits n ∧
  is_permutation n (2*n) ∧
  is_permutation n (3*n) ∧
  is_permutation n (4*n) ∧
  is_permutation n (5*n) ∧
  is_permutation n (6*n) ∧
  n = 142857 := by sorry

end NUMINAMATH_CALUDE_unique_cyclic_number_l1373_137322


namespace NUMINAMATH_CALUDE_first_group_men_count_l1373_137391

/-- Represents the number of men in the first group -/
def first_group_men : ℕ := 30

/-- Represents the number of days worked by the first group -/
def first_group_days : ℕ := 12

/-- Represents the number of hours worked per day by the first group -/
def first_group_hours_per_day : ℕ := 8

/-- Represents the length of road (in km) asphalted by the first group -/
def first_group_road_length : ℕ := 1

/-- Represents the number of men in the second group -/
def second_group_men : ℕ := 20

/-- Represents the number of days worked by the second group -/
def second_group_days : ℝ := 19.2

/-- Represents the number of hours worked per day by the second group -/
def second_group_hours_per_day : ℕ := 15

/-- Represents the length of road (in km) asphalted by the second group -/
def second_group_road_length : ℕ := 2

/-- Theorem stating that the number of men in the first group is 30 -/
theorem first_group_men_count : 
  first_group_men * first_group_days * first_group_hours_per_day * second_group_road_length = 
  second_group_men * second_group_days * second_group_hours_per_day * first_group_road_length :=
by sorry

end NUMINAMATH_CALUDE_first_group_men_count_l1373_137391


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2939_l1373_137374

theorem smallest_prime_factor_of_2939 :
  (Nat.minFac 2939 = 13) := by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2939_l1373_137374


namespace NUMINAMATH_CALUDE_rectangle_area_scientific_notation_l1373_137344

theorem rectangle_area_scientific_notation :
  let side1 : ℝ := 3 * 10^3
  let side2 : ℝ := 400
  let area : ℝ := side1 * side2
  area = 1.2 * 10^6 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_scientific_notation_l1373_137344


namespace NUMINAMATH_CALUDE_work_completion_time_l1373_137327

/-- 
If two workers can complete a job together in a certain time, 
and one worker can complete it alone in a known time, 
we can determine how long it takes the other worker to complete the job alone.
-/
theorem work_completion_time 
  (total_work : ℝ) 
  (time_together time_a time_b : ℝ) 
  (h1 : time_together > 0)
  (h2 : time_a > 0)
  (h3 : time_b > 0)
  (h4 : total_work / time_together = total_work / time_a + total_work / time_b)
  (h5 : time_together = 5)
  (h6 : time_a = 10) :
  time_b = 10 := by
sorry

end NUMINAMATH_CALUDE_work_completion_time_l1373_137327


namespace NUMINAMATH_CALUDE_inequality_proof_l1373_137389

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c = 1 → a^2 + b^2 + c^2 ≥ 1/3) ∧
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1373_137389
