import Mathlib

namespace NUMINAMATH_CALUDE_grocery_store_distance_l1555_155557

theorem grocery_store_distance (distance_house_to_park : ℝ) 
                               (distance_park_to_store : ℝ) 
                               (total_distance : ℝ) : ℝ := by
  have h1 : distance_house_to_park = 5 := by sorry
  have h2 : distance_park_to_store = 3 := by sorry
  have h3 : total_distance = 16 := by sorry
  
  let distance_store_to_house := total_distance - distance_house_to_park - distance_park_to_store
  
  have h4 : distance_store_to_house = 
            total_distance - distance_house_to_park - distance_park_to_store := by rfl
  
  exact distance_store_to_house

end NUMINAMATH_CALUDE_grocery_store_distance_l1555_155557


namespace NUMINAMATH_CALUDE_problem_solid_surface_area_l1555_155565

/-- Represents a solid formed by unit cubes -/
structure CubeSolid where
  base_length : ℕ
  base_width : ℕ
  top_length : ℕ
  top_width : ℕ
  base_height : ℕ := 1
  top_height : ℕ := 1

/-- Calculates the surface area of the CubeSolid -/
def surface_area (solid : CubeSolid) : ℕ :=
  let base_area := solid.base_length * solid.base_width
  let top_area := solid.top_length * solid.top_width
  let exposed_base := 2 * base_area
  let exposed_sides := 2 * (solid.base_length * solid.base_height + solid.base_width * solid.base_height)
  let exposed_top := base_area - top_area + top_area
  let exposed_top_sides := 2 * (solid.top_length * solid.top_height + solid.top_width * solid.top_height)
  exposed_base + exposed_sides + exposed_top + exposed_top_sides

/-- The specific solid described in the problem -/
def problem_solid : CubeSolid := {
  base_length := 4
  base_width := 2
  top_length := 2
  top_width := 2
}

theorem problem_solid_surface_area :
  surface_area problem_solid = 36 := by
  sorry

end NUMINAMATH_CALUDE_problem_solid_surface_area_l1555_155565


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l1555_155535

/-- A circle with radius 2, center on the positive x-axis, and tangent to the y-axis -/
structure TangentCircle where
  center : ℝ × ℝ
  radius : ℝ
  center_on_x_axis : center.2 = 0
  positive_x : center.1 > 0
  radius_is_two : radius = 2
  tangent_to_y : center.1 = radius

/-- The equation of the circle is x² + y² - 4x = 0 -/
theorem tangent_circle_equation (c : TangentCircle) :
  ∀ x y : ℝ, (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 ↔ x^2 + y^2 - 4*x = 0 :=
by sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l1555_155535


namespace NUMINAMATH_CALUDE_direction_vector_of_determinant_line_l1555_155529

/-- Given a line in 2D space defined by the determinant equation |x y; 2 1| = 3,
    prove that (-2, -1) is a direction vector of this line. -/
theorem direction_vector_of_determinant_line :
  let line := {(x, y) : ℝ × ℝ | x - 2*y = 3}
  ((-2 : ℝ), -1) ∈ {v : ℝ × ℝ | ∃ (t : ℝ), ∀ (p q : ℝ × ℝ), p ∈ line → q ∈ line → ∃ (s : ℝ), q.1 - p.1 = s * v.1 ∧ q.2 - p.2 = s * v.2} :=
by sorry

end NUMINAMATH_CALUDE_direction_vector_of_determinant_line_l1555_155529


namespace NUMINAMATH_CALUDE_division_with_specific_endings_impossible_l1555_155510

theorem division_with_specific_endings_impossible :
  ¬∃ (a b c d : ℕ), 
    a = b * c + d ∧
    a % 10 = 9 ∧
    b % 10 = 7 ∧
    c % 10 = 3 ∧
    d = 1 := by
  sorry

end NUMINAMATH_CALUDE_division_with_specific_endings_impossible_l1555_155510


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l1555_155532

theorem polynomial_division_remainder : ∃ q : Polynomial ℤ, 
  (X^5 - 1) * (X^3 - 1) = (X^3 + X^2 + X + 1) * q + (2*X + 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l1555_155532


namespace NUMINAMATH_CALUDE_rowing_speed_ratio_l1555_155513

/-- Given that a man takes twice as long to row a distance against a stream as to row the same distance with the stream, prove that the ratio of the boat's speed in still water to the stream's speed is 3:1. -/
theorem rowing_speed_ratio 
  (B : ℝ) -- Speed of the boat in still water
  (S : ℝ) -- Speed of the stream
  (h : B > S) -- Assumption that the boat's speed is greater than the stream's speed
  (h1 : (1 / (B - S)) = 2 * (1 / (B + S))) -- Time against stream is twice time with stream
  : B / S = 3 := by
  sorry

end NUMINAMATH_CALUDE_rowing_speed_ratio_l1555_155513


namespace NUMINAMATH_CALUDE_complex_number_evaluation_l1555_155517

theorem complex_number_evaluation :
  let i : ℂ := Complex.I
  ((1 - i) * i^2) / (1 + 2*i) = 1/5 + 3/5*i := by
  sorry

end NUMINAMATH_CALUDE_complex_number_evaluation_l1555_155517


namespace NUMINAMATH_CALUDE_lcm_24_90_128_l1555_155548

theorem lcm_24_90_128 : Nat.lcm (Nat.lcm 24 90) 128 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_90_128_l1555_155548


namespace NUMINAMATH_CALUDE_polynomial_simplification_l1555_155505

variable (p : ℝ)

theorem polynomial_simplification :
  (6 * p^4 + 2 * p^3 - 8 * p + 9) + (-3 * p^3 + 7 * p^2 - 5 * p - 1) =
  6 * p^4 - p^3 + 7 * p^2 - 13 * p + 8 := by
sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l1555_155505


namespace NUMINAMATH_CALUDE_max_min_difference_z_l1555_155501

theorem max_min_difference_z (x y z : ℝ) 
  (sum_eq : x + y + z = 6)
  (sum_squares_eq : x^2 + y^2 + z^2 = 24) :
  ∃ (z_max z_min : ℝ),
    (∀ w, w = x ∨ w = y ∨ w = z → z_min ≤ w ∧ w ≤ z_max) ∧
    z_max - z_min = 4 :=
sorry

end NUMINAMATH_CALUDE_max_min_difference_z_l1555_155501


namespace NUMINAMATH_CALUDE_smallest_number_divisibility_l1555_155589

theorem smallest_number_divisibility (n : ℕ) : n = 4722 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k₁ k₂ k₃ k₄ : ℕ, 
    m + 3 = 27 * k₁ ∧ 
    m + 3 = 35 * k₂ ∧ 
    m + 3 = 25 * k₃ ∧ 
    m + 3 = 21 * k₄)) ∧ 
  (∃ k₁ k₂ k₃ k₄ : ℕ, 
    n + 3 = 27 * k₁ ∧ 
    n + 3 = 35 * k₂ ∧ 
    n + 3 = 25 * k₃ ∧ 
    n + 3 = 21 * k₄) := by
  sorry

#check smallest_number_divisibility

end NUMINAMATH_CALUDE_smallest_number_divisibility_l1555_155589


namespace NUMINAMATH_CALUDE_sin_315_degrees_l1555_155528

theorem sin_315_degrees : Real.sin (315 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_315_degrees_l1555_155528


namespace NUMINAMATH_CALUDE_cos_sum_17th_roots_unity_l1555_155540

theorem cos_sum_17th_roots_unity :
  Real.cos (2 * Real.pi / 17) + Real.cos (8 * Real.pi / 17) + Real.cos (14 * Real.pi / 17) = (Real.sqrt 17 - 1) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_17th_roots_unity_l1555_155540


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l1555_155594

/-- An even function that is increasing on (-∞, 0] -/
def EvenIncreasingNonPositive (f : ℝ → ℝ) : Prop :=
  (∀ x, f x = f (-x)) ∧ 
  (∀ x y, x ≤ y ∧ y ≤ 0 → f x ≤ f y)

/-- The theorem statement -/
theorem solution_set_of_inequality 
  (f : ℝ → ℝ) 
  (h : EvenIncreasingNonPositive f) :
  {x : ℝ | f (x - 1) ≥ f 1} = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l1555_155594


namespace NUMINAMATH_CALUDE_straws_paper_difference_l1555_155573

theorem straws_paper_difference :
  let straws : ℕ := 15
  let paper : ℕ := 7
  straws - paper = 8 := by sorry

end NUMINAMATH_CALUDE_straws_paper_difference_l1555_155573


namespace NUMINAMATH_CALUDE_abc_def_ratio_l1555_155522

theorem abc_def_ratio (a b c d e f : ℝ) 
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 6) :
  a * b * c / (d * e * f) = 1 / 12 := by
sorry

end NUMINAMATH_CALUDE_abc_def_ratio_l1555_155522


namespace NUMINAMATH_CALUDE_round_to_nearest_thousandth_l1555_155546

/-- The decimal representation of the number to be rounded -/
def number : ℚ := 67 + 36 / 99

/-- Rounding a rational number to the nearest thousandth -/
def round_to_thousandth (q : ℚ) : ℚ := 
  (⌊q * 1000 + 1/2⌋ : ℤ) / 1000

/-- Theorem stating that rounding 67.363636... to the nearest thousandth equals 67.364 -/
theorem round_to_nearest_thousandth :
  round_to_thousandth number = 67364 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_round_to_nearest_thousandth_l1555_155546


namespace NUMINAMATH_CALUDE_f_properties_implications_l1555_155592

/-- A function satisfying the given properties -/
def f_properties (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = f x) ∧  -- even function
  (∀ x, f x = f (2 - x)) ∧  -- symmetric about x = 1
  (∀ x₁ x₂, x₁ ∈ Set.Icc 0 (1/2) → x₂ ∈ Set.Icc 0 (1/2) → f (x₁ + x₂) = f x₁ * f x₂) ∧
  f 1 = 2

theorem f_properties_implications {f : ℝ → ℝ} (hf : f_properties f) :
  f (1/2) = Real.sqrt 2 ∧ f (1/4) = Real.sqrt (Real.sqrt 2) ∧ ∀ x, f x = f (x + 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_implications_l1555_155592


namespace NUMINAMATH_CALUDE_seating_arrangements_correct_l1555_155521

/-- The number of ways to arrange four people in five chairs, with the first chair always occupied -/
def seating_arrangements : ℕ := 120

/-- The number of chairs in the row -/
def num_chairs : ℕ := 5

/-- The number of people to be seated -/
def num_people : ℕ := 4

/-- Theorem stating that the number of seating arrangements is correct -/
theorem seating_arrangements_correct : 
  seating_arrangements = (num_chairs - 1) * (num_chairs - 2) * (num_chairs - 3) * (num_chairs - 4) :=
by sorry

end NUMINAMATH_CALUDE_seating_arrangements_correct_l1555_155521


namespace NUMINAMATH_CALUDE_jake_has_one_more_balloon_l1555_155585

/-- The number of balloons Allan brought to the park -/
def allan_initial : ℕ := 2

/-- The number of balloons Allan bought at the park -/
def allan_bought : ℕ := 3

/-- The number of balloons Jake brought to the park -/
def jake_balloons : ℕ := 6

/-- Allan's total number of balloons in the park -/
def allan_total : ℕ := allan_initial + allan_bought

/-- The difference between Jake's balloons and Allan's total balloons -/
def balloon_difference : ℤ := jake_balloons - allan_total

theorem jake_has_one_more_balloon : balloon_difference = 1 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_one_more_balloon_l1555_155585


namespace NUMINAMATH_CALUDE_starting_lineup_combinations_l1555_155500

def team_size : ℕ := 15
def lineup_size : ℕ := 6
def guaranteed_players : ℕ := 2

theorem starting_lineup_combinations :
  Nat.choose (team_size - guaranteed_players) (lineup_size - guaranteed_players) = 715 := by
  sorry

end NUMINAMATH_CALUDE_starting_lineup_combinations_l1555_155500


namespace NUMINAMATH_CALUDE_quadratic_complete_square_sum_l1555_155596

/-- Given a quadratic equation x^2 - 2x + m = 0 that can be written as (x-1)^2 = n
    after completing the square, prove that m + n = 1 -/
theorem quadratic_complete_square_sum (m n : ℝ) : 
  (∀ x, x^2 - 2*x + m = 0 ↔ (x - 1)^2 = n) → m + n = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_sum_l1555_155596


namespace NUMINAMATH_CALUDE_percentage_problem_l1555_155574

theorem percentage_problem (x : ℝ) :
  (15 / 100) * (30 / 100) * (50 / 100) * x = 126 →
  x = 5600 := by
sorry

end NUMINAMATH_CALUDE_percentage_problem_l1555_155574


namespace NUMINAMATH_CALUDE_intersection_point_and_lines_l1555_155555

/-- Given two lines that intersect at point P, this theorem proves:
    1. The equation of a line passing through P and parallel to a given line
    2. The equation of a line passing through P that maximizes the distance from the origin --/
theorem intersection_point_and_lines (x y : ℝ) :
  (2 * x + y = 8) →
  (x - 2 * y = -1) →
  (∃ P : ℝ × ℝ, P.1 = x ∧ P.2 = y) →
  (∃ l₁ : ℝ → ℝ → Prop, l₁ x y ↔ 4 * x - 3 * y = 6) ∧
  (∃ l₂ : ℝ → ℝ → Prop, l₂ x y ↔ 3 * x + 2 * y = 13) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_and_lines_l1555_155555


namespace NUMINAMATH_CALUDE_solve_for_x_and_y_l1555_155518

theorem solve_for_x_and_y :
  ∀ x y : ℝ,
  (15 + 24 + x + y) / 4 = 20 →
  x - y = 6 →
  x = 23.5 ∧ y = 17.5 := by
sorry

end NUMINAMATH_CALUDE_solve_for_x_and_y_l1555_155518


namespace NUMINAMATH_CALUDE_machine_theorem_l1555_155572

def machine_step (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 3 * n + 1

def machine_4_steps (n : ℕ) : ℕ :=
  machine_step (machine_step (machine_step (machine_step n)))

theorem machine_theorem :
  ∀ n : ℕ, n > 0 → (machine_4_steps n = 10 ↔ n = 3 ∨ n = 160) := by
  sorry

end NUMINAMATH_CALUDE_machine_theorem_l1555_155572


namespace NUMINAMATH_CALUDE_additional_cars_needed_l1555_155503

def current_cars : ℕ := 23
def cars_per_row : ℕ := 6

theorem additional_cars_needed :
  let next_multiple := (current_cars + cars_per_row - 1) / cars_per_row * cars_per_row
  next_multiple - current_cars = 1 := by sorry

end NUMINAMATH_CALUDE_additional_cars_needed_l1555_155503


namespace NUMINAMATH_CALUDE_smallest_n_for_zoe_play_l1555_155579

def can_zoe_play (n : ℕ) : Prop :=
  ∀ (yvan_first : ℕ) (h_yvan_first : yvan_first ≤ n),
    ∃ (zoe_first : ℕ) (zoe_last : ℕ),
      zoe_first < zoe_last ∧
      zoe_last ≤ n ∧
      zoe_first ≠ yvan_first ∧
      zoe_last ≠ yvan_first ∧
      ∀ (yvan_second : ℕ) (yvan_second_last : ℕ),
        yvan_second < yvan_second_last ∧
        yvan_second_last ≤ n ∧
        yvan_second ≠ yvan_first ∧
        yvan_second_last ≠ yvan_first ∧
        yvan_second ∉ Set.Icc zoe_first zoe_last ∧
        yvan_second_last ∉ Set.Icc zoe_first zoe_last →
        ∃ (zoe_second : ℕ) (zoe_second_last : ℕ),
          zoe_second < zoe_second_last ∧
          zoe_second_last ≤ n ∧
          zoe_second ∉ Set.Icc zoe_first zoe_last ∪ Set.Icc yvan_second yvan_second_last ∧
          zoe_second_last ∉ Set.Icc zoe_first zoe_last ∪ Set.Icc yvan_second yvan_second_last ∧
          zoe_second_last - zoe_second = 3

theorem smallest_n_for_zoe_play :
  (∀ k < 14, ¬ can_zoe_play k) ∧ can_zoe_play 14 := by sorry

end NUMINAMATH_CALUDE_smallest_n_for_zoe_play_l1555_155579


namespace NUMINAMATH_CALUDE_pool_completion_theorem_l1555_155514

/-- A pool with blue and red tiles that needs to be completed -/
structure Pool :=
  (blue_tiles : ℕ)
  (red_tiles : ℕ)
  (total_required : ℕ)

/-- Calculate the number of additional tiles needed to complete the pool -/
def additional_tiles_needed (p : Pool) : ℕ :=
  p.total_required - (p.blue_tiles + p.red_tiles)

/-- Theorem stating that for a pool with 48 blue tiles, 32 red tiles, 
    and a total requirement of 100 tiles, 20 additional tiles are needed -/
theorem pool_completion_theorem :
  let p : Pool := { blue_tiles := 48, red_tiles := 32, total_required := 100 }
  additional_tiles_needed p = 20 := by
  sorry

end NUMINAMATH_CALUDE_pool_completion_theorem_l1555_155514


namespace NUMINAMATH_CALUDE_circle_equation_l1555_155544

/-- Given a circle with center (t, t^2/4) and radius |t|, if it's tangent to y-axis and y = -1, 
    then its equation is x^2 + y^2 ± 4x - 2y + 1 = 0 -/
theorem circle_equation (t : ℝ) (h : t ≠ 0) :
  let center := (t, t^2/4)
  let radius := |t|
  (∃ (x : ℝ), (x - t)^2 + (0 - t^2/4)^2 = t^2) →  -- tangent to y-axis
  (1 + t^2/4)^2 = t^2 →                           -- tangent to y = -1
  ∃ (sign : ℝ) (h_sign : sign = 1 ∨ sign = -1),
    ∀ (x y : ℝ), (x - t)^2 + (y - t^2/4)^2 = t^2 ↔ 
      x^2 + y^2 + sign * 4*x - 2*y + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_l1555_155544


namespace NUMINAMATH_CALUDE_HN_passes_through_fixed_point_l1555_155553

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 3 + y^2 / 4 = 1

-- Define the line segment AB
def line_AB (x y : ℝ) : Prop := y = 2/3 * x - 2

-- Define a point on the ellipse
def on_ellipse (p : ℝ × ℝ) : Prop := ellipse p.1 p.2

-- Define a point on line AB
def on_line_AB (p : ℝ × ℝ) : Prop := line_AB p.1 p.2

-- Define the property of T being on the line parallel to x-axis through M
def T_on_parallel_line (M T : ℝ × ℝ) : Prop := M.2 = T.2

-- Define the property of H satisfying MT = TH
def H_satisfies_MT_eq_TH (M T H : ℝ × ℝ) : Prop := 
  (H.1 - T.1 = T.1 - M.1) ∧ (H.2 - T.2 = T.2 - M.2)

-- Main theorem
theorem HN_passes_through_fixed_point :
  ∀ (M N T H : ℝ × ℝ),
  on_ellipse M → on_ellipse N →
  on_line_AB T →
  T_on_parallel_line M T →
  H_satisfies_MT_eq_TH M T H →
  ∃ (t : ℝ), (1 - t) * H.1 + t * N.1 = 0 ∧ (1 - t) * H.2 + t * N.2 = -2 :=
sorry

end NUMINAMATH_CALUDE_HN_passes_through_fixed_point_l1555_155553


namespace NUMINAMATH_CALUDE_pams_bank_theorem_l1555_155512

def pams_bank_problem (current_balance end_year_withdrawal initial_balance : ℕ) : Prop :=
  let end_year_balance := current_balance + end_year_withdrawal
  let ratio := end_year_balance / initial_balance
  ratio = 19 / 8

theorem pams_bank_theorem :
  pams_bank_problem 950 250 400 := by
  sorry

end NUMINAMATH_CALUDE_pams_bank_theorem_l1555_155512


namespace NUMINAMATH_CALUDE_value_of_N_l1555_155551

theorem value_of_N : ∃ N : ℝ, (0.20 * N = 0.30 * 5000) ∧ (N = 7500) := by
  sorry

end NUMINAMATH_CALUDE_value_of_N_l1555_155551


namespace NUMINAMATH_CALUDE_alia_has_40_markers_l1555_155552

-- Define the number of markers for each person
def steve_markers : ℕ := 60
def austin_markers : ℕ := steve_markers / 3
def alia_markers : ℕ := 2 * austin_markers

-- Theorem statement
theorem alia_has_40_markers : alia_markers = 40 := by
  sorry

end NUMINAMATH_CALUDE_alia_has_40_markers_l1555_155552


namespace NUMINAMATH_CALUDE_h2o_formation_in_neutralization_l1555_155542

/-- Represents a chemical substance -/
structure Substance where
  name : String
  moles : ℝ

/-- Represents a chemical reaction -/
structure Reaction where
  reactants : List Substance
  products : List Substance

/-- Given a balanced chemical equation and the amounts of reactants, 
    calculate the amount of a specific product formed -/
def calculateProductAmount (reaction : Reaction) (product : Substance) : ℝ :=
  sorry

theorem h2o_formation_in_neutralization :
  let hch3co2 := Substance.mk "HCH3CO2" 1
  let naoh := Substance.mk "NaOH" 1
  let h2o := Substance.mk "H2O" 1
  let nach3co2 := Substance.mk "NaCH3CO2" 1
  let reaction := Reaction.mk [hch3co2, naoh] [nach3co2, h2o]
  calculateProductAmount reaction h2o = 1 := by
  sorry

end NUMINAMATH_CALUDE_h2o_formation_in_neutralization_l1555_155542


namespace NUMINAMATH_CALUDE_partial_fraction_sum_l1555_155577

open Real

theorem partial_fraction_sum : ∃ (A B C D E : ℝ),
  (∀ x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ x ≠ -2 ∧ x ≠ -3 ∧ x ≠ -5 →
    1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) =
    A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5)) ∧
  A + B + C + D + E = 1 / 6 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_l1555_155577


namespace NUMINAMATH_CALUDE_right_triangle_sides_l1555_155598

theorem right_triangle_sides : ∃! (a b c : ℕ), 
  ((a = 3 ∧ b = 4 ∧ c = 5) ∨ 
   (a = 2 ∧ b = 3 ∧ c = 4) ∨ 
   (a = 4 ∧ b = 5 ∧ c = 6) ∨ 
   (a = 5 ∧ b = 6 ∧ c = 7)) ∧ 
  a^2 + b^2 = c^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sides_l1555_155598


namespace NUMINAMATH_CALUDE_perpendicular_lines_l1555_155593

theorem perpendicular_lines (b : ℚ) : 
  (∀ x y : ℚ, 2 * x + 3 * y + 4 = 0 → ∃ m₁ : ℚ, y = m₁ * x + (-4/3)) →
  (∀ x y : ℚ, b * x + 3 * y + 4 = 0 → ∃ m₂ : ℚ, y = m₂ * x + (-4/3)) →
  (∃ m₁ m₂ : ℚ, m₁ * m₂ = -1) →
  b = -9/2 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_l1555_155593


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1555_155545

/-- The quadratic function f(x) = ax² + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The proposition that the solution set of ax² + bx + c > 0 is {x | x < -2 or x > 4} -/
def solution_set (a b c : ℝ) : Prop :=
  ∀ x, a * x^2 + b * x + c > 0 ↔ (x < -2 ∨ x > 4)

theorem quadratic_inequality (a b c : ℝ) (h : solution_set a b c) :
  f a b c 2 < f a b c (-1) ∧ f a b c (-1) < f a b c 5 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1555_155545


namespace NUMINAMATH_CALUDE_square_difference_area_l1555_155580

theorem square_difference_area (a b : ℝ) (h : a > b) :
  (a ^ 2 - b ^ 2 : ℝ) = (Real.sqrt (a ^ 2 - b ^ 2)) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_area_l1555_155580


namespace NUMINAMATH_CALUDE_not_divisible_by_twelve_l1555_155508

theorem not_divisible_by_twelve (m : ℕ) (h1 : m > 0) 
  (h2 : ∃ (j : ℤ), (1 : ℚ) / 3 + (1 : ℚ) / 4 + (1 : ℚ) / 9 + (1 : ℚ) / m = j) : 
  ¬(12 ∣ m) := by
sorry

end NUMINAMATH_CALUDE_not_divisible_by_twelve_l1555_155508


namespace NUMINAMATH_CALUDE_red_balloons_count_l1555_155595

theorem red_balloons_count (total : ℕ) (green : ℕ) (h1 : total = 17) (h2 : green = 9) :
  total - green = 8 := by
  sorry

end NUMINAMATH_CALUDE_red_balloons_count_l1555_155595


namespace NUMINAMATH_CALUDE_complex_expression_value_l1555_155506

theorem complex_expression_value : 
  2.5 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_value_l1555_155506


namespace NUMINAMATH_CALUDE_fraction_arrangement_equals_two_l1555_155525

theorem fraction_arrangement_equals_two : ∃ (f : ℚ → ℚ → ℚ → ℚ → ℚ), f (1/4) (1/4) (1/4) (1/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_arrangement_equals_two_l1555_155525


namespace NUMINAMATH_CALUDE_jacket_price_reduction_l1555_155519

theorem jacket_price_reduction (initial_reduction : ℝ) (final_increase : ℝ) (special_reduction : ℝ) : 
  initial_reduction = 25 →
  final_increase = 48.148148148148145 →
  (1 - initial_reduction / 100) * (1 - special_reduction / 100) * (1 + final_increase / 100) = 1 →
  special_reduction = 10 := by
sorry

end NUMINAMATH_CALUDE_jacket_price_reduction_l1555_155519


namespace NUMINAMATH_CALUDE_f_always_negative_iff_a_in_range_l1555_155588

/-- A quadratic function f(x) = ax^2 + ax - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a * x - 1

/-- The property that f(x) is always less than 0 on ℝ -/
def always_negative (a : ℝ) : Prop := ∀ x, f a x < 0

/-- Theorem stating that f(x) is always negative if and only if a is in the interval (-4, 0] -/
theorem f_always_negative_iff_a_in_range :
  ∀ a : ℝ, always_negative a ↔ -4 < a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_f_always_negative_iff_a_in_range_l1555_155588


namespace NUMINAMATH_CALUDE_cost_of_tax_free_items_l1555_155547

/-- Calculates the cost of tax-free items given total amount spent, sales tax paid, and tax rate -/
theorem cost_of_tax_free_items
  (total_spent : ℝ)
  (sales_tax : ℝ)
  (tax_rate : ℝ)
  (h_total : total_spent = 25)
  (h_tax : sales_tax = 0.30)
  (h_rate : tax_rate = 0.06)
  : ∃ (cost_tax_free : ℝ), cost_tax_free = 20 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_tax_free_items_l1555_155547


namespace NUMINAMATH_CALUDE_probability_of_three_successes_l1555_155558

def n : ℕ := 7
def k : ℕ := 3
def p : ℚ := 1/3

theorem probability_of_three_successes :
  Nat.choose n k * p^k * (1-p)^(n-k) = 560/2187 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_three_successes_l1555_155558


namespace NUMINAMATH_CALUDE_one_contribution_before_john_l1555_155536

-- Define the problem parameters
def john_donation : ℝ := 100
def new_average : ℝ := 75
def increase_percentage : ℝ := 0.5

-- Define the theorem
theorem one_contribution_before_john
  (n : ℝ) -- Initial number of contributions
  (A : ℝ) -- Initial average contribution
  (h1 : A + increase_percentage * A = new_average) -- New average is 50% higher
  (h2 : n * A + john_donation = (n + 1) * new_average) -- Total amount equality
  : n = 1 := by
  sorry


end NUMINAMATH_CALUDE_one_contribution_before_john_l1555_155536


namespace NUMINAMATH_CALUDE_max_perimeter_special_triangle_l1555_155584

/-- A triangle with integer side lengths, where one side is twice another, and the third side is 10 -/
structure SpecialTriangle where
  x : ℕ
  side1 : ℕ := x
  side2 : ℕ := 2 * x
  side3 : ℕ := 10

/-- The perimeter of a SpecialTriangle -/
def perimeter (t : SpecialTriangle) : ℕ := t.side1 + t.side2 + t.side3

/-- The triangle inequality for SpecialTriangle -/
def is_valid (t : SpecialTriangle) : Prop :=
  t.side1 + t.side2 > t.side3 ∧
  t.side1 + t.side3 > t.side2 ∧
  t.side2 + t.side3 > t.side1

/-- The theorem stating the maximum perimeter of a valid SpecialTriangle -/
theorem max_perimeter_special_triangle :
  ∃ (t : SpecialTriangle), is_valid t ∧
  ∀ (t' : SpecialTriangle), is_valid t' → perimeter t' ≤ perimeter t ∧
  perimeter t = 37 := by sorry

end NUMINAMATH_CALUDE_max_perimeter_special_triangle_l1555_155584


namespace NUMINAMATH_CALUDE_donna_episodes_per_weekday_l1555_155559

theorem donna_episodes_per_weekday : 
  ∀ (weekday_episodes : ℕ),
  weekday_episodes > 0 →
  5 * weekday_episodes + 2 * (3 * weekday_episodes) = 88 →
  weekday_episodes = 8 := by
sorry

end NUMINAMATH_CALUDE_donna_episodes_per_weekday_l1555_155559


namespace NUMINAMATH_CALUDE_quadratic_solutions_fractional_solution_l1555_155533

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 + 3*x + 1 = 0

-- Define the fractional equation
def fractional_equation (x : ℝ) : Prop := (4*x)/(x-2) - 1 = 3/(2-x)

-- Theorem for the quadratic equation solutions
theorem quadratic_solutions :
  ∃ x1 x2 : ℝ, 
    quadratic_equation x1 ∧ 
    quadratic_equation x2 ∧ 
    x1 = (-3 + Real.sqrt 5) / 2 ∧ 
    x2 = (-3 - Real.sqrt 5) / 2 :=
sorry

-- Theorem for the fractional equation solution
theorem fractional_solution :
  ∃ x : ℝ, fractional_equation x ∧ x = -5/3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_solutions_fractional_solution_l1555_155533


namespace NUMINAMATH_CALUDE_log_inequality_condition_l1555_155568

theorem log_inequality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ a b, Real.log a > Real.log b → 2*a > 2*b) ∧
  ¬(∀ a b, 2*a > 2*b → Real.log a > Real.log b) :=
sorry

end NUMINAMATH_CALUDE_log_inequality_condition_l1555_155568


namespace NUMINAMATH_CALUDE_radish_distribution_l1555_155583

theorem radish_distribution (total : ℕ) (groups : ℕ) (first_basket : ℕ) 
  (h1 : total = 88)
  (h2 : groups = 4)
  (h3 : first_basket = 37)
  (h4 : total % groups = 0) : 
  (total - first_basket) - first_basket = 14 := by
  sorry

end NUMINAMATH_CALUDE_radish_distribution_l1555_155583


namespace NUMINAMATH_CALUDE_strawberry_area_l1555_155541

theorem strawberry_area (garden_size : ℝ) (fruit_ratio : ℝ) (strawberry_ratio : ℝ) : 
  garden_size = 64 → 
  fruit_ratio = 1/2 → 
  strawberry_ratio = 1/4 → 
  garden_size * fruit_ratio * strawberry_ratio = 8 := by
sorry

end NUMINAMATH_CALUDE_strawberry_area_l1555_155541


namespace NUMINAMATH_CALUDE_pattern_proof_l1555_155582

theorem pattern_proof (x : ℝ) (hx : x > 0) 
  (h1 : x + 1 / x ≥ 2)
  (h2 : x + 4 / x^2 ≥ 3)
  (h3 : x + 27 / x^3 ≥ 4)
  (h4 : ∃ a : ℝ, x + a / x^4 ≥ 5) :
  ∃ a : ℝ, x + a / x^4 ≥ 5 ∧ a = 256 := by
sorry

end NUMINAMATH_CALUDE_pattern_proof_l1555_155582


namespace NUMINAMATH_CALUDE_gary_chickens_l1555_155575

theorem gary_chickens (initial_chickens : ℕ) : 
  (∃ (current_chickens : ℕ), 
    current_chickens = 8 * initial_chickens ∧ 
    6 * 7 * current_chickens = 1344) → 
  initial_chickens = 4 := by
sorry

end NUMINAMATH_CALUDE_gary_chickens_l1555_155575


namespace NUMINAMATH_CALUDE_cake_two_sided_icing_count_l1555_155550

/-- Represents a cube cake with icing on specific faces -/
structure CakeCube where
  size : Nat
  icedFaces : Finset (Fin 3)

/-- Counts the number of 1×1×1 subcubes with icing on exactly two sides -/
def countTwoSidedIcingCubes (cake : CakeCube) : Nat :=
  sorry

/-- The main theorem stating that a 5×5×5 cake with icing on top, front, and back
    has exactly 12 subcubes with icing on two sides when cut into 1×1×1 cubes -/
theorem cake_two_sided_icing_count :
  let cake : CakeCube := { size := 5, icedFaces := {0, 1, 2} }
  countTwoSidedIcingCubes cake = 12 := by
  sorry

end NUMINAMATH_CALUDE_cake_two_sided_icing_count_l1555_155550


namespace NUMINAMATH_CALUDE_count_five_ruble_coins_l1555_155597

theorem count_five_ruble_coins 
  (total_coins : ℕ) 
  (not_two_ruble : ℕ) 
  (not_ten_ruble : ℕ) 
  (not_one_ruble : ℕ) 
  (h1 : total_coins = 25)
  (h2 : not_two_ruble = 19)
  (h3 : not_ten_ruble = 20)
  (h4 : not_one_ruble = 16) :
  total_coins - ((total_coins - not_two_ruble) + (total_coins - not_ten_ruble) + (total_coins - not_one_ruble)) = 5 := by
sorry

end NUMINAMATH_CALUDE_count_five_ruble_coins_l1555_155597


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1555_155523

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_ninth_term 
  (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 5 + a 7 = 16) 
  (h_third : a 3 = 4) : 
  a 9 = 12 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l1555_155523


namespace NUMINAMATH_CALUDE_total_deflection_is_zero_l1555_155526

/-- The total deflection of a beam passing through two mirrors -/
def total_deflection (β : ℝ) : ℝ :=
  2 * β - 2 * β

/-- Theorem: The total deflection is zero -/
theorem total_deflection_is_zero (β : ℝ) :
  total_deflection β = 0 := by
  sorry

#check total_deflection_is_zero

end NUMINAMATH_CALUDE_total_deflection_is_zero_l1555_155526


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1555_155591

theorem sufficient_not_necessary (a b : ℝ) : 
  (a^2 + b^2 = 0 → a * b = 0) ∧ 
  ∃ a b : ℝ, a * b = 0 ∧ a^2 + b^2 ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1555_155591


namespace NUMINAMATH_CALUDE_find_p_l1555_155516

-- Define the quadratic equation
def quadratic_eq (p : ℝ) (x : ℝ) : ℝ := x^2 - 5*p*x + 2*p^3

-- Define a and b as roots of the quadratic equation
def roots_condition (a b p : ℝ) : Prop := 
  quadratic_eq p a = 0 ∧ quadratic_eq p b = 0

-- Define the condition that a and b are non-zero
def non_zero_condition (a b : ℝ) : Prop := a ≠ 0 ∧ b ≠ 0

-- Define the condition for unique root
def unique_root_condition (a b : ℝ) : Prop := 
  ∃! x, x^2 - a*x + b = 0

-- Main theorem
theorem find_p (a b p : ℝ) : 
  non_zero_condition a b → 
  roots_condition a b p → 
  unique_root_condition a b → 
  p = 3 := by sorry

end NUMINAMATH_CALUDE_find_p_l1555_155516


namespace NUMINAMATH_CALUDE_cubic_roots_classification_l1555_155537

/-- The discriminant of the cubic equation x³ + px + q = 0 -/
def discriminant (p q : ℝ) : ℝ := 4 * p^3 + 27 * q^2

/-- Theorem about the nature of roots for the cubic equation x³ + px + q = 0 -/
theorem cubic_roots_classification (p q : ℝ) :
  (discriminant p q > 0 → ∃ (x : ℂ), x^3 + p*x + q = 0 ∧ (∀ y : ℂ, y^3 + p*y + q = 0 → y = x ∨ y.im ≠ 0)) ∧
  (discriminant p q < 0 → ∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ x^3 + p*x + q = 0 ∧ y^3 + p*y + q = 0 ∧ z^3 + p*z + q = 0) ∧
  (discriminant p q = 0 ∧ p = 0 ∧ q = 0 → ∃ (x : ℝ), ∀ y : ℝ, y^3 + p*y + q = 0 → y = x) ∧
  (discriminant p q = 0 ∧ (p ≠ 0 ∨ q ≠ 0) → ∃ (x y : ℝ), x ≠ y ∧ x^3 + p*x + q = 0 ∧ y^3 + p*y + q = 0 ∧ 
    ∀ z : ℝ, z^3 + p*z + q = 0 → z = x ∨ z = y) :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_classification_l1555_155537


namespace NUMINAMATH_CALUDE_jillian_apartment_size_l1555_155504

/-- The cost per square foot of apartment rentals in Rivertown -/
def cost_per_sqft : ℚ := 1.20

/-- Jillian's maximum monthly budget for rent -/
def max_budget : ℚ := 720

/-- The largest apartment size Jillian should consider -/
def largest_apartment_size : ℚ := max_budget / cost_per_sqft

theorem jillian_apartment_size :
  largest_apartment_size = 600 :=
by sorry

end NUMINAMATH_CALUDE_jillian_apartment_size_l1555_155504


namespace NUMINAMATH_CALUDE_laundry_charge_per_shirt_l1555_155562

theorem laundry_charge_per_shirt 
  (total_trousers : ℕ) 
  (cost_per_trouser : ℚ) 
  (total_bill : ℚ) 
  (total_shirts : ℕ) : 
  (total_bill - total_trousers * cost_per_trouser) / total_shirts = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_laundry_charge_per_shirt_l1555_155562


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1555_155530

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a = 1/4 → ∀ x : ℝ, x > 0 → x + a/x ≥ 1) ∧
  (∃ a : ℝ, a > 1/4 ∧ ∀ x : ℝ, x > 0 → x + a/x ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1555_155530


namespace NUMINAMATH_CALUDE_parabola_properties_l1555_155531

/-- A parabola with equation y = ax^2 + (2m-6)x + 1 passing through (1, 2m-4) -/
def Parabola (a m : ℝ) : ℝ → ℝ := λ x => a * x^2 + (2*m - 6) * x + 1

/-- Points on the parabola -/
def PointsOnParabola (m : ℝ) : (ℝ × ℝ) × (ℝ × ℝ) × (ℝ × ℝ) :=
  ((-m, Parabola 3 m (-m)), (m, Parabola 3 m m), (m+2, Parabola 3 m (m+2)))

theorem parabola_properties (m : ℝ) :
  let (y1, y2, y3) := (Parabola 3 m (-m), Parabola 3 m m, Parabola 3 m (m+2))
  Parabola 3 m 1 = 2*m - 4 ∧ 
  y2 < y3 ∧ y3 ≤ y1 →
  (3 : ℝ) = 3 ∧
  (3 - m : ℝ) = -((2*m - 6) / (2*3)) ∧
  1 < m ∧ m ≤ 2 := by sorry

#check parabola_properties

end NUMINAMATH_CALUDE_parabola_properties_l1555_155531


namespace NUMINAMATH_CALUDE_two_digit_number_interchange_l1555_155515

theorem two_digit_number_interchange (x y : ℕ) : 
  x ≥ 1 ∧ x ≤ 9 ∧ y ≥ 0 ∧ y ≤ 9 ∧ x - y = 5 → 
  (10 * x + y) - (10 * y + x) = 45 :=
by sorry

end NUMINAMATH_CALUDE_two_digit_number_interchange_l1555_155515


namespace NUMINAMATH_CALUDE_washing_machines_removed_count_l1555_155587

/-- Represents the number of washing machines removed from a shipping container --/
def washing_machines_removed (num_crates : ℕ) (boxes_per_crate : ℕ) (initial_machines_per_box : ℕ) (machines_removed_per_box : ℕ) : ℕ :=
  num_crates * boxes_per_crate * machines_removed_per_box

/-- Theorem stating that 60 washing machines were removed from the shipping container --/
theorem washing_machines_removed_count : 
  washing_machines_removed 10 6 4 1 = 60 := by
  sorry

end NUMINAMATH_CALUDE_washing_machines_removed_count_l1555_155587


namespace NUMINAMATH_CALUDE_minimize_sum_distances_l1555_155567

/-- A structure representing a point on a line --/
structure Point where
  x : ℝ

/-- The distance between two points on a line --/
def distance (p q : Point) : ℝ := abs (p.x - q.x)

/-- The theorem stating that Q₅ minimizes the sum of distances --/
theorem minimize_sum_distances 
  (Q₁ Q₂ Q₃ Q₄ Q₅ Q₆ Q₇ Q₈ Q₉ : Point)
  (h_order : Q₁.x < Q₂.x ∧ Q₂.x < Q₃.x ∧ Q₃.x < Q₄.x ∧ Q₄.x < Q₅.x ∧ 
             Q₅.x < Q₆.x ∧ Q₆.x < Q₇.x ∧ Q₇.x < Q₈.x ∧ Q₈.x < Q₉.x)
  (h_fixed : Q₁.x ≠ Q₉.x)
  (h_not_midpoint : Q₅.x ≠ (Q₁.x + Q₉.x) / 2) :
  ∀ Q : Point, 
    distance Q Q₁ + distance Q Q₂ + distance Q Q₃ + distance Q Q₄ + 
    distance Q Q₅ + distance Q Q₆ + distance Q Q₇ + distance Q Q₈ + 
    distance Q Q₉ 
    ≥ 
    distance Q₅ Q₁ + distance Q₅ Q₂ + distance Q₅ Q₃ + distance Q₅ Q₄ + 
    distance Q₅ Q₅ + distance Q₅ Q₆ + distance Q₅ Q₇ + distance Q₅ Q₈ + 
    distance Q₅ Q₉ :=
by sorry

end NUMINAMATH_CALUDE_minimize_sum_distances_l1555_155567


namespace NUMINAMATH_CALUDE_sequence_sum_unique_value_l1555_155524

def is_strictly_increasing (s : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → s n < s m

theorem sequence_sum_unique_value
  (a b : ℕ → ℕ)
  (h_a_incr : is_strictly_increasing a)
  (h_b_incr : is_strictly_increasing b)
  (h_eq : a 10 = b 10)
  (h_lt : a 10 < 2017)
  (h_a_rec : ∀ n : ℕ, a (n + 2) = a (n + 1) + a n)
  (h_b_rec : ∀ n : ℕ, b (n + 1) = 2 * b n) :
  a 1 + b 1 = 5 := by
sorry

end NUMINAMATH_CALUDE_sequence_sum_unique_value_l1555_155524


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l1555_155578

/-- The area of an equilateral triangle with altitude √8 units is (32√3)/3 square units. -/
theorem equilateral_triangle_area (altitude : ℝ) (h : altitude = Real.sqrt 8) :
  let side := (2 * altitude) / Real.sqrt 3
  let area := (Real.sqrt 3 / 4) * side^2
  area = (32 * Real.sqrt 3) / 3 := by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l1555_155578


namespace NUMINAMATH_CALUDE_chair_count_sequence_l1555_155569

theorem chair_count_sequence (a : ℕ → ℕ) 
  (h1 : a 1 = 14)
  (h2 : a 2 = 23)
  (h3 : a 3 = 32)
  (h5 : a 5 = 50)
  (h6 : a 6 = 59)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → a (n + 1) - a n = a 2 - a 1) :
  a 4 = 41 := by
  sorry

end NUMINAMATH_CALUDE_chair_count_sequence_l1555_155569


namespace NUMINAMATH_CALUDE_sum_of_cubes_product_l1555_155561

theorem sum_of_cubes_product : ∃ x y : ℤ, x^3 + y^3 = 35 ∧ x * y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_product_l1555_155561


namespace NUMINAMATH_CALUDE_conic_focal_distance_l1555_155549

/-- The focal distance of a conic curve x^2 + y^2/m = 1, where m is the geometric mean of 2 and 8 -/
theorem conic_focal_distance (m : ℝ) : 
  (m^2 = 2 * 8) →  -- m is the geometric mean between 2 and 8
  let focal_distance := 
    if m > 0 then 2 * Real.sqrt 3  -- Ellipse case
    else 2 * Real.sqrt 5           -- Hyperbola case
  (∃ (x y : ℝ), x^2 + y^2/m = 1) →  -- The conic curve exists
  focal_distance = 2 * Real.sqrt 3 ∨ focal_distance = 2 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_conic_focal_distance_l1555_155549


namespace NUMINAMATH_CALUDE_intersection_point_l1555_155581

/-- The point of intersection of two lines in a 2D plane. -/
structure IntersectionPoint where
  x : ℝ
  y : ℝ

/-- First line: y = 2x -/
def line1 (p : IntersectionPoint) : Prop := p.y = 2 * p.x

/-- Second line: x + y = 3 -/
def line2 (p : IntersectionPoint) : Prop := p.x + p.y = 3

/-- The intersection point of the two lines -/
def intersection : IntersectionPoint := ⟨1, 2⟩

/-- Theorem: The point (1, 2) is the unique intersection of the lines y = 2x and x + y = 3 -/
theorem intersection_point :
  line1 intersection ∧ line2 intersection ∧
  ∀ p : IntersectionPoint, line1 p ∧ line2 p → p = intersection :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l1555_155581


namespace NUMINAMATH_CALUDE_stating_isosceles_triangle_base_height_l1555_155502

/-- Represents an isosceles triangle with leg length a -/
structure IsoscelesTriangle (a : ℝ) where
  (a_pos : a > 0)

/-- The height from one leg to the other leg forms a 30° angle -/
def height_angle (t : IsoscelesTriangle a) : ℝ := 30

/-- The height from the base of the isosceles triangle -/
def base_height (t : IsoscelesTriangle a) : Set ℝ :=
  {h | h = (Real.sqrt 3 / 2) * a ∨ h = (1 / 2) * a}

/-- 
  Theorem stating that for an isosceles triangle with leg length a, 
  where the height from one leg to the other leg forms a 30° angle, 
  the height from the base is either (√3/2)a or (1/2)a.
-/
theorem isosceles_triangle_base_height (a : ℝ) (t : IsoscelesTriangle a) :
  ∀ h, h ∈ base_height t ↔ 
    (h = (Real.sqrt 3 / 2) * a ∨ h = (1 / 2) * a) ∧ 
    height_angle t = 30 := by
  sorry

end NUMINAMATH_CALUDE_stating_isosceles_triangle_base_height_l1555_155502


namespace NUMINAMATH_CALUDE_somu_present_age_l1555_155511

-- Define Somu's age and his father's age
def somu_age : ℕ := sorry
def father_age : ℕ := sorry

-- State the theorem
theorem somu_present_age :
  (somu_age = father_age / 3) ∧
  (somu_age - 8 = (father_age - 8) / 5) →
  somu_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_somu_present_age_l1555_155511


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1555_155556

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the hyperbola
def hyperbola (x y a b : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

-- State the theorem
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), parabola x y ∧ hyperbola x y a b) →
  (∀ (x y : ℝ), asymptotes x y) →
  (∀ (x y : ℝ), hyperbola x y 1 (Real.sqrt 3)) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l1555_155556


namespace NUMINAMATH_CALUDE_dragons_games_count_l1555_155507

theorem dragons_games_count :
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (0.4 : ℝ) * initial_games →
    (initial_wins + 5 : ℝ) / (initial_games + 8) = 0.55 →
    initial_games + 8 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_dragons_games_count_l1555_155507


namespace NUMINAMATH_CALUDE_binary_1011001_equals_quaternary_1121_l1555_155527

def binary_to_decimal (b : List Bool) : ℕ :=
  b.foldr (λ x acc => 2 * acc + if x then 1 else 0) 0

def decimal_to_quaternary (n : ℕ) : List (Fin 4) :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List (Fin 4) :=
    if m = 0 then [] else (m % 4) :: aux (m / 4)
  aux n |>.reverse

theorem binary_1011001_equals_quaternary_1121 :
  decimal_to_quaternary (binary_to_decimal [true, false, true, true, false, false, true]) = [1, 1, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_binary_1011001_equals_quaternary_1121_l1555_155527


namespace NUMINAMATH_CALUDE_total_water_needed_is_112_l1555_155599

/-- Calculates the total gallons of water needed for Nicole's fish tanks in four weeks -/
def water_needed_in_four_weeks : ℕ :=
  let num_tanks : ℕ := 4
  let first_tank_gallons : ℕ := 8
  let num_first_type_tanks : ℕ := 2
  let num_second_type_tanks : ℕ := num_tanks - num_first_type_tanks
  let second_tank_gallons : ℕ := first_tank_gallons - 2
  let weeks : ℕ := 4
  
  let weekly_total : ℕ := 
    first_tank_gallons * num_first_type_tanks + 
    second_tank_gallons * num_second_type_tanks
  
  weekly_total * weeks

/-- Theorem stating that the total gallons of water needed in four weeks is 112 -/
theorem total_water_needed_is_112 : water_needed_in_four_weeks = 112 := by
  sorry

end NUMINAMATH_CALUDE_total_water_needed_is_112_l1555_155599


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l1555_155563

theorem circle_area_from_circumference : 
  ∀ (r : ℝ), (2 * π * r = 36 * π) → (π * r^2 = 324 * π) := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l1555_155563


namespace NUMINAMATH_CALUDE_smallest_number_greater_than_digit_sum_by_1755_l1555_155570

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_number_greater_than_digit_sum_by_1755 :
  (∀ m : ℕ, m < 1770 → m ≠ sum_of_digits m + 1755) ∧
  1770 = sum_of_digits 1770 + 1755 := by
  sorry

end NUMINAMATH_CALUDE_smallest_number_greater_than_digit_sum_by_1755_l1555_155570


namespace NUMINAMATH_CALUDE_not_all_data_sets_have_regression_equation_l1555_155520

-- Define a type for data sets
structure DataSet where
  -- Add necessary fields to represent a data set
  dummy : Unit

-- Define a predicate for the existence of a regression equation
def has_regression_equation (ds : DataSet) : Prop :=
  -- Add necessary conditions for a data set to have a regression equation
  sorry

-- Theorem stating that not every data set has a regression equation
theorem not_all_data_sets_have_regression_equation :
  ¬ (∀ ds : DataSet, has_regression_equation ds) := by
  sorry

end NUMINAMATH_CALUDE_not_all_data_sets_have_regression_equation_l1555_155520


namespace NUMINAMATH_CALUDE_n_has_9_digits_l1555_155564

/-- The smallest positive integer satisfying the given conditions -/
def n : ℕ := sorry

/-- n is divisible by 30 -/
axiom n_div_30 : 30 ∣ n

/-- n^2 is a perfect cube -/
axiom n_sq_cube : ∃ k : ℕ, n^2 = k^3

/-- n^3 is a perfect square -/
axiom n_cube_square : ∃ k : ℕ, n^3 = k^2

/-- n is the smallest positive integer satisfying the conditions -/
axiom n_smallest : ∀ m : ℕ, m < n → ¬(30 ∣ m ∧ (∃ k : ℕ, m^2 = k^3) ∧ (∃ k : ℕ, m^3 = k^2))

/-- Function to count the number of digits in a natural number -/
def count_digits (x : ℕ) : ℕ := sorry

theorem n_has_9_digits : count_digits n = 9 := by sorry

end NUMINAMATH_CALUDE_n_has_9_digits_l1555_155564


namespace NUMINAMATH_CALUDE_sum_equals_product_l1555_155576

theorem sum_equals_product (x : ℝ) (h : x ≠ 1) :
  ∃! y : ℝ, x + y = x * y ∧ y = x / (x - 1) := by sorry

end NUMINAMATH_CALUDE_sum_equals_product_l1555_155576


namespace NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l1555_155571

/-- The focal length of a hyperbola with equation x²/a² - y²/b² = 1 is 2√(a² + b²) -/
theorem hyperbola_focal_length (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let focal_length := 2 * Real.sqrt (a^2 + b^2)
  focal_length = 2 * Real.sqrt 10 →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 ↔ x^2 / 7 - y^2 / 3 = 1) :=
by sorry

/-- The focal length of the hyperbola x²/7 - y²/3 = 1 is 2√10 -/
theorem specific_hyperbola_focal_length :
  let focal_length := 2 * Real.sqrt ((Real.sqrt 7)^2 + (Real.sqrt 3)^2)
  focal_length = 2 * Real.sqrt 10 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focal_length_specific_hyperbola_focal_length_l1555_155571


namespace NUMINAMATH_CALUDE_inequality_proof_l1555_155539

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 < (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ∧
  (a / Real.sqrt (a^2 + b^2)) + (b / Real.sqrt (b^2 + c^2)) + (c / Real.sqrt (c^2 + a^2)) ≤ 3 * Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1555_155539


namespace NUMINAMATH_CALUDE_rational_inequality_l1555_155566

theorem rational_inequality (a b : ℚ) 
  (h1 : |a| < |b|) 
  (h2 : a > 0) 
  (h3 : b < 0) : 
  b < -a ∧ -a < a ∧ a < -b := by
  sorry

end NUMINAMATH_CALUDE_rational_inequality_l1555_155566


namespace NUMINAMATH_CALUDE_train_length_l1555_155554

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 27 → time = 16 → speed * time * (5 / 18) = 120 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1555_155554


namespace NUMINAMATH_CALUDE_largest_common_term_l1555_155586

theorem largest_common_term (a : ℕ) : a ≤ 150 ∧ 
  (∃ n : ℕ, a = 2 + 5 * n) ∧ 
  (∃ m : ℕ, a = 5 + 8 * m) ∧
  (∀ b : ℕ, b ≤ 150 → 
    (∃ k : ℕ, b = 2 + 5 * k) → 
    (∃ l : ℕ, b = 5 + 8 * l) → 
    b ≤ a) →
  a = 117 := by sorry

end NUMINAMATH_CALUDE_largest_common_term_l1555_155586


namespace NUMINAMATH_CALUDE_range_of_m_l1555_155560

theorem range_of_m (x m : ℝ) : 
  (∀ x, (1/3 < x ∧ x < 1/2) → |x - m| < 1) →
  (-1/2 ≤ m ∧ m ≤ 4/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1555_155560


namespace NUMINAMATH_CALUDE_park_outer_boundary_diameter_l1555_155534

/-- The diameter of the outer boundary of a circular park structure -/
def outer_boundary_diameter (pond_diameter : ℝ) (picnic_width : ℝ) (track_width : ℝ) : ℝ :=
  pond_diameter + 2 * (picnic_width + track_width)

/-- Theorem stating the diameter of the outer boundary of the cycling track -/
theorem park_outer_boundary_diameter :
  outer_boundary_diameter 16 10 4 = 44 := by
  sorry

end NUMINAMATH_CALUDE_park_outer_boundary_diameter_l1555_155534


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1555_155538

theorem gcd_of_three_numbers : Nat.gcd 4557 (Nat.gcd 1953 5115) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1555_155538


namespace NUMINAMATH_CALUDE_part_one_part_two_l1555_155590

-- Define the sets A and B
def A : Set ℝ := {x | (x + 2) * (x - 3) < 0}
def B (a : ℝ) : Set ℝ := {x | x - a > 0}

-- Theorem for part (1)
theorem part_one : 
  (Set.univ \ A) ∪ (B 1) = {x : ℝ | x ≤ -2 ∨ x > 1} := by sorry

-- Theorem for part (2)
theorem part_two : 
  ∀ a : ℝ, A ⊆ B a ↔ a ≤ -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1555_155590


namespace NUMINAMATH_CALUDE_yellow_probability_is_correct_l1555_155509

/-- Represents the contents of a bag of marbles -/
structure BagContents where
  white : ℕ := 0
  black : ℕ := 0
  yellow : ℕ := 0
  blue : ℕ := 0
  green : ℕ := 0

/-- Calculates the total number of marbles in a bag -/
def BagContents.total (bag : BagContents) : ℕ :=
  bag.white + bag.black + bag.yellow + bag.blue + bag.green

/-- The contents of Bag A -/
def bagA : BagContents := { white := 4, black := 5 }

/-- The contents of Bag B -/
def bagB : BagContents := { yellow := 5, blue := 3, green := 2 }

/-- The contents of Bag C -/
def bagC : BagContents := { yellow := 2, blue := 5 }

/-- The probability of drawing a yellow marble as the second marble -/
def yellowProbability : ℚ :=
  (bagA.white * bagB.yellow / (bagA.total * bagB.total) : ℚ) +
  (bagA.black * bagC.yellow / (bagA.total * bagC.total) : ℚ)

theorem yellow_probability_is_correct :
  yellowProbability = 8 / 21 := by
  sorry

end NUMINAMATH_CALUDE_yellow_probability_is_correct_l1555_155509


namespace NUMINAMATH_CALUDE_cistern_fill_time_l1555_155543

/-- Represents the time taken to fill a cistern given the rates of three pipes -/
def fill_time (rate1 rate2 rate3 : ℚ) : ℚ :=
  1 / (rate1 + rate2 + rate3)

/-- Theorem stating that given pipes with specific rates will fill the cistern in 7.5 hours -/
theorem cistern_fill_time :
  fill_time (1/10) (1/12) (-1/20) = 15/2 := by
  sorry

end NUMINAMATH_CALUDE_cistern_fill_time_l1555_155543
