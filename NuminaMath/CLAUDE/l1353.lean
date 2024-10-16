import Mathlib

namespace NUMINAMATH_CALUDE_club_members_count_l1353_135330

/-- The number of female members in the club -/
def female_members : ℕ := 12

/-- The number of male members in the club -/
def male_members : ℕ := female_members / 2

/-- The total number of members in the club -/
def total_members : ℕ := female_members + male_members

/-- Proof that the total number of members in the club is 18 -/
theorem club_members_count : total_members = 18 := by
  sorry

end NUMINAMATH_CALUDE_club_members_count_l1353_135330


namespace NUMINAMATH_CALUDE_min_value_theorem_l1353_135332

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 1) :
  3 / a + 2 / b ≥ 25 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 1 ∧ 3 / a₀ + 2 / b₀ = 25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1353_135332


namespace NUMINAMATH_CALUDE_square_side_length_average_l1353_135308

theorem square_side_length_average (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 16) (h₂ : a₂ = 49) (h₃ : a₃ = 169) :
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l1353_135308


namespace NUMINAMATH_CALUDE_ratio_characterization_l1353_135304

/-- Given points A, B, and M on a line, where M ≠ B, this theorem characterizes the position of M based on the ratio AM:BM -/
theorem ratio_characterization (A B M M1 M2 : ℝ) : 
  (M ≠ B) →
  (A < B) →
  (A < M1) → (M1 < B) →
  (A < M2) → (B < M2) →
  (A - M1 = 2 * (M1 - B)) →
  (M2 - A = 2 * (B - A)) →
  (((M - A) > 2 * (B - M) ↔ (M1 < M ∧ M < M2 ∧ M ≠ B)) ∧
   ((M - A) < 2 * (B - M) ↔ (M < M1 ∨ M2 < M))) :=
by sorry

end NUMINAMATH_CALUDE_ratio_characterization_l1353_135304


namespace NUMINAMATH_CALUDE_trajectory_is_ellipse_l1353_135337

/-- The trajectory of point P(x,y) moving such that its distance from the line x=-4 
    is twice its distance from the fixed point F(-1,0) -/
def trajectory (x y : ℝ) : Prop :=
  let F := ((-1 : ℝ), (0 : ℝ))
  let d := |x + 4|
  let PF := Real.sqrt ((x + 1)^2 + y^2)
  d = 2 * PF ∧ x^2 / 4 + y^2 / 3 = 1

/-- The theorem stating that the trajectory satisfies the ellipse equation -/
theorem trajectory_is_ellipse (x y : ℝ) : 
  trajectory x y ↔ x^2 / 4 + y^2 / 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_trajectory_is_ellipse_l1353_135337


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_2551_l1353_135315

theorem smallest_prime_factor_of_2551 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ 2551 ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ 2551 → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_2551_l1353_135315


namespace NUMINAMATH_CALUDE_symmetry_line_theorem_l1353_135373

/-- Circle represented by its equation -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Line represented by its equation -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Define Circle O -/
def circle_O : Circle :=
  { equation := λ x y => x^2 + y^2 = 4 }

/-- Define Circle C -/
def circle_C : Circle :=
  { equation := λ x y => x^2 + y^2 + 4*x - 4*y + 4 = 0 }

/-- Define the line of symmetry -/
def line_of_symmetry : Line :=
  { equation := λ x y => x - y + 2 = 0 }

/-- Function to check if a line is the line of symmetry between two circles -/
def is_line_of_symmetry (l : Line) (c1 c2 : Circle) : Prop :=
  sorry -- Definition of symmetry between circles with respect to a line

/-- Theorem stating that the given line is the line of symmetry between Circle O and Circle C -/
theorem symmetry_line_theorem :
  is_line_of_symmetry line_of_symmetry circle_O circle_C := by
  sorry

end NUMINAMATH_CALUDE_symmetry_line_theorem_l1353_135373


namespace NUMINAMATH_CALUDE_product_of_real_parts_complex_equation_l1353_135327

theorem product_of_real_parts_complex_equation : ∃ (z₁ z₂ : ℂ),
  (z₁^2 - 2*z₁ = Complex.I) ∧
  (z₂^2 - 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (z₁.re * z₂.re = (1 - Real.sqrt 2) / 2) := by
sorry

end NUMINAMATH_CALUDE_product_of_real_parts_complex_equation_l1353_135327


namespace NUMINAMATH_CALUDE_lcm_from_product_and_gcd_l1353_135328

theorem lcm_from_product_and_gcd (a b : ℕ+) 
  (h_product : a * b = 17820)
  (h_gcd : Nat.gcd a b = 12) :
  Nat.lcm a b = 1485 := by
  sorry

end NUMINAMATH_CALUDE_lcm_from_product_and_gcd_l1353_135328


namespace NUMINAMATH_CALUDE_cubic_root_fraction_equality_l1353_135397

theorem cubic_root_fraction_equality (x : ℝ) (h : x^3 + x - 1 = 0) :
  (x^4 - 2*x^3 + x^2 - 3*x + 5) / (x^5 - x^2 - x + 2) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_fraction_equality_l1353_135397


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l1353_135375

/-- Represents a 2D point --/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a line in parametric form --/
structure ParametricLine where
  origin : Point
  direction : Point

/-- The first line --/
def line1 : ParametricLine :=
  { origin := { x := 2, y := 3 },
    direction := { x := 3, y := 4 } }

/-- The second line --/
def line2 : ParametricLine :=
  { origin := { x := 6, y := 1 },
    direction := { x := 5, y := -1 } }

/-- The proposed intersection point --/
def intersectionPoint : Point :=
  { x := 20/23, y := 27/23 }

/-- Function to get a point on a parametric line given a parameter --/
def pointOnLine (line : ParametricLine) (t : ℚ) : Point :=
  { x := line.origin.x + t * line.direction.x,
    y := line.origin.y + t * line.direction.y }

/-- Theorem stating that the given point is the unique intersection of the two lines --/
theorem intersection_point_is_unique :
  ∃! t u, pointOnLine line1 t = intersectionPoint ∧ pointOnLine line2 u = intersectionPoint :=
sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l1353_135375


namespace NUMINAMATH_CALUDE_tomato_suggestion_count_tomato_suggestion_count_proof_l1353_135338

theorem tomato_suggestion_count : ℕ → ℕ → ℕ → Prop :=
  fun bacon_count difference tomato_count =>
    (bacon_count = tomato_count + difference) →
    (bacon_count = 337 ∧ difference = 314) →
    tomato_count = 23

theorem tomato_suggestion_count_proof :
  ∃ (tomato_count : ℕ), tomato_suggestion_count 337 314 tomato_count :=
sorry

end NUMINAMATH_CALUDE_tomato_suggestion_count_tomato_suggestion_count_proof_l1353_135338


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_one_f_at_specific_angle_l1353_135325

-- Problem 1
theorem trigonometric_expression_equals_one :
  Real.sin (-120 * Real.pi / 180) * Real.cos (210 * Real.pi / 180) +
  Real.cos (-300 * Real.pi / 180) * Real.sin (-330 * Real.pi / 180) = 1 := by
sorry

-- Problem 2
noncomputable def f (α : Real) : Real :=
  (2 * Real.sin (Real.pi + α) * Real.cos (Real.pi - α) - Real.cos (Real.pi + α)) /
  (1 + Real.sin α ^ 2 + Real.cos ((3 * Real.pi) / 2 + α) - Real.sin ((Real.pi / 2 + α) ^ 2))

theorem f_at_specific_angle (h : 1 + 2 * Real.sin (-23 * Real.pi / 6) ≠ 0) :
  f (-23 * Real.pi / 6) = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_one_f_at_specific_angle_l1353_135325


namespace NUMINAMATH_CALUDE_power_product_simplification_l1353_135318

theorem power_product_simplification (a : ℝ) : ((-2 * a)^2) * (a^4) = 4 * (a^6) := by
  sorry

end NUMINAMATH_CALUDE_power_product_simplification_l1353_135318


namespace NUMINAMATH_CALUDE_sandys_shorts_cost_l1353_135312

theorem sandys_shorts_cost (total_spent shirt_cost jacket_cost : ℚ)
  (h1 : shirt_cost = 12.14)
  (h2 : jacket_cost = 7.43)
  (h3 : total_spent = 33.56) :
  total_spent - shirt_cost - jacket_cost = 13.99 := by
sorry

end NUMINAMATH_CALUDE_sandys_shorts_cost_l1353_135312


namespace NUMINAMATH_CALUDE_product_remainder_mod_three_l1353_135360

theorem product_remainder_mod_three (a b : ℕ) : 
  a % 3 = 1 → b % 3 = 2 → (a * b) % 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_three_l1353_135360


namespace NUMINAMATH_CALUDE_expression_evaluation_l1353_135329

theorem expression_evaluation :
  let x : ℝ := 2 * Real.sqrt 3
  (x - Real.sqrt 2) * (x + Real.sqrt 2) + x * (x - 1) = 22 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1353_135329


namespace NUMINAMATH_CALUDE_problem_stack_surface_area_l1353_135376

/-- Represents a solid formed by stacking unit cubes -/
structure CubeStack where
  base_length : ℕ
  base_width : ℕ
  base_height : ℕ
  top_cube : Bool

/-- Calculates the surface area of a CubeStack -/
def surface_area (stack : CubeStack) : ℕ :=
  sorry

/-- The specific cube stack described in the problem -/
def problem_stack : CubeStack :=
  { base_length := 3
  , base_width := 3
  , base_height := 1
  , top_cube := true }

/-- Theorem stating that the surface area of the problem_stack is 34 square units -/
theorem problem_stack_surface_area :
  surface_area problem_stack = 34 :=
sorry

end NUMINAMATH_CALUDE_problem_stack_surface_area_l1353_135376


namespace NUMINAMATH_CALUDE_text_messages_difference_l1353_135377

theorem text_messages_difference (last_week : ℕ) (total : ℕ) : last_week = 111 → total = 283 → total - last_week - last_week = 61 := by
  sorry

end NUMINAMATH_CALUDE_text_messages_difference_l1353_135377


namespace NUMINAMATH_CALUDE_inequality_proof_l1353_135388

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  1 / (x^3 * y) + 1 / (y^3 * z) + 1 / (z^3 * x) ≥ x * y + y * z + z * x := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1353_135388


namespace NUMINAMATH_CALUDE_expression_evaluation_l1353_135362

theorem expression_evaluation : 6^4 - 4 * 6^3 + 6^2 - 2 * 6 + 1 = 457 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1353_135362


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_product_l1353_135344

/-- Given an ellipse and a hyperbola with specified foci, prove that the product of their semi-axes lengths is √868.5 -/
theorem ellipse_hyperbola_product (a b : ℝ) : 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → (x = 0 ∧ y = 5) ∨ (x = 0 ∧ y = -5)) →
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x = 8 ∧ y = 0) ∨ (x = -8 ∧ y = 0)) →
  |a * b| = Real.sqrt 868.5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_product_l1353_135344


namespace NUMINAMATH_CALUDE_weekend_rain_probability_l1353_135301

theorem weekend_rain_probability (p_saturday p_sunday : ℝ) 
  (h1 : p_saturday = 0.6)
  (h2 : p_sunday = 0.7)
  (h3 : 0 ≤ p_saturday ∧ p_saturday ≤ 1)
  (h4 : 0 ≤ p_sunday ∧ p_sunday ≤ 1) :
  1 - (1 - p_saturday) * (1 - p_sunday) = 0.88 := by
sorry

end NUMINAMATH_CALUDE_weekend_rain_probability_l1353_135301


namespace NUMINAMATH_CALUDE_technician_average_salary_l1353_135302

/-- Calculates the average salary of technicians in a workshop --/
theorem technician_average_salary
  (total_workers : ℕ)
  (total_average : ℝ)
  (non_tech_average : ℝ)
  (num_technicians : ℕ)
  (h1 : total_workers = 14)
  (h2 : total_average = 10000)
  (h3 : non_tech_average = 8000)
  (h4 : num_technicians = 7) :
  (total_workers * total_average - (total_workers - num_technicians) * non_tech_average) / num_technicians = 12000 := by
sorry

end NUMINAMATH_CALUDE_technician_average_salary_l1353_135302


namespace NUMINAMATH_CALUDE_number_puzzle_l1353_135389

theorem number_puzzle (x y : ℝ) : x = 33 → (x / 4) + y = 15 → y = 6.75 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1353_135389


namespace NUMINAMATH_CALUDE_triathlon_bike_speed_l1353_135391

/-- Triathlon problem -/
theorem triathlon_bike_speed 
  (total_time : ℝ) 
  (swim_speed swim_distance : ℝ) 
  (run_speed run_distance : ℝ) 
  (bike_distance : ℝ) :
  total_time = 2.5 →
  swim_speed = 2 →
  swim_distance = 0.25 →
  run_speed = 5 →
  run_distance = 3 →
  bike_distance = 20 →
  ∃ bike_speed : ℝ, 
    (swim_distance / swim_speed + run_distance / run_speed + bike_distance / bike_speed = total_time) ∧
    (abs (bike_speed - 11.27) < 0.01) := by
  sorry

end NUMINAMATH_CALUDE_triathlon_bike_speed_l1353_135391


namespace NUMINAMATH_CALUDE_probability_at_least_six_heads_l1353_135303

/-- A sequence of 8 coin flips -/
def CoinFlipSequence := Fin 8 → Bool

/-- The total number of possible outcomes for 8 coin flips -/
def totalOutcomes : ℕ := 256

/-- Checks if a sequence has at least 6 consecutive heads -/
def hasAtLeastSixConsecutiveHeads (s : CoinFlipSequence) : Prop :=
  ∃ i, i + 5 < 8 ∧ (∀ j, i ≤ j ∧ j ≤ i + 5 → s j = true)

/-- The number of sequences with at least 6 consecutive heads -/
def favorableOutcomes : ℕ := 13

/-- The probability of getting at least 6 consecutive heads in 8 fair coin flips -/
def probabilityAtLeastSixHeads : ℚ := favorableOutcomes / totalOutcomes

theorem probability_at_least_six_heads :
  probabilityAtLeastSixHeads = 13 / 256 :=
sorry

end NUMINAMATH_CALUDE_probability_at_least_six_heads_l1353_135303


namespace NUMINAMATH_CALUDE_certain_white_ball_draw_l1353_135368

/-- A box containing only white balls -/
structure WhiteBallBox where
  total_balls : ℕ
  all_white : Bool

/-- Drawing balls from the box -/
def draw_balls (box : WhiteBallBox) (num_drawn : ℕ) : Prop :=
  num_drawn ≤ box.total_balls

/-- The event of drawing white balls -/
def white_ball_event (box : WhiteBallBox) (num_drawn : ℕ) : Prop :=
  box.all_white ∧ draw_balls box num_drawn

/-- Theorem: Drawing 2 white balls from a box of 5 white balls is certain -/
theorem certain_white_ball_draw :
  ∀ (box : WhiteBallBox),
    box.total_balls = 5 →
    box.all_white = true →
    white_ball_event box 2 := by
  sorry

end NUMINAMATH_CALUDE_certain_white_ball_draw_l1353_135368


namespace NUMINAMATH_CALUDE_largest_whole_number_less_than_120_over_8_l1353_135323

theorem largest_whole_number_less_than_120_over_8 :
  ∃ (x : ℕ), x = 14 ∧ (∀ y : ℕ, 8 * y < 120 → y ≤ x) :=
by sorry

end NUMINAMATH_CALUDE_largest_whole_number_less_than_120_over_8_l1353_135323


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1353_135353

def P : Set ℝ := {x | x ≤ 1}
def Q : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}

theorem intersection_of_P_and_Q : P ∩ Q = {x | -1 ≤ x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l1353_135353


namespace NUMINAMATH_CALUDE_desk_final_price_l1353_135390

/-- Calculates the final price of an auctioned item given initial price, price increase per bid, and number of bids -/
def final_price (initial_price : ℕ) (price_increase : ℕ) (num_bids : ℕ) : ℕ :=
  initial_price + price_increase * num_bids

/-- Theorem stating the final price of the desk after the bidding war -/
theorem desk_final_price :
  final_price 15 5 10 = 65 := by
  sorry

end NUMINAMATH_CALUDE_desk_final_price_l1353_135390


namespace NUMINAMATH_CALUDE_peg_arrangement_count_l1353_135355

/-- The number of ways to distribute colored pegs on a square board. -/
def peg_arrangements : ℕ :=
  let red_pegs := 6
  let green_pegs := 5
  let blue_pegs := 4
  let yellow_pegs := 3
  let orange_pegs := 2
  let board_size := 6
  Nat.factorial red_pegs * Nat.factorial green_pegs * Nat.factorial blue_pegs *
  Nat.factorial yellow_pegs * Nat.factorial orange_pegs

/-- Theorem stating the number of valid peg arrangements. -/
theorem peg_arrangement_count :
  peg_arrangements = 12441600 := by
  sorry

end NUMINAMATH_CALUDE_peg_arrangement_count_l1353_135355


namespace NUMINAMATH_CALUDE_problem_statement_l1353_135382

theorem problem_statement : (-5/12)^2023 * (12/5)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l1353_135382


namespace NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l1353_135361

theorem max_value_theorem (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  3 * x + 4 * y + 6 * z ≤ Real.sqrt 53 :=
by sorry

theorem max_value_achieved (x y z : ℝ) (h : 9 * x^2 + 4 * y^2 + 25 * z^2 = 1) :
  ∃ (x' y' z' : ℝ), 9 * x'^2 + 4 * y'^2 + 25 * z'^2 = 1 ∧ 3 * x' + 4 * y' + 6 * z' = Real.sqrt 53 :=
by sorry

end NUMINAMATH_CALUDE_max_value_theorem_max_value_achieved_l1353_135361


namespace NUMINAMATH_CALUDE_no_sum_of_cubes_l1353_135393

theorem no_sum_of_cubes (n : ℕ) : ¬∃ (x y : ℕ), 10^(3*n + 1) = x^3 + y^3 := by
  sorry

end NUMINAMATH_CALUDE_no_sum_of_cubes_l1353_135393


namespace NUMINAMATH_CALUDE_unique_x_value_l1353_135314

/-- Binary operation ⋆ on pairs of integers -/
def star : (ℤ × ℤ) → (ℤ × ℤ) → (ℤ × ℤ) := 
  fun (a, b) (c, d) ↦ (a + c, b - d)

/-- Theorem stating the unique value of x satisfying the equation -/
theorem unique_x_value : 
  ∃! x : ℤ, star (x, 4) (2, 1) = star (5, 2) (1, -3) := by sorry

end NUMINAMATH_CALUDE_unique_x_value_l1353_135314


namespace NUMINAMATH_CALUDE_sector_central_angle_l1353_135366

/-- Given a circular sector with area 1 and perimeter 4, its central angle is 2 radians -/
theorem sector_central_angle (S : ℝ) (P : ℝ) (α : ℝ) :
  S = 1 →  -- area of the sector
  P = 4 →  -- perimeter of the sector
  S = (1/2) * α * (P - α)^2 / α^2 →  -- area formula for a sector
  α = 2 := by sorry

end NUMINAMATH_CALUDE_sector_central_angle_l1353_135366


namespace NUMINAMATH_CALUDE_one_plus_sqrt3i_in_M_l1353_135335

/-- The set M of complex numbers with magnitude 2 -/
def M : Set ℂ := {z : ℂ | Complex.abs z = 2}

/-- Proof that 1 + √3i belongs to M -/
theorem one_plus_sqrt3i_in_M : (1 : ℂ) + Complex.I * Real.sqrt 3 ∈ M := by
  sorry

end NUMINAMATH_CALUDE_one_plus_sqrt3i_in_M_l1353_135335


namespace NUMINAMATH_CALUDE_add_fractions_l1353_135364

theorem add_fractions : (1 : ℚ) / 4 + (3 : ℚ) / 8 = (5 : ℚ) / 8 := by
  sorry

end NUMINAMATH_CALUDE_add_fractions_l1353_135364


namespace NUMINAMATH_CALUDE_ace_king_probability_l1353_135348

/-- The number of cards in a standard deck -/
def deck_size : ℕ := 52

/-- The number of Aces in a standard deck -/
def num_aces : ℕ := 4

/-- The number of Kings in a standard deck -/
def num_kings : ℕ := 4

/-- The probability of drawing an Ace followed by a King from a standard deck -/
theorem ace_king_probability : 
  (num_aces : ℚ) / deck_size * num_kings / (deck_size - 1) = 4 / 663 := by
sorry

end NUMINAMATH_CALUDE_ace_king_probability_l1353_135348


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1353_135320

-- Define the function f
def f (a x : ℝ) : ℝ := |x + a| + |x - 1|

-- Theorem 1
theorem solution_set_when_a_is_2 :
  {x : ℝ | f 2 x ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} := by sorry

-- Theorem 2
theorem range_of_a :
  ∀ a : ℝ, (∃ x₀ : ℝ, f a x₀ ≤ |2*a - 1|) → 0 ≤ a ∧ a ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_2_range_of_a_l1353_135320


namespace NUMINAMATH_CALUDE_correct_growth_rate_equation_l1353_135343

/-- Represents the monthly average growth rate of sales volume for a product -/
def monthly_growth_rate (march_sales may_sales : ℝ) (x : ℝ) : Prop :=
  x > 0 ∧ 10 * (1 + x)^2 = 11.5 ∧ may_sales = march_sales * (1 + x)^2

/-- Theorem stating that the given equation correctly represents the monthly average growth rate -/
theorem correct_growth_rate_equation :
  ∃ x : ℝ, monthly_growth_rate 100000 115000 x :=
sorry

end NUMINAMATH_CALUDE_correct_growth_rate_equation_l1353_135343


namespace NUMINAMATH_CALUDE_log_base_value_l1353_135384

theorem log_base_value (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x > 0, f x = Real.log x / Real.log a) (h2 : f 4 = 2) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_log_base_value_l1353_135384


namespace NUMINAMATH_CALUDE_least_even_p_for_square_three_is_solution_least_even_p_is_three_l1353_135313

theorem least_even_p_for_square (p : ℕ) : 
  (∃ n : ℕ, 300 * p = n^2) ∧ Even p → p ≥ 3 :=
by sorry

theorem three_is_solution : 
  (∃ n : ℕ, 300 * 3 = n^2) ∧ Even 3 :=
by sorry

theorem least_even_p_is_three : 
  (∃ p : ℕ, (∃ n : ℕ, 300 * p = n^2) ∧ Even p ∧ 
  (∀ q : ℕ, (∃ m : ℕ, 300 * q = m^2) ∧ Even q → p ≤ q)) ∧
  (∀ p : ℕ, (∃ n : ℕ, 300 * p = n^2) ∧ Even p ∧ 
  (∀ q : ℕ, (∃ m : ℕ, 300 * q = m^2) ∧ Even q → p ≤ q) → p = 3) :=
by sorry

end NUMINAMATH_CALUDE_least_even_p_for_square_three_is_solution_least_even_p_is_three_l1353_135313


namespace NUMINAMATH_CALUDE_one_face_colored_cubes_125_l1353_135379

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  edge_length : ℕ
  num_colors : ℕ

/-- Calculates the number of small cubes with only one face colored -/
def one_face_colored_cubes (c : CutCube) : ℕ :=
  (c.edge_length - 2)^2 * c.num_colors

/-- Theorem: A cube cut into 125 smaller cubes with 6 different colored faces has 54 small cubes with only one face colored -/
theorem one_face_colored_cubes_125 :
  ∀ c : CutCube, c.edge_length = 5 → c.num_colors = 6 → one_face_colored_cubes c = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_one_face_colored_cubes_125_l1353_135379


namespace NUMINAMATH_CALUDE_nancy_tuition_ratio_l1353_135341

/-- Calculates the ratio of student loan to scholarship for Nancy's university tuition --/
theorem nancy_tuition_ratio :
  let tuition : ℚ := 22000
  let parents_contribution : ℚ := tuition / 2
  let scholarship : ℚ := 3000
  let work_hours : ℚ := 200
  let hourly_rate : ℚ := 10
  let work_earnings : ℚ := work_hours * hourly_rate
  let total_available : ℚ := parents_contribution + scholarship + work_earnings
  let loan_amount : ℚ := tuition - total_available
  loan_amount / scholarship = 2 := by sorry

end NUMINAMATH_CALUDE_nancy_tuition_ratio_l1353_135341


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_77_over_6_l1353_135305

/-- Represents a square with a given side length -/
structure Square where
  sideLength : ℝ

/-- Represents the arrangement of three squares -/
structure SquareArrangement where
  square1 : Square
  square2 : Square
  square3 : Square

/-- Calculates the area of the quadrilateral formed in the square arrangement -/
def quadrilateralArea (arrangement : SquareArrangement) : ℝ :=
  sorry

theorem quadrilateral_area_is_77_over_6 :
  let arrangement := SquareArrangement.mk
    (Square.mk 3)
    (Square.mk 5)
    (Square.mk 7)
  quadrilateralArea arrangement = 77 / 6 := by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_77_over_6_l1353_135305


namespace NUMINAMATH_CALUDE_proposition_relationship_l1353_135365

theorem proposition_relationship (x y : ℝ) :
  (∀ x y, x + y ≠ 8 → (x ≠ 2 ∨ y ≠ 6)) ∧
  (∃ x y, (x ≠ 2 ∨ y ≠ 6) ∧ x + y = 8) :=
by sorry

end NUMINAMATH_CALUDE_proposition_relationship_l1353_135365


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l1353_135396

def a : ℝ × ℝ := (3, 1)
def b : ℝ × ℝ := (1, 0)

def c (k : ℝ) : ℝ × ℝ := (a.1 + k * b.1, a.2 + k * b.2)

theorem perpendicular_vectors (k : ℝ) :
  (a.1 * (a.1 + k * b.1) + a.2 * (a.2 + k * b.2) = 0) → k = -10/3 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l1353_135396


namespace NUMINAMATH_CALUDE_student_count_incorrect_l1353_135324

theorem student_count_incorrect : ¬ ∃ k : ℕ, 18 + 17 * k = 2012 := by
  sorry

end NUMINAMATH_CALUDE_student_count_incorrect_l1353_135324


namespace NUMINAMATH_CALUDE_no_valid_license_plate_divisible_by_8_l1353_135347

/-- Represents a 4-digit number of the form aaab -/
structure LicensePlate where
  a : Nat
  b : Nat
  h1 : a < 10
  h2 : b < 10

/-- Checks if a number is prime -/
def isPrime (n : Nat) : Prop := n > 1 ∧ ∀ d : Nat, d > 1 → d < n → ¬(n % d = 0)

/-- The theorem to be proved -/
theorem no_valid_license_plate_divisible_by_8 :
  ¬∃ (plate : LicensePlate),
    (∀ (child_age : Nat), child_age ≥ 1 → child_age ≤ 10 → (1000 * plate.a + 100 * plate.a + 10 * plate.a + plate.b) % child_age = 0) ∧
    isPrime (10 * plate.a + plate.b) ∧
    (1000 * plate.a + 100 * plate.a + 10 * plate.a + plate.b) % 8 = 0 :=
by sorry


end NUMINAMATH_CALUDE_no_valid_license_plate_divisible_by_8_l1353_135347


namespace NUMINAMATH_CALUDE_bug_traversal_12_25_l1353_135398

/-- The number of tiles a bug traverses when walking diagonally across a rectangular floor -/
def bugTraversal (width length : ℕ) : ℕ :=
  width + length - Nat.gcd width length

theorem bug_traversal_12_25 :
  bugTraversal 12 25 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bug_traversal_12_25_l1353_135398


namespace NUMINAMATH_CALUDE_sandwich_bread_slices_l1353_135399

theorem sandwich_bread_slices 
  (total_sandwiches : ℕ) 
  (bread_packs : ℕ) 
  (slices_per_pack : ℕ) 
  (h1 : total_sandwiches = 8)
  (h2 : bread_packs = 4)
  (h3 : slices_per_pack = 4) :
  (bread_packs * slices_per_pack) / total_sandwiches = 2 := by
  sorry

end NUMINAMATH_CALUDE_sandwich_bread_slices_l1353_135399


namespace NUMINAMATH_CALUDE_two_digit_sums_of_six_powers_of_two_l1353_135359

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def is_sum_of_six_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    a < b ∧ b < c ∧ c < d ∧ d < e ∧ e < f ∧
    n = 2^0 + 2^a + 2^b + 2^c + 2^d + 2^e + 2^f

theorem two_digit_sums_of_six_powers_of_two :
  ∃! (s : Finset ℕ),
    (∀ n ∈ s, is_two_digit n ∧ is_sum_of_six_powers_of_two n) ∧
    s.card = 2 :=
sorry

end NUMINAMATH_CALUDE_two_digit_sums_of_six_powers_of_two_l1353_135359


namespace NUMINAMATH_CALUDE_product_zero_l1353_135395

theorem product_zero (a x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉ x₁₀ x₁₁ x₁₂ x₁₃ : ℤ) 
  (h1 : a = (1 + x₁) * (1 + x₂) * (1 + x₃) * (1 + x₄) * (1 + x₅) * (1 + x₆) * (1 + x₇) * 
           (1 + x₈) * (1 + x₉) * (1 + x₁₀) * (1 + x₁₁) * (1 + x₁₂) * (1 + x₁₃))
  (h2 : a = (1 - x₁) * (1 - x₂) * (1 - x₃) * (1 - x₄) * (1 - x₅) * (1 - x₆) * (1 - x₇) * 
           (1 - x₈) * (1 - x₉) * (1 - x₁₀) * (1 - x₁₁) * (1 - x₁₂) * (1 - x₁₃)) :
  a * x₁ * x₂ * x₃ * x₄ * x₅ * x₆ * x₇ * x₈ * x₉ * x₁₀ * x₁₁ * x₁₂ * x₁₃ = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_zero_l1353_135395


namespace NUMINAMATH_CALUDE_solution_pairs_count_l1353_135385

theorem solution_pairs_count : 
  ∃! n : ℕ, n = (Finset.filter (fun p : ℕ × ℕ => 
    4 * p.1 + 7 * p.2 = 600 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 601) (Finset.range 601))).card ∧ n = 22 :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_count_l1353_135385


namespace NUMINAMATH_CALUDE_no_three_digit_number_satisfies_conditions_l1353_135319

/-- Function to check if digits are different and in ascending order -/
def digits_ascending_different (n : ℕ) : Prop := sorry

/-- Theorem stating that no three-digit number satisfies the given conditions -/
theorem no_three_digit_number_satisfies_conditions :
  ¬ ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    digits_ascending_different n ∧
    digits_ascending_different (n^2) ∧
    digits_ascending_different (n^3) := by
  sorry

end NUMINAMATH_CALUDE_no_three_digit_number_satisfies_conditions_l1353_135319


namespace NUMINAMATH_CALUDE_james_meditation_sessions_l1353_135381

/-- Calculates the number of meditation sessions per day given the session duration and weekly meditation time. -/
def meditation_sessions_per_day (session_duration : ℕ) (weekly_meditation_hours : ℕ) : ℕ :=
  let minutes_per_week : ℕ := weekly_meditation_hours * 60
  let minutes_per_day : ℕ := minutes_per_week / 7
  minutes_per_day / session_duration

/-- Theorem stating that given the specified conditions, the number of meditation sessions per day is 2. -/
theorem james_meditation_sessions :
  meditation_sessions_per_day 30 7 = 2 :=
by sorry

end NUMINAMATH_CALUDE_james_meditation_sessions_l1353_135381


namespace NUMINAMATH_CALUDE_log_square_problem_l1353_135342

theorem log_square_problem (x y : ℝ) 
  (hx_pos : x > 0) (hy_pos : y > 0)
  (hx_neq_one : x ≠ 1) (hy_neq_one : y ≠ 1)
  (h_log : Real.log x / Real.log 2 = Real.log 8 / Real.log y)
  (h_prod : x * y = 128) :
  (Real.log (x / y) / Real.log 2) ^ 2 = 37 := by
sorry

end NUMINAMATH_CALUDE_log_square_problem_l1353_135342


namespace NUMINAMATH_CALUDE_jebbs_take_home_pay_l1353_135363

/-- Calculates the take-home pay given the total pay and tax rate -/
def takeHomePay (totalPay : ℝ) (taxRate : ℝ) : ℝ :=
  totalPay * (1 - taxRate)

/-- Theorem stating that given a total pay of 650 and a tax rate of 10%, the take-home pay is 585 -/
theorem jebbs_take_home_pay :
  takeHomePay 650 0.1 = 585 := by
  sorry

end NUMINAMATH_CALUDE_jebbs_take_home_pay_l1353_135363


namespace NUMINAMATH_CALUDE_system_solutions_l1353_135383

/-- The system of equations -/
def system (x y z t : ℝ) : Prop :=
  x * y * z = x + y + z ∧
  y * z * t = y + z + t ∧
  z * t * x = z + t + x ∧
  t * x * y = t + x + y

/-- The set of solutions to the system -/
def solutions : Set (ℝ × ℝ × ℝ × ℝ) :=
  {(0, 0, 0, 0), (Real.sqrt 3, Real.sqrt 3, Real.sqrt 3, Real.sqrt 3), 
   (-Real.sqrt 3, -Real.sqrt 3, -Real.sqrt 3, -Real.sqrt 3)}

/-- Theorem stating that the solutions are correct and complete -/
theorem system_solutions :
  ∀ x y z t : ℝ, system x y z t ↔ (x, y, z, t) ∈ solutions :=
sorry


end NUMINAMATH_CALUDE_system_solutions_l1353_135383


namespace NUMINAMATH_CALUDE_smallest_g_is_correct_l1353_135372

/-- The smallest positive integer g such that 3150 * g is a perfect square -/
def smallest_g : ℕ := 14

/-- 3150 * g is a perfect square -/
def is_perfect_square (g : ℕ) : Prop :=
  ∃ n : ℕ, 3150 * g = n^2

theorem smallest_g_is_correct :
  (is_perfect_square smallest_g) ∧
  (∀ g : ℕ, 0 < g ∧ g < smallest_g → ¬(is_perfect_square g)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_g_is_correct_l1353_135372


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1353_135349

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (q > 0) →
  (∀ n : ℕ, a (n + 1) = q * a n) →
  (a 3 * a 7 = 4 * (a 4)^2) →
  q = 2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1353_135349


namespace NUMINAMATH_CALUDE_ending_number_of_range_l1353_135317

theorem ending_number_of_range (n : ℕ) (h1 : n = 10) (h2 : ∀ k ∈ Finset.range n, 15 + 5 * k ∣ 5) :
  15 + 5 * (n - 1) = 60 :=
sorry

end NUMINAMATH_CALUDE_ending_number_of_range_l1353_135317


namespace NUMINAMATH_CALUDE_theater_attendance_l1353_135321

/-- Calculates the total number of attendees given ticket prices and revenue --/
def total_attendees (adult_price child_price : ℕ) (total_revenue : ℕ) (num_children : ℕ) : ℕ :=
  let num_adults := (total_revenue - child_price * num_children) / adult_price
  num_adults + num_children

/-- Theorem stating that under the given conditions, the total number of attendees is 280 --/
theorem theater_attendance : total_attendees 60 25 14000 80 = 280 := by
  sorry

end NUMINAMATH_CALUDE_theater_attendance_l1353_135321


namespace NUMINAMATH_CALUDE_T_sum_zero_l1353_135354

def T (n : ℕ) : ℤ :=
  if n % 2 = 0 then
    -n / 2
  else
    (n + 1) / 2

theorem T_sum_zero : T 18 + T 34 + T 51 = 0 := by
  sorry

end NUMINAMATH_CALUDE_T_sum_zero_l1353_135354


namespace NUMINAMATH_CALUDE_tom_toy_cost_proof_l1353_135378

def tom_toy_cost (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) : ℕ :=
  (initial_money - game_cost) / num_toys

theorem tom_toy_cost_proof (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) 
  (h1 : initial_money = 57)
  (h2 : game_cost = 49)
  (h3 : num_toys = 2)
  (h4 : initial_money > game_cost) :
  tom_toy_cost initial_money game_cost num_toys = 4 := by
  sorry

end NUMINAMATH_CALUDE_tom_toy_cost_proof_l1353_135378


namespace NUMINAMATH_CALUDE_smallest_superior_discount_l1353_135346

def successive_discount (single_discount : ℝ) (times : ℕ) : ℝ :=
  1 - (1 - single_discount) ^ times

theorem smallest_superior_discount : ∃ (n : ℕ), n = 37 ∧
  (∀ (m : ℕ), m < n →
    (m : ℝ) / 100 ≤ successive_discount (12 / 100) 3 ∨
    (m : ℝ) / 100 ≤ successive_discount (20 / 100) 2 ∨
    (m : ℝ) / 100 ≤ successive_discount (8 / 100) 4) ∧
  (37 : ℝ) / 100 > successive_discount (12 / 100) 3 ∧
  (37 : ℝ) / 100 > successive_discount (20 / 100) 2 ∧
  (37 : ℝ) / 100 > successive_discount (8 / 100) 4 :=
sorry

end NUMINAMATH_CALUDE_smallest_superior_discount_l1353_135346


namespace NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1353_135309

theorem greatest_of_three_consecutive_integers (x y z : ℤ) : 
  (y = x + 1) → (z = y + 1) → (x + y + z = 39) → 
  (max x (max y z) = 14) :=
by sorry

end NUMINAMATH_CALUDE_greatest_of_three_consecutive_integers_l1353_135309


namespace NUMINAMATH_CALUDE_product_of_polynomials_l1353_135310

theorem product_of_polynomials (g h : ℚ) : 
  (∀ d : ℚ, (5*d^2 - 2*d + g) * (2*d^2 + h*d - 3) = 10*d^4 - 19*d^3 + g*d^2 + d - 6) →
  g + h = -1/2 := by
sorry

end NUMINAMATH_CALUDE_product_of_polynomials_l1353_135310


namespace NUMINAMATH_CALUDE_not_three_equal_root_equation_three_equal_root_with_negative_one_root_three_equal_root_on_line_l1353_135300

/-- A quadratic equation is a three equal root equation if one root is 1/3 of the other --/
def is_three_equal_root_equation (a b c : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0 ∧ x₁ = (1/3) * x₂

/-- The first part of the problem --/
theorem not_three_equal_root_equation : ¬ is_three_equal_root_equation 1 (-8) 11 := by
  sorry

/-- The second part of the problem --/
theorem three_equal_root_with_negative_one_root (b c : ℤ) :
  is_three_equal_root_equation 1 b c ∧ (∃ x : ℝ, x^2 + b*x + c = 0 ∧ x = -1) → b = 4 ∧ c = 3 := by
  sorry

/-- The third part of the problem --/
theorem three_equal_root_on_line (m n : ℝ) :
  n = 2*m + 1 ∧ is_three_equal_root_equation m n 2 → m = 3/2 ∨ m = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_not_three_equal_root_equation_three_equal_root_with_negative_one_root_three_equal_root_on_line_l1353_135300


namespace NUMINAMATH_CALUDE_factorial_ratio_l1353_135311

theorem factorial_ratio : Nat.factorial 10 / Nat.factorial 9 = 10 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l1353_135311


namespace NUMINAMATH_CALUDE_line_satisfies_conditions_l1353_135380

-- Define the lines
def line1 (x y : ℝ) : Prop := x - 2*y + 4 = 0
def line2 (x y : ℝ) : Prop := x + y - 2 = 0
def line3 (x y : ℝ) : Prop := 3*x - 4*y + 7 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := 10*x + 13*y - 26 = 0

-- Theorem statement
theorem line_satisfies_conditions :
  -- The result line passes through the intersection of line1 and line2
  (∃ x y : ℝ, line1 x y ∧ line2 x y ∧ result_line x y) ∧
  -- The result line passes through the point (3, -2)
  (result_line 3 (-2)) ∧
  -- The result line is perpendicular to line3
  (∃ m1 m2 : ℝ, 
    (∀ x y : ℝ, line3 x y → y = m1 * x + (7 / 4)) ∧
    (∀ x y : ℝ, result_line x y → y = m2 * x + (26 / 10)) ∧
    m1 * m2 = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_satisfies_conditions_l1353_135380


namespace NUMINAMATH_CALUDE_loggers_required_is_eight_l1353_135387

/-- Represents the number of loggers required to cut down all trees in a forest under specific conditions. -/
def number_of_loggers (forest_length : ℕ) (forest_width : ℕ) (trees_per_square_mile : ℕ) 
  (trees_per_day : ℕ) (days_per_month : ℕ) (months_to_complete : ℕ) : ℕ :=
  (forest_length * forest_width * trees_per_square_mile) / 
  (trees_per_day * days_per_month * months_to_complete)

/-- Theorem stating that the number of loggers required under the given conditions is 8. -/
theorem loggers_required_is_eight :
  number_of_loggers 4 6 600 6 30 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_loggers_required_is_eight_l1353_135387


namespace NUMINAMATH_CALUDE_power_inequality_l1353_135326

theorem power_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  x^5 + y^5 - (x^4*y + x*y^4) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l1353_135326


namespace NUMINAMATH_CALUDE_certain_number_problem_l1353_135358

theorem certain_number_problem : ∃ x : ℚ, (5/6 : ℚ) * x = (5/16 : ℚ) * x + 100 ∧ x = 192 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_problem_l1353_135358


namespace NUMINAMATH_CALUDE_inverse_proportion_comparison_l1353_135345

theorem inverse_proportion_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h1 : k > 0) 
  (h2 : y₁ = k / (-2)) 
  (h3 : y₂ = k / (-1)) : 
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_comparison_l1353_135345


namespace NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1353_135374

/-- Calculate the simple interest rate given the principal and annual interest -/
theorem simple_interest_rate_calculation
  (principal : ℝ) 
  (annual_interest : ℝ) 
  (h1 : principal = 9000)
  (h2 : annual_interest = 810) :
  (annual_interest / principal) * 100 = 9 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_rate_calculation_l1353_135374


namespace NUMINAMATH_CALUDE_rectangular_prism_dimensions_l1353_135351

/-- Proves that a rectangular prism with given conditions has length 9 and width 3 -/
theorem rectangular_prism_dimensions :
  ∀ l w h : ℝ,
  l = 3 * w →
  h = 12 →
  Real.sqrt (l^2 + w^2 + h^2) = 15 →
  l = 9 ∧ w = 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_prism_dimensions_l1353_135351


namespace NUMINAMATH_CALUDE_right_trapezoid_bases_l1353_135339

/-- 
Given a right trapezoid with lateral sides c and d (c < d), if a line parallel to its bases 
splits it into two smaller trapezoids each with an inscribed circle, then the bases of the 
original trapezoid are (√(d+c) + √(d-c))² / 4 and (√(d+c) - √(d-c))² / 4.
-/
theorem right_trapezoid_bases (c d : ℝ) (h : c < d) : 
  ∃ (x y z : ℝ), 
    x > 0 ∧ y > 0 ∧ z > 0 ∧
    (∃ (r₁ r₂ : ℝ), r₁ > 0 ∧ r₂ > 0 ∧ 
      y^2 = x * z ∧
      c + d = x + 2*y + z) →
    x = ((Real.sqrt (d+c) - Real.sqrt (d-c))^2) / 4 ∧
    z = ((Real.sqrt (d+c) + Real.sqrt (d-c))^2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_trapezoid_bases_l1353_135339


namespace NUMINAMATH_CALUDE_probability_arithmetic_progression_l1353_135350

def dice_sides := 4

def is_arithmetic_progression (a b c : ℕ) : Prop :=
  (b = a + 1 ∧ c = b + 1) ∨ (a = b + 1 ∧ b = c + 1)

def favorable_outcomes : ℕ := 12

def total_outcomes : ℕ := dice_sides ^ 3

theorem probability_arithmetic_progression :
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_arithmetic_progression_l1353_135350


namespace NUMINAMATH_CALUDE_sqrt_12_minus_n_integer_l1353_135370

theorem sqrt_12_minus_n_integer (n : ℕ) : 
  (∃ k : ℕ, k^2 = 12 - n) → n ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_12_minus_n_integer_l1353_135370


namespace NUMINAMATH_CALUDE_absolute_value_simplification_l1353_135334

theorem absolute_value_simplification : |(-4^2 + 7)| = 9 := by sorry

end NUMINAMATH_CALUDE_absolute_value_simplification_l1353_135334


namespace NUMINAMATH_CALUDE_sum_of_2010_3_array_remainder_of_sum_l1353_135316

/-- Definition of the sum of a pq-array --/
def pq_array_sum (p q : ℕ) : ℚ :=
  (1 / (1 - 1 / (2 * p))) * (1 / (1 - 1 / q))

/-- Theorem stating the sum of the specific 1/2010,3-array --/
theorem sum_of_2010_3_array :
  pq_array_sum 2010 3 = 6030 / 4019 := by
  sorry

/-- Theorem for the remainder when numerator + denominator is divided by 2010 --/
theorem remainder_of_sum :
  (6030 + 4019) % 2010 = 1009 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_2010_3_array_remainder_of_sum_l1353_135316


namespace NUMINAMATH_CALUDE_polynomial_roots_l1353_135336

/-- The polynomial x^3 - 3x^2 - x + 3 -/
def p (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

/-- The roots of the polynomial -/
def roots : Set ℝ := {1, -1, 3}

theorem polynomial_roots : 
  (∀ x ∈ roots, p x = 0) ∧ 
  (∀ x : ℝ, p x = 0 → x ∈ roots) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1353_135336


namespace NUMINAMATH_CALUDE_three_std_dev_below_mean_l1353_135352

/-- Represents a normal distribution --/
structure NormalDistribution where
  mean : ℝ
  stdDev : ℝ

/-- Calculates the value that is n standard deviations below the mean --/
def valueBelow (nd : NormalDistribution) (n : ℝ) : ℝ :=
  nd.mean - n * nd.stdDev

/-- Theorem: For a normal distribution with standard deviation 2 and mean 51,
    the value 3 standard deviations below the mean is 45 --/
theorem three_std_dev_below_mean (nd : NormalDistribution) 
    (h1 : nd.stdDev = 2) 
    (h2 : nd.mean = 51) : 
    valueBelow nd 3 = 45 := by
  sorry

end NUMINAMATH_CALUDE_three_std_dev_below_mean_l1353_135352


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1353_135307

theorem polynomial_coefficient_sum (A B C D : ℝ) : 
  (∀ x : ℝ, (x + 3) * (3 * x^2 - 2 * x + 7) = A * x^3 + B * x^2 + C * x + D) →
  A + B + C + D = 32 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l1353_135307


namespace NUMINAMATH_CALUDE_doris_monthly_expenses_l1353_135306

/-- Calculates Doris's monthly expenses based on her work schedule and hourly rate. -/
def monthly_expenses (hourly_rate : ℕ) (weekday_hours : ℕ) (saturday_hours : ℕ) (weeks : ℕ) : ℕ :=
  let weekly_hours := weekday_hours * 5 + saturday_hours
  let weekly_earnings := hourly_rate * weekly_hours
  weekly_earnings * weeks

/-- Theorem stating that Doris's monthly expenses are $1200 given her work schedule and hourly rate. -/
theorem doris_monthly_expenses :
  monthly_expenses 20 3 5 3 = 1200 := by
  sorry

end NUMINAMATH_CALUDE_doris_monthly_expenses_l1353_135306


namespace NUMINAMATH_CALUDE_high_school_math_team_payment_l1353_135392

theorem high_school_math_team_payment (B : ℕ) : 
  B < 10 → (100 + 10 * B + 3) % 13 = 0 → B = 4 := by
sorry

end NUMINAMATH_CALUDE_high_school_math_team_payment_l1353_135392


namespace NUMINAMATH_CALUDE_total_plants_grown_l1353_135394

def eggplants_per_packet : ℕ := 14
def sunflowers_per_packet : ℕ := 10
def tomatoes_per_packet : ℕ := 16
def peas_per_packet : ℕ := 20

def eggplant_packets : ℕ := 4
def sunflower_packets : ℕ := 6
def tomato_packets : ℕ := 5
def pea_packets : ℕ := 7

def spring_growth_rate : ℚ := 7/10
def summer_growth_rate : ℚ := 4/5

theorem total_plants_grown (
  eggplants_per_packet sunflowers_per_packet tomatoes_per_packet peas_per_packet : ℕ)
  (eggplant_packets sunflower_packets tomato_packets pea_packets : ℕ)
  (spring_growth_rate summer_growth_rate : ℚ) :
  ⌊(eggplants_per_packet * eggplant_packets : ℚ) * spring_growth_rate⌋ +
  ⌊(peas_per_packet * pea_packets : ℚ) * spring_growth_rate⌋ +
  ⌊(sunflowers_per_packet * sunflower_packets : ℚ) * summer_growth_rate⌋ +
  ⌊(tomatoes_per_packet * tomato_packets : ℚ) * summer_growth_rate⌋ = 249 :=
by sorry

end NUMINAMATH_CALUDE_total_plants_grown_l1353_135394


namespace NUMINAMATH_CALUDE_polyhedron_sum_theorem_l1353_135331

/-- Represents a convex polyhedron --/
structure ConvexPolyhedron where
  V : ℕ  -- number of vertices
  E : ℕ  -- number of edges
  F : ℕ  -- number of faces
  euler_formula : V - E + F = 2

/-- Represents the face configuration of a polyhedron --/
structure FaceConfig where
  T : ℕ  -- number of triangles meeting at each vertex
  H : ℕ  -- number of hexagons meeting at each vertex

theorem polyhedron_sum_theorem (p : ConvexPolyhedron) (fc : FaceConfig)
  (h_faces : p.F = 50)
  (h_vertex_config : fc.T = 3 ∧ fc.H = 2) :
  100 * fc.H + 10 * fc.T + p.V = 230 := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_sum_theorem_l1353_135331


namespace NUMINAMATH_CALUDE_triangle_side_b_l1353_135369

theorem triangle_side_b (a c S : ℝ) (h1 : a = 5) (h2 : c = 2) (h3 : S = 4) :
  ∃ b : ℝ, (b = Real.sqrt 17 ∨ b = Real.sqrt 41) ∧
    S = (1/2) * a * c * Real.sqrt (1 - ((a^2 + c^2 - b^2) / (2*a*c))^2) :=
sorry

end NUMINAMATH_CALUDE_triangle_side_b_l1353_135369


namespace NUMINAMATH_CALUDE_triangular_field_yield_l1353_135322

/-- Proves that a triangular field with given dimensions and harvest yields 1 ton per hectare -/
theorem triangular_field_yield (base : ℝ) (height_factor : ℝ) (total_harvest : ℝ) :
  base = 200 →
  height_factor = 1.2 →
  total_harvest = 2.4 →
  let height := height_factor * base
  let area_sq_meters := (1 / 2) * base * height
  let area_hectares := area_sq_meters / 10000
  total_harvest / area_hectares = 1 := by sorry

end NUMINAMATH_CALUDE_triangular_field_yield_l1353_135322


namespace NUMINAMATH_CALUDE_quartic_sum_to_quadratic_sum_l1353_135357

theorem quartic_sum_to_quadratic_sum (x : ℝ) (h : 45 = x^4 + 1/x^4) : 
  x^2 + 1/x^2 = Real.sqrt 47 := by
  sorry

end NUMINAMATH_CALUDE_quartic_sum_to_quadratic_sum_l1353_135357


namespace NUMINAMATH_CALUDE_star_37_25_l1353_135371

-- Define the star operation
def star (x y : ℝ) : ℝ := x * y + 3

-- State the theorem
theorem star_37_25 :
  (∀ (x : ℝ), x > 0 → star (star x 1) x = star x (star 1 x)) →
  star 1 1 = 4 →
  star 37 25 = 928 := by sorry

end NUMINAMATH_CALUDE_star_37_25_l1353_135371


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l1353_135386

/-- The y-intercept of the tangent line to f(x) = ax - ln x at x = 1 is 1 -/
theorem tangent_line_y_intercept (a : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ a * x - Real.log x
  let f' : ℝ → ℝ := λ x ↦ a - 1 / x
  let tangent_slope : ℝ := f' 1
  let tangent_point : ℝ × ℝ := (1, f 1)
  let tangent_line : ℝ → ℝ := λ x ↦ tangent_slope * (x - tangent_point.1) + tangent_point.2
  tangent_line 0 = 1 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l1353_135386


namespace NUMINAMATH_CALUDE_students_with_glasses_and_watches_l1353_135333

theorem students_with_glasses_and_watches (n : ℕ) 
  (glasses : ℚ) (watches : ℚ) (neither : ℚ) (both : ℕ) :
  glasses = 3/5 →
  watches = 5/6 →
  neither = 1/10 →
  (n : ℚ) * glasses + (n : ℚ) * watches - (n : ℚ) + (n : ℚ) * neither = (n : ℚ) →
  both = 16 :=
by
  sorry

#check students_with_glasses_and_watches

end NUMINAMATH_CALUDE_students_with_glasses_and_watches_l1353_135333


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_ten_thirds_l1353_135367

theorem sqrt_x_div_sqrt_y_equals_ten_thirds (x y : ℝ) 
  (h : (1/2)^2 + (1/3)^2 = ((13 * x) / (41 * y)) * ((1/4)^2 + (1/5)^2)) : 
  Real.sqrt x / Real.sqrt y = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_ten_thirds_l1353_135367


namespace NUMINAMATH_CALUDE_ten_stairs_ways_l1353_135340

def stair_ways (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 1
  | 2 => 2
  | 3 => 4
  | m + 4 => stair_ways m + stair_ways (m + 1) + stair_ways (m + 2) + stair_ways (m + 3)

theorem ten_stairs_ways : stair_ways 10 = 401 := by
  sorry

end NUMINAMATH_CALUDE_ten_stairs_ways_l1353_135340


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l1353_135356

/-- A geometric sequence -/
def GeometricSequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

/-- The product of the first n terms of a sequence -/
def SequenceProduct (b : ℕ → ℝ) (n : ℕ) : ℝ :=
  (List.range n).foldl (λ acc i => acc * b (i + 1)) 1

theorem geometric_sequence_property (b : ℕ → ℝ) (h : GeometricSequence b) (h7 : b 7 = 1) :
  ∀ n : ℕ+, n < 13 → SequenceProduct b n = SequenceProduct b (13 - n) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l1353_135356
