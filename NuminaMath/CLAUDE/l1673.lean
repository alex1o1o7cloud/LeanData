import Mathlib

namespace NUMINAMATH_CALUDE_incompatible_inequalities_l1673_167327

theorem incompatible_inequalities (a b c d : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ¬(a + b < c + d ∧ 
    (a + b) * (c + d) < a * b + c * d ∧ 
    (a + b) * c * d < (c + d) * a * b) :=
by sorry

end NUMINAMATH_CALUDE_incompatible_inequalities_l1673_167327


namespace NUMINAMATH_CALUDE_bent_strips_odd_l1673_167350

/-- Represents a paper strip covering two unit squares on the cube's surface -/
structure Strip where
  isBent : Bool

/-- Represents a 9x9x9 cube covered with 2x1 paper strips -/
structure Cube where
  strips : List Strip

/-- The number of unit squares on the surface of a 9x9x9 cube -/
def surfaceSquares : Nat := 6 * 9 * 9

/-- Theorem: The number of bent strips covering a 9x9x9 cube is odd -/
theorem bent_strips_odd (cube : Cube) (h1 : cube.strips.length * 2 = surfaceSquares) : 
  Odd (cube.strips.filter Strip.isBent).length := by
  sorry


end NUMINAMATH_CALUDE_bent_strips_odd_l1673_167350


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1673_167338

-- Define the sets M and N
def M : Set ℝ := {x | -3 < x ∧ x ≤ 5}
def N : Set ℝ := {x | x < -5 ∨ x > 5}

-- State the theorem
theorem union_of_M_and_N : 
  M ∪ N = {x : ℝ | x < -5 ∨ x > -3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1673_167338


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l1673_167344

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (1 + x, 1 - 3*x)
  let b : ℝ × ℝ := (2, -1)
  are_parallel a b → x = 3/5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l1673_167344


namespace NUMINAMATH_CALUDE_sum_divisible_by_twelve_l1673_167349

theorem sum_divisible_by_twelve (b : ℤ) : 
  ∃ k : ℤ, 6 * b * (b + 1) = 12 * k := by sorry

end NUMINAMATH_CALUDE_sum_divisible_by_twelve_l1673_167349


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l1673_167330

/-- An ellipse with parametric equations x = 3cos(θ) and y = 2sin(θ) -/
structure Ellipse where
  x : ℝ → ℝ
  y : ℝ → ℝ
  h_x : ∀ θ, x θ = 3 * Real.cos θ
  h_y : ∀ θ, y θ = 2 * Real.sin θ

/-- The length of the major axis of an ellipse -/
def major_axis_length (e : Ellipse) : ℝ := 6

/-- Theorem: The length of the major axis of the given ellipse is 6 -/
theorem ellipse_major_axis_length (e : Ellipse) : 
  major_axis_length e = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l1673_167330


namespace NUMINAMATH_CALUDE_event_probability_l1673_167356

theorem event_probability (P_A P_A_and_B P_A_or_B : ℝ) 
  (h1 : P_A = 0.4)
  (h2 : P_A_and_B = 0.25)
  (h3 : P_A_or_B = 0.8) :
  ∃ P_B : ℝ, P_B = 0.65 ∧ P_A_or_B = P_A + P_B - P_A_and_B :=
by
  sorry

end NUMINAMATH_CALUDE_event_probability_l1673_167356


namespace NUMINAMATH_CALUDE_adults_in_group_l1673_167379

theorem adults_in_group (children : ℕ) (meal_cost : ℕ) (total_bill : ℕ) (adults : ℕ) : 
  children = 5 → 
  meal_cost = 3 → 
  total_bill = 21 → 
  adults * meal_cost + children * meal_cost = total_bill → 
  adults = 2 := by
sorry

end NUMINAMATH_CALUDE_adults_in_group_l1673_167379


namespace NUMINAMATH_CALUDE_not_perp_planes_implies_no_perp_line_l1673_167378

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the "line within plane" relation
variable (line_in_plane : Line → Plane → Prop)

-- Theorem statement
theorem not_perp_planes_implies_no_perp_line (α β : Plane) :
  ¬(∀ (α β : Plane), ¬(perp_planes α β) → ∀ (l : Line), line_in_plane l α → ¬(perp_line_plane l β)) :=
sorry

end NUMINAMATH_CALUDE_not_perp_planes_implies_no_perp_line_l1673_167378


namespace NUMINAMATH_CALUDE_distribute_eq_choose_l1673_167318

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes -/
def distribute (n k : ℕ+) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

/-- Theorem stating that the number of ways to distribute n indistinguishable objects
    into k distinct boxes is equal to (n+k-1) choose (k-1) -/
theorem distribute_eq_choose (n k : ℕ+) :
  distribute n k = Nat.choose (n + k - 1) (k - 1) := by
  sorry

end NUMINAMATH_CALUDE_distribute_eq_choose_l1673_167318


namespace NUMINAMATH_CALUDE_somu_age_proof_l1673_167362

/-- Somu's present age -/
def somu_age : ℕ := 12

/-- Somu's father's present age -/
def father_age : ℕ := 3 * somu_age

theorem somu_age_proof :
  (somu_age = father_age / 3) ∧
  (somu_age - 6 = (father_age - 6) / 5) →
  somu_age = 12 := by
sorry

end NUMINAMATH_CALUDE_somu_age_proof_l1673_167362


namespace NUMINAMATH_CALUDE_choose_four_from_nine_l1673_167377

theorem choose_four_from_nine : Nat.choose 9 4 = 126 := by
  sorry

end NUMINAMATH_CALUDE_choose_four_from_nine_l1673_167377


namespace NUMINAMATH_CALUDE_classroom_window_2023_l1673_167394

/-- Represents a digit as seen through a transparent surface --/
inductive MirroredDigit
| Zero
| Two
| Three

/-- Represents the appearance of a number when viewed through a transparent surface --/
def mirror_number (n : List Nat) : List MirroredDigit :=
  sorry

/-- The property of being viewed from the opposite side of a transparent surface --/
def viewed_from_opposite_side (original : List Nat) (mirrored : List MirroredDigit) : Prop :=
  mirror_number original = mirrored.reverse

theorem classroom_window_2023 :
  viewed_from_opposite_side [2, 0, 2, 3] [MirroredDigit.Three, MirroredDigit.Two, MirroredDigit.Zero, MirroredDigit.Two] :=
by sorry

end NUMINAMATH_CALUDE_classroom_window_2023_l1673_167394


namespace NUMINAMATH_CALUDE_anayet_speed_l1673_167399

/-- Calculates Anayet's speed given the total distance, Amoli's speed and driving time,
    Anayet's driving time, and the remaining distance. -/
theorem anayet_speed
  (total_distance : ℝ)
  (amoli_speed : ℝ)
  (amoli_time : ℝ)
  (anayet_time : ℝ)
  (remaining_distance : ℝ)
  (h1 : total_distance = 369)
  (h2 : amoli_speed = 42)
  (h3 : amoli_time = 3)
  (h4 : anayet_time = 2)
  (h5 : remaining_distance = 121) :
  (total_distance - (amoli_speed * amoli_time) - remaining_distance) / anayet_time = 61 :=
by sorry

end NUMINAMATH_CALUDE_anayet_speed_l1673_167399


namespace NUMINAMATH_CALUDE_angle_triple_supplement_l1673_167302

theorem angle_triple_supplement (x : ℝ) : x = 3 * (180 - x) → x = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_triple_supplement_l1673_167302


namespace NUMINAMATH_CALUDE_p_geq_q_l1673_167353

theorem p_geq_q (a b : ℝ) (h : a > 2) : a + 1 / (a - 2) ≥ -b^2 - 2*b + 3 := by
  sorry

end NUMINAMATH_CALUDE_p_geq_q_l1673_167353


namespace NUMINAMATH_CALUDE_ellipse_major_axis_l1673_167360

/-- The equation of an ellipse -/
def ellipse_equation (x y : ℝ) : Prop := x^2 + 9*y^2 = 9

/-- The length of the major axis of the ellipse -/
def major_axis_length : ℝ := 6

/-- Theorem: The length of the major axis of the ellipse x^2 + 9y^2 = 9 is 6 -/
theorem ellipse_major_axis :
  ∀ x y : ℝ, ellipse_equation x y → major_axis_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_l1673_167360


namespace NUMINAMATH_CALUDE_tangent_line_at_point_one_three_l1673_167361

def f (x : ℝ) : ℝ := x^3 - x + 3

theorem tangent_line_at_point_one_three :
  let p : ℝ × ℝ := (1, 3)
  let m : ℝ := (deriv f) p.1
  (λ (x y : ℝ) => 2*x - y + 1 = 0) = (λ (x y : ℝ) => y - p.2 = m * (x - p.1)) := by sorry

end NUMINAMATH_CALUDE_tangent_line_at_point_one_three_l1673_167361


namespace NUMINAMATH_CALUDE_basketball_handshakes_l1673_167385

/-- Number of players in each team -/
def team_size : ℕ := 6

/-- Number of teams -/
def num_teams : ℕ := 2

/-- Number of referees -/
def num_referees : ℕ := 3

/-- Total number of players -/
def total_players : ℕ := team_size * num_teams

/-- Function to calculate the number of handshakes between two teams -/
def inter_team_handshakes : ℕ := team_size * team_size

/-- Function to calculate the number of handshakes within a team -/
def intra_team_handshakes : ℕ := team_size.choose 2

/-- Function to calculate the number of handshakes between players and referees -/
def player_referee_handshakes : ℕ := total_players * num_referees

/-- The total number of handshakes in the basketball game -/
def total_handshakes : ℕ := 
  inter_team_handshakes + 
  (intra_team_handshakes * num_teams) + 
  player_referee_handshakes

theorem basketball_handshakes : total_handshakes = 102 := by
  sorry

end NUMINAMATH_CALUDE_basketball_handshakes_l1673_167385


namespace NUMINAMATH_CALUDE_tangent_line_speed_l1673_167337

theorem tangent_line_speed 
  (a T R L x : ℝ) 
  (h_pos : a > 0 ∧ T > 0 ∧ R > 0 ∧ L > 0)
  (h_eq : (a * T) / (a * T - R) = (L + x) / x) :
  x / T = a * L / R :=
sorry

end NUMINAMATH_CALUDE_tangent_line_speed_l1673_167337


namespace NUMINAMATH_CALUDE_one_cut_divides_two_squares_equally_l1673_167335

-- Define a square
structure Square where
  center : ℝ × ℝ
  side_length : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Function to check if a line passes through a point
def line_passes_through_point (l : Line) (p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p = (l.point1.1 + t * (l.point2.1 - l.point1.1), 
               l.point1.2 + t * (l.point2.2 - l.point1.2))

-- Function to check if a line divides a square into two equal parts
def line_divides_square_equally (l : Line) (s : Square) : Prop :=
  line_passes_through_point l s.center

-- Theorem statement
theorem one_cut_divides_two_squares_equally 
  (s1 s2 : Square) (l : Line) : 
  line_passes_through_point l s1.center → 
  line_passes_through_point l s2.center → 
  line_divides_square_equally l s1 ∧ line_divides_square_equally l s2 :=
sorry

end NUMINAMATH_CALUDE_one_cut_divides_two_squares_equally_l1673_167335


namespace NUMINAMATH_CALUDE_barry_sotter_magic_l1673_167390

/-- The increase factor for day k --/
def increase_factor (k : ℕ) : ℚ := (k + 3) / (k + 2)

/-- The overall increase factor after n days --/
def overall_increase (n : ℕ) : ℚ := (n + 3) / 3

theorem barry_sotter_magic (n : ℕ) : overall_increase n = 50 → n = 147 := by
  sorry

end NUMINAMATH_CALUDE_barry_sotter_magic_l1673_167390


namespace NUMINAMATH_CALUDE_no_integer_solutions_l1673_167319

theorem no_integer_solutions : ¬ ∃ (x y : ℤ), 14 * x^2 + 15 * y^2 = 7^2000 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solutions_l1673_167319


namespace NUMINAMATH_CALUDE_temperature_conversion_l1673_167328

theorem temperature_conversion (t k : ℚ) (f : ℚ) : 
  t = f * (k - 32) → t = 50 → k = 122 → f = 5/9 := by
  sorry

end NUMINAMATH_CALUDE_temperature_conversion_l1673_167328


namespace NUMINAMATH_CALUDE_total_notebooks_is_303_l1673_167354

/-- The total number of notebooks in a classroom with specific distribution of notebooks among students. -/
def total_notebooks : ℕ :=
  let total_students : ℕ := 60
  let students_with_5 : ℕ := total_students / 4
  let students_with_3 : ℕ := total_students / 5
  let students_with_7 : ℕ := total_students / 3
  let students_with_4 : ℕ := total_students - (students_with_5 + students_with_3 + students_with_7)
  (students_with_5 * 5) + (students_with_3 * 3) + (students_with_7 * 7) + (students_with_4 * 4)

theorem total_notebooks_is_303 : total_notebooks = 303 := by
  sorry

end NUMINAMATH_CALUDE_total_notebooks_is_303_l1673_167354


namespace NUMINAMATH_CALUDE_remainder_1897_2048_mod_600_l1673_167386

theorem remainder_1897_2048_mod_600 : (1897 * 2048) % 600 = 256 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1897_2048_mod_600_l1673_167386


namespace NUMINAMATH_CALUDE_min_sum_of_square_areas_l1673_167396

theorem min_sum_of_square_areas (wire_length : ℝ) (h : wire_length = 16) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ wire_length ∧
  (x^2 + (wire_length - x)^2 ≥ 8 ∧
   ∀ (y : ℝ), 0 ≤ y ∧ y ≤ wire_length →
     y^2 + (wire_length - y)^2 ≥ x^2 + (wire_length - x)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_sum_of_square_areas_l1673_167396


namespace NUMINAMATH_CALUDE_square_8x_minus_5_l1673_167363

theorem square_8x_minus_5 (x : ℝ) (h : 8 * x^2 + 7 = 12 * x + 17) : (8 * x - 5)^2 = 465 := by
  sorry

end NUMINAMATH_CALUDE_square_8x_minus_5_l1673_167363


namespace NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1673_167311

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem thirtieth_term_of_sequence (a₁ a₂ a₃ : ℝ) (h1 : a₁ = 3) (h2 : a₂ = 13) (h3 : a₃ = 23) :
  arithmetic_sequence a₁ (a₂ - a₁) 30 = 293 := by
sorry

end NUMINAMATH_CALUDE_thirtieth_term_of_sequence_l1673_167311


namespace NUMINAMATH_CALUDE_quadratic_function_j_value_l1673_167346

/-- A quadratic function with integer coefficients -/
def QuadraticFunction (a b c : ℤ) : ℝ → ℝ := fun x ↦ (a * x^2 : ℝ) + (b * x : ℝ) + (c : ℝ)

theorem quadratic_function_j_value
  (a b c : ℤ)
  (h1 : QuadraticFunction a b c 1 = 0)
  (h2 : QuadraticFunction a b c (-1) = 0)
  (h3 : 70 < QuadraticFunction a b c 7 ∧ QuadraticFunction a b c 7 < 90)
  (h4 : 110 < QuadraticFunction a b c 8 ∧ QuadraticFunction a b c 8 < 140)
  (h5 : ∃ j : ℤ, 1000 * j < QuadraticFunction a b c 50 ∧ QuadraticFunction a b c 50 < 1000 * (j + 1)) :
  ∃ j : ℤ, j = 4 ∧ 1000 * j < QuadraticFunction a b c 50 ∧ QuadraticFunction a b c 50 < 1000 * (j + 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_j_value_l1673_167346


namespace NUMINAMATH_CALUDE_marcus_earnings_theorem_l1673_167331

/-- Calculates the after-tax earnings for Marcus over two weeks -/
def marcusEarnings (hoursWeek1 hoursWeek2 : ℕ) (extraEarnings : ℚ) (taxRate : ℚ) : ℚ :=
  let hourlyWage := extraEarnings / (hoursWeek2 - hoursWeek1)
  let totalHours := hoursWeek1 + hoursWeek2
  let totalEarnings := hourlyWage * totalHours
  totalEarnings * (1 - taxRate)

/-- Theorem stating that Marcus's earnings after tax for the two weeks is $293.40 -/
theorem marcus_earnings_theorem :
  marcusEarnings 20 30 65.20 0.1 = 293.40 := by
  sorry

end NUMINAMATH_CALUDE_marcus_earnings_theorem_l1673_167331


namespace NUMINAMATH_CALUDE_triangle_properties_l1673_167301

/-- Properties of a triangle ABC with given circumradius, one side length, and ratio of other sides -/
theorem triangle_properties (R a t : ℝ) (h_R : R > 0) (h_a : a > 0) (h_t : t > 0) :
  ∃ (b c : ℝ) (A B C : ℝ),
    b = 2 * R * Real.sin B ∧
    c = b / t ∧
    A = Real.arcsin (a / (2 * R)) ∧
    B = Real.arctan ((t * Real.sin A) / (1 - t * Real.cos A)) ∧
    C = π - A - B :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l1673_167301


namespace NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1673_167308

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3*x - 15 → x ≥ 8 ∧ 8 < 3*8 - 15 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_satisfying_inequality_l1673_167308


namespace NUMINAMATH_CALUDE_julia_cakes_l1673_167375

theorem julia_cakes (x : ℕ) : 
  (x * 6 - 3 = 21) → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_julia_cakes_l1673_167375


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1673_167303

theorem complex_modulus_problem : Complex.abs ((Complex.I + 1) / Complex.I) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1673_167303


namespace NUMINAMATH_CALUDE_hcf_of_4_and_18_l1673_167393

theorem hcf_of_4_and_18 :
  let a : ℕ := 4
  let b : ℕ := 18
  let lcm_ab : ℕ := 36
  Nat.lcm a b = lcm_ab →
  Nat.gcd a b = 2 := by
sorry

end NUMINAMATH_CALUDE_hcf_of_4_and_18_l1673_167393


namespace NUMINAMATH_CALUDE_coin_problem_l1673_167369

/-- Represents the number of different values that can be produced with given coins -/
def different_values (five_cent_coins ten_cent_coins : ℕ) : ℕ :=
  29 - five_cent_coins

theorem coin_problem (total_coins : ℕ) (distinct_values : ℕ) 
  (h1 : total_coins = 15)
  (h2 : distinct_values = 26) :
  ∃ (five_cent_coins ten_cent_coins : ℕ),
    five_cent_coins + ten_cent_coins = total_coins ∧
    different_values five_cent_coins ten_cent_coins = distinct_values ∧
    ten_cent_coins = 12 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l1673_167369


namespace NUMINAMATH_CALUDE_two_white_balls_probability_l1673_167313

/-- The probability of drawing two white balls without replacement from a box containing 8 white balls and 7 black balls is 4/15. -/
theorem two_white_balls_probability :
  let total_balls : ℕ := 8 + 7
  let white_balls : ℕ := 8
  let black_balls : ℕ := 7
  let prob_first_white : ℚ := white_balls / total_balls
  let prob_second_white : ℚ := (white_balls - 1) / (total_balls - 1)
  prob_first_white * prob_second_white = 4 / 15 := by
sorry

end NUMINAMATH_CALUDE_two_white_balls_probability_l1673_167313


namespace NUMINAMATH_CALUDE_final_sum_theorem_l1673_167329

theorem final_sum_theorem (x y R : ℝ) (h : x + y = R) :
  3 * (x + 5) + 3 * (y + 5) = 3 * R + 30 :=
by sorry

end NUMINAMATH_CALUDE_final_sum_theorem_l1673_167329


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l1673_167341

theorem no_solutions_in_interval (x : Real) : 
  x ∈ Set.Icc 0 Real.pi → 
  ¬(Real.sin (Real.pi * Real.cos x) = Real.cos (Real.pi * Real.sin x)) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l1673_167341


namespace NUMINAMATH_CALUDE_log_equation_proof_l1673_167368

-- Define the common logarithm (base 10)
noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_equation_proof :
  (log 5) ^ 2 + log 2 * log 50 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_equation_proof_l1673_167368


namespace NUMINAMATH_CALUDE_limit_sequence_equals_one_over_e_l1673_167384

theorem limit_sequence_equals_one_over_e :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |((10 * n - 3) / (10 * n - 1)) ^ (5 * n) - (1 / Real.exp 1)| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_sequence_equals_one_over_e_l1673_167384


namespace NUMINAMATH_CALUDE_binary_sum_equals_1100000_l1673_167345

/-- Converts a list of bits to its decimal representation -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Represents a binary number as a list of booleans -/
def Binary := List Bool

theorem binary_sum_equals_1100000 :
  let a : Binary := [true, false, false, true, true]  -- 11001₂
  let b : Binary := [true, true, true]                -- 111₂
  let c : Binary := [false, false, true, false, true] -- 10100₂
  let d : Binary := [true, true, true, true]          -- 1111₂
  let e : Binary := [true, true, false, false, true, true] -- 110011₂
  let sum : Binary := [false, false, false, false, false, true, true] -- 1100000₂
  binaryToDecimal a + binaryToDecimal b + binaryToDecimal c +
  binaryToDecimal d + binaryToDecimal e = binaryToDecimal sum := by
  sorry


end NUMINAMATH_CALUDE_binary_sum_equals_1100000_l1673_167345


namespace NUMINAMATH_CALUDE_min_sides_convex_polygon_l1673_167371

/-- A convex polygon is a closed planar figure with straight sides. -/
structure ConvexPolygon where
  sides : ℕ
  is_convex : Bool

/-- Theorem: The minimum number of sides for a convex polygon is 3. -/
theorem min_sides_convex_polygon :
  ∀ p : ConvexPolygon, p.is_convex → p.sides ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_min_sides_convex_polygon_l1673_167371


namespace NUMINAMATH_CALUDE_infinite_geometric_series_second_term_l1673_167325

theorem infinite_geometric_series_second_term
  (r : ℝ) (S : ℝ) (h_r : r = 1/4) (h_S : S = 20) :
  let a := S * (1 - r)
  a * r = 15/4 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_second_term_l1673_167325


namespace NUMINAMATH_CALUDE_courtyard_paving_l1673_167314

def courtyard_length : ℝ := 25
def courtyard_width : ℝ := 16
def brick_length : ℝ := 0.2
def brick_width : ℝ := 0.1

theorem courtyard_paving :
  (courtyard_length * courtyard_width) / (brick_length * brick_width) = 20000 := by
  sorry

end NUMINAMATH_CALUDE_courtyard_paving_l1673_167314


namespace NUMINAMATH_CALUDE_cleaning_hourly_rate_l1673_167306

/-- Calculates the hourly rate for cleaning rooms in a building -/
theorem cleaning_hourly_rate
  (floors : ℕ)
  (rooms_per_floor : ℕ)
  (hours_per_room : ℕ)
  (total_earnings : ℕ)
  (h1 : floors = 4)
  (h2 : rooms_per_floor = 10)
  (h3 : hours_per_room = 6)
  (h4 : total_earnings = 3600) :
  total_earnings / (floors * rooms_per_floor * hours_per_room) = 15 := by
  sorry

#check cleaning_hourly_rate

end NUMINAMATH_CALUDE_cleaning_hourly_rate_l1673_167306


namespace NUMINAMATH_CALUDE_larger_number_problem_l1673_167326

theorem larger_number_problem (L S : ℕ) 
  (h1 : L - S = 2415)
  (h2 : L = 21 * S + 15) : 
  L = 2535 := by
sorry

end NUMINAMATH_CALUDE_larger_number_problem_l1673_167326


namespace NUMINAMATH_CALUDE_sequence_general_term_l1673_167397

/-- Given a sequence {a_n} where S_n represents the sum of the first n terms,
    prove that the general term a_n can be expressed as a_1 + (n-1)d,
    where d is the common difference (a_2 - a_1). -/
theorem sequence_general_term (a : ℕ → ℝ) (S : ℕ → ℝ) (n : ℕ) :
  (∀ k, S k = k / 2 * (a 1 + a k)) →
  ∃ d : ℝ, d = a 2 - a 1 ∧ ∀ m, a m = a 1 + (m - 1) * d :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1673_167397


namespace NUMINAMATH_CALUDE_sum_of_three_squares_l1673_167382

/-- The value of a triangle -/
def triangle_value : ℝ := sorry

/-- The value of a square -/
def square_value : ℝ := sorry

/-- The sum of three triangles and two squares equals 18 -/
axiom eq1 : 3 * triangle_value + 2 * square_value = 18

/-- The sum of two triangles and three squares equals 22 -/
axiom eq2 : 2 * triangle_value + 3 * square_value = 22

/-- The sum of three squares equals 18 -/
theorem sum_of_three_squares : 3 * square_value = 18 := by sorry

end NUMINAMATH_CALUDE_sum_of_three_squares_l1673_167382


namespace NUMINAMATH_CALUDE_speed_calculation_l1673_167339

-- Define the distance in meters
def distance_meters : ℝ := 375.03

-- Define the time in seconds
def time_seconds : ℝ := 25

-- Define the conversion factor from m/s to km/h
def mps_to_kmph : ℝ := 3.6

-- Theorem to prove
theorem speed_calculation :
  let speed_mps := distance_meters / time_seconds
  let speed_kmph := speed_mps * mps_to_kmph
  ∃ ε > 0, |speed_kmph - 54.009| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_speed_calculation_l1673_167339


namespace NUMINAMATH_CALUDE_smallest_clock_equivalent_hour_l1673_167348

theorem smallest_clock_equivalent_hour : ∃ (n : ℕ), n > 10 ∧ n % 12 = (n^2) % 12 ∧ ∀ (m : ℕ), m > 10 ∧ m < n → m % 12 ≠ (m^2) % 12 :=
  sorry

end NUMINAMATH_CALUDE_smallest_clock_equivalent_hour_l1673_167348


namespace NUMINAMATH_CALUDE_range_of_m_l1673_167387

-- Define the conditions
def p (x : ℝ) : Prop := |1 - (x - 1) / 3| ≤ 2
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- State the theorem
theorem range_of_m (m : ℝ) :
  (∀ x, p x → q x m) ∧  -- q is necessary for p
  (∃ x, p x ∧ ¬(q x m)) ∧  -- q is not sufficient for p
  (m > 0) →
  m ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l1673_167387


namespace NUMINAMATH_CALUDE_beatty_theorem_l1673_167359

theorem beatty_theorem (α β : ℝ) (hα : Irrational α) (hβ : Irrational β) 
  (hpos_α : α > 0) (hpos_β : β > 0) (h_sum : 1/α + 1/β = 1) :
  (∀ k : ℕ+, ∃! n : ℕ+, k = ⌊n * α⌋ ∨ k = ⌊n * β⌋) ∧ 
  (∀ k : ℕ+, ¬(∃ m n : ℕ+, k = ⌊m * α⌋ ∧ k = ⌊n * β⌋)) := by
  sorry

end NUMINAMATH_CALUDE_beatty_theorem_l1673_167359


namespace NUMINAMATH_CALUDE_interview_bounds_l1673_167320

theorem interview_bounds (students : ℕ) (junior_high : ℕ) (teachers : ℕ) (table_tennis : ℕ) (basketball : ℕ)
  (h1 : students = 6)
  (h2 : junior_high = 4)
  (h3 : teachers = 2)
  (h4 : table_tennis = 5)
  (h5 : basketball = 2)
  (h6 : junior_high ≤ students) :
  ∃ (min max : ℕ),
    (min = students + teachers) ∧
    (max = students - junior_high + teachers + table_tennis + basketball + junior_high) ∧
    (min = 8) ∧
    (max = 15) ∧
    (∀ n : ℕ, n ≥ min ∧ n ≤ max) := by
  sorry

end NUMINAMATH_CALUDE_interview_bounds_l1673_167320


namespace NUMINAMATH_CALUDE_olivers_money_l1673_167317

/-- Oliver's money calculation -/
theorem olivers_money (initial_amount spent_amount received_amount : ℕ) :
  initial_amount = 33 →
  spent_amount = 4 →
  received_amount = 32 →
  initial_amount - spent_amount + received_amount = 61 :=
by
  sorry

end NUMINAMATH_CALUDE_olivers_money_l1673_167317


namespace NUMINAMATH_CALUDE_triangle_not_divisible_into_trapeziums_l1673_167357

-- Define a shape as a type
inductive Shape
| Rectangle
| Square
| RegularHexagon
| Trapezium
| Triangle

-- Define a trapezium
def isTrapezium (s : Shape) : Prop :=
  ∃ (sides : ℕ), sides = 4 ∧ ∃ (parallel_sides : ℕ), parallel_sides ≥ 1

-- Define the property of being divisible into two trapeziums by a single straight line
def isDivisibleIntoTwoTrapeziums (s : Shape) : Prop :=
  ∃ (part1 part2 : Shape), isTrapezium part1 ∧ isTrapezium part2

-- State the theorem
theorem triangle_not_divisible_into_trapeziums :
  ¬(isDivisibleIntoTwoTrapeziums Shape.Triangle) :=
sorry

end NUMINAMATH_CALUDE_triangle_not_divisible_into_trapeziums_l1673_167357


namespace NUMINAMATH_CALUDE_lost_card_value_l1673_167373

theorem lost_card_value (n : ℕ) (h1 : n > 0) (h2 : (n * (n + 1)) / 2 - 101 ≤ n) : 
  (n * (n + 1)) / 2 - 101 = 4 :=
by sorry

end NUMINAMATH_CALUDE_lost_card_value_l1673_167373


namespace NUMINAMATH_CALUDE_sport_gender_relationship_l1673_167305

/-- The critical value of K² for P(K² ≥ k) = 0.05 -/
def critical_value : ℝ := 3.841

/-- The observed value of K² -/
def observed_value : ℝ := 4.892

/-- The significance level -/
def significance_level : ℝ := 0.05

/-- The sample size -/
def sample_size : ℕ := 200

/-- Theorem stating that the observed value exceeds the critical value,
    allowing us to conclude a relationship between liking the sport and gender
    with 1 - significance_level confidence -/
theorem sport_gender_relationship :
  observed_value > critical_value →
  ∃ (confidence_level : ℝ), confidence_level = 1 - significance_level ∧
    confidence_level > 0.95 ∧
    (∃ (relationship : Prop), relationship) :=
by
  sorry

end NUMINAMATH_CALUDE_sport_gender_relationship_l1673_167305


namespace NUMINAMATH_CALUDE_nine_digit_repeat_gcd_l1673_167398

theorem nine_digit_repeat_gcd : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), 100 ≤ m ∧ m < 1000 → 
    (∃ (k : ℕ), k = m * 1001001 ∧ 
      Nat.gcd n k = n)) ∧ 
  (∀ (d : ℕ), d > n → 
    ∃ (m₁ m₂ : ℕ), 100 ≤ m₁ ∧ m₁ < 1000 ∧ 100 ≤ m₂ ∧ m₂ < 1000 ∧ 
      Nat.gcd (m₁ * 1001001) (m₂ * 1001001) < d) :=
by sorry

end NUMINAMATH_CALUDE_nine_digit_repeat_gcd_l1673_167398


namespace NUMINAMATH_CALUDE_remainder_divisibility_l1673_167304

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 13 * k + 4) → (∃ m : ℤ, N = 39 * m + 4) :=
sorry

end NUMINAMATH_CALUDE_remainder_divisibility_l1673_167304


namespace NUMINAMATH_CALUDE_negative_one_power_difference_l1673_167381

theorem negative_one_power_difference : (-1 : ℤ)^5 - (-1 : ℤ)^4 = -2 := by
  sorry

end NUMINAMATH_CALUDE_negative_one_power_difference_l1673_167381


namespace NUMINAMATH_CALUDE_urn_theorem_l1673_167312

/-- Represents the state of the two urns -/
structure UrnState where
  urn1 : ℕ
  urn2 : ℕ

/-- Represents the transfer rule between urns -/
def transfer (state : UrnState) : UrnState :=
  if state.urn1 % 2 = 0 then
    UrnState.mk (state.urn1 / 2) (state.urn2 + state.urn1 / 2)
  else if state.urn2 % 2 = 0 then
    UrnState.mk (state.urn1 + state.urn2 / 2) (state.urn2 / 2)
  else
    state

theorem urn_theorem (p k : ℕ) (h1 : Prime p) (h2 : Prime (2 * p + 1)) (h3 : k < 2 * p + 1) :
  ∃ (n : ℕ) (state : UrnState),
    state.urn1 + state.urn2 = 2 * p + 1 ∧
    (transfer^[n] state).urn1 = k ∨ (transfer^[n] state).urn2 = k :=
  sorry

end NUMINAMATH_CALUDE_urn_theorem_l1673_167312


namespace NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1673_167322

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 4) 
  (h2 : a^2 + b^2 = 30) : 
  a * b = 32 := by
sorry

end NUMINAMATH_CALUDE_product_of_difference_and_sum_of_squares_l1673_167322


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l1673_167355

/-- The area of the quadrilateral formed by connecting the midpoints of a rectangle -/
theorem midpoint_quadrilateral_area (w l : ℝ) (hw : w = 10) (hl : l = 14) :
  let midpoint_quad_area := (w / 2) * (l / 2)
  midpoint_quad_area = 35 := by sorry

end NUMINAMATH_CALUDE_midpoint_quadrilateral_area_l1673_167355


namespace NUMINAMATH_CALUDE_day_150_previous_year_is_friday_l1673_167307

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  value : ℕ
  isLeapYear : Bool

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def advanceDays (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

theorem day_150_previous_year_is_friday 
  (N : Year) 
  (h1 : N.isLeapYear = true) 
  (h2 : advanceDays DayOfWeek.Sunday 249 = DayOfWeek.Friday) : 
  advanceDays DayOfWeek.Sunday 149 = DayOfWeek.Friday :=
sorry

end NUMINAMATH_CALUDE_day_150_previous_year_is_friday_l1673_167307


namespace NUMINAMATH_CALUDE_no_real_roots_condition_l1673_167340

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := x^2 + 2*x - m

-- Define the discriminant of the quadratic equation
def discriminant (m : ℝ) : ℝ := 2^2 - 4*(1)*(-m)

-- Theorem statement
theorem no_real_roots_condition (m : ℝ) :
  (∀ x, quadratic_equation x m ≠ 0) ↔ m < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_condition_l1673_167340


namespace NUMINAMATH_CALUDE_intersection_right_isosceles_l1673_167389

-- Define the universe set of all triangles
def Triangle : Type := sorry

-- Define the property of being a right triangle
def IsRight (t : Triangle) : Prop := sorry

-- Define the property of being an isosceles triangle
def IsIsosceles (t : Triangle) : Prop := sorry

-- Define the set of right triangles
def RightTriangles : Set Triangle := {t : Triangle | IsRight t}

-- Define the set of isosceles triangles
def IsoscelesTriangles : Set Triangle := {t : Triangle | IsIsosceles t}

-- Define the property of being both right and isosceles
def IsRightAndIsosceles (t : Triangle) : Prop := IsRight t ∧ IsIsosceles t

-- Define the set of isosceles right triangles
def IsoscelesRightTriangles : Set Triangle := {t : Triangle | IsRightAndIsosceles t}

-- Theorem statement
theorem intersection_right_isosceles :
  RightTriangles ∩ IsoscelesTriangles = IsoscelesRightTriangles := by sorry

end NUMINAMATH_CALUDE_intersection_right_isosceles_l1673_167389


namespace NUMINAMATH_CALUDE_accurate_counting_requires_shaking_l1673_167324

/-- Represents a yeast cell -/
structure YeastCell where
  id : ℕ

/-- Represents a culture fluid containing yeast cells -/
structure CultureFluid where
  cells : List YeastCell

/-- Represents a test tube containing culture fluid -/
structure TestTube where
  fluid : CultureFluid

/-- Represents a hemocytometer for counting cells -/
structure Hemocytometer where
  volume : ℝ
  count : CultureFluid → ℕ

/-- Yeast is a unicellular fungus -/
axiom yeast_is_unicellular : ∀ (y : YeastCell), true

/-- Yeast is a facultative anaerobe -/
axiom yeast_is_facultative_anaerobe : ∀ (y : YeastCell), true

/-- Yeast distribution in culture fluid is uneven -/
axiom yeast_distribution_uneven : ∀ (cf : CultureFluid), true

/-- A hemocytometer is used for counting yeast cells -/
axiom hemocytometer_used : ∃ (h : Hemocytometer), true

/-- Shaking the test tube before sampling leads to accurate counting -/
theorem accurate_counting_requires_shaking (tt : TestTube) (h : Hemocytometer) :
  (∀ (sample : CultureFluid), h.count sample = h.count tt.fluid) ↔ 
  (∃ (shaken_tt : TestTube), shaken_tt.fluid = tt.fluid ∧ 
    ∀ (sample : CultureFluid), h.count sample = h.count shaken_tt.fluid) :=
sorry

end NUMINAMATH_CALUDE_accurate_counting_requires_shaking_l1673_167324


namespace NUMINAMATH_CALUDE_light_ray_equation_l1673_167367

-- Define the circle M
def circle_M (x y : ℝ) : Prop :=
  x^2 + y^2 - 4*x - 4*y + 7 = 0

-- Define the point A
def point_A : ℝ × ℝ := (-3, 3)

-- Define the x-axis
def x_axis (x y : ℝ) : Prop := y = 0

-- Define the reflected ray equation
def reflected_ray (x y : ℝ) : Prop :=
  4*x - 3*y + 9 = 0

-- Theorem statement
theorem light_ray_equation :
  ∃ (x₀ y₀ : ℝ),
    -- The ray passes through point A
    reflected_ray x₀ y₀ ∧ (x₀, y₀) = point_A ∧
    -- The ray intersects the x-axis
    ∃ (x₁ : ℝ), reflected_ray x₁ 0 ∧
    -- The ray is tangent to circle M
    ∃ (x₂ y₂ : ℝ), circle_M x₂ y₂ ∧ reflected_ray x₂ y₂ ∧
      ∀ (x y : ℝ), circle_M x y → (x - x₂)^2 + (y - y₂)^2 ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_light_ray_equation_l1673_167367


namespace NUMINAMATH_CALUDE_no_valid_polygon_pairs_l1673_167372

theorem no_valid_polygon_pairs : ¬∃ (y l : ℕ), 
  (∃ (k : ℕ), y = 30 * k) ∧ 
  (l > 1) ∧
  (∃ (n : ℕ), y = 180 - 360 / n) ∧
  (∃ (m : ℕ), l * y = 180 - 360 / m) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_polygon_pairs_l1673_167372


namespace NUMINAMATH_CALUDE_base_conversion_problem_l1673_167358

theorem base_conversion_problem : 
  (∃ (S : Finset ℕ), 
    (∀ b ∈ S, b ≥ 2 ∧ b^3 ≤ 250 ∧ 250 < b^4) ∧ 
    (∀ b : ℕ, b ≥ 2 → b^3 ≤ 250 → 250 < b^4 → b ∈ S) ∧
    Finset.card S = 2) := by sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l1673_167358


namespace NUMINAMATH_CALUDE_largest_value_between_2_and_3_l1673_167323

theorem largest_value_between_2_and_3 (x : ℝ) (h : 2 < x ∧ x < 3) :
  x^2 ≥ x ∧ x^2 ≥ 3*x ∧ x^2 ≥ Real.sqrt x ∧ x^2 ≥ 1/x :=
by sorry

end NUMINAMATH_CALUDE_largest_value_between_2_and_3_l1673_167323


namespace NUMINAMATH_CALUDE_distance_between_foci_l1673_167365

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 3)^2 + (y + 4)^2) + Real.sqrt ((x + 5)^2 + (y - 8)^2) = 20

-- Define the foci of the ellipse
def focus1 : ℝ × ℝ := (3, -4)
def focus2 : ℝ × ℝ := (-5, 8)

-- Theorem stating the distance between foci
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 4 * Real.sqrt 13 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_foci_l1673_167365


namespace NUMINAMATH_CALUDE_quilt_patch_cost_is_450_l1673_167376

/-- Calculates the total cost of patches for a quilt with given dimensions and patch pricing. -/
def quilt_patch_cost (quilt_length : ℕ) (quilt_width : ℕ) (patch_area : ℕ) 
                     (initial_patch_cost : ℕ) (initial_patch_count : ℕ) : ℕ :=
  let total_area := quilt_length * quilt_width
  let total_patches := total_area / patch_area
  let initial_cost := initial_patch_count * initial_patch_cost
  let remaining_patches := total_patches - initial_patch_count
  let remaining_cost := remaining_patches * (initial_patch_cost / 2)
  initial_cost + remaining_cost

/-- The total cost of patches for a 16-foot by 20-foot quilt with specified patch pricing is $450. -/
theorem quilt_patch_cost_is_450 : 
  quilt_patch_cost 16 20 4 10 10 = 450 := by
  sorry

end NUMINAMATH_CALUDE_quilt_patch_cost_is_450_l1673_167376


namespace NUMINAMATH_CALUDE_rectangular_solid_volume_l1673_167336

theorem rectangular_solid_volume
  (a b c : ℕ+)
  (h1 : a * b - c * a - b * c = 1)
  (h2 : c * a = b * c + 1) :
  a * b * c = 6 :=
sorry

end NUMINAMATH_CALUDE_rectangular_solid_volume_l1673_167336


namespace NUMINAMATH_CALUDE_no_intersection_l1673_167310

/-- The line equation 3x + 4y = 12 -/
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12

/-- The circle equation x^2 + y^2 = 4 -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The number of intersection points between the line and the circle -/
def intersection_count : ℕ := 0

theorem no_intersection :
  ∀ x y : ℝ, ¬(line_eq x y ∧ circle_eq x y) :=
by sorry

end NUMINAMATH_CALUDE_no_intersection_l1673_167310


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_l1673_167352

theorem sqrt_2_irrational : ¬ ∃ (a b : ℤ), b ≠ 0 ∧ (a / b : ℚ)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_l1673_167352


namespace NUMINAMATH_CALUDE_x_squared_plus_5xy_plus_y_squared_l1673_167316

theorem x_squared_plus_5xy_plus_y_squared (x y : ℝ) 
  (h1 : x * y = 4) 
  (h2 : x - y = 5) : 
  x^2 + 5*x*y + y^2 = 53 := by
sorry

end NUMINAMATH_CALUDE_x_squared_plus_5xy_plus_y_squared_l1673_167316


namespace NUMINAMATH_CALUDE_multiply_and_add_l1673_167395

theorem multiply_and_add : (23 * 37) + 16 = 867 := by
  sorry

end NUMINAMATH_CALUDE_multiply_and_add_l1673_167395


namespace NUMINAMATH_CALUDE_square_perimeter_l1673_167392

theorem square_perimeter (rectangleA_perimeter : ℝ) (squareB_area_ratio : ℝ) :
  rectangleA_perimeter = 30 →
  squareB_area_ratio = 1/3 →
  ∃ (rectangleA_length rectangleA_width : ℝ),
    rectangleA_length > 0 ∧
    rectangleA_width > 0 ∧
    2 * (rectangleA_length + rectangleA_width) = rectangleA_perimeter ∧
    ∃ (squareB_side : ℝ),
      squareB_side > 0 ∧
      squareB_side^2 = squareB_area_ratio * (rectangleA_length * rectangleA_width) ∧
      4 * squareB_side = 12 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_square_perimeter_l1673_167392


namespace NUMINAMATH_CALUDE_initial_machines_count_l1673_167374

/-- The number of pens produced by a group of machines in a given time -/
structure PenProduction where
  machines : ℕ
  pens : ℕ
  minutes : ℕ

/-- The rate of pen production per minute for a given number of machines -/
def production_rate (p : PenProduction) : ℚ :=
  p.pens / (p.machines * p.minutes)

theorem initial_machines_count (total_rate : ℕ) (sample : PenProduction) :
  sample.machines * total_rate = sample.pens * production_rate sample →
  total_rate = 240 →
  sample = { machines := 5, pens := 750, minutes := 5 } →
  ∃ n : ℕ, n * (production_rate sample) = total_rate ∧ n = 8 := by
sorry

end NUMINAMATH_CALUDE_initial_machines_count_l1673_167374


namespace NUMINAMATH_CALUDE_triangle_area_proof_l1673_167347

/-- The length of a li in meters -/
def li_to_meters : ℝ := 500

/-- The sides of the triangle in li -/
def side1 : ℝ := 5
def side2 : ℝ := 12
def side3 : ℝ := 13

/-- The area of the triangle in square kilometers -/
def triangle_area : ℝ := 7.5

theorem triangle_area_proof :
  let side1_m := side1 * li_to_meters
  let side2_m := side2 * li_to_meters
  let side3_m := side3 * li_to_meters
  side1_m ^ 2 + side2_m ^ 2 = side3_m ^ 2 →
  (1 / 2) * side1_m * side2_m / 1000000 = triangle_area := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_proof_l1673_167347


namespace NUMINAMATH_CALUDE_max_flow_increase_l1673_167351

/-- Represents a water purification system with two sections of pipes -/
structure WaterSystem :=
  (pipes_AB : ℕ)
  (pipes_BC : ℕ)
  (flow_increase : ℝ)

/-- The theorem stating the maximum flow rate increase -/
theorem max_flow_increase (system : WaterSystem) 
  (h1 : system.pipes_AB = 10)
  (h2 : system.pipes_BC = 10)
  (h3 : system.flow_increase = 40) : 
  ∃ (max_increase : ℝ), max_increase = 200 ∧ 
  ∀ (new_system : WaterSystem), 
    new_system.pipes_AB + new_system.pipes_BC = system.pipes_AB + system.pipes_BC →
    new_system.flow_increase ≤ max_increase :=
sorry

end NUMINAMATH_CALUDE_max_flow_increase_l1673_167351


namespace NUMINAMATH_CALUDE_problem_statement_l1673_167309

theorem problem_statement (a b c : ℝ) (h : a + 10 = b + 12 ∧ b + 12 = c + 15) :
  a^2 + b^2 + c^2 - a*b - b*c - a*c = 38 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l1673_167309


namespace NUMINAMATH_CALUDE_john_unanswered_questions_l1673_167366

/-- Represents the scoring systems and John's scores -/
structure ScoringSystem where
  new_correct : ℤ
  new_wrong : ℤ
  new_unanswered : ℤ
  old_start : ℤ
  old_correct : ℤ
  old_wrong : ℤ
  total_questions : ℕ
  new_score : ℤ
  old_score : ℤ

/-- Calculates the number of unanswered questions based on the scoring system -/
def unanswered_questions (s : ScoringSystem) : ℕ :=
  sorry

/-- Theorem stating that for the given scoring system, John left 2 questions unanswered -/
theorem john_unanswered_questions :
  let s : ScoringSystem := {
    new_correct := 6,
    new_wrong := -1,
    new_unanswered := 3,
    old_start := 25,
    old_correct := 5,
    old_wrong := -2,
    total_questions := 30,
    new_score := 105,
    old_score := 95
  }
  unanswered_questions s = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_unanswered_questions_l1673_167366


namespace NUMINAMATH_CALUDE_exists_quadratic_sequence_l1673_167391

/-- A quadratic sequence is a finite sequence of integers where the absolute difference
    between consecutive terms is equal to the square of their position. -/
def IsQuadraticSequence (a : ℕ → ℤ) (n : ℕ) : Prop :=
  ∀ i : ℕ, i ≤ n → |a i - a (i - 1)| = i^2

/-- For any two integers, there exists a quadratic sequence connecting them. -/
theorem exists_quadratic_sequence (b c : ℤ) :
  ∃ (n : ℕ) (a : ℕ → ℤ), a 0 = b ∧ a n = c ∧ IsQuadraticSequence a n :=
sorry

end NUMINAMATH_CALUDE_exists_quadratic_sequence_l1673_167391


namespace NUMINAMATH_CALUDE_marathon_solution_l1673_167364

def marathon_problem (dean_time : ℝ) : Prop :=
  let micah_time := dean_time * (3/2)
  let jake_time := micah_time * (4/3)
  let nia_time := micah_time * 2
  let eliza_time := dean_time * (4/5)
  let average_time := (dean_time + micah_time + jake_time + nia_time + eliza_time) / 5
  dean_time = 9 ∧ average_time = 15.14

theorem marathon_solution :
  ∃ (dean_time : ℝ), marathon_problem dean_time :=
by
  sorry

end NUMINAMATH_CALUDE_marathon_solution_l1673_167364


namespace NUMINAMATH_CALUDE_arithmetic_trapezoid_area_l1673_167333

/-- Represents a trapezoid with bases and altitude in arithmetic progression -/
structure ArithmeticTrapezoid where
  b : ℝ  -- altitude
  d : ℝ  -- common difference

/-- The area of an arithmetic trapezoid is b^2 -/
theorem arithmetic_trapezoid_area (t : ArithmeticTrapezoid) : 
  (1 / 2 : ℝ) * ((t.b + t.d) + (t.b - t.d)) * t.b = t.b^2 := by
  sorry

#check arithmetic_trapezoid_area

end NUMINAMATH_CALUDE_arithmetic_trapezoid_area_l1673_167333


namespace NUMINAMATH_CALUDE_four_bb_two_divisible_by_9_l1673_167370

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

def digit_sum (B : ℕ) : ℕ :=
  4 + B + B + 2

theorem four_bb_two_divisible_by_9 (B : ℕ) (h1 : B < 10) :
  is_divisible_by_9 (4000 + 100 * B + 10 * B + 2) ↔ B = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_four_bb_two_divisible_by_9_l1673_167370


namespace NUMINAMATH_CALUDE_kenneth_initial_money_l1673_167332

/-- The amount of money Kenneth had initially -/
def initial_money : ℕ := 50

/-- The number of baguettes Kenneth bought -/
def num_baguettes : ℕ := 2

/-- The cost of each baguette in dollars -/
def cost_baguette : ℕ := 2

/-- The number of water bottles Kenneth bought -/
def num_water : ℕ := 2

/-- The cost of each water bottle in dollars -/
def cost_water : ℕ := 1

/-- The amount of money Kenneth has left after the purchase -/
def money_left : ℕ := 44

/-- Theorem stating that Kenneth's initial money equals $50 -/
theorem kenneth_initial_money :
  initial_money = 
    num_baguettes * cost_baguette + 
    num_water * cost_water + 
    money_left := by sorry

end NUMINAMATH_CALUDE_kenneth_initial_money_l1673_167332


namespace NUMINAMATH_CALUDE_odd_periodic_function_property_l1673_167334

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem odd_periodic_function_property (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : has_period f 5) 
  (h_f1 : f 1 = 1) 
  (h_f2 : f 2 = 2) : 
  f 3 - f 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_property_l1673_167334


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l1673_167380

theorem arithmetic_calculations :
  (24 - (-16) + (-25) - 15 = 0) ∧
  ((-81) + 2.25 * (4/9) / (-16) = -81 - 1/16) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l1673_167380


namespace NUMINAMATH_CALUDE_max_eggs_l1673_167342

theorem max_eggs (x : ℕ) : 
  x < 200 ∧ 
  x % 3 = 2 ∧ 
  x % 4 = 3 ∧ 
  x % 5 = 4 ∧
  (∀ y : ℕ, y < 200 ∧ y % 3 = 2 ∧ y % 4 = 3 ∧ y % 5 = 4 → y ≤ x) →
  x = 179 :=
by sorry

end NUMINAMATH_CALUDE_max_eggs_l1673_167342


namespace NUMINAMATH_CALUDE_min_turns_for_1000_pieces_l1673_167321

/-- Represents the state of the game with black and white pieces on a circumference. -/
structure GameState where
  black : ℕ
  white : ℕ

/-- Represents a player's turn in the game. -/
inductive Turn
  | PlayerA
  | PlayerB

/-- Defines the rules for removing pieces based on the current player's turn. -/
def removePieces (state : GameState) (turn : Turn) : GameState :=
  match turn with
  | Turn.PlayerA => { black := state.black, white := state.white + 2 * state.black }
  | Turn.PlayerB => { black := state.black + 2 * state.white, white := state.white }

/-- Checks if the game has ended (only one color remains). -/
def isGameOver (state : GameState) : Bool :=
  state.black = 0 || state.white = 0

/-- Calculates the minimum number of turns required to end the game. -/
def minTurnsToEnd (initialState : GameState) : ℕ :=
  sorry

/-- Theorem stating that for 1000 initial pieces, the minimum number of turns to end the game is 8. -/
theorem min_turns_for_1000_pieces :
  ∃ (black white : ℕ), black + white = 1000 ∧ minTurnsToEnd { black := black, white := white } = 8 :=
  sorry

end NUMINAMATH_CALUDE_min_turns_for_1000_pieces_l1673_167321


namespace NUMINAMATH_CALUDE_yellow_mugs_count_l1673_167383

/-- Represents the number of mugs of each color in Hannah's collection --/
structure MugCollection where
  red : ℕ
  blue : ℕ
  yellow : ℕ
  other : ℕ

/-- The conditions of Hannah's mug collection --/
def hannahsMugs : MugCollection → Prop
  | m => m.red + m.blue + m.yellow + m.other = 40 ∧
         m.blue = 3 * m.red ∧
         m.red = m.yellow / 2 ∧
         m.other = 4

theorem yellow_mugs_count (m : MugCollection) (h : hannahsMugs m) : m.yellow = 12 := by
  sorry

#check yellow_mugs_count

end NUMINAMATH_CALUDE_yellow_mugs_count_l1673_167383


namespace NUMINAMATH_CALUDE_product_mod_25_l1673_167315

theorem product_mod_25 (n : ℕ) : 
  (105 * 86 * 97 ≡ n [ZMOD 25]) → 
  (0 ≤ n ∧ n < 25) → 
  n = 10 := by
sorry

end NUMINAMATH_CALUDE_product_mod_25_l1673_167315


namespace NUMINAMATH_CALUDE_not_magical_2099_l1673_167300

/-- A year is magical if there exists a month and day such that their sum equals the last two digits of the year. -/
def isMagicalYear (year : ℕ) : Prop :=
  ∃ (month day : ℕ), 
    1 ≤ month ∧ month ≤ 12 ∧
    1 ≤ day ∧ day ≤ 31 ∧
    month + day = year % 100

/-- 2099 is not a magical year. -/
theorem not_magical_2099 : ¬ isMagicalYear 2099 := by
  sorry

#check not_magical_2099

end NUMINAMATH_CALUDE_not_magical_2099_l1673_167300


namespace NUMINAMATH_CALUDE_min_value_of_f_l1673_167343

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + |x - a| + 1

-- State the theorem
theorem min_value_of_f (a : ℝ) : 
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), f a x ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1673_167343


namespace NUMINAMATH_CALUDE_eagles_falcons_games_l1673_167388

theorem eagles_falcons_games (N : ℕ) : 
  (∀ n : ℕ, n < N → (3 + n : ℚ) / (7 + n) < 9/10) ∧ 
  (3 + N : ℚ) / (7 + N) ≥ 9/10 → 
  N = 33 :=
sorry

end NUMINAMATH_CALUDE_eagles_falcons_games_l1673_167388
