import Mathlib

namespace dividend_calculation_l1201_120165

theorem dividend_calculation (dividend quotient remainder divisor : ℕ) 
  (h1 : divisor = 28)
  (h2 : quotient = 7)
  (h3 : remainder = 11)
  (h4 : dividend = divisor * quotient + remainder) :
  dividend = 207 := by
  sorry

end dividend_calculation_l1201_120165


namespace inverse_g_at_negative_seven_sixty_four_l1201_120103

open Real

noncomputable def g (x : ℝ) : ℝ := (x^5 - 1) / 4

theorem inverse_g_at_negative_seven_sixty_four :
  g⁻¹ (-7/64) = (9/16)^(1/5) :=
by sorry

end inverse_g_at_negative_seven_sixty_four_l1201_120103


namespace quadratic_inequality_always_positive_l1201_120170

theorem quadratic_inequality_always_positive (c : ℝ) :
  (∀ x : ℝ, x^2 + x + c > 0) ↔ c > 1/4 := by sorry

end quadratic_inequality_always_positive_l1201_120170


namespace no_fixed_point_for_h_h_condition_l1201_120153

-- Define the function h
def h (x : ℝ) : ℝ := x - 6

-- Theorem statement
theorem no_fixed_point_for_h : ¬ ∃ x : ℝ, h x = x := by
  sorry

-- Condition from the original problem
theorem h_condition (x : ℝ) : h (3 * x + 2) = 3 * x - 4 := by
  sorry

end no_fixed_point_for_h_h_condition_l1201_120153


namespace circle_equation_l1201_120123

/-- The standard equation of a circle with center (h, k) and radius r is (x - h)^2 + (y - k)^2 = r^2 -/
def standard_circle_equation (x y h k r : ℝ) : Prop :=
  (x - h)^2 + (y - k)^2 = r^2

/-- The center of the circle -/
def center : ℝ × ℝ := (-3, 4)

/-- The radius of the circle -/
def radius : ℝ := 2

/-- Theorem: The equation of the circle with center (-3, 4) and radius 2 is (x+3)^2 + (y-4)^2 = 4 -/
theorem circle_equation (x y : ℝ) :
  standard_circle_equation x y center.1 center.2 radius ↔ (x + 3)^2 + (y - 4)^2 = 4 := by
  sorry

end circle_equation_l1201_120123


namespace min_area_ratio_l1201_120127

-- Define the triangles
structure EquilateralTriangle :=
  (A B C : ℝ × ℝ)

structure RightTriangle :=
  (D E F : ℝ × ℝ)

-- Define the conditions
def inscribed (rt : RightTriangle) (et : EquilateralTriangle) : Prop :=
  sorry

def right_angle (rt : RightTriangle) : Prop :=
  sorry

def angle_edf_30 (rt : RightTriangle) : Prop :=
  sorry

-- Define the area ratio
def area_ratio (rt : RightTriangle) (et : EquilateralTriangle) : ℝ :=
  sorry

-- Theorem statement
theorem min_area_ratio 
  (et : EquilateralTriangle) 
  (rt : RightTriangle) 
  (h1 : inscribed rt et) 
  (h2 : right_angle rt) 
  (h3 : angle_edf_30 rt) :
  ∃ (min_ratio : ℝ), 
    (∀ (rt' : RightTriangle), inscribed rt' et → right_angle rt' → angle_edf_30 rt' → 
      area_ratio rt' et ≥ min_ratio) ∧ 
    min_ratio = 3/14 :=
  sorry

end min_area_ratio_l1201_120127


namespace simplify_expression_l1201_120125

theorem simplify_expression : (2^8 + 4^5) * (2^3 - (-2)^3)^2 = 327680 := by
  sorry

end simplify_expression_l1201_120125


namespace pentagon_reassembly_l1201_120159

/-- Given a 10x15 rectangle cut into two congruent pentagons and reassembled into a larger rectangle,
    prove that one-third of the longer side of the new rectangle is 5√2. -/
theorem pentagon_reassembly (original_length original_width : ℝ) 
                            (new_length new_width : ℝ) (y : ℝ) : 
  original_length = 10 →
  original_width = 15 →
  new_length * new_width = original_length * original_width →
  y = new_length / 3 →
  y = 5 * Real.sqrt 2 := by
  sorry


end pentagon_reassembly_l1201_120159


namespace polynomial_roots_sum_l1201_120150

theorem polynomial_roots_sum (p q : ℝ) : 
  (∃ a b c d : ℕ+, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    ∀ x : ℝ, x^4 - 10*x^3 + p*x^2 - q*x + 24 = 0 ↔ (x = a ∨ x = b ∨ x = c ∨ x = d)) →
  p + q = 85 := by
sorry

end polynomial_roots_sum_l1201_120150


namespace frustum_center_height_for_specific_pyramid_l1201_120117

/-- Represents a rectangular pyramid with a parallel cut -/
structure CutPyramid where
  base_length : ℝ
  base_width : ℝ
  height : ℝ
  volume_ratio : ℝ  -- ratio of smaller pyramid to whole pyramid

/-- Calculate the distance from the center of the frustum's circumsphere to the base -/
def frustum_center_height (p : CutPyramid) : ℝ :=
  sorry

/-- The main theorem -/
theorem frustum_center_height_for_specific_pyramid :
  let p : CutPyramid := {
    base_length := 15,
    base_width := 20,
    height := 30,
    volume_ratio := 1/9
  }
  abs (frustum_center_height p - 25.73) < 0.01 := by
  sorry

end frustum_center_height_for_specific_pyramid_l1201_120117


namespace decimal_to_fraction_l1201_120196

theorem decimal_to_fraction : (0.34 : ℚ) = 17 / 50 := by sorry

end decimal_to_fraction_l1201_120196


namespace ellipse_m_value_l1201_120132

/-- Given an ellipse with equation x²/4 + y²/m = 1, foci on the x-axis, 
    and eccentricity 1/2, prove that m = 3 -/
theorem ellipse_m_value (m : ℝ) 
  (h1 : ∀ (x y : ℝ), x^2/4 + y^2/m = 1 → (∃ (a b c : ℝ), a^2 = 4 ∧ b^2 = m ∧ c^2 = a^2 - b^2))
  (h2 : ∃ (e : ℝ), e = 1/2 ∧ e^2 = (4 - m)/4) : 
  m = 3 := by sorry

end ellipse_m_value_l1201_120132


namespace remainder_4015_div_32_l1201_120188

theorem remainder_4015_div_32 : 4015 % 32 = 15 := by
  sorry

end remainder_4015_div_32_l1201_120188


namespace hoseok_english_score_l1201_120162

theorem hoseok_english_score 
  (korean_math_avg : ℝ) 
  (all_subjects_avg : ℝ) 
  (h1 : korean_math_avg = 88)
  (h2 : all_subjects_avg = 90) :
  ∃ (korean math english : ℝ),
    (korean + math) / 2 = korean_math_avg ∧
    (korean + math + english) / 3 = all_subjects_avg ∧
    english = 94 := by
  sorry

end hoseok_english_score_l1201_120162


namespace distance_between_foci_l1201_120156

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop := x^2 / 45 + y^2 / 5 = 9

-- Theorem statement
theorem distance_between_foci :
  ∃ (a b c : ℝ), 
    (∀ x y, ellipse_equation x y ↔ x^2 / a^2 + y^2 / b^2 = 1) ∧
    c^2 = a^2 - b^2 ∧
    2 * c = 12 * Real.sqrt 10 :=
sorry

end distance_between_foci_l1201_120156


namespace prob_three_even_out_of_five_l1201_120143

-- Define a fair 20-sided die
def fair_20_sided_die : Finset ℕ := Finset.range 20

-- Define the probability of rolling an even number on a fair 20-sided die
def prob_even (d : Finset ℕ) : ℚ :=
  (d.filter (λ x => x % 2 = 0)).card / d.card

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of dice we want to show even
def num_even : ℕ := 3

-- Theorem statement
theorem prob_three_even_out_of_five :
  prob_even fair_20_sided_die = 1/2 →
  (num_dice.choose num_even : ℚ) * (1/2)^num_dice = 5/16 := by
  sorry

end prob_three_even_out_of_five_l1201_120143


namespace ten_coin_flips_sequences_l1201_120167

/-- The number of distinct sequences when flipping a coin n times -/
def coin_flip_sequences (n : ℕ) : ℕ := 2^n

/-- Theorem: The number of distinct sequences when flipping a coin 10 times is 1024 -/
theorem ten_coin_flips_sequences : coin_flip_sequences 10 = 1024 := by
  sorry

end ten_coin_flips_sequences_l1201_120167


namespace perpendicular_line_plane_condition_l1201_120124

-- Define the basic types
variable (Point Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_plane_condition
  (m n : Line) (α β : Plane)
  (h1 : parallel m n)
  (h2 : perpendicular_line_plane n β)
  (h3 : parallel_plane α β) :
  perpendicular_line_plane m α :=
sorry

end perpendicular_line_plane_condition_l1201_120124


namespace contrapositive_equivalence_l1201_120129

theorem contrapositive_equivalence (a b : ℝ) :
  (((a + b = 1) → (a^2 + b^2 ≥ 1/2)) ↔ ((a^2 + b^2 < 1/2) → (a + b ≠ 1))) :=
by sorry

end contrapositive_equivalence_l1201_120129


namespace barium_oxide_moles_l1201_120169

-- Define the chemical reaction
structure Reaction where
  bao : ℝ    -- moles of Barium oxide
  h2o : ℝ    -- moles of Water
  baoh2 : ℝ  -- moles of Barium hydroxide

-- Define the reaction conditions
def reaction_conditions (r : Reaction) : Prop :=
  r.h2o = 1 ∧ r.baoh2 = r.bao

-- Theorem statement
theorem barium_oxide_moles (e : ℝ) :
  ∀ r : Reaction, reaction_conditions r → r.baoh2 = e → r.bao = e :=
by
  sorry

end barium_oxide_moles_l1201_120169


namespace divisibility_equivalence_l1201_120155

theorem divisibility_equivalence (n m : ℤ) : 
  (∃ k : ℤ, 2*n + 5*m = 9*k) ↔ (∃ l : ℤ, 5*n + 8*m = 9*l) := by
  sorry

end divisibility_equivalence_l1201_120155


namespace line_segment_point_sum_l1201_120175

/-- Given a line y = -2/3x + 6 that crosses the x-axis at P and y-axis at Q,
    and a point T(r, s) on line segment PQ, prove that if the area of triangle POQ
    is four times the area of triangle TOP, then r + s = 8.25. -/
theorem line_segment_point_sum (r s : ℝ) : 
  let line := fun (x : ℝ) ↦ -2/3 * x + 6
  let P := (9, 0)
  let Q := (0, 6)
  let T := (r, s)
  (T.1 ≥ 0 ∧ T.1 ≤ 9) →  -- T is on line segment PQ
  (T.2 = line T.1) →  -- T is on the line
  (1/2 * 9 * 6 = 4 * (1/2 * 9 * s)) →  -- Area condition
  r + s = 8.25 :=
by sorry

end line_segment_point_sum_l1201_120175


namespace complex_fraction_simplification_l1201_120171

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (2 - 2 * i) / (1 + 4 * i) = -6 / 17 - (10 / 17) * i :=
by
  -- The proof would go here, but we'll skip it
  sorry

end complex_fraction_simplification_l1201_120171


namespace angie_tax_payment_l1201_120112

/-- Represents Angie's monthly finances -/
structure AngieFinances where
  salary : ℕ
  necessities : ℕ
  leftOver : ℕ

/-- Calculates Angie's tax payment based on her finances -/
def taxPayment (finances : AngieFinances) : ℕ :=
  finances.salary - finances.necessities - finances.leftOver

/-- Theorem stating that Angie's tax payment is $20 given her financial situation -/
theorem angie_tax_payment :
  let finances : AngieFinances := { salary := 80, necessities := 42, leftOver := 18 }
  taxPayment finances = 20 := by
  sorry


end angie_tax_payment_l1201_120112


namespace watermelon_slices_l1201_120136

/-- The number of slices in a watermelon, given the number of seeds per slice and the total number of seeds. -/
def number_of_slices (black_seeds_per_slice : ℕ) (white_seeds_per_slice : ℕ) (total_seeds : ℕ) : ℕ :=
  total_seeds / (black_seeds_per_slice + white_seeds_per_slice)

/-- Theorem stating that the number of slices is 40 given the conditions of the problem. -/
theorem watermelon_slices :
  number_of_slices 20 20 1600 = 40 := by
  sorry

end watermelon_slices_l1201_120136


namespace linear_function_passes_through_point_l1201_120164

/-- The linear function f(x) = -2x - 6 passes through the point (-4, 2) -/
theorem linear_function_passes_through_point :
  let f : ℝ → ℝ := λ x => -2 * x - 6
  f (-4) = 2 := by sorry

end linear_function_passes_through_point_l1201_120164


namespace frequency_calculation_l1201_120180

theorem frequency_calculation (sample_size : ℕ) (area_percentage : ℚ) (h1 : sample_size = 50) (h2 : area_percentage = 16/100) :
  (sample_size : ℚ) * area_percentage = 8 := by
  sorry

end frequency_calculation_l1201_120180


namespace fraction_equality_l1201_120190

theorem fraction_equality (x y : ℝ) (h : x / y = 5 / 3) :
  x / (y - x) = -3 / 2 ∧ x / (y - x) ≠ 5 / 2 := by
  sorry

end fraction_equality_l1201_120190


namespace equivalent_statements_l1201_120185

variable (P Q : Prop)

theorem equivalent_statements :
  ((P → Q) ↔ (¬Q → ¬P)) ∧ ((P → Q) ↔ (¬P ∨ Q)) :=
sorry

end equivalent_statements_l1201_120185


namespace elijah_masking_tape_l1201_120151

/-- The amount of masking tape needed for Elijah's room -/
def masking_tape_needed (narrow_wall_width : ℕ) (wide_wall_width : ℕ) : ℕ :=
  2 * narrow_wall_width + 2 * wide_wall_width

/-- Theorem: The amount of masking tape needed for Elijah's room is 20 meters -/
theorem elijah_masking_tape : masking_tape_needed 4 6 = 20 := by
  sorry

end elijah_masking_tape_l1201_120151


namespace expression_evaluation_l1201_120109

theorem expression_evaluation : 
  3 + 2 * Real.sqrt 3 + (3 + 2 * Real.sqrt 3)⁻¹ + (2 * Real.sqrt 3 - 3)⁻¹ = 3 + (10 * Real.sqrt 3) / 3 :=
by sorry

end expression_evaluation_l1201_120109


namespace exactly_one_correct_proposition_l1201_120177

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel relation between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the parallel relation between lines
variable (parallel_line_line : Line → Line → Prop)

-- Define the perpendicular relation between lines and planes
variable (perpendicular_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perpendicular_line_line : Line → Line → Prop)

-- Define the subset relation for a line in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the "not subset" relation for a line not in a plane
variable (line_not_in_plane : Line → Plane → Prop)

theorem exactly_one_correct_proposition (a b : Line) (M : Plane) : 
  (∃! i : Fin 4, 
    (i = 0 → (parallel_line_plane a M ∧ parallel_line_plane b M → parallel_line_line a b)) ∧
    (i = 1 → (line_in_plane b M ∧ line_not_in_plane a M ∧ parallel_line_line a b → parallel_line_plane a M)) ∧
    (i = 2 → (perpendicular_line_line a b ∧ line_in_plane b M → perpendicular_line_plane a M)) ∧
    (i = 3 → (perpendicular_line_plane a M ∧ perpendicular_line_line a b → parallel_line_plane b M))) :=
by sorry

end exactly_one_correct_proposition_l1201_120177


namespace linear_inequality_solution_l1201_120119

theorem linear_inequality_solution (x : ℝ) : (2 * x - 1 ≥ 3) ↔ (x ≥ 2) := by
  sorry

end linear_inequality_solution_l1201_120119


namespace cylinder_surface_area_l1201_120138

/-- The surface area of a cylinder given its unfolded lateral surface dimensions -/
theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0)
  (h_is_6pi : h = 6 * Real.pi) (w_is_4pi : w = 4 * Real.pi) :
  let r := min (h / (2 * Real.pi)) (w / (2 * Real.pi))
  let surface_area := h * w + 2 * Real.pi * r^2
  surface_area = 24 * Real.pi^2 + 18 * Real.pi ∨
  surface_area = 24 * Real.pi^2 + 8 * Real.pi :=
sorry

end cylinder_surface_area_l1201_120138


namespace min_value_problem_l1201_120140

theorem min_value_problem (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 := by
  sorry

end min_value_problem_l1201_120140


namespace unique_sequence_coefficients_l1201_120148

/-- Given two distinct roots of a characteristic equation and two initial terms of a sequence,
    there exists a unique pair of coefficients that generates the entire sequence. -/
theorem unique_sequence_coefficients
  (x₁ x₂ : ℝ) (a₀ a₁ : ℝ) (h : x₁ ≠ x₂) :
  ∃! (c₁ c₂ : ℝ), ∀ (n : ℕ), c₁ * x₁^n + c₂ * x₂^n = 
    if n = 0 then a₀ else if n = 1 then a₁ else c₁ * x₁^n + c₂ * x₂^n :=
sorry

end unique_sequence_coefficients_l1201_120148


namespace sequence_general_term_l1201_120160

theorem sequence_general_term (a : ℕ → ℝ) :
  (a 1 = 1) →
  (∀ n : ℕ, n > 1 → a n = 2 * a (n - 1) + 1) →
  (∀ n : ℕ, n > 0 → a n = 2^n - 1) :=
by sorry

end sequence_general_term_l1201_120160


namespace positive_real_inequality_l1201_120173

theorem positive_real_inequality (x : ℝ) (h : x > 0) : x + 1/x ≥ 2 := by
  sorry

end positive_real_inequality_l1201_120173


namespace fraction_division_addition_l1201_120172

theorem fraction_division_addition : (3 / 7 : ℚ) / 4 + 1 / 2 = 17 / 28 := by
  sorry

end fraction_division_addition_l1201_120172


namespace number_problem_l1201_120195

theorem number_problem (x : ℝ) (h : x - (3/4) * x = 100) : (1/4) * x = 100 := by
  sorry

end number_problem_l1201_120195


namespace sock_probability_theorem_l1201_120147

/-- Represents the number of pairs of socks for each color -/
structure SockPairs :=
  (blue : ℕ)
  (red : ℕ)
  (green : ℕ)

/-- Calculates the probability of picking two socks of the same color -/
def probabilitySameColor (pairs : SockPairs) : ℚ :=
  let totalSocks := 2 * (pairs.blue + pairs.red + pairs.green)
  let blueProbability := (2 * pairs.blue * (2 * pairs.blue - 1)) / (totalSocks * (totalSocks - 1))
  let redProbability := (2 * pairs.red * (2 * pairs.red - 1)) / (totalSocks * (totalSocks - 1))
  let greenProbability := (2 * pairs.green * (2 * pairs.green - 1)) / (totalSocks * (totalSocks - 1))
  blueProbability + redProbability + greenProbability

/-- Theorem: The probability of picking two socks of the same color is 77/189 -/
theorem sock_probability_theorem (pairs : SockPairs) 
  (h1 : pairs.blue = 8) 
  (h2 : pairs.red = 4) 
  (h3 : pairs.green = 2) : 
  probabilitySameColor pairs = 77 / 189 := by
  sorry

#eval probabilitySameColor { blue := 8, red := 4, green := 2 }

end sock_probability_theorem_l1201_120147


namespace hydras_always_live_l1201_120114

/-- Represents the number of new heads a hydra can grow in a week -/
inductive NewHeads
  | five : NewHeads
  | seven : NewHeads

/-- The state of the hydras after a certain number of weeks -/
structure HydraState where
  weeks : ℕ
  totalHeads : ℕ

/-- The initial state of the hydras -/
def initialState : HydraState :=
  { weeks := 0, totalHeads := 2016 + 2017 }

/-- The change in total heads after one week -/
def weeklyChange (a b : NewHeads) : ℕ :=
  match a, b with
  | NewHeads.five, NewHeads.five => 6
  | NewHeads.five, NewHeads.seven => 8
  | NewHeads.seven, NewHeads.five => 8
  | NewHeads.seven, NewHeads.seven => 10

/-- The state transition function -/
def nextState (state : HydraState) (a b : NewHeads) : HydraState :=
  { weeks := state.weeks + 1
  , totalHeads := state.totalHeads + weeklyChange a b }

theorem hydras_always_live :
  ∀ (state : HydraState), state.totalHeads % 2 = 1 →
    ∀ (a b : NewHeads), (nextState state a b).totalHeads % 2 = 1 :=
sorry

end hydras_always_live_l1201_120114


namespace ferris_wheel_capacity_l1201_120128

theorem ferris_wheel_capacity (total_capacity : ℕ) (num_seats : ℕ) (people_per_seat : ℕ) 
  (h1 : total_capacity = 4)
  (h2 : num_seats = 2)
  (h3 : people_per_seat * num_seats = total_capacity) :
  people_per_seat = 2 := by
sorry

end ferris_wheel_capacity_l1201_120128


namespace distribute_six_balls_four_boxes_l1201_120113

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 65 ways to distribute 6 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_six_balls_four_boxes : distribute_balls 6 4 = 65 := by sorry

end distribute_six_balls_four_boxes_l1201_120113


namespace consecutive_integers_sum_l1201_120107

theorem consecutive_integers_sum (n : ℕ) (h : n > 0) :
  (6 * n + 15 = 2013) → (n + 5 = 338) := by
  sorry

end consecutive_integers_sum_l1201_120107


namespace exactly_fifteen_numbers_l1201_120174

/-- Represents a three-digit positive integer in base 10 -/
def ThreeDigitInteger (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : ℕ :=
  sorry

/-- Converts a natural number to its base-8 representation -/
def toBase8 (n : ℕ) : ℕ :=
  sorry

/-- Checks if the two rightmost digits of two numbers are the same -/
def sameLastTwoDigits (a b : ℕ) : Prop :=
  a % 100 = b % 100

/-- The main theorem stating that there are exactly 15 numbers satisfying the condition -/
theorem exactly_fifteen_numbers :
  ∃! (s : Finset ℕ),
    Finset.card s = 15 ∧
    ∀ n, n ∈ s ↔ 
      ThreeDigitInteger n ∧
      sameLastTwoDigits (toBase7 n * toBase8 n) (3 * n) :=
  sorry


end exactly_fifteen_numbers_l1201_120174


namespace geometric_sequence_common_ratio_l1201_120108

/-- An increasing geometric sequence -/
def IsIncreasingGeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a (n + 1) = a n * q) ∧ (q > 1) ∧ (a 1 > 0)

/-- The theorem stating the common ratio of the geometric sequence -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) (q : ℝ) 
  (h_geom : IsIncreasingGeometricSequence a q) 
  (h_sum : a 1 + a 4 = 9) 
  (h_prod : a 2 * a 3 = 8) : 
  q = 2 := by
sorry

end geometric_sequence_common_ratio_l1201_120108


namespace p_or_q_necessary_not_sufficient_l1201_120115

theorem p_or_q_necessary_not_sufficient (p q : Prop) :
  (¬¬p → (p ∨ q)) ∧ ¬((p ∨ q) → ¬¬p) := by
  sorry

end p_or_q_necessary_not_sufficient_l1201_120115


namespace service_center_location_l1201_120131

/-- Represents a highway with exits and a service center -/
structure Highway where
  third_exit : ℝ
  seventh_exit : ℝ
  twelfth_exit : ℝ
  service_center : ℝ

/-- Theorem stating the location of the service center on the highway -/
theorem service_center_location (h : Highway) 
  (h_third : h.third_exit = 30)
  (h_seventh : h.seventh_exit = 90)
  (h_twelfth : h.twelfth_exit = 195)
  (h_service : h.service_center = h.third_exit + 2/3 * (h.seventh_exit - h.third_exit)) :
  h.service_center = 70 := by
  sorry

#check service_center_location

end service_center_location_l1201_120131


namespace pen_difference_after_four_weeks_l1201_120144

/-- The difference in pens between Alex and Jane after 4 weeks -/
def pen_difference (A B : ℕ) (X Y : ℝ) (M N : ℕ) : ℕ :=
  M - N

/-- Theorem stating the difference in pens after 4 weeks -/
theorem pen_difference_after_four_weeks 
  (A B : ℕ) (X Y : ℝ) (M N : ℕ) 
  (hM : M = A * X^4) 
  (hN : N = B * Y^4) :
  pen_difference A B X Y M N = M - N :=
by
  sorry

end pen_difference_after_four_weeks_l1201_120144


namespace figure_area_l1201_120152

/-- The area of a rectangle given its width and height -/
def rectangle_area (width : ℕ) (height : ℕ) : ℕ := width * height

/-- The total area of three rectangles -/
def total_area (rect1_width rect1_height rect2_width rect2_height rect3_width rect3_height : ℕ) : ℕ :=
  rectangle_area rect1_width rect1_height +
  rectangle_area rect2_width rect2_height +
  rectangle_area rect3_width rect3_height

/-- Theorem: The total area of the figure is 71 square units -/
theorem figure_area :
  total_area 7 7 3 2 4 4 = 71 := by
  sorry

end figure_area_l1201_120152


namespace sum_of_numbers_l1201_120134

theorem sum_of_numbers : 4321 + 3214 + 2143 - 1432 = 8246 := by
  sorry

end sum_of_numbers_l1201_120134


namespace pond_eyes_count_l1201_120139

/-- The number of eyes for each frog -/
def frog_eyes : ℕ := 2

/-- The number of eyes for each crocodile -/
def crocodile_eyes : ℕ := 2

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 20

/-- The number of crocodiles in the pond -/
def num_crocodiles : ℕ := 10

/-- The total number of animal eyes in the pond -/
def total_eyes : ℕ := num_frogs * frog_eyes + num_crocodiles * crocodile_eyes

theorem pond_eyes_count : total_eyes = 60 := by
  sorry

end pond_eyes_count_l1201_120139


namespace min_value_F_l1201_120163

/-- The function F(x, y) -/
def F (x y : ℝ) : ℝ := 6*y + 8*x - 9

/-- The constraint equation -/
def constraint (x y : ℝ) : Prop := x^2 + y^2 + 25 = 10*(x + y)

/-- Theorem stating that the minimum value of F(x, y) is 11 given the constraint -/
theorem min_value_F :
  ∃ (min : ℝ), min = 11 ∧
  (∀ x y : ℝ, constraint x y → F x y ≥ min) ∧
  (∃ x y : ℝ, constraint x y ∧ F x y = min) :=
sorry

end min_value_F_l1201_120163


namespace isosceles_triangle_perimeter_perimeter_is_22_l1201_120116

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22 -/
theorem isosceles_triangle_perimeter : ℝ → Prop :=
  fun perimeter =>
    ∀ a b c : ℝ,
      a = 9 ∧ b = 9 ∧ c = 4 →
      a + b > c ∧ b + c > a ∧ c + a > b →  -- Triangle inequality
      a = b →  -- Isosceles condition
      perimeter = a + b + c

/-- The perimeter of the isosceles triangle is 22 -/
theorem perimeter_is_22 : isosceles_triangle_perimeter 22 := by
  sorry

end isosceles_triangle_perimeter_perimeter_is_22_l1201_120116


namespace remaining_cheese_calories_l1201_120101

/-- Calculates the remaining calories in a block of cheese after a portion is removed -/
theorem remaining_cheese_calories (length width height : ℝ) 
  (calorie_density : ℝ) (eaten_side_length : ℝ) : 
  length = 4 → width = 8 → height = 2 → calorie_density = 110 → eaten_side_length = 2 →
  (length * width * height - eaten_side_length ^ 3) * calorie_density = 6160 := by
  sorry

#check remaining_cheese_calories

end remaining_cheese_calories_l1201_120101


namespace nickels_to_dimes_ratio_l1201_120176

/-- Represents the number of coins of each type in Tommy's collection -/
structure CoinCollection where
  pennies : ℕ
  nickels : ℕ
  dimes : ℕ
  quarters : ℕ

/-- Defines Tommy's coin collection based on the given conditions -/
def tommys_collection : CoinCollection where
  pennies := 40
  nickels := 100
  dimes := 50
  quarters := 4

/-- Theorem stating the ratio of nickels to dimes in Tommy's collection -/
theorem nickels_to_dimes_ratio (c : CoinCollection) 
  (h1 : c.dimes = c.pennies + 10)
  (h2 : c.quarters = 4)
  (h3 : c.pennies = 10 * c.quarters)
  (h4 : c.nickels = 100) :
  c.nickels / c.dimes = 2 := by
  sorry

#check nickels_to_dimes_ratio tommys_collection

end nickels_to_dimes_ratio_l1201_120176


namespace expression_simplification_l1201_120181

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (2 * a^2 - 9 * b^2) / (3 * a * b) - (6 * a * b - 9 * b^2) / (4 * a * b - 3 * a^2) =
  2 * (a^2 - 9 * b^2) / (3 * a * b) :=
by sorry

end expression_simplification_l1201_120181


namespace min_value_theorem_l1201_120120

theorem min_value_theorem (m n : ℝ) (h1 : 2 * n + m = 4) (h2 : m > 0) (h3 : n > 0) :
  2 / m + 1 / n ≥ 2 := by
  sorry

end min_value_theorem_l1201_120120


namespace cos_three_pi_fourth_plus_two_alpha_l1201_120166

theorem cos_three_pi_fourth_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π / 8 - α) = 1 / 6) : 
  Real.cos (3 * π / 4 + 2 * α) = 17 / 18 := by
  sorry

end cos_three_pi_fourth_plus_two_alpha_l1201_120166


namespace monomial_degree_l1201_120186

/-- Given that (a-2)x^2y^(|a|+1) is a monomial of degree 5 in x and y, and (a-2) ≠ 0, prove that a = -2 -/
theorem monomial_degree (a : ℤ) : 
  (∃ (x y : ℝ), (a - 2) * x^2 * y^(|a| + 1) ≠ 0) →  -- (a-2)x^2y^(|a|+1) is a monomial
  (2 + |a| + 1 = 5) →  -- The degree of the monomial in x and y is 5
  (a - 2 ≠ 0) →  -- (a-2) ≠ 0
  a = -2 := by
sorry


end monomial_degree_l1201_120186


namespace money_ratio_problem_l1201_120130

/-- Given the ratio of money between Ravi and Giri, and the amounts of money
    Ravi and Kiran have, prove the ratio of money between Giri and Kiran. -/
theorem money_ratio_problem (ravi_giri_ratio : ℚ) (ravi_money kiran_money : ℕ) :
  ravi_giri_ratio = 6 / 7 →
  ravi_money = 36 →
  kiran_money = 105 →
  ∃ (giri_money : ℕ), 
    (ravi_money : ℚ) / giri_money = ravi_giri_ratio ∧
    (giri_money : ℚ) / kiran_money = 2 / 5 :=
by sorry

end money_ratio_problem_l1201_120130


namespace distance_foci_to_asymptotes_l1201_120102

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the foci
def foci : Set (ℝ × ℝ) := {(5, 0), (-5, 0)}

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := (3 * x - 4 * y = 0) ∨ (3 * x + 4 * y = 0)

-- Theorem statement
theorem distance_foci_to_asymptotes :
  ∀ (f : ℝ × ℝ) (x y : ℝ),
  f ∈ foci → asymptotes x y →
  ∃ (d : ℝ), d = 3 ∧ d = |3 * f.1 + 4 * f.2| / Real.sqrt 25 :=
sorry

end distance_foci_to_asymptotes_l1201_120102


namespace total_groups_is_1026_l1201_120158

/-- The number of boys in the class -/
def num_boys : ℕ := 9

/-- The number of girls in the class -/
def num_girls : ℕ := 12

/-- The size of each group -/
def group_size : ℕ := 3

/-- Calculate the number of combinations of n items taken k at a time -/
def combinations (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.choose n k

/-- Calculate the number of groups with 2 boys and 1 girl -/
def groups_2boys1girl : ℕ :=
  combinations num_boys 2 * combinations num_girls 1

/-- Calculate the number of groups with 2 girls and 1 boy -/
def groups_2girls1boy : ℕ :=
  combinations num_girls 2 * combinations num_boys 1

/-- The total number of possible groups -/
def total_groups : ℕ :=
  groups_2boys1girl + groups_2girls1boy

/-- Theorem stating that the total number of possible groups is 1026 -/
theorem total_groups_is_1026 : total_groups = 1026 := by
  sorry

end total_groups_is_1026_l1201_120158


namespace notebooks_distribution_l1201_120193

theorem notebooks_distribution (C : ℕ) (N : ℕ) : 
  (N / C = C / 8) →  -- Condition 1
  (N / (C / 2) = 16) →  -- Condition 2
  N = 512 := by sorry

end notebooks_distribution_l1201_120193


namespace unique_a_value_l1201_120126

-- Define the sets M and N as functions of a
def M (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def N (a : ℝ) : Set ℝ := {-1, a, 3}

-- State the theorem
theorem unique_a_value : ∃! a : ℝ, (M a ∩ N a = {3} ∧ a ≠ -1) := by
  sorry

end unique_a_value_l1201_120126


namespace midpoint_coordinate_sum_l1201_120161

/-- Given a line segment CD with midpoint M(5,4) and one endpoint C(7,-2),
    the sum of the coordinates of the other endpoint D is 13. -/
theorem midpoint_coordinate_sum :
  ∀ (D : ℝ × ℝ),
  (5, 4) = ((7, -2) + D) / 2 →
  D.1 + D.2 = 13 :=
by sorry

end midpoint_coordinate_sum_l1201_120161


namespace a_formula_T_formula_l1201_120189

-- Define the arithmetic sequence a_n
def a (n : ℕ) : ℚ := sorry

-- Define the sum of the first n terms
def S (n : ℕ) : ℚ := sorry

-- Define the conditions
axiom S3_eq_0 : S 3 = 0
axiom S5_eq_neg5 : S 5 = -5

-- Theorem 1: General formula for a_n
theorem a_formula (n : ℕ) : a n = -n + 2 := sorry

-- Define the sequence 1 / (a_{2n-1} * a_{2n+1})
def b (n : ℕ) : ℚ := 1 / (a (2*n - 1) * a (2*n + 1))

-- Define the sum of the first n terms of b
def T (n : ℕ) : ℚ := sorry

-- Theorem 2: Sum of the first n terms of b
theorem T_formula (n : ℕ) : T n = n / (1 - 2*n) := sorry

end a_formula_T_formula_l1201_120189


namespace sum_first_last_33_l1201_120118

/-- A sequence of ten terms -/
def Sequence := Fin 10 → ℕ

/-- The property that C (the third term) is 7 -/
def third_is_seven (s : Sequence) : Prop := s 2 = 7

/-- The property that the sum of any three consecutive terms is 40 -/
def consecutive_sum_40 (s : Sequence) : Prop :=
  ∀ i, i < 8 → s i + s (i + 1) + s (i + 2) = 40

/-- The main theorem: If C is 7 and the sum of any three consecutive terms is 40,
    then A + J = 33 -/
theorem sum_first_last_33 (s : Sequence) 
  (h1 : third_is_seven s) (h2 : consecutive_sum_40 s) : s 0 + s 9 = 33 := by
  sorry

end sum_first_last_33_l1201_120118


namespace same_solution_implies_b_value_l1201_120106

theorem same_solution_implies_b_value :
  ∀ (x b : ℚ),
  (3 * x + 9 = 0) ∧ (b * x + 15 = 5) →
  b = 10 / 3 := by
sorry

end same_solution_implies_b_value_l1201_120106


namespace first_player_wins_l1201_120142

/-- Represents the game state with k piles of stones -/
structure GameState where
  k : ℕ+
  n : Fin k → ℕ

/-- Defines the set of winning positions -/
def WinningPositions : Set GameState :=
  sorry

/-- Defines a valid move in the game -/
def ValidMove (s₁ s₂ : GameState) : Prop :=
  sorry

/-- Theorem stating the winning condition for the first player -/
theorem first_player_wins (s : GameState) :
  s ∈ WinningPositions ↔
    ∃ (s' : GameState), ValidMove s s' ∧ 
      ∀ (s'' : GameState), ValidMove s' s'' → s'' ∈ WinningPositions :=
by sorry

end first_player_wins_l1201_120142


namespace second_discount_percentage_l1201_120105

theorem second_discount_percentage
  (initial_price : ℝ)
  (first_discount : ℝ)
  (final_price : ℝ)
  (h1 : initial_price = 400)
  (h2 : first_discount = 25)
  (h3 : final_price = 240)
  : ∃ (second_discount : ℝ),
    final_price = initial_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 20 := by
  sorry

end second_discount_percentage_l1201_120105


namespace two_plus_three_equals_eight_is_proposition_l1201_120145

/-- A statement is a proposition if it can be judged as either true or false. -/
def is_proposition (statement : Prop) : Prop :=
  (statement ∨ ¬statement) ∧ ¬(statement ∧ ¬statement)

/-- The statement "2 + 3 = 8" is a proposition. -/
theorem two_plus_three_equals_eight_is_proposition :
  is_proposition (2 + 3 = 8) := by
  sorry

end two_plus_three_equals_eight_is_proposition_l1201_120145


namespace negation_of_universal_proposition_l1201_120104

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, 2 * x^2 + 1 > 0)) ↔ (∃ x : ℝ, 2 * x^2 + 1 ≤ 0) := by
  sorry

end negation_of_universal_proposition_l1201_120104


namespace parallelepiped_diagonal_bounds_l1201_120149

/-- Regular tetrahedron with side length and height 1 -/
structure Tetrahedron where
  side_length : ℝ
  height : ℝ
  is_regular : side_length = 1 ∧ height = 1

/-- Rectangular parallelepiped inscribed in the tetrahedron -/
structure Parallelepiped (t : Tetrahedron) where
  base_area : ℝ
  base_in_tetrahedron_base : Prop
  opposite_vertex_on_lateral_surface : Prop
  diagonal : ℝ

/-- Theorem stating the bounds of the parallelepiped's diagonal -/
theorem parallelepiped_diagonal_bounds (t : Tetrahedron) (p : Parallelepiped t) :
  (0 < p.base_area ∧ p.base_area ≤ 1/18 →
    Real.sqrt (2/3 - 2*p.base_area) ≤ p.diagonal ∧ p.diagonal < Real.sqrt (2 - 2*p.base_area)) ∧
  ((7 + 2*Real.sqrt 6)/25 ≤ p.base_area ∧ p.base_area < 1/2 →
    Real.sqrt (1 - 2*Real.sqrt (2*p.base_area) + 4*p.base_area) ≤ p.diagonal ∧
    p.diagonal < Real.sqrt (1 - 2*Real.sqrt p.base_area + 3*p.base_area)) ∧
  (1/2 ≤ p.base_area ∧ p.base_area < 1 →
    Real.sqrt (2*p.base_area) < p.diagonal ∧
    p.diagonal ≤ Real.sqrt (1 - 2*Real.sqrt p.base_area + 3*p.base_area)) := by
  sorry

end parallelepiped_diagonal_bounds_l1201_120149


namespace roosters_count_l1201_120110

theorem roosters_count (total_chickens egg_laying_hens non_egg_laying_hens : ℕ) 
  (h1 : total_chickens = 325)
  (h2 : egg_laying_hens = 277)
  (h3 : non_egg_laying_hens = 20) :
  total_chickens - (egg_laying_hens + non_egg_laying_hens) = 28 :=
by sorry

end roosters_count_l1201_120110


namespace unique_a_value_l1201_120191

def A (a : ℝ) : Set ℝ := {1, a}
def B : Set ℝ := {1, 3}

theorem unique_a_value (a : ℝ) (h : A a ∪ B = {1, 2, 3}) : a = 2 := by
  sorry

end unique_a_value_l1201_120191


namespace ages_sum_l1201_120197

theorem ages_sum (a b c : ℕ) : 
  a = 16 + b + c → 
  a^2 = 1632 + (b + c)^2 → 
  a + b + c = 102 := by
sorry

end ages_sum_l1201_120197


namespace thousand_chime_date_l1201_120178

/-- Represents a date --/
structure Date :=
  (year : Nat)
  (month : Nat)
  (day : Nat)

/-- Represents a time --/
structure Time :=
  (hour : Nat)
  (minute : Nat)

/-- Represents the chiming pattern of the clock --/
def clockChime (hour : Nat) (minute : Nat) : Nat :=
  if minute == 30 then 1
  else if minute == 0 then (if hour == 0 || hour == 12 then 12 else hour)
  else 0

/-- Calculates the number of chimes from a given start date and time to a given end date and time --/
def countChimes (startDate : Date) (startTime : Time) (endDate : Date) (endTime : Time) : Nat :=
  sorry -- Implementation details omitted

/-- The theorem to be proved --/
theorem thousand_chime_date :
  let startDate := Date.mk 2003 2 26
  let startTime := Time.mk 10 15
  let endDate := Date.mk 2003 3 7
  countChimes startDate startTime endDate (Time.mk 23 59) ≥ 1000 ∧
  ∀ (d : Date), d.year == 2003 ∧ d.month == 3 ∧ d.day < 7 →
    countChimes startDate startTime d (Time.mk 23 59) < 1000 :=
by sorry

end thousand_chime_date_l1201_120178


namespace parabola_focus_hyperbola_vertex_asymptote_distance_l1201_120179

-- Define the parabola
def parabola (a : ℝ) (x y : ℝ) : Prop := x = a * y^2 ∧ a ≠ 0

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 12 - y^2 / 4 = 1

-- Theorem for the focus of the parabola
theorem parabola_focus (a : ℝ) :
  ∃ (x y : ℝ), parabola a x y → (x = 1 / (4 * a) ∧ y = 0) :=
sorry

-- Theorem for the distance from vertex to asymptote of the hyperbola
theorem hyperbola_vertex_asymptote_distance :
  ∃ (d : ℝ), (∀ x y : ℝ, hyperbola x y → d = Real.sqrt 30 / 5) :=
sorry

end parabola_focus_hyperbola_vertex_asymptote_distance_l1201_120179


namespace algebraic_expression_value_l1201_120192

theorem algebraic_expression_value (x y : ℝ) 
  (h1 : x * y = -2) 
  (h2 : x + y = 4) : 
  x^2 * y + x * y^2 = -8 := by
sorry

end algebraic_expression_value_l1201_120192


namespace worker_speed_reduction_l1201_120122

theorem worker_speed_reduction (usual_time : ℕ) (delay : ℕ) : 
  usual_time = 60 → delay = 12 → 
  (usual_time : ℚ) / (usual_time + delay) = 5 / 6 := by sorry

end worker_speed_reduction_l1201_120122


namespace hyperbola_eccentricity_l1201_120182

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- The x-coordinate of one focus -/
  focus_x : ℝ
  /-- The y-coordinate of one focus -/
  focus_y : ℝ
  /-- The slope of one asymptote -/
  asymptote_slope : ℝ

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ :=
  sorry

/-- Theorem: The eccentricity of a hyperbola with one focus at (5,0) and one asymptote with slope 3/4 is 5/4 -/
theorem hyperbola_eccentricity :
  let h : Hyperbola := { focus_x := 5, focus_y := 0, asymptote_slope := 3/4 }
  eccentricity h = 5/4 := by
  sorry

end hyperbola_eccentricity_l1201_120182


namespace alex_coin_distribution_l1201_120135

/-- The minimum number of additional coins needed -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  (friends * (friends + 1)) / 2 - initial_coins

/-- Theorem stating the minimum number of additional coins needed for Alex -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ) 
  (h1 : friends = 15) (h2 : initial_coins = 100) : 
  min_additional_coins friends initial_coins = 20 := by
  sorry

end alex_coin_distribution_l1201_120135


namespace complete_square_constant_l1201_120183

theorem complete_square_constant (a h k : ℚ) :
  (∀ x, x^2 - 7*x = a*(x - h)^2 + k) →
  k = -49/4 := by
sorry

end complete_square_constant_l1201_120183


namespace supermarket_spending_l1201_120141

theorem supermarket_spending (total : ℚ) : 
  (1/2 : ℚ) * total + (1/3 : ℚ) * total + (1/10 : ℚ) * total + 5 = total →
  total = 75 := by
sorry

end supermarket_spending_l1201_120141


namespace interval_representation_l1201_120100

def open_closed_interval (a b : ℝ) : Set ℝ := {x | a < x ∧ x ≤ b}

theorem interval_representation :
  open_closed_interval (-3) 2 = {x : ℝ | -3 < x ∧ x ≤ 2} := by sorry

end interval_representation_l1201_120100


namespace max_servings_emily_l1201_120184

/-- Represents the recipe requirements for 4 servings -/
structure Recipe :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (water : ℚ)
  (milk : ℚ)

/-- Represents Emily's available ingredients -/
structure Available :=
  (chocolate : ℚ)
  (sugar : ℚ)
  (milk : ℚ)

def recipe : Recipe :=
  { chocolate := 3
  , sugar := 1/2
  , water := 2
  , milk := 3 }

def emily : Available :=
  { chocolate := 9
  , sugar := 3
  , milk := 10 }

/-- Calculates the number of servings possible for a given ingredient -/
def servings_for_ingredient (recipe_amount : ℚ) (available_amount : ℚ) : ℚ :=
  (available_amount / recipe_amount) * 4

theorem max_servings_emily :
  let chocolate_servings := servings_for_ingredient recipe.chocolate emily.chocolate
  let sugar_servings := servings_for_ingredient recipe.sugar emily.sugar
  let milk_servings := servings_for_ingredient recipe.milk emily.milk
  min chocolate_servings (min sugar_servings milk_servings) = 12 := by
  sorry

end max_servings_emily_l1201_120184


namespace final_eraser_count_l1201_120121

def initial_erasers : Float := 95.0
def bought_erasers : Float := 42.0

theorem final_eraser_count :
  initial_erasers + bought_erasers = 137.0 := by
  sorry

end final_eraser_count_l1201_120121


namespace namjoon_candies_l1201_120154

/-- The number of candies Namjoon gave to Yoongi -/
def candies_given : ℕ := 18

/-- The number of candies left over -/
def candies_left : ℕ := 16

/-- The total number of candies Namjoon had in the beginning -/
def total_candies : ℕ := candies_given + candies_left

theorem namjoon_candies : total_candies = 34 := by
  sorry

end namjoon_candies_l1201_120154


namespace roots_of_equation_l1201_120198

def f (x : ℝ) : ℝ := x^10 - 5*x^8 + 4*x^6 - 64*x^4 + 320*x^2 - 256

theorem roots_of_equation :
  {x : ℝ | f x = 0} = {-2, -1, 1, 2} := by sorry

end roots_of_equation_l1201_120198


namespace dog_groupings_count_l1201_120111

/-- The number of ways to divide 12 dogs into three groups -/
def dog_groupings : ℕ :=
  let total_dogs : ℕ := 12
  let group1_size : ℕ := 4  -- Fluffy's group
  let group2_size : ℕ := 5  -- Nipper's group
  let group3_size : ℕ := 3
  let remaining_dogs : ℕ := total_dogs - 2  -- Excluding Fluffy and Nipper
  (Nat.choose remaining_dogs (group1_size - 1)) * (Nat.choose (remaining_dogs - (group1_size - 1)) (group2_size - 1))

/-- Theorem stating the number of ways to divide the dogs is 4200 -/
theorem dog_groupings_count : dog_groupings = 4200 := by
  sorry

end dog_groupings_count_l1201_120111


namespace library_average_MB_per_hour_l1201_120133

/-- Calculates the average megabytes per hour of music in a digital library -/
def averageMBPerHour (days : ℕ) (totalMB : ℕ) : ℕ :=
  let hoursPerDay : ℕ := 24
  let totalHours : ℕ := days * hoursPerDay
  let exactAverage : ℚ := totalMB / totalHours
  (exactAverage + 1/2).floor.toNat

/-- Proves that the average megabytes per hour for the given library is 67 MB -/
theorem library_average_MB_per_hour :
  averageMBPerHour 15 24000 = 67 := by
  sorry

end library_average_MB_per_hour_l1201_120133


namespace smallest_integer_y_l1201_120199

theorem smallest_integer_y : ∃ y : ℤ, (y : ℚ) / 4 + 3 / 7 > 2 / 3 ∧ ∀ z : ℤ, (z : ℚ) / 4 + 3 / 7 > 2 / 3 → y ≤ z :=
  sorry

end smallest_integer_y_l1201_120199


namespace smoothie_mix_amount_l1201_120137

/-- The amount of smoothie mix in ounces per packet -/
def smoothie_mix_per_packet (total_smoothies : ℕ) (smoothie_size : ℕ) (total_packets : ℕ) : ℚ :=
  (total_smoothies * smoothie_size : ℚ) / total_packets

theorem smoothie_mix_amount : 
  smoothie_mix_per_packet 150 12 180 = 10 := by
  sorry

end smoothie_mix_amount_l1201_120137


namespace complex_fraction_simplification_l1201_120187

theorem complex_fraction_simplification :
  (5 : ℂ) / (2 - Complex.I) = 2 + Complex.I := by sorry

end complex_fraction_simplification_l1201_120187


namespace parallel_vectors_m_value_l1201_120168

def vector_a : ℝ × ℝ := (2, 1)
def vector_b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem parallel_vectors_m_value :
  ∀ m : ℝ, (∃ k : ℝ, vector_a = k • (vector_b m)) → m = -2 := by
  sorry

end parallel_vectors_m_value_l1201_120168


namespace problem_statement_l1201_120157

theorem problem_statement (x n f : ℝ) : 
  x = (3 + Real.sqrt 8)^500 →
  n = ⌊x⌋ →
  f = x - n →
  x * (1 - f) = 1 := by sorry

end problem_statement_l1201_120157


namespace chocolate_gain_percent_l1201_120194

theorem chocolate_gain_percent :
  ∀ (C S : ℝ),
  C > 0 →
  S > 0 →
  35 * C = 21 * S →
  ((S - C) / C) * 100 = 200 / 3 :=
by
  sorry

end chocolate_gain_percent_l1201_120194


namespace movie_theater_revenue_is_6600_l1201_120146

/-- Calculates the total revenue of a movie theater given ticket prices and quantities sold --/
def movie_theater_revenue (matinee_price evening_price three_d_price : ℕ) 
                          (matinee_sold evening_sold three_d_sold : ℕ) : ℕ :=
  matinee_price * matinee_sold + evening_price * evening_sold + three_d_price * three_d_sold

/-- Theorem stating that the movie theater's revenue is $6600 given the specified prices and quantities --/
theorem movie_theater_revenue_is_6600 :
  movie_theater_revenue 5 12 20 200 300 100 = 6600 := by
  sorry

end movie_theater_revenue_is_6600_l1201_120146
