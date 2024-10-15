import Mathlib

namespace NUMINAMATH_CALUDE_solution_pairs_l328_32819

theorem solution_pairs : 
  ∃ (S : Set (ℕ × ℕ)), 
    S = {(0, 0), (1, 0)} ∧ 
    ∀ (a b : ℕ) (x : ℝ), 
      (a, b) ∈ S ↔ 
        (-2 * (a : ℝ) + (b : ℝ)^2 = Real.cos (π * (a : ℝ) + (b : ℝ)^2) - 1 ∧
         (b : ℝ)^2 = Real.cos (2 * π * (a : ℝ) + (b : ℝ)^2) - 1) :=
by sorry

end NUMINAMATH_CALUDE_solution_pairs_l328_32819


namespace NUMINAMATH_CALUDE_congruence_solution_l328_32811

theorem congruence_solution (p q : Nat) (n : Nat) : 
  Nat.Prime p → Nat.Prime q → Odd p → Odd q → n > 1 →
  (q ^ (n + 2) % (p ^ n) = 3 ^ (n + 2) % (p ^ n)) →
  (p ^ (n + 2) % (q ^ n) = 3 ^ (n + 2) % (q ^ n)) →
  (p = 3 ∧ q = 3) := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l328_32811


namespace NUMINAMATH_CALUDE_curve_is_line_segment_l328_32824

-- Define the parametric equations
def x (t : ℝ) : ℝ := 3 * t^2 + 2
def y (t : ℝ) : ℝ := t^2 - 1

-- Define the parameter range
def t_range : Set ℝ := {t | 0 ≤ t ∧ t ≤ 5}

-- Theorem statement
theorem curve_is_line_segment :
  ∃ (a b c : ℝ), ∀ t ∈ t_range, a * x t + b * y t + c = 0 ∧
  ∃ (x_min x_max : ℝ), (∀ t ∈ t_range, x_min ≤ x t ∧ x t ≤ x_max) ∧
  x_min < x_max :=
sorry

end NUMINAMATH_CALUDE_curve_is_line_segment_l328_32824


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l328_32869

theorem empty_solution_set_iff_a_in_range (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*x + 3 > a^2 - 2*a - 1) ↔ (-1 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_a_in_range_l328_32869


namespace NUMINAMATH_CALUDE_trigonometric_equality_iff_sum_pi_half_l328_32810

open Real

theorem trigonometric_equality_iff_sum_pi_half 
  (α β : ℝ) (k : ℕ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : k > 0) :
  (sin α)^(k+2) / (cos β)^k + (cos α)^(k+2) / (sin β)^k = 1 ↔ α + β = π/2 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equality_iff_sum_pi_half_l328_32810


namespace NUMINAMATH_CALUDE_parabola_equation_l328_32866

/-- A parabola with vertex (h, k) and vertical axis of symmetry has the form y = a(x-h)^2 + k -/
def is_vertical_parabola (a h k : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = a * (x - h)^2 + k

theorem parabola_equation (f : ℝ → ℝ) :
  (∀ x, f x = -3 * x^2 + 18 * x - 22) →
  is_vertical_parabola (-3) 3 5 f ∧
  f 2 = 2 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l328_32866


namespace NUMINAMATH_CALUDE_fourth_root_l328_32889

/-- The polynomial function defined by the given coefficients -/
def f (b c x : ℝ) : ℝ := b*x^4 + (b + 3*c)*x^3 + (c - 4*b)*x^2 + (19 - b)*x - 2

theorem fourth_root (b c : ℝ) 
  (h1 : f b c (-3) = 0)
  (h2 : f b c 4 = 0)
  (h3 : f b c 2 = 0) :
  ∃ x, x ≠ -3 ∧ x ≠ 4 ∧ x ≠ 2 ∧ f b c x = 0 ∧ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_root_l328_32889


namespace NUMINAMATH_CALUDE_water_fraction_proof_l328_32849

def initial_water : ℚ := 18
def initial_total : ℚ := 20
def removal_amount : ℚ := 5
def num_iterations : ℕ := 3

def water_fraction_after_iterations : ℚ := 
  (initial_water / initial_total) * ((initial_total - removal_amount) / initial_total) ^ num_iterations

theorem water_fraction_proof : water_fraction_after_iterations = 243 / 640 := by
  sorry

end NUMINAMATH_CALUDE_water_fraction_proof_l328_32849


namespace NUMINAMATH_CALUDE_arithmetic_sum_1_to_19_l328_32845

/-- Sum of an arithmetic sequence -/
def arithmetic_sum (a₁ aₙ : ℕ) (d : ℕ) : ℕ := 
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

/-- Proof that the sum of the arithmetic sequence 1, 3, 5, ..., 17, 19 is 100 -/
theorem arithmetic_sum_1_to_19 : arithmetic_sum 1 19 2 = 100 := by
  sorry

#eval arithmetic_sum 1 19 2

end NUMINAMATH_CALUDE_arithmetic_sum_1_to_19_l328_32845


namespace NUMINAMATH_CALUDE_min_games_for_prediction_l328_32885

/-- Represents the chess tournament setup -/
structure ChessTournament where
  white_rook : ℕ  -- number of students from "White Rook" school
  black_elephant : ℕ  -- number of students from "Black Elephant" school
  total_games : ℕ  -- total number of games to be played

/-- Checks if the tournament setup is valid -/
def is_valid_tournament (t : ChessTournament) : Prop :=
  t.white_rook * t.black_elephant = t.total_games

/-- Represents the state of the tournament after some games -/
structure TournamentState where
  tournament : ChessTournament
  games_played : ℕ

/-- Checks if Sasha can predict a participant in the next game -/
def can_predict_participant (state : TournamentState) : Prop :=
  state.games_played ≥ state.tournament.total_games - state.tournament.black_elephant

/-- The main theorem to be proved -/
theorem min_games_for_prediction (t : ChessTournament) 
    (h_valid : is_valid_tournament t) 
    (h_white : t.white_rook = 15) 
    (h_black : t.black_elephant = 20) : 
    ∀ n : ℕ, can_predict_participant ⟨t, n⟩ ↔ n ≥ 280 := by
  sorry

end NUMINAMATH_CALUDE_min_games_for_prediction_l328_32885


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_plus_one_less_than_zero_l328_32818

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, P x) ↔ (∀ x : ℝ, ¬ P x) :=
by sorry

theorem negation_of_square_plus_one_less_than_zero :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_square_plus_one_less_than_zero_l328_32818


namespace NUMINAMATH_CALUDE_problem_statement_l328_32807

theorem problem_statement (a b c d e : ℝ) 
  (h1 : (a + c) * (a + d) = e)
  (h2 : (b + c) * (b + d) = e)
  (h3 : e ≠ 0)
  (h4 : a ≠ b) :
  (a + c) * (b + c) - (a + d) * (b + d) = 0 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l328_32807


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_product_l328_32848

theorem consecutive_negative_integers_product (n : ℤ) :
  n < 0 ∧ (n + 1) < 0 ∧ n * (n + 1) = 2240 →
  |n - (n + 1)| = 1 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_product_l328_32848


namespace NUMINAMATH_CALUDE_probability_of_triangle_in_decagon_l328_32883

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
structure RegularDecagon where
  -- No specific properties needed for this problem

/-- The number of diagonals in a regular decagon -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 3 diagonals from the total number of diagonals -/
def total_diagonal_choices : ℕ := Nat.choose num_diagonals 3

/-- The number of ways to choose 4 points from 10 points -/
def four_point_choices : ℕ := Nat.choose 10 4

/-- The number of ways to choose 3 points out of 4 points -/
def three_out_of_four : ℕ := Nat.choose 4 3

/-- The number of triangle-forming sets of diagonals -/
def triangle_forming_sets : ℕ := four_point_choices * three_out_of_four

theorem probability_of_triangle_in_decagon (d : RegularDecagon) :
  (triangle_forming_sets : ℚ) / total_diagonal_choices = 840 / 6545 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_triangle_in_decagon_l328_32883


namespace NUMINAMATH_CALUDE_division_remainder_l328_32806

theorem division_remainder (N : ℕ) : N = 7 * 5 + 0 → N % 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l328_32806


namespace NUMINAMATH_CALUDE_michaels_matchsticks_l328_32825

theorem michaels_matchsticks (total : ℕ) : 
  (30 * 10 + 20 * 15 + 10 * 25 : ℕ) = (2 * total) / 3 → total = 1275 := by
  sorry

end NUMINAMATH_CALUDE_michaels_matchsticks_l328_32825


namespace NUMINAMATH_CALUDE_area_of_triangle_BDE_l328_32804

noncomputable section

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Distance between two points in 3D space -/
def distance (p q : Point3D) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

/-- Angle between three points in 3D space -/
def angle (p q r : Point3D) : ℝ := sorry

/-- Check if two lines are parallel in 3D space -/
def parallel_lines (p1 q1 p2 q2 : Point3D) : Prop := sorry

/-- Check if a plane is parallel to a line in 3D space -/
def plane_parallel_to_line (p1 p2 p3 l1 l2 : Point3D) : Prop := sorry

/-- Calculate the area of a triangle given its three vertices -/
def triangle_area (p q r : Point3D) : ℝ := sorry

theorem area_of_triangle_BDE (A B C D E : Point3D)
  (h1 : distance A B = 3)
  (h2 : distance B C = 3)
  (h3 : distance C D = 3)
  (h4 : distance D E = 3)
  (h5 : distance E A = 3)
  (h6 : angle A B C = Real.pi / 2)
  (h7 : angle C D E = Real.pi / 2)
  (h8 : angle D E A = Real.pi / 2)
  (h9 : plane_parallel_to_line A C D B E) :
  triangle_area B D E = 9 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_BDE_l328_32804


namespace NUMINAMATH_CALUDE_min_neg_half_third_l328_32850

theorem min_neg_half_third : min (-1/2 : ℚ) (-1/3) = -1/2 := by sorry

end NUMINAMATH_CALUDE_min_neg_half_third_l328_32850


namespace NUMINAMATH_CALUDE_total_spent_is_200_l328_32803

/-- The amount Pete and Raymond each received in cents -/
def initial_amount : ℕ := 250

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The number of nickels Pete spent -/
def pete_nickels_spent : ℕ := 4

/-- The number of dimes Raymond has left -/
def raymond_dimes_left : ℕ := 7

/-- Theorem: The total amount spent by Pete and Raymond is 200 cents -/
theorem total_spent_is_200 : 
  (pete_nickels_spent * nickel_value) + 
  (initial_amount - (raymond_dimes_left * dime_value)) = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_200_l328_32803


namespace NUMINAMATH_CALUDE_solution_quadratic_equation_l328_32808

theorem solution_quadratic_equation :
  ∀ x : ℝ, (x - 2)^2 = 3*(x - 2) ↔ x = 2 ∨ x = 5 := by sorry

end NUMINAMATH_CALUDE_solution_quadratic_equation_l328_32808


namespace NUMINAMATH_CALUDE_least_reducible_fraction_l328_32871

theorem least_reducible_fraction :
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ m : ℕ, m > 0 → m < n → ¬(Nat.gcd (m - 17) (6 * m + 7) > 1)) ∧
  Nat.gcd (n - 17) (6 * n + 7) > 1 ∧
  n = 126 := by
sorry

end NUMINAMATH_CALUDE_least_reducible_fraction_l328_32871


namespace NUMINAMATH_CALUDE_smallest_n_no_sum_of_powers_is_square_l328_32867

theorem smallest_n_no_sum_of_powers_is_square : ∃ (n : ℕ), n > 1 ∧
  (∀ (m k : ℕ), ¬∃ (a : ℕ), n^m + n^k = a^2) ∧
  (∀ (n' : ℕ), 1 < n' ∧ n' < n →
    ∃ (m k a : ℕ), n'^m + n'^k = a^2) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_n_no_sum_of_powers_is_square_l328_32867


namespace NUMINAMATH_CALUDE_kids_played_correct_l328_32874

/-- The number of kids Julia played with on each day --/
structure KidsPlayed where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ

/-- The conditions of the problem --/
def satisfiesConditions (k : KidsPlayed) : Prop :=
  k.tuesday = 14 ∧
  k.wednesday = k.tuesday + (k.tuesday / 4 + 1) ∧
  k.thursday = 2 * k.wednesday - 4 ∧
  k.monday = k.tuesday + 8

/-- The theorem to prove --/
theorem kids_played_correct : 
  ∃ (k : KidsPlayed), satisfiesConditions k ∧ 
    k.monday = 22 ∧ k.tuesday = 14 ∧ k.wednesday = 18 ∧ k.thursday = 32 := by
  sorry

end NUMINAMATH_CALUDE_kids_played_correct_l328_32874


namespace NUMINAMATH_CALUDE_zoe_pool_cleaning_earnings_l328_32821

/-- Represents Zoe's earnings and babysitting frequencies -/
structure ZoeEarnings where
  total : ℕ
  zachary_earnings : ℕ
  julie_freq : ℕ
  zachary_freq : ℕ
  chloe_freq : ℕ

/-- Calculates Zoe's earnings from pool cleaning -/
def pool_cleaning_earnings (e : ZoeEarnings) : ℕ :=
  e.total - (e.zachary_earnings * (1 + 3 + 5))

/-- Theorem stating that Zoe's pool cleaning earnings are $2,600 -/
theorem zoe_pool_cleaning_earnings :
  ∀ e : ZoeEarnings,
    e.total = 8000 ∧
    e.zachary_earnings = 600 ∧
    e.julie_freq = 3 * e.zachary_freq ∧
    e.zachary_freq * 5 = e.chloe_freq →
    pool_cleaning_earnings e = 2600 :=
by
  sorry


end NUMINAMATH_CALUDE_zoe_pool_cleaning_earnings_l328_32821


namespace NUMINAMATH_CALUDE_solution_set_range_of_a_l328_32878

-- Part 1
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem solution_set (x : ℝ) : f x ≥ 2 ↔ x ≤ 1/2 ∨ x ≥ 5/2 := by sorry

-- Part 2
def g (a x : ℝ) : ℝ := |x - 1| + |x - a|

theorem range_of_a (a : ℝ) : 
  (a > 1 ∧ ∀ x, g a x + |x - 1| ≥ 1) ↔ a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_range_of_a_l328_32878


namespace NUMINAMATH_CALUDE_square_coverage_l328_32844

/-- The smallest number of 3-by-4 rectangles needed to cover a square region exactly -/
def min_rectangles : ℕ := 12

/-- The side length of the square region -/
def square_side : ℕ := 12

/-- The width of each rectangle -/
def rectangle_width : ℕ := 3

/-- The height of each rectangle -/
def rectangle_height : ℕ := 4

theorem square_coverage :
  (square_side * square_side) = (min_rectangles * rectangle_width * rectangle_height) ∧
  (square_side % rectangle_width = 0) ∧
  (square_side % rectangle_height = 0) ∧
  ∀ n : ℕ, n < min_rectangles →
    (n * rectangle_width * rectangle_height) < (square_side * square_side) :=
by sorry

end NUMINAMATH_CALUDE_square_coverage_l328_32844


namespace NUMINAMATH_CALUDE_triangle_problem_l328_32895

noncomputable def f (x φ : Real) : Real := 2 * Real.sin x * (Real.cos (φ / 2))^2 + Real.cos x * Real.sin φ - Real.sin x

theorem triangle_problem (φ : Real) (A B C : Real) (a b c : Real) :
  (0 < φ) ∧ (φ < Real.pi) ∧
  (∀ x, f x φ ≥ f Real.pi φ) ∧
  (a = 1) ∧ (b = Real.sqrt 2) ∧
  (f A φ = Real.sqrt 3 / 2) ∧
  (a / Real.sin A = b / Real.sin B) ∧
  (A + B + C = Real.pi) →
  (φ = Real.pi / 2) ∧
  (∀ x, f x φ = Real.cos x) ∧
  (C = 7 * Real.pi / 12) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l328_32895


namespace NUMINAMATH_CALUDE_same_acquaintance_count_l328_32822

theorem same_acquaintance_count (n : ℕ) (h : n > 0) :
  ∃ (i j : Fin n) (k : ℕ), i ≠ j ∧
  (∃ (f : Fin n → ℕ), (∀ x, f x < n) ∧ f i = k ∧ f j = k) :=
by sorry

end NUMINAMATH_CALUDE_same_acquaintance_count_l328_32822


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_values_l328_32880

def A : Set ℝ := {x | x^2 - x - 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equality_implies_a_values (a : ℝ) :
  A ∩ B a = B a → a = 0 ∨ a = -1 ∨ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_values_l328_32880


namespace NUMINAMATH_CALUDE_complex_equation_solution_l328_32892

theorem complex_equation_solution (z : ℂ) : z * (1 - Complex.I) = 2 → z = 1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l328_32892


namespace NUMINAMATH_CALUDE_min_jumps_to_visit_all_l328_32881

/-- Represents a jump on the circle -/
inductive Jump
| Two : Jump  -- Jump by 2 points
| Three : Jump  -- Jump by 3 points

/-- The number of points on the circle -/
def numPoints : Nat := 2016

/-- A sequence of jumps -/
def JumpSequence := List Jump

/-- Function to calculate the total distance covered by a sequence of jumps -/
def totalDistance (seq : JumpSequence) : Nat :=
  seq.foldl (fun acc jump => acc + match jump with
    | Jump.Two => 2
    | Jump.Three => 3) 0

/-- Predicate to check if a sequence of jumps visits all points and returns to start -/
def isValidSequence (seq : JumpSequence) : Prop :=
  totalDistance seq % numPoints = 0 ∧ seq.length ≥ numPoints

/-- The main theorem -/
theorem min_jumps_to_visit_all :
  ∃ (seq : JumpSequence), isValidSequence seq ∧ seq.length = 2017 ∧
  (∀ (other : JumpSequence), isValidSequence other → seq.length ≤ other.length) :=
sorry

end NUMINAMATH_CALUDE_min_jumps_to_visit_all_l328_32881


namespace NUMINAMATH_CALUDE_proposition_implication_l328_32842

theorem proposition_implication (P : ℕ → Prop) :
  (∀ k : ℕ, k > 0 → (P k → P (k + 1))) →
  (¬ P 5) →
  (¬ P 4) :=
sorry

end NUMINAMATH_CALUDE_proposition_implication_l328_32842


namespace NUMINAMATH_CALUDE_carriage_sharing_problem_l328_32828

theorem carriage_sharing_problem (x : ℝ) : 
  (x > 0) →                            -- Ensure positive number of people
  (x / 3 + 2 = (x - 9) / 2) →           -- The equation to be proved
  (∃ n : ℕ, x = n) →                    -- Ensure x is a natural number
  (x / 3 + 2 : ℝ) = (x - 9) / 2 :=      -- The equation represents the problem
by
  sorry

end NUMINAMATH_CALUDE_carriage_sharing_problem_l328_32828


namespace NUMINAMATH_CALUDE_largest_number_l328_32859

/-- Converts a number from base b to base 10 --/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * b^i) 0

/-- The value of 85 in base 9 --/
def num1 : Nat := to_base_10 [5, 8] 9

/-- The value of 210 in base 6 --/
def num2 : Nat := to_base_10 [0, 1, 2] 6

/-- The value of 1000 in base 4 --/
def num3 : Nat := to_base_10 [0, 0, 0, 1] 4

/-- The value of 111111 in base 2 --/
def num4 : Nat := to_base_10 [1, 1, 1, 1, 1, 1] 2

theorem largest_number : num2 > num1 ∧ num2 > num3 ∧ num2 > num4 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l328_32859


namespace NUMINAMATH_CALUDE_line_equal_intercepts_l328_32865

/-- 
Given a line mx - y - 3 - m = 0, if its intercepts on the x-axis and y-axis are equal, 
then m = -3 or m = -1.
-/
theorem line_equal_intercepts (m : ℝ) : 
  (∃ (a : ℝ), a ≠ 0 ∧ m * a - 3 - m = 0 ∧ -3 - m = a) → 
  (m = -3 ∨ m = -1) :=
by sorry

end NUMINAMATH_CALUDE_line_equal_intercepts_l328_32865


namespace NUMINAMATH_CALUDE_smallest_n_for_floor_equation_l328_32886

theorem smallest_n_for_floor_equation : 
  ∀ n : ℕ, n < 7 → ¬∃ x : ℤ, ⌊(10 : ℝ)^n / x⌋ = 2006 ∧ 
  ∃ x : ℤ, ⌊(10 : ℝ)^7 / x⌋ = 2006 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_floor_equation_l328_32886


namespace NUMINAMATH_CALUDE_proposition_1_proposition_2_not_always_true_proposition_3_proposition_4_l328_32854

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularLP : Line → Plane → Prop)
variable (perpendicularPP : Plane → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (subset : Line → Plane → Prop)

-- Define the lines and planes
variable (a b : Line)
variable (α β : Plane)

-- Assume the lines and planes are distinct
variable (h_distinct_lines : a ≠ b)
variable (h_distinct_planes : α ≠ β)

-- Proposition 1
theorem proposition_1 : 
  perpendicular a b → perpendicularLP a α → ¬contains α b → parallel b α :=
sorry

-- Proposition 2 (not necessarily true)
theorem proposition_2_not_always_true : 
  ¬(∀ (a : Line) (α β : Plane), parallel a α → perpendicularPP α β → perpendicularLP a β) :=
sorry

-- Proposition 3
theorem proposition_3 : 
  perpendicularLP a β → perpendicularPP α β → (parallel a α ∨ subset a α) :=
sorry

-- Proposition 4
theorem proposition_4 : 
  perpendicular a b → perpendicularLP a α → perpendicularLP b β → perpendicularPP α β :=
sorry

end NUMINAMATH_CALUDE_proposition_1_proposition_2_not_always_true_proposition_3_proposition_4_l328_32854


namespace NUMINAMATH_CALUDE_dave_tickets_l328_32873

/-- The number of tickets Dave spent on the stuffed tiger -/
def tickets_spent : ℕ := 43

/-- The number of tickets Dave had left after the purchase -/
def tickets_left : ℕ := 55

/-- The initial number of tickets Dave had -/
def initial_tickets : ℕ := tickets_spent + tickets_left

theorem dave_tickets : initial_tickets = 98 := by sorry

end NUMINAMATH_CALUDE_dave_tickets_l328_32873


namespace NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l328_32801

-- Define the hyperbola
def is_hyperbola (x y m : ℝ) : Prop := x^2 - y^2 / m^2 = 1

-- Define the condition that m is positive
def m_positive (m : ℝ) : Prop := m > 0

-- Define the distance from focus to asymptote
def focus_asymptote_distance (m : ℝ) : ℝ := m

-- Theorem statement
theorem hyperbola_focus_asymptote_distance (m : ℝ) :
  m_positive m →
  (∃ x y, is_hyperbola x y m) →
  focus_asymptote_distance m = 4 →
  m = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_asymptote_distance_l328_32801


namespace NUMINAMATH_CALUDE_rose_additional_money_l328_32814

def paintbrush_cost : ℚ := 2.4
def paints_cost : ℚ := 9.2
def easel_cost : ℚ := 6.5
def rose_money : ℚ := 7.1

theorem rose_additional_money :
  paintbrush_cost + paints_cost + easel_cost - rose_money = 11 := by sorry

end NUMINAMATH_CALUDE_rose_additional_money_l328_32814


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l328_32812

theorem circle_equation_k_value (k : ℝ) :
  (∀ x y : ℝ, x^2 + 12*x + y^2 + 8*y - k = 0 ↔ (x + 6)^2 + (y + 4)^2 = 25) →
  k = -27 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l328_32812


namespace NUMINAMATH_CALUDE_fraction_equality_l328_32870

theorem fraction_equality (m n p q : ℚ) 
  (h1 : m / n = 12)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 8) :
  m / q = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l328_32870


namespace NUMINAMATH_CALUDE_function_symmetry_l328_32863

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom functional_equation : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

-- State the theorem
theorem function_symmetry : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_function_symmetry_l328_32863


namespace NUMINAMATH_CALUDE_range_of_2a_minus_b_l328_32826

theorem range_of_2a_minus_b (a b : ℝ) (h : -1 < a ∧ a < b ∧ b < 2) :
  ∀ x, x ∈ Set.Ioo (-4 : ℝ) 2 ↔ ∃ a b, -1 < a ∧ a < b ∧ b < 2 ∧ x = 2*a - b :=
by sorry

end NUMINAMATH_CALUDE_range_of_2a_minus_b_l328_32826


namespace NUMINAMATH_CALUDE_abs_x_minus_one_leq_one_iff_x_leq_two_l328_32823

theorem abs_x_minus_one_leq_one_iff_x_leq_two :
  ∀ x : ℝ, |x - 1| ≤ 1 ↔ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_abs_x_minus_one_leq_one_iff_x_leq_two_l328_32823


namespace NUMINAMATH_CALUDE_work_problem_underdetermined_l328_32896

-- Define the work rate of one man and one woman
variable (m w : ℝ)

-- Define the unknown number of men
variable (x : ℝ)

-- Condition 1: x men or 12 women can do the work in 20 days
def condition1 : Prop := x * m * 20 = 12 * w * 20

-- Condition 2: 6 men and 11 women can do the work in 12 days
def condition2 : Prop := (6 * m + 11 * w) * 12 = 1

-- Theorem: The conditions are insufficient to uniquely determine x
theorem work_problem_underdetermined :
  ∃ (m1 w1 x1 : ℝ) (m2 w2 x2 : ℝ),
    condition1 m1 w1 x1 ∧ condition2 m1 w1 ∧
    condition1 m2 w2 x2 ∧ condition2 m2 w2 ∧
    x1 ≠ x2 :=
sorry

end NUMINAMATH_CALUDE_work_problem_underdetermined_l328_32896


namespace NUMINAMATH_CALUDE_gas_station_lighter_price_l328_32802

/-- The cost of a single lighter at the gas station -/
def gas_station_price : ℝ := 1.75

/-- The cost of a pack of 12 lighters on Amazon -/
def amazon_pack_price : ℝ := 5

/-- The number of lighters in a pack on Amazon -/
def lighters_per_pack : ℕ := 12

/-- The number of lighters Amanda is considering buying -/
def total_lighters : ℕ := 24

/-- The amount saved by buying online instead of at the gas station -/
def savings : ℝ := 32

theorem gas_station_lighter_price :
  gas_station_price = 1.75 ∧
  amazon_pack_price * (total_lighters / lighters_per_pack) + savings =
    gas_station_price * total_lighters :=
by sorry

end NUMINAMATH_CALUDE_gas_station_lighter_price_l328_32802


namespace NUMINAMATH_CALUDE_platform_length_l328_32827

/-- The length of a platform given train crossing times -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 39 →
  pole_time = 18 →
  ∃ platform_length : ℝ,
    platform_length = 350 ∧
    (train_length + platform_length) / platform_time = train_length / pole_time :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l328_32827


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l328_32893

theorem complex_fraction_equality (a b : ℂ) (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : a^2 - a*b + b^2 = 0) : 
  (a^8 + b^8) / (a^2 + b^2)^4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l328_32893


namespace NUMINAMATH_CALUDE_height_difference_l328_32832

-- Define the heights in inches
def dog_height : ℕ := 24
def carter_height : ℕ := 2 * dog_height
def betty_height : ℕ := 3 * 12  -- 3 feet converted to inches

-- Theorem statement
theorem height_difference : carter_height - betty_height = 12 := by
  sorry

end NUMINAMATH_CALUDE_height_difference_l328_32832


namespace NUMINAMATH_CALUDE_range_of_f_l328_32864

def f (x : ℝ) : ℝ := x^4 + 6*x^2 + 9

theorem range_of_f :
  {y : ℝ | ∃ x ≥ 0, f x = y} = {y : ℝ | y ≥ 9} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l328_32864


namespace NUMINAMATH_CALUDE_job_completion_proof_l328_32894

/-- The number of days it takes to complete the job with the initial number of machines -/
def initial_days : ℝ := 12

/-- The number of days it takes to complete the job after adding more machines -/
def new_days : ℝ := 8

/-- The number of additional machines added -/
def additional_machines : ℕ := 6

/-- The number of machines initially working on the job -/
def initial_machines : ℕ := 12

theorem job_completion_proof :
  ∀ (rate : ℝ),
  rate > 0 →
  (initial_machines : ℝ) * rate * initial_days = 1 →
  ((initial_machines : ℝ) + additional_machines) * rate * new_days = 1 →
  initial_machines = 12 := by
  sorry

end NUMINAMATH_CALUDE_job_completion_proof_l328_32894


namespace NUMINAMATH_CALUDE_f_properties_l328_32820

-- Define the function f(x)
def f (x : ℝ) : ℝ := 4 * x^3 - 3 * x^2 - 18 * x + 5

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 12 * x^2 - 6 * x - 18

theorem f_properties :
  (f' (-1) = 0) ∧
  (f' (3/2) = 0) ∧
  (∀ x ∈ Set.Ioo (-1 : ℝ) (3/2), f' x < 0) ∧
  (∀ x ∈ Set.Iic (-1 : ℝ), f' x > 0) ∧
  (∀ x ∈ Set.Ioi (3/2 : ℝ), f' x > 0) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≤ 16) ∧
  (∀ x ∈ Set.Icc (-1 : ℝ) 2, f x ≥ -61/4) ∧
  (f (-1) = 16) ∧
  (f (3/2) = -61/4) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l328_32820


namespace NUMINAMATH_CALUDE_first_winner_of_both_prizes_l328_32891

theorem first_winner_of_both_prizes (n : ℕ) : 
  (n % 5 = 0 ∧ n % 7 = 0) → n ≥ 35 :=
by sorry

end NUMINAMATH_CALUDE_first_winner_of_both_prizes_l328_32891


namespace NUMINAMATH_CALUDE_largest_initial_number_l328_32809

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 ∉ {x : ℕ | a ∣ x ∨ b ∣ x ∨ c ∣ x ∨ d ∣ x ∨ e ∣ x} ∧
    189 + a + b + c + d + e = 200 ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (a' b' c' d' e' : ℕ),
        n ∉ {x : ℕ | a' ∣ x ∨ b' ∣ x ∨ c' ∣ x ∨ d' ∣ x ∨ e' ∣ x} ∧
        n + a' + b' + c' + d' + e' = 200 :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l328_32809


namespace NUMINAMATH_CALUDE_bill_processing_error_l328_32857

theorem bill_processing_error (x y : ℕ) : 
  10 ≤ x ∧ x ≤ 99 ∧ 10 ≤ y ∧ y ≤ 99 →
  100 * y + x - (100 * x + y) = 2970 →
  y = x + 30 ∧ 10 ≤ x ∧ x ≤ 69 :=
by sorry

end NUMINAMATH_CALUDE_bill_processing_error_l328_32857


namespace NUMINAMATH_CALUDE_solution_set_l328_32898

theorem solution_set (x : ℝ) : 4 ≤ x / (3 * x - 7) ∧ x / (3 * x - 7) < 9 ↔ 63 / 26 < x ∧ x ≤ 28 / 11 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l328_32898


namespace NUMINAMATH_CALUDE_expected_red_lights_proof_l328_32877

/-- The number of intersections with traffic lights -/
def num_intersections : ℕ := 3

/-- The probability of encountering a red light at each intersection -/
def red_light_probability : ℝ := 0.3

/-- The events of encountering a red light at each intersection are independent -/
axiom events_independent : True

/-- The expected number of red lights encountered -/
def expected_red_lights : ℝ := num_intersections * red_light_probability

theorem expected_red_lights_proof :
  expected_red_lights = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_expected_red_lights_proof_l328_32877


namespace NUMINAMATH_CALUDE_product_of_roots_l328_32861

theorem product_of_roots (x : ℝ) : (x + 2) * (x - 3) = 14 → ∃ y : ℝ, (x + 2) * (x - 3) = 14 ∧ (x * y = -20) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l328_32861


namespace NUMINAMATH_CALUDE_solution_satisfies_equation_l328_32800

variable (x : ℝ) (y : ℝ → ℝ) (C : ℝ)

noncomputable def solution (x : ℝ) (y : ℝ → ℝ) (C : ℝ) : Prop :=
  x * y x - 1 / (x * y x) - 2 * Real.log (abs (y x)) = C

def differential_equation (x : ℝ) (y : ℝ → ℝ) : Prop :=
  (1 + x^2 * (y x)^2) * y x + (x * y x - 1)^2 * x * (deriv y x) = 0

theorem solution_satisfies_equation :
  solution x y C → differential_equation x y :=
sorry

end NUMINAMATH_CALUDE_solution_satisfies_equation_l328_32800


namespace NUMINAMATH_CALUDE_min_distance_to_line_l328_32839

theorem min_distance_to_line (x y : ℝ) : 
  (x + 2)^2 + (y - 3)^2 = 1 → 
  ∃ (min : ℝ), min = 15 ∧ ∀ (a b : ℝ), (a + 2)^2 + (b - 3)^2 = 1 → |3*a + 4*b - 26| ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l328_32839


namespace NUMINAMATH_CALUDE_divisibility_properties_l328_32846

theorem divisibility_properties (n : ℤ) :
  -- Part (a)
  (n = 3 → ∃ m₁ m₂ : ℤ, m₁ = -5 ∧ m₂ = 9 ∧
    ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 ↔ m = m₁ ∨ m = m₂) ∧
  -- Part (b)
  (∃ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0) ∧
  -- Part (c)
  (∃ k : ℕ, ∀ m : ℤ, (m^2 + n^2 + 1) % (m - n + 1) = 0 ∧ (m^2 + n^2 + 1) % (m + n + 1) = 0 → m ≤ k) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_properties_l328_32846


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_ratio_l328_32862

theorem arithmetic_sequence_cosine_ratio (a : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a 9 - a 8) →  -- arithmetic sequence condition
  a 8 = 8 →                            -- given condition
  a 9 = 8 + π / 3 →                    -- given condition
  (Real.cos (a 5) + Real.cos (a 7)) / Real.cos (a 6) = 1 := by
    sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_ratio_l328_32862


namespace NUMINAMATH_CALUDE_polynomial_roots_l328_32876

theorem polynomial_roots : 
  let p (x : ℝ) := 10*x^4 - 55*x^3 + 96*x^2 - 55*x + 10
  ∀ x : ℝ, p x = 0 ↔ (x = 2 ∨ x = 1/2 ∨ x = 1) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l328_32876


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l328_32838

theorem sum_of_squares_and_square_of_sum : (3 + 7)^2 + (3^2 + 7^2 + 5^2) = 183 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l328_32838


namespace NUMINAMATH_CALUDE_olympic_rings_area_l328_32829

/-- Olympic Ring -/
structure OlympicRing where
  outer_diameter : ℝ
  inner_diameter : ℝ

/-- Olympic Emblem -/
structure OlympicEmblem where
  rings : Fin 5 → OlympicRing
  hypotenuse : ℝ

/-- The area covered by the Olympic rings -/
def area_covered (e : OlympicEmblem) : ℝ :=
  sorry

/-- The theorem statement -/
theorem olympic_rings_area (e : OlympicEmblem)
  (h1 : ∀ i : Fin 5, (e.rings i).outer_diameter = 22)
  (h2 : ∀ i : Fin 5, (e.rings i).inner_diameter = 18)
  (h3 : e.hypotenuse = 24) :
  ∃ ε > 0, |area_covered e - 592| < ε :=
sorry

end NUMINAMATH_CALUDE_olympic_rings_area_l328_32829


namespace NUMINAMATH_CALUDE_nested_root_equality_l328_32887

theorem nested_root_equality (a b c : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) :
  (∀ N : ℝ, N ≠ 1 → (N^(1/a + 1/(a*b) + 1/(a*b*c)) = N^(15/24))) →
  c = 8 := by
sorry

end NUMINAMATH_CALUDE_nested_root_equality_l328_32887


namespace NUMINAMATH_CALUDE_sweater_price_after_discounts_l328_32855

/-- Calculates the final price of an item after two successive discounts -/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  originalPrice * (1 - discount1) * (1 - discount2)

/-- Theorem: The final price of a $240 sweater after 60% and 30% discounts is $67.20 -/
theorem sweater_price_after_discounts :
  finalPrice 240 0.6 0.3 = 67.2 := by
  sorry

end NUMINAMATH_CALUDE_sweater_price_after_discounts_l328_32855


namespace NUMINAMATH_CALUDE_point_below_line_l328_32860

/-- A point P(a, 3) is below the line 2x - y = 3 if and only if a < 3 -/
theorem point_below_line (a : ℝ) : 
  (2 * a - 3 < 3) ↔ (a < 3) := by sorry

end NUMINAMATH_CALUDE_point_below_line_l328_32860


namespace NUMINAMATH_CALUDE_power_of_four_l328_32875

theorem power_of_four (k : ℝ) : (4 : ℝ) ^ (2 * k + 2) = 400 → (4 : ℝ) ^ k = 5 := by
  sorry

end NUMINAMATH_CALUDE_power_of_four_l328_32875


namespace NUMINAMATH_CALUDE_stock_worth_calculation_l328_32884

/-- Calculates the total worth of a stock given specific selling conditions and overall loss --/
theorem stock_worth_calculation (stock_value : ℝ) : 
  (0.2 * stock_value * 1.2 + 0.8 * stock_value * 0.9) - stock_value = -500 → 
  stock_value = 12500 := by
  sorry

#check stock_worth_calculation

end NUMINAMATH_CALUDE_stock_worth_calculation_l328_32884


namespace NUMINAMATH_CALUDE_arithmetic_sequence_index_l328_32882

/-- Given an arithmetic sequence {a_n} with first term a₁ = 1 and common difference d = 5,
    prove that if a_n = 2016, then n = 404. -/
theorem arithmetic_sequence_index (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ k : ℕ, a (k + 1) - a k = 5)
  (h3 : a n = 2016) :
  n = 404 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_index_l328_32882


namespace NUMINAMATH_CALUDE_train_crossing_pole_time_l328_32840

/-- Proves that a train with a given length and speed takes a specific time to cross a pole -/
theorem train_crossing_pole_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 250 →
  train_speed_kmh = 100 →
  crossing_time = 90 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_pole_time

end NUMINAMATH_CALUDE_train_crossing_pole_time_l328_32840


namespace NUMINAMATH_CALUDE_tan_sum_range_l328_32868

theorem tan_sum_range (m : ℝ) (α β : ℝ) : 
  (∃ (x y : ℝ), x ≠ y ∧ 
    m * x^2 - 2 * x * Real.sqrt (7 * m - 3) + 2 * m = 0 ∧
    m * y^2 - 2 * y * Real.sqrt (7 * m - 3) + 2 * m = 0 ∧
    x = Real.tan α ∧ y = Real.tan β) →
  ∃ (l u : ℝ), l = -(7 * Real.sqrt 3) / 3 ∧ u = -2 * Real.sqrt 2 ∧
    Real.tan (α + β) ∈ Set.Icc l u :=
sorry

end NUMINAMATH_CALUDE_tan_sum_range_l328_32868


namespace NUMINAMATH_CALUDE_prism_volume_l328_32853

-- Define a right rectangular prism
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the volume of a rectangular prism
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

-- Define the face areas of a rectangular prism
def sideFaceArea (p : RectangularPrism) : ℝ := p.a * p.b
def frontFaceArea (p : RectangularPrism) : ℝ := p.b * p.c
def bottomFaceArea (p : RectangularPrism) : ℝ := p.a * p.c

-- Theorem: The volume of the prism is 12 cubic inches
theorem prism_volume (p : RectangularPrism) 
  (h1 : sideFaceArea p = 18) 
  (h2 : frontFaceArea p = 12) 
  (h3 : bottomFaceArea p = 8) : 
  volume p = 12 := by
  sorry


end NUMINAMATH_CALUDE_prism_volume_l328_32853


namespace NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l328_32833

-- Define the function
def f (x : ℝ) : ℝ := 2 * |x - 3| + 5

-- State the theorem
theorem minimum_point_of_translated_graph :
  ∃! (x : ℝ), ∀ (y : ℝ), f y ≥ f x ∧ f x = 5 ∧ x = 3 :=
sorry

end NUMINAMATH_CALUDE_minimum_point_of_translated_graph_l328_32833


namespace NUMINAMATH_CALUDE_sqrt_n_divisors_characterization_l328_32835

def has_sqrt_n_divisors (n : ℕ) : Prop :=
  (Nat.divisors n).card = Nat.sqrt n

theorem sqrt_n_divisors_characterization :
  ∀ n : ℕ, has_sqrt_n_divisors n ↔ n = 1 ∨ n = 9 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_n_divisors_characterization_l328_32835


namespace NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l328_32879

theorem square_root_of_factorial_fraction : 
  Real.sqrt (Nat.factorial 9 / 210) = 216 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_factorial_fraction_l328_32879


namespace NUMINAMATH_CALUDE_hybrid_car_percentage_l328_32851

/-- Proves that the percentage of hybrid cars in a dealership is 60% -/
theorem hybrid_car_percentage
  (total_cars : ℕ)
  (hybrids_with_full_headlights : ℕ)
  (hybrid_one_headlight_percent : ℚ)
  (h1 : total_cars = 600)
  (h2 : hybrids_with_full_headlights = 216)
  (h3 : hybrid_one_headlight_percent = 40 / 100) :
  (hybrids_with_full_headlights / (1 - hybrid_one_headlight_percent) : ℚ) / total_cars = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_hybrid_car_percentage_l328_32851


namespace NUMINAMATH_CALUDE_dot_product_on_trajectory_l328_32856

/-- The trajectory E in the xy-plane -/
def TrajectoryE (x y : ℝ) : Prop :=
  |((x + 2)^2 + y^2).sqrt - ((x - 2)^2 + y^2).sqrt| = 2

/-- Point A -/
def A : ℝ × ℝ := (-2, 0)

/-- Point B -/
def B : ℝ × ℝ := (2, 0)

/-- Theorem stating that for any point C on trajectory E with BC perpendicular to x-axis,
    the dot product of AC and BC equals 9 -/
theorem dot_product_on_trajectory (C : ℝ × ℝ) (hC : TrajectoryE C.1 C.2)
  (hPerp : C.1 = B.1) : (C.1 - A.1) * (C.1 - B.1) + (C.2 - A.2) * (C.2 - B.2) = 9 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_on_trajectory_l328_32856


namespace NUMINAMATH_CALUDE_polygon_angle_sum_l328_32813

theorem polygon_angle_sum (n : ℕ) (x : ℝ) : 
  n ≥ 3 → 
  0 < x → 
  x < 180 → 
  (n - 2) * 180 + x = 1350 → 
  n = 9 ∧ x = 90 := by
sorry

end NUMINAMATH_CALUDE_polygon_angle_sum_l328_32813


namespace NUMINAMATH_CALUDE_hexagon_area_theorem_l328_32837

/-- Regular hexagon with vertices A and C -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a regular hexagon given its vertices A and C -/
def hexagon_area (h : RegularHexagon) : ℝ :=
  sorry

theorem hexagon_area_theorem (h : RegularHexagon) 
  (h_A : h.A = (0, 0)) 
  (h_C : h.C = (8, 2)) : 
  hexagon_area h = 34 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_area_theorem_l328_32837


namespace NUMINAMATH_CALUDE_find_C_l328_32899

theorem find_C (A B C : ℤ) (h1 : A = 509) (h2 : A = B + 197) (h3 : C = B - 125) : C = 187 := by
  sorry

end NUMINAMATH_CALUDE_find_C_l328_32899


namespace NUMINAMATH_CALUDE_gcd_5_factorial_7_factorial_l328_32816

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcd_5_factorial_7_factorial : 
  Nat.gcd (factorial 5) (factorial 7) = factorial 5 := by sorry

end NUMINAMATH_CALUDE_gcd_5_factorial_7_factorial_l328_32816


namespace NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l328_32817

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exists_greater_than_one_l328_32817


namespace NUMINAMATH_CALUDE_sophie_savings_l328_32872

/-- Represents the amount of money saved in a year by not buying dryer sheets -/
def money_saved (loads_per_week : ℕ) (sheets_per_load : ℕ) (sheets_per_box : ℕ) (cost_per_box : ℚ) (weeks_per_year : ℕ) : ℚ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := (sheets_per_year + sheets_per_box - 1) / sheets_per_box
  boxes_per_year * cost_per_box

/-- The amount of money Sophie saves in a year by not buying dryer sheets is $11.00 -/
theorem sophie_savings : 
  money_saved 4 1 104 (11/2) 52 = 11 := by
  sorry

end NUMINAMATH_CALUDE_sophie_savings_l328_32872


namespace NUMINAMATH_CALUDE_floor_sqrt_12_squared_l328_32805

theorem floor_sqrt_12_squared : ⌊Real.sqrt 12⌋^2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_12_squared_l328_32805


namespace NUMINAMATH_CALUDE_field_length_is_48_l328_32831

-- Define the field and pond
def field_width : ℝ := sorry
def field_length : ℝ := 2 * field_width
def pond_side : ℝ := 8

-- Define the areas
def field_area : ℝ := field_length * field_width
def pond_area : ℝ := pond_side^2

-- State the theorem
theorem field_length_is_48 :
  field_length = 2 * field_width ∧
  pond_side = 8 ∧
  pond_area = (1/18) * field_area →
  field_length = 48 := by sorry

end NUMINAMATH_CALUDE_field_length_is_48_l328_32831


namespace NUMINAMATH_CALUDE_average_salary_theorem_l328_32858

def salary_A : ℕ := 9000
def salary_B : ℕ := 5000
def salary_C : ℕ := 11000
def salary_D : ℕ := 7000
def salary_E : ℕ := 9000

def total_salary : ℕ := salary_A + salary_B + salary_C + salary_D + salary_E
def num_people : ℕ := 5

theorem average_salary_theorem :
  (total_salary : ℚ) / num_people = 8200 := by sorry

end NUMINAMATH_CALUDE_average_salary_theorem_l328_32858


namespace NUMINAMATH_CALUDE_negative_two_less_than_negative_two_thirds_l328_32847

theorem negative_two_less_than_negative_two_thirds : -2 < -(2/3) := by
  sorry

end NUMINAMATH_CALUDE_negative_two_less_than_negative_two_thirds_l328_32847


namespace NUMINAMATH_CALUDE_c_range_l328_32890

def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

theorem c_range (c : ℝ) : c > 0 →
  (is_decreasing (fun x ↦ c^x) ∨ (∀ x ∈ Set.Icc 0 2, x + c > 2)) ∧
  ¬(is_decreasing (fun x ↦ c^x) ∧ (∀ x ∈ Set.Icc 0 2, x + c > 2)) →
  (0 < c ∧ c < 1) ∨ c > 2 :=
by sorry

end NUMINAMATH_CALUDE_c_range_l328_32890


namespace NUMINAMATH_CALUDE_base6_arithmetic_sum_l328_32815

/-- Converts a base 6 number to base 10 --/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Converts a base 10 number to base 6 --/
def base10ToBase6 (n : ℕ) : ℕ := sorry

/-- Calculates the number of terms in the arithmetic sequence --/
def numTerms (first last step : ℕ) : ℕ := (last - first) / step + 1

/-- Calculates the sum of an arithmetic sequence --/
def arithmeticSum (n first last : ℕ) : ℕ := n * (first + last) / 2

theorem base6_arithmetic_sum :
  let first := base6ToBase10 2
  let last := base6ToBase10 50
  let step := base6ToBase10 2
  let n := numTerms first last step
  let sum := arithmeticSum n first last
  base10ToBase6 sum = 1040 := by sorry

end NUMINAMATH_CALUDE_base6_arithmetic_sum_l328_32815


namespace NUMINAMATH_CALUDE_cocktail_theorem_l328_32834

def cocktail_proof (initial_volume : ℝ) (jasmine_percent : ℝ) (rose_percent : ℝ) (mint_percent : ℝ)
  (added_jasmine : ℝ) (added_rose : ℝ) (added_mint : ℝ) (added_plain : ℝ) : Prop :=
  let initial_jasmine := initial_volume * jasmine_percent
  let initial_rose := initial_volume * rose_percent
  let initial_mint := initial_volume * mint_percent
  let new_jasmine := initial_jasmine + added_jasmine
  let new_rose := initial_rose + added_rose
  let new_mint := initial_mint + added_mint
  let new_volume := initial_volume + added_jasmine + added_rose + added_mint + added_plain
  let new_percent := (new_jasmine + new_rose + new_mint) / new_volume * 100
  new_percent = 21.91

theorem cocktail_theorem :
  cocktail_proof 150 0.03 0.05 0.02 12 9 3 4 := by
  sorry

end NUMINAMATH_CALUDE_cocktail_theorem_l328_32834


namespace NUMINAMATH_CALUDE_fraction_equality_l328_32836

theorem fraction_equality : (12 : ℚ) / (8 * 75) = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l328_32836


namespace NUMINAMATH_CALUDE_points_are_coplanar_l328_32852

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem points_are_coplanar
  (h_nonzero : e₁ ≠ 0 ∧ e₂ ≠ 0)
  (h_not_collinear : ¬ ∃ (k : ℝ), e₁ = k • e₂)
  (h_AB : B - A = e₁ + e₂)
  (h_AC : C - A = -3 • e₁ + 7 • e₂)
  (h_AD : D - A = 2 • e₁ - 3 • e₂) :
  ∃ (a b c d : ℝ), a • (B - A) + b • (C - A) + c • (D - A) = d • (0 : V) :=
sorry

end NUMINAMATH_CALUDE_points_are_coplanar_l328_32852


namespace NUMINAMATH_CALUDE_combination_equality_l328_32841

theorem combination_equality (x : ℕ) : (Nat.choose 5 3 + Nat.choose 5 4 = Nat.choose x 4) ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_combination_equality_l328_32841


namespace NUMINAMATH_CALUDE_cherry_tree_leaves_l328_32888

/-- The number of cherry trees originally planned to be planted -/
def originalTreeCount : ℕ := 7

/-- The actual number of cherry trees planted -/
def actualTreeCount : ℕ := 2 * originalTreeCount

/-- The number of leaves dropped by each tree during fall -/
def leavesPerTree : ℕ := 100

/-- The total number of leaves falling from all cherry trees -/
def totalLeaves : ℕ := actualTreeCount * leavesPerTree

theorem cherry_tree_leaves :
  totalLeaves = 1400 := by sorry

end NUMINAMATH_CALUDE_cherry_tree_leaves_l328_32888


namespace NUMINAMATH_CALUDE_not_enough_apples_for_pie_l328_32830

theorem not_enough_apples_for_pie (tessa_initial : Real) (anita_gave : Real) (pie_requirement : Real) : 
  tessa_initial = 4.75 → anita_gave = 5.5 → pie_requirement = 12.25 → tessa_initial + anita_gave < pie_requirement :=
by
  sorry

end NUMINAMATH_CALUDE_not_enough_apples_for_pie_l328_32830


namespace NUMINAMATH_CALUDE_factor_expression_l328_32843

theorem factor_expression (y : ℝ) : y * (y + 3) + 2 * (y + 3) = (y + 2) * (y + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l328_32843


namespace NUMINAMATH_CALUDE_Fe2O3_weight_l328_32897

/-- The atomic weight of iron in g/mol -/
def atomic_weight_Fe : ℝ := 55.845

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 15.999

/-- The number of iron atoms in Fe2O3 -/
def Fe_count : ℕ := 2

/-- The number of oxygen atoms in Fe2O3 -/
def O_count : ℕ := 3

/-- The number of moles of Fe2O3 -/
def moles_Fe2O3 : ℝ := 8

/-- The molecular weight of Fe2O3 in g/mol -/
def molecular_weight_Fe2O3 : ℝ := Fe_count * atomic_weight_Fe + O_count * atomic_weight_O

/-- The total weight of Fe2O3 in grams -/
def total_weight_Fe2O3 : ℝ := moles_Fe2O3 * molecular_weight_Fe2O3

theorem Fe2O3_weight : total_weight_Fe2O3 = 1277.496 := by sorry

end NUMINAMATH_CALUDE_Fe2O3_weight_l328_32897
