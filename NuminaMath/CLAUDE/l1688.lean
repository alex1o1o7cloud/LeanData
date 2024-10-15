import Mathlib

namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1688_168813

/-- A geometric sequence with first term 512 and 8th term 2 has 6th term equal to 16 -/
theorem geometric_sequence_sixth_term : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = a n * (a 8 / a 7)) →  -- Geometric sequence property
  a 1 = 512 →                            -- First term is 512
  a 8 = 2 →                              -- 8th term is 2
  a 6 = 16 :=                            -- 6th term is 16
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l1688_168813


namespace NUMINAMATH_CALUDE_solution_of_equation_l1688_168817

theorem solution_of_equation : ∃! x : ℝ, (3 / (x - 2) - 1 = 0) ∧ (x = 5) := by
  sorry

end NUMINAMATH_CALUDE_solution_of_equation_l1688_168817


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1688_168841

theorem p_necessary_not_sufficient_for_q :
  (∃ x, x < 2 ∧ ¬(-2 < x ∧ x < 2)) ∧
  (∀ x, -2 < x ∧ x < 2 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l1688_168841


namespace NUMINAMATH_CALUDE_min_tangent_length_l1688_168874

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 3 = 0

-- Define point A
def point_A : ℝ × ℝ := (-1, 1)

-- Define the property that P is outside C
def outside_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 4*x - 2*y + 3 > 0

-- Define the tangent condition (|PM| = |PA|)
def tangent_condition (x y : ℝ) : Prop :=
  ∃ (mx my : ℝ), circle_C mx my ∧
  (x - mx)^2 + (y - my)^2 = (x + 1)^2 + (y - 1)^2

-- Theorem statement
theorem min_tangent_length :
  ∃ (min_length : ℝ),
    (∀ (x y : ℝ), outside_circle x y → tangent_condition x y →
      (x + 1)^2 + (y - 1)^2 ≥ min_length^2) ∧
    (∃ (x y : ℝ), outside_circle x y ∧ tangent_condition x y ∧
      (x + 1)^2 + (y - 1)^2 = min_length^2) ∧
    min_length = 1/2 :=
sorry

end NUMINAMATH_CALUDE_min_tangent_length_l1688_168874


namespace NUMINAMATH_CALUDE_triangle_existence_theorem_l1688_168871

/-- The sum of angles in a triangle is 180 degrees -/
axiom triangle_angle_sum : ℝ → ℝ → ℝ → Prop

/-- A right angle is 90 degrees -/
def is_right_angle (angle : ℝ) : Prop := angle = 90

/-- An acute angle is less than 90 degrees -/
def is_acute_angle (angle : ℝ) : Prop := angle < 90

/-- An equilateral triangle has three equal angles -/
def is_equilateral_triangle (a b c : ℝ) : Prop := a = b ∧ b = c

theorem triangle_existence_theorem :
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → b = 60 → c = 60 → False) ∧
  (∃ a b c : ℝ, triangle_angle_sum a b c ∧ is_equilateral_triangle a b c ∧ a = 60) ∧
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → is_right_angle b → is_right_angle c → False) ∧
  (∃ a b c : ℝ, triangle_angle_sum a b c ∧ is_equilateral_triangle a b c ∧ is_acute_angle a) ∧
  (∀ a b c : ℝ, triangle_angle_sum a b c → is_right_angle a → b = 45 → c = 15 → False) :=
by sorry

end NUMINAMATH_CALUDE_triangle_existence_theorem_l1688_168871


namespace NUMINAMATH_CALUDE_inscribed_triangle_theorem_l1688_168867

-- Define the parabola C: y^2 = 4x
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the point M
def M : ℝ × ℝ := (1, 2)

-- Define the fixed point P
def P : ℝ × ℝ := (5, -2)

-- Define a line passing through two points
def line_through (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p = (1 - t) • p1 + t • p2}

-- Define a right triangle
def right_triangle (A B M : ℝ × ℝ) : Prop :=
  (A.1 - M.1) * (B.1 - M.1) + (A.2 - M.2) * (B.2 - M.2) = 0

-- Define the theorem
theorem inscribed_triangle_theorem (A B : ℝ × ℝ) :
  parabola A.1 A.2 →
  parabola B.1 B.2 →
  right_triangle A B M →
  (∃ (N : ℝ × ℝ), N ∈ line_through A B ∧ 
    (N.1 - M.1) * (B.1 - A.1) + (N.2 - M.2) * (B.2 - A.2) = 0) →
  (P ∈ line_through A B) ∧
  (∀ (x y : ℝ), x ≠ 1 → ((x - 3)^2 + y^2 = 8 ↔ 
    (∃ (A' B' : ℝ × ℝ), parabola A'.1 A'.2 ∧ parabola B'.1 B'.2 ∧
      right_triangle A' B' M ∧ (x, y) ∈ line_through A' B' ∧
      (x - M.1) * (B'.1 - A'.1) + (y - M.2) * (B'.2 - A'.2) = 0))) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_triangle_theorem_l1688_168867


namespace NUMINAMATH_CALUDE_specific_polygon_properties_l1688_168880

/-- Represents a regular polygon with given properties -/
structure RegularPolygon where
  total_angle_sum : ℝ
  known_angle : ℝ
  num_sides : ℕ
  remaining_angle : ℝ

/-- Theorem about a specific regular polygon -/
theorem specific_polygon_properties :
  let p := RegularPolygon.mk 3420 160 21 163
  p.num_sides = 21 ∧
  p.remaining_angle = 163 ∧
  p.total_angle_sum = 180 * (p.num_sides - 2) ∧
  p.total_angle_sum = p.known_angle + (p.num_sides - 1) * p.remaining_angle :=
by sorry

end NUMINAMATH_CALUDE_specific_polygon_properties_l1688_168880


namespace NUMINAMATH_CALUDE_leak_emptying_time_l1688_168877

/-- Given a pipe that fills a tank in 12 hours and a leak that causes the tank to take 20 hours to fill when both are active, prove that the leak alone will empty the full tank in 30 hours. -/
theorem leak_emptying_time (pipe_fill_rate : ℝ) (combined_fill_rate : ℝ) (leak_empty_rate : ℝ) :
  pipe_fill_rate = 1 / 12 →
  combined_fill_rate = 1 / 20 →
  pipe_fill_rate - leak_empty_rate = combined_fill_rate →
  1 / leak_empty_rate = 30 := by
  sorry

#check leak_emptying_time

end NUMINAMATH_CALUDE_leak_emptying_time_l1688_168877


namespace NUMINAMATH_CALUDE_point_inside_circle_l1688_168885

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if a point is inside a circle -/
def isInside (p : ℝ × ℝ) (c : Circle) : Prop :=
  Real.sqrt ((p.1 - c.center.1)^2 + (p.2 - c.center.2)^2) < c.radius

theorem point_inside_circle (O : Circle) (P : ℝ × ℝ) 
    (h1 : O.radius = 5)
    (h2 : Real.sqrt ((P.1 - O.center.1)^2 + (P.2 - O.center.2)^2) = 4) :
  isInside P O := by
  sorry

end NUMINAMATH_CALUDE_point_inside_circle_l1688_168885


namespace NUMINAMATH_CALUDE_bus_calculation_l1688_168899

theorem bus_calculation (total_students : ℕ) (capacity_40 capacity_30 : ℕ) : 
  total_students = 186 → capacity_40 = 40 → capacity_30 = 30 →
  (Nat.ceil (total_students / capacity_40) = 5 ∧
   Nat.ceil (total_students / capacity_30) = 7) := by
  sorry

#check bus_calculation

end NUMINAMATH_CALUDE_bus_calculation_l1688_168899


namespace NUMINAMATH_CALUDE_election_theorem_l1688_168838

theorem election_theorem (winner_percentage : ℝ) (winner_margin : ℕ) (winner_votes : ℕ) :
  winner_percentage = 0.62 →
  winner_votes = 992 →
  winner_margin = 384 →
  ∃ (total_votes : ℕ) (runner_up_votes : ℕ),
    total_votes = 1600 ∧
    runner_up_votes = 608 ∧
    winner_votes = winner_percentage * total_votes ∧
    winner_votes = runner_up_votes + winner_margin :=
by
  sorry

#check election_theorem

end NUMINAMATH_CALUDE_election_theorem_l1688_168838


namespace NUMINAMATH_CALUDE_inequality_1_inequality_2_l1688_168854

-- Define the solution set for the first inequality
def solution_set_1 : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem for the first inequality
theorem inequality_1 : ∀ x : ℝ, -x^2 + 4*x + 5 < 0 ↔ x ∈ solution_set_1 := by sorry

-- Define the solution set for the second inequality
def solution_set_2 (a : ℝ) : Set ℝ :=
  if a = -1 then ∅ 
  else if a > -1 then {x | -1 < x ∧ x < a}
  else {x | a < x ∧ x < -1}

-- Theorem for the second inequality
theorem inequality_2 : ∀ a x : ℝ, x^2 + (1-a)*x - a < 0 ↔ x ∈ solution_set_2 a := by sorry

end NUMINAMATH_CALUDE_inequality_1_inequality_2_l1688_168854


namespace NUMINAMATH_CALUDE_volume_increase_when_quadrupled_l1688_168882

/-- Given a cylindrical container, when all its dimensions are quadrupled, 
    its volume increases by a factor of 64. -/
theorem volume_increase_when_quadrupled (r h V : ℝ) :
  V = π * r^2 * h →
  (π * (4*r)^2 * (4*h)) = 64 * V :=
by sorry

end NUMINAMATH_CALUDE_volume_increase_when_quadrupled_l1688_168882


namespace NUMINAMATH_CALUDE_rectangle_triangle_perimeter_l1688_168893

/-- A rectangle ABCD with an equilateral triangle CMN where M is on AB -/
structure RectangleWithTriangle where
  /-- Length of rectangle ABCD -/
  length : ℝ
  /-- Width of rectangle ABCD -/
  width : ℝ
  /-- Distance AM, where M is the point on AB where the triangle meets the rectangle -/
  x : ℝ

/-- The perimeter of the equilateral triangle CMN in the RectangleWithTriangle -/
def triangle_perimeter (r : RectangleWithTriangle) : ℝ :=
  3 * (r.x^2 + 1)

theorem rectangle_triangle_perimeter 
  (r : RectangleWithTriangle) 
  (h1 : r.length = 2) 
  (h2 : r.width = 1) 
  (h3 : 0 ≤ r.x) 
  (h4 : r.x ≤ 2) : 
  ∃ (p : ℝ), triangle_perimeter r = p := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_perimeter_l1688_168893


namespace NUMINAMATH_CALUDE_star_equation_solution_l1688_168802

/-- Custom star operation -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a

/-- Theorem: If 4 ⋆ x = 46, then x = 50/7 -/
theorem star_equation_solution (x : ℝ) (h : star 4 x = 46) : x = 50/7 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1688_168802


namespace NUMINAMATH_CALUDE_certain_number_equation_l1688_168849

theorem certain_number_equation (x : ℝ) : x = 25 ↔ 0.8 * 45 = 4/5 * x + 16 := by sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1688_168849


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1688_168821

-- Define the number of sides in a nonagon
def nonagon_sides : ℕ := 9

-- Define the number of diagonals in a nonagon
def nonagon_diagonals : ℕ := 27

-- Define a function to calculate combinations
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem statement
theorem nonagon_diagonal_intersection_probability :
  let total_diagonal_pairs := choose nonagon_diagonals 2
  let intersecting_diagonal_pairs := choose nonagon_sides 4
  (intersecting_diagonal_pairs : ℚ) / total_diagonal_pairs = 14 / 39 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l1688_168821


namespace NUMINAMATH_CALUDE_coeff_x4_when_sum_64_l1688_168866

-- Define the binomial coefficient function
def binomial_coeff (n k : ℕ) : ℕ := sorry

-- Define the function to calculate the sum of binomial coefficients
def sum_binomial_coeff (n : ℕ) : ℕ := sorry

-- Define the function to calculate the coefficient of x^4 in the expansion
def coeff_x4 (n : ℕ) : ℤ := sorry

-- Theorem statement
theorem coeff_x4_when_sum_64 (n : ℕ) :
  sum_binomial_coeff n = 64 → coeff_x4 n = -12 := by sorry

end NUMINAMATH_CALUDE_coeff_x4_when_sum_64_l1688_168866


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l1688_168848

/-- The minimum number of additional coins needed to distribute distinct, positive numbers of coins to a given number of friends, starting with a given number of coins. -/
def min_additional_coins (num_friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := (num_friends * (num_friends + 1)) / 2
  if required_coins > initial_coins then
    required_coins - initial_coins
  else
    0

/-- The problem statement as a theorem -/
theorem alex_coin_distribution :
  min_additional_coins 15 97 = 23 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l1688_168848


namespace NUMINAMATH_CALUDE_unique_intersection_values_l1688_168896

-- Define the set of complex numbers that satisfy |z - 2| = 3|z + 2|
def S : Set ℂ := {z : ℂ | Complex.abs (z - 2) = 3 * Complex.abs (z + 2)}

-- Define a function that returns the set of intersection points between S and |z| = k
def intersection (k : ℝ) : Set ℂ := S ∩ {z : ℂ | Complex.abs z = k}

-- State the theorem
theorem unique_intersection_values :
  ∀ k : ℝ, (∃! z : ℂ, z ∈ intersection k) ↔ (k = 1 ∨ k = 4) :=
by sorry

end NUMINAMATH_CALUDE_unique_intersection_values_l1688_168896


namespace NUMINAMATH_CALUDE_cube_plus_minus_one_divisible_by_seven_l1688_168858

theorem cube_plus_minus_one_divisible_by_seven (n : ℤ) (h : ¬ 7 ∣ n) :
  7 ∣ (n^3 - 1) ∨ 7 ∣ (n^3 + 1) := by
  sorry

end NUMINAMATH_CALUDE_cube_plus_minus_one_divisible_by_seven_l1688_168858


namespace NUMINAMATH_CALUDE_compound_interest_problem_l1688_168839

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time - principal

/-- Total amount calculation -/
def total_amount (principal : ℝ) (interest : ℝ) : ℝ :=
  principal + interest

/-- Theorem statement -/
theorem compound_interest_problem (P : ℝ) :
  compound_interest P 0.04 2 = 326.40 →
  total_amount P 326.40 = 4326.40 := by
sorry

#eval compound_interest 4000 0.04 2
#eval total_amount 4000 326.40

end NUMINAMATH_CALUDE_compound_interest_problem_l1688_168839


namespace NUMINAMATH_CALUDE_scientific_notation_of_9600000_l1688_168835

/-- Proves that 9600000 is equal to 9.6 × 10^6 -/
theorem scientific_notation_of_9600000 : 9600000 = 9.6 * (10 ^ 6) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_9600000_l1688_168835


namespace NUMINAMATH_CALUDE_givenSampleIsValidSystematic_l1688_168846

/-- Checks if a list of integers represents a valid systematic sample -/
def isValidSystematicSample (sample : List Nat) (populationSize : Nat) : Prop :=
  let n := sample.length
  ∃ k : Nat,
    k > 0 ∧
    (∀ i : Fin n, sample[i] = k * (i + 1)) ∧
    sample.all (· ≤ populationSize)

/-- The given sample -/
def givenSample : List Nat := [3, 13, 23, 33, 43]

/-- The theorem stating that the given sample is a valid systematic sample -/
theorem givenSampleIsValidSystematic :
  isValidSystematicSample givenSample 50 := by
  sorry


end NUMINAMATH_CALUDE_givenSampleIsValidSystematic_l1688_168846


namespace NUMINAMATH_CALUDE_product_plus_number_equals_93_l1688_168872

theorem product_plus_number_equals_93 : ∃ x : ℤ, (-11 * -8) + x = 93 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_product_plus_number_equals_93_l1688_168872


namespace NUMINAMATH_CALUDE_f_sum_equals_two_l1688_168875

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.sqrt (1 + 9 * x^2) - 3 * x) + 1

theorem f_sum_equals_two : f (Real.log 2) + f (Real.log (1/2)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_equals_two_l1688_168875


namespace NUMINAMATH_CALUDE_icosahedron_edge_ratio_l1688_168864

/-- An icosahedron with edge length a -/
structure Icosahedron where
  a : ℝ
  a_pos : 0 < a

/-- A regular octahedron -/
structure RegularOctahedron where
  edge_length : ℝ
  edge_length_pos : 0 < edge_length

/-- Given two icosahedrons, this function returns true if six vertices 
    can be chosen from them to form a regular octahedron -/
def can_form_octahedron (i1 i2 : Icosahedron) : Prop := sorry

theorem icosahedron_edge_ratio 
  (i1 i2 : Icosahedron) 
  (h : can_form_octahedron i1 i2) : 
  i1.a / i2.a = (Real.sqrt 5 + 1) / 2 := by sorry

end NUMINAMATH_CALUDE_icosahedron_edge_ratio_l1688_168864


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1688_168830

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1688_168830


namespace NUMINAMATH_CALUDE_book_reading_time_l1688_168898

theorem book_reading_time (chapters : ℕ) (total_pages : ℕ) (pages_per_day : ℕ) : 
  chapters = 41 → total_pages = 450 → pages_per_day = 15 → 
  (total_pages / pages_per_day : ℕ) = 30 := by
sorry

end NUMINAMATH_CALUDE_book_reading_time_l1688_168898


namespace NUMINAMATH_CALUDE_clown_mobile_distribution_l1688_168890

theorem clown_mobile_distribution (total_clowns : ℕ) (num_mobiles : ℕ) (clowns_per_mobile : ℕ) :
  total_clowns = 140 →
  num_mobiles = 5 →
  total_clowns = num_mobiles * clowns_per_mobile →
  clowns_per_mobile = 28 := by
  sorry

end NUMINAMATH_CALUDE_clown_mobile_distribution_l1688_168890


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l1688_168843

theorem estimate_sqrt_expression :
  ∀ (x : ℝ), (1.4 < Real.sqrt 2 ∧ Real.sqrt 2 < 1.5) →
  (6 < (3 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt (1/3) ∧
   (3 * Real.sqrt 6 + Real.sqrt 12) * Real.sqrt (1/3) < 7) := by
  sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l1688_168843


namespace NUMINAMATH_CALUDE_inverse_of_A_squared_l1688_168833

theorem inverse_of_A_squared (A : Matrix (Fin 2) (Fin 2) ℝ) : 
  A⁻¹ = !![3, 8; -2, -5] → (A^2)⁻¹ = !![(-7), (-16); 4, 9] := by
  sorry

end NUMINAMATH_CALUDE_inverse_of_A_squared_l1688_168833


namespace NUMINAMATH_CALUDE_remainder_99_36_mod_100_l1688_168855

theorem remainder_99_36_mod_100 : 99^36 % 100 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_99_36_mod_100_l1688_168855


namespace NUMINAMATH_CALUDE_total_lives_calculation_video_game_lives_proof_l1688_168884

theorem total_lives_calculation (initial_players : Nat) (additional_players : Nat) (lives_per_player : Nat) : Nat :=
  by
  -- Define the total number of players
  let total_players := initial_players + additional_players
  
  -- Calculate the total number of lives
  let total_lives := total_players * lives_per_player
  
  -- Prove that the total number of lives is 24
  have h : total_lives = 24 := by
    -- Replace with actual proof
    sorry
  
  -- Return the result
  exact total_lives

-- Define the specific values from the problem
def initial_friends : Nat := 2
def new_players : Nat := 2
def lives_per_player : Nat := 6

-- Theorem to prove the specific case
theorem video_game_lives_proof : 
  total_lives_calculation initial_friends new_players lives_per_player = 24 :=
by
  -- Replace with actual proof
  sorry

end NUMINAMATH_CALUDE_total_lives_calculation_video_game_lives_proof_l1688_168884


namespace NUMINAMATH_CALUDE_abc_fraction_value_l1688_168868

theorem abc_fraction_value (a b c : ℝ) 
  (h1 : a * b / (a + b) = 1 / 3)
  (h2 : b * c / (b + c) = 1 / 4)
  (h3 : a * c / (c + a) = 1 / 5) :
  24 * a * b * c / (a * b + b * c + c * a) = 4 := by
sorry

end NUMINAMATH_CALUDE_abc_fraction_value_l1688_168868


namespace NUMINAMATH_CALUDE_xy_xz_yz_bounds_l1688_168814

theorem xy_xz_yz_bounds (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) :
  (∃ (a b c : ℝ), a + b + c = x + y + z ∧ a * b + b * c + c * a = 27) ∧
  (∃ (d e f : ℝ), d + e + f = x + y + z ∧ d * e + e * f + f * d = 0) ∧
  (∀ (u v w : ℝ), u + v + w = x + y + z → u * v + v * w + w * u ≤ 27) ∧
  (∀ (u v w : ℝ), u + v + w = x + y + z → u * v + v * w + w * u ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_xy_xz_yz_bounds_l1688_168814


namespace NUMINAMATH_CALUDE_det_special_matrix_l1688_168808

/-- The determinant of the matrix [[y+2, y-1, y+1], [y+1, y+2, y-1], [y-1, y+1, y+2]] 
    is equal to 6y^2 + 23y + 14 for any real number y. -/
theorem det_special_matrix (y : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![y + 2, y - 1, y + 1],
    ![y + 1, y + 2, y - 1],
    ![y - 1, y + 1, y + 2]
  ]
  Matrix.det M = 6 * y^2 + 23 * y + 14 := by
  sorry

end NUMINAMATH_CALUDE_det_special_matrix_l1688_168808


namespace NUMINAMATH_CALUDE_stack_map_front_view_l1688_168895

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Represents the top view of a stack map -/
structure StackMap :=
  (column1 : Column)
  (column2 : Column)
  (column3 : Column)

/-- Returns the maximum height of a column -/
def maxHeight (c : Column) : Nat :=
  c.foldl max 0

/-- Returns the front view of a stack map -/
def frontView (s : StackMap) : List Nat :=
  [maxHeight s.column1, maxHeight s.column2, maxHeight s.column3]

/-- The given stack map -/
def givenStackMap : StackMap :=
  { column1 := [3, 2]
  , column2 := [2, 4, 2]
  , column3 := [5, 2] }

theorem stack_map_front_view :
  frontView givenStackMap = [3, 4, 5] := by
  sorry

end NUMINAMATH_CALUDE_stack_map_front_view_l1688_168895


namespace NUMINAMATH_CALUDE_exchange_rates_problem_l1688_168883

theorem exchange_rates_problem (drum wife leopard_skin : ℕ) : 
  (2 * drum + 3 * wife + leopard_skin = 111) →
  (3 * drum + 4 * wife = 2 * leopard_skin + 8) →
  (leopard_skin % 2 = 0) →
  (drum = 20 ∧ wife = 9 ∧ leopard_skin = 44) := by
  sorry

end NUMINAMATH_CALUDE_exchange_rates_problem_l1688_168883


namespace NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1688_168879

/-- In a right-angled triangle ABC, prove that arctan(a/(b+c)) + arctan(c/(a+b)) = π/4 -/
theorem right_triangle_arctan_sum (a b c : ℝ) (h : a^2 + c^2 = b^2) :
  Real.arctan (a / (b + c)) + Real.arctan (c / (a + b)) = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_arctan_sum_l1688_168879


namespace NUMINAMATH_CALUDE_final_price_calculation_l1688_168859

/-- Calculates the final price of an item after applying discounts and tax -/
theorem final_price_calculation (original_price : ℝ) 
  (first_discount_rate : ℝ) (second_discount_rate : ℝ) (tax_rate : ℝ) : 
  original_price = 200 ∧ 
  first_discount_rate = 0.5 ∧ 
  second_discount_rate = 0.25 ∧ 
  tax_rate = 0.1 → 
  original_price * (1 - first_discount_rate) * (1 - second_discount_rate) * (1 + tax_rate) = 82.5 := by
  sorry

#check final_price_calculation

end NUMINAMATH_CALUDE_final_price_calculation_l1688_168859


namespace NUMINAMATH_CALUDE_mens_wages_l1688_168853

/-- Given that 5 men are equal to W women, W women are equal to 8 boys,
    and the total earnings of all (5 men + W women + 8 boys) is 180 Rs,
    prove that each man's wage is 36 Rs. -/
theorem mens_wages (W : ℕ) : 
  (5 : ℕ) = W → -- 5 men are equal to W women
  W = 8 → -- W women are equal to 8 boys
  (5 : ℕ) * x + W * x + 8 * x = 180 → -- total earnings equation
  x = 36 := by sorry

end NUMINAMATH_CALUDE_mens_wages_l1688_168853


namespace NUMINAMATH_CALUDE_sin_80_gt_sqrt3_sin_10_l1688_168842

theorem sin_80_gt_sqrt3_sin_10 : Real.sin (80 * π / 180) > Real.sqrt 3 * Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_80_gt_sqrt3_sin_10_l1688_168842


namespace NUMINAMATH_CALUDE_triangle_side_length_validity_l1688_168832

theorem triangle_side_length_validity 
  (a b c : ℝ) 
  (ha : a = 5) 
  (hb : b = 8) 
  (hc : c = 6) : 
  a + b > c ∧ a + c > b ∧ b + c > a :=
sorry

end NUMINAMATH_CALUDE_triangle_side_length_validity_l1688_168832


namespace NUMINAMATH_CALUDE_sam_need_change_probability_l1688_168822

/-- The number of toys in the vending machine -/
def num_toys : ℕ := 9

/-- The price of Sam's favorite toy in half-dollars -/
def favorite_toy_price : ℕ := 5

/-- The number of half-dollar coins Sam has -/
def sam_coins : ℕ := 10

/-- The probability of Sam needing to break the twenty-dollar bill -/
def probability_need_change : ℚ := 55 / 63

/-- Theorem stating the probability of Sam needing to break the twenty-dollar bill -/
theorem sam_need_change_probability :
  let total_arrangements := (num_toys.factorial : ℚ)
  let favorable_outcomes := ((num_toys - 1).factorial : ℚ) + ((num_toys - 2).factorial : ℚ) + ((num_toys - 3).factorial : ℚ)
  (1 - favorable_outcomes / total_arrangements) = probability_need_change := by
  sorry


end NUMINAMATH_CALUDE_sam_need_change_probability_l1688_168822


namespace NUMINAMATH_CALUDE_max_diff_even_digit_numbers_l1688_168801

/-- A function that checks if a natural number has all even digits -/
def all_even_digits (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d % 2 = 0

/-- A function that checks if a natural number has at least one odd digit -/
def has_odd_digit (n : ℕ) : Prop :=
  ∃ d, d ∈ n.digits 10 ∧ d % 2 = 1

/-- The theorem stating the maximum difference between two 6-digit numbers with all even digits -/
theorem max_diff_even_digit_numbers :
  ∃ (a b : ℕ),
    100000 ≤ a ∧ a < b ∧ b < 1000000 ∧
    all_even_digits a ∧
    all_even_digits b ∧
    (∀ k, a < k ∧ k < b → has_odd_digit k) ∧
    b - a = 111112 ∧
    (∀ a' b', 100000 ≤ a' ∧ a' < b' ∧ b' < 1000000 ∧
              all_even_digits a' ∧
              all_even_digits b' ∧
              (∀ k, a' < k ∧ k < b' → has_odd_digit k) →
              b' - a' ≤ 111112) :=
by sorry

end NUMINAMATH_CALUDE_max_diff_even_digit_numbers_l1688_168801


namespace NUMINAMATH_CALUDE_triangle_angle_matrix_det_zero_l1688_168861

/-- The determinant of a specific matrix formed by angles of a triangle is zero -/
theorem triangle_angle_matrix_det_zero (A B C : Real) 
  (h : A + B + C = Real.pi) : -- Condition that A, B, C are angles of a triangle
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![
    ![Real.exp A, Real.exp (-A), 1],
    ![Real.exp B, Real.exp (-B), 1],
    ![Real.exp C, Real.exp (-C), 1]
  ]
  Matrix.det M = 0 := by
  sorry


end NUMINAMATH_CALUDE_triangle_angle_matrix_det_zero_l1688_168861


namespace NUMINAMATH_CALUDE_geometric_sequence_floor_frac_l1688_168811

theorem geometric_sequence_floor_frac (x : ℝ) : 
  x ≠ 0 →
  let floor_x := ⌊x⌋
  let frac_x := x - floor_x
  (frac_x * floor_x = floor_x * x) →
  x = (5 + Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_floor_frac_l1688_168811


namespace NUMINAMATH_CALUDE_donut_distribution_l1688_168888

/-- The number of ways to distribute n items among k categories,
    with at least one item in each category. -/
def distribute_with_minimum (n k : ℕ) : ℕ :=
  Nat.choose (n - k + (k - 1)) (k - 1)

/-- Theorem: There are 35 ways to distribute 8 donuts among 4 types,
    with at least one donut of each type. -/
theorem donut_distribution : distribute_with_minimum 8 4 = 35 := by
  sorry

#eval distribute_with_minimum 8 4

end NUMINAMATH_CALUDE_donut_distribution_l1688_168888


namespace NUMINAMATH_CALUDE_additional_curtain_material_l1688_168878

-- Define the room height in feet
def room_height_feet : ℕ := 8

-- Define the desired curtain length in inches
def desired_curtain_length : ℕ := 101

-- Define the conversion factor from feet to inches
def feet_to_inches : ℕ := 12

-- Theorem to prove the additional material needed
theorem additional_curtain_material :
  desired_curtain_length - (room_height_feet * feet_to_inches) = 5 := by
  sorry

end NUMINAMATH_CALUDE_additional_curtain_material_l1688_168878


namespace NUMINAMATH_CALUDE_remainder_of_least_number_l1688_168837

theorem remainder_of_least_number (n : ℕ) (h1 : n = 261) (h2 : ∀ m < n, m % 37 ≠ n % 37 ∨ m % 7 ≠ n % 7) : n % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_least_number_l1688_168837


namespace NUMINAMATH_CALUDE_modulus_of_z_l1688_168860

theorem modulus_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_modulus_of_z_l1688_168860


namespace NUMINAMATH_CALUDE_symmetry_sum_l1688_168851

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are equal
    and their y-coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (A B : ℝ × ℝ) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

/-- If point A(m, 3) is symmetric to point B(2, n) with respect to the x-axis,
    then m + n = -1. -/
theorem symmetry_sum (m n : ℝ) :
  symmetric_wrt_x_axis (m, 3) (2, n) → m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetry_sum_l1688_168851


namespace NUMINAMATH_CALUDE_largest_number_value_l1688_168865

theorem largest_number_value
  (a b c : ℝ)
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 100)
  (h_larger_diff : c - b = 10)
  (h_smaller_diff : b - a = 5) :
  c = 125 / 3 := by
sorry

end NUMINAMATH_CALUDE_largest_number_value_l1688_168865


namespace NUMINAMATH_CALUDE_base_edges_same_color_l1688_168816

/-- A color type representing red or green -/
inductive Color
| Red
| Green

/-- A vertex of the prism -/
structure Vertex where
  base : Bool  -- True for top base, False for bottom base
  index : Fin 5

/-- An edge of the prism -/
structure Edge where
  v1 : Vertex
  v2 : Vertex

/-- A prism with pentagonal bases -/
structure Prism where
  /-- The color of each edge -/
  edge_color : Edge → Color
  /-- Ensure that any triangle has edges of different colors -/
  triangle_property : ∀ (v1 v2 v3 : Vertex),
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v3 ≠ v1 →
    (edge_color ⟨v1, v2⟩ ≠ edge_color ⟨v2, v3⟩ ∨
     edge_color ⟨v2, v3⟩ ≠ edge_color ⟨v3, v1⟩ ∨
     edge_color ⟨v3, v1⟩ ≠ edge_color ⟨v1, v2⟩)

/-- The main theorem -/
theorem base_edges_same_color (p : Prism) :
  (∀ (i j : Fin 5), p.edge_color ⟨⟨true, i⟩, ⟨true, j⟩⟩ = p.edge_color ⟨⟨false, i⟩, ⟨false, j⟩⟩) :=
sorry

end NUMINAMATH_CALUDE_base_edges_same_color_l1688_168816


namespace NUMINAMATH_CALUDE_student_distribution_theorem_l1688_168828

/-- The number of ways to distribute students among attractions -/
def distribute_students (n m k : ℕ) : ℕ :=
  Nat.choose n k * (m - 1)^(n - k)

/-- Theorem: The number of ways to distribute 6 students among 6 attractions,
    where exactly 2 students visit a specific attraction, is C₆² × 5⁴ -/
theorem student_distribution_theorem :
  distribute_students 6 6 2 = Nat.choose 6 2 * 5^4 := by
  sorry

end NUMINAMATH_CALUDE_student_distribution_theorem_l1688_168828


namespace NUMINAMATH_CALUDE_jelly_bean_problem_l1688_168894

theorem jelly_bean_problem (initial_red : ℚ) (initial_green : ℚ) (initial_blue : ℚ)
  (removed : ℚ) (final_blue_percentage : ℚ) (final_red_percentage : ℚ) :
  initial_red = 54 / 100 →
  initial_green = 30 / 100 →
  initial_blue = 16 / 100 →
  initial_red + initial_green + initial_blue = 1 →
  removed ≥ 0 →
  removed ≤ min initial_red initial_green →
  final_blue_percentage = 20 / 100 →
  final_blue_percentage = initial_blue / (1 - 2 * removed) →
  final_red_percentage = (initial_red - removed) / (1 - 2 * removed) →
  final_red_percentage = 55 / 100 :=
by sorry

end NUMINAMATH_CALUDE_jelly_bean_problem_l1688_168894


namespace NUMINAMATH_CALUDE_xyz_mod_8_l1688_168826

theorem xyz_mod_8 (x y z : ℕ) : 
  x < 8 → y < 8 → z < 8 → x > 0 → y > 0 → z > 0 →
  (x * y * z) % 8 = 1 → 
  (3 * z) % 8 = 5 → 
  (7 * y) % 8 = (4 + y) % 8 → 
  (x + y + z) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_xyz_mod_8_l1688_168826


namespace NUMINAMATH_CALUDE_solution_values_l1688_168819

/-- A quadratic function f(x) = ax^2 - 2(a+1)x + b where a and b are real numbers. -/
def f (a b x : ℝ) : ℝ := a * x^2 - 2 * (a + 1) * x + b

/-- The property that the solution set of f(x) < 0 is (1,2) -/
def solution_set_property (a b : ℝ) : Prop :=
  ∀ x, f a b x < 0 ↔ 1 < x ∧ x < 2

/-- Theorem stating that if the solution set property holds, then a = 2 and b = 4 -/
theorem solution_values (a b : ℝ) (h : solution_set_property a b) : a = 2 ∧ b = 4 := by
  sorry


end NUMINAMATH_CALUDE_solution_values_l1688_168819


namespace NUMINAMATH_CALUDE_point_on_line_l1688_168840

/-- Given two points (m, n) and (m + a, n + 1.5) on the line x = 2y + 5, prove that a = 3 -/
theorem point_on_line (m n a : ℝ) : 
  (m = 2 * n + 5) → 
  (m + a = 2 * (n + 1.5) + 5) → 
  a = 3 := by sorry

end NUMINAMATH_CALUDE_point_on_line_l1688_168840


namespace NUMINAMATH_CALUDE_xiao_ming_banknote_combinations_l1688_168823

def is_valid_combination (x y z : ℕ) : Prop :=
  x + 2*y + 5*z = 18 ∧ x + y + z ≤ 10 ∧ (x > 0 ∧ y > 0) ∨ (x > 0 ∧ z > 0) ∨ (y > 0 ∧ z > 0)

def count_valid_combinations : ℕ := sorry

theorem xiao_ming_banknote_combinations : count_valid_combinations = 9 := by sorry

end NUMINAMATH_CALUDE_xiao_ming_banknote_combinations_l1688_168823


namespace NUMINAMATH_CALUDE_neon_signs_blink_together_l1688_168852

theorem neon_signs_blink_together (a b : ℕ) (ha : a = 9) (hb : b = 15) : 
  Nat.lcm a b = 45 := by
  sorry

end NUMINAMATH_CALUDE_neon_signs_blink_together_l1688_168852


namespace NUMINAMATH_CALUDE_evaluate_expression_l1688_168807

theorem evaluate_expression (b : ℝ) : 
  let x : ℝ := b + 9
  (x - b + 5) = 14 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1688_168807


namespace NUMINAMATH_CALUDE_valid_arrangement_probability_l1688_168897

-- Define the number of teachers and days
def num_teachers : ℕ := 6
def num_days : ℕ := 3
def teachers_per_day : ℕ := 2

-- Define the teachers who have restrictions
structure RestrictedTeacher where
  name : String
  restricted_day : ℕ

-- Define the specific restrictions
def wang : RestrictedTeacher := ⟨"Wang", 2⟩
def li : RestrictedTeacher := ⟨"Li", 3⟩

-- Define the probability function
def probability_of_valid_arrangement (t : ℕ) (d : ℕ) (tpd : ℕ) 
  (r1 r2 : RestrictedTeacher) : ℚ :=
  7/15

-- State the theorem
theorem valid_arrangement_probability :
  probability_of_valid_arrangement num_teachers num_days teachers_per_day wang li = 7/15 := by
  sorry

end NUMINAMATH_CALUDE_valid_arrangement_probability_l1688_168897


namespace NUMINAMATH_CALUDE_bus_seat_capacity_l1688_168827

/-- Represents the seating arrangement and capacity of a bus -/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  total_capacity : ℕ

/-- Calculates the number of people each regular seat can hold -/
def seats_capacity (bus : BusSeating) : ℚ :=
  (bus.total_capacity - bus.back_seat_capacity) / (bus.left_seats + bus.right_seats)

/-- Theorem stating that for the given bus configuration, each seat can hold 3 people -/
theorem bus_seat_capacity :
  let bus : BusSeating := {
    left_seats := 15,
    right_seats := 12,
    back_seat_capacity := 7,
    total_capacity := 88
  }
  seats_capacity bus = 3 := by sorry

end NUMINAMATH_CALUDE_bus_seat_capacity_l1688_168827


namespace NUMINAMATH_CALUDE_derivative_at_two_l1688_168815

open Real

theorem derivative_at_two (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x > 0, f x = x^2 + 3 * x * (deriv f 2) - log x) : 
  deriv f 2 = -7/4 := by
  sorry

end NUMINAMATH_CALUDE_derivative_at_two_l1688_168815


namespace NUMINAMATH_CALUDE_intersection_M_N_l1688_168847

def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1688_168847


namespace NUMINAMATH_CALUDE_rectangle_segment_length_l1688_168825

/-- Given a rectangle ABCD with side lengths AB = 6 and BC = 5,
    and a segment GH through B perpendicular to DB,
    with A on DG and C on DH, prove that GH = 11√61/6 -/
theorem rectangle_segment_length (A B C D G H : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let DB := Real.sqrt ((D.1 - B.1)^2 + (D.2 - B.2)^2)
  let GH := Real.sqrt ((G.1 - H.1)^2 + (G.2 - H.2)^2)
  AB = 6 →
  BC = 5 →
  (G.1 - H.1) * (D.1 - B.1) + (G.2 - H.2) * (D.2 - B.2) = 0 →  -- GH ⟂ DB
  ∃ t₁ : ℝ, A = t₁ • (G - D) + D →  -- A lies on DG
  ∃ t₂ : ℝ, C = t₂ • (H - D) + D →  -- C lies on DH
  GH = 11 * Real.sqrt 61 / 6 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_segment_length_l1688_168825


namespace NUMINAMATH_CALUDE_probability_even_product_l1688_168881

def range_start : ℕ := 6
def range_end : ℕ := 18

def is_in_range (n : ℕ) : Prop := range_start ≤ n ∧ n ≤ range_end

def total_integers : ℕ := range_end - range_start + 1

def total_combinations : ℕ := (total_integers * (total_integers - 1)) / 2

def count_even_in_range : ℕ := (range_end - range_start) / 2 + 1

def count_odd_in_range : ℕ := total_integers - count_even_in_range

def combinations_with_odd_product : ℕ := (count_odd_in_range * (count_odd_in_range - 1)) / 2

def combinations_with_even_product : ℕ := total_combinations - combinations_with_odd_product

theorem probability_even_product : 
  (combinations_with_even_product : ℚ) / total_combinations = 9 / 13 := by sorry

end NUMINAMATH_CALUDE_probability_even_product_l1688_168881


namespace NUMINAMATH_CALUDE_minimal_leasing_cost_l1688_168857

/-- Represents the daily production and cost of equipment types -/
structure EquipmentType where
  productA : ℕ
  productB : ℕ
  cost : ℕ

/-- Represents the company's production requirements -/
structure Requirements where
  minProductA : ℕ
  minProductB : ℕ

/-- Calculates the total production and cost for a given number of days of each equipment type -/
def calculateProduction (typeA : EquipmentType) (typeB : EquipmentType) (daysA : ℕ) (daysB : ℕ) : ℕ × ℕ × ℕ :=
  (daysA * typeA.productA + daysB * typeB.productA,
   daysA * typeB.productB + daysB * typeB.productB,
   daysA * typeA.cost + daysB * typeB.cost)

/-- Checks if the production meets the requirements -/
def meetsRequirements (prod : ℕ × ℕ × ℕ) (req : Requirements) : Prop :=
  prod.1 ≥ req.minProductA ∧ prod.2.1 ≥ req.minProductB

/-- Theorem stating that the minimal leasing cost is 2000 yuan -/
theorem minimal_leasing_cost 
  (typeA : EquipmentType)
  (typeB : EquipmentType)
  (req : Requirements)
  (h1 : typeA.productA = 5)
  (h2 : typeA.productB = 10)
  (h3 : typeA.cost = 200)
  (h4 : typeB.productA = 6)
  (h5 : typeB.productB = 20)
  (h6 : typeB.cost = 300)
  (h7 : req.minProductA = 50)
  (h8 : req.minProductB = 140) :
  ∃ (daysA daysB : ℕ), 
    let prod := calculateProduction typeA typeB daysA daysB
    meetsRequirements prod req ∧ 
    prod.2.2 = 2000 ∧
    (∀ (x y : ℕ), 
      let otherProd := calculateProduction typeA typeB x y
      meetsRequirements otherProd req → otherProd.2.2 ≥ 2000) := by
  sorry


end NUMINAMATH_CALUDE_minimal_leasing_cost_l1688_168857


namespace NUMINAMATH_CALUDE_proposition_b_is_true_l1688_168804

theorem proposition_b_is_true : 3 > 4 ∨ 3 < 4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_is_true_l1688_168804


namespace NUMINAMATH_CALUDE_intersection_theorem_l1688_168803

-- Define the four lines
def line1 (x y : ℚ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℚ) : Prop := x + 3 * y = 3
def line3 (x y : ℚ) : Prop := 6 * x - 4 * y = 2
def line4 (x y : ℚ) : Prop := 5 * x - 15 * y = 15

-- Define the set of intersection points
def intersection_points : Set (ℚ × ℚ) :=
  {(18/11, 13/11), (21/11, 8/11)}

-- Define a function to check if a point lies on at least two lines
def on_at_least_two_lines (p : ℚ × ℚ) : Prop :=
  let (x, y) := p
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line1 x y ∧ line4 x y) ∨
  (line2 x y ∧ line3 x y) ∨ (line2 x y ∧ line4 x y) ∨ (line3 x y ∧ line4 x y)

-- Theorem statement
theorem intersection_theorem :
  {p : ℚ × ℚ | on_at_least_two_lines p} = intersection_points := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l1688_168803


namespace NUMINAMATH_CALUDE_not_all_prime_l1688_168863

theorem not_all_prime (a₁ a₂ a₃ : ℕ) : 
  a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₂ ≠ a₃ →
  a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 →
  a₁ ∣ (a₂ + a₃ + a₂ * a₃) →
  a₂ ∣ (a₃ + a₁ + a₃ * a₁) →
  a₃ ∣ (a₁ + a₂ + a₁ * a₂) →
  ¬(Prime a₁ ∧ Prime a₂ ∧ Prime a₃) :=
by sorry

end NUMINAMATH_CALUDE_not_all_prime_l1688_168863


namespace NUMINAMATH_CALUDE_prob_three_non_defective_l1688_168800

/-- The probability of selecting 3 non-defective pencils from a box of 7 pencils, where 2 are defective. -/
theorem prob_three_non_defective (total : Nat) (defective : Nat) (selected : Nat) :
  total = 7 →
  defective = 2 →
  selected = 3 →
  (Nat.choose (total - defective) selected : ℚ) / (Nat.choose total selected : ℚ) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_non_defective_l1688_168800


namespace NUMINAMATH_CALUDE_inclined_plane_friction_l1688_168889

/-- The coefficient of friction between a block and an inclined plane -/
theorem inclined_plane_friction (P F_up F_down : ℝ) (α : ℝ) (μ : ℝ) :
  F_up = 3 * F_down →
  F_up + F_down = P →
  F_up = P * Real.sin α + μ * P * Real.cos α →
  F_down = P * Real.sin α - μ * P * Real.cos α →
  μ = Real.sqrt 3 / 6 := by
sorry

end NUMINAMATH_CALUDE_inclined_plane_friction_l1688_168889


namespace NUMINAMATH_CALUDE_circle_tangent_line_l1688_168850

theorem circle_tangent_line (a : ℝ) : 
  (∃ (x y : ℝ), x - y + 1 = 0 ∧ x^2 + y^2 - 2*x + 1 - a = 0 ∧ 
  ∀ (x' y' : ℝ), x' - y' + 1 = 0 → x'^2 + y'^2 - 2*x' + 1 - a ≥ 0) → 
  a = 2 := by
sorry

end NUMINAMATH_CALUDE_circle_tangent_line_l1688_168850


namespace NUMINAMATH_CALUDE_arun_age_is_60_l1688_168818

/-- Proves that Arun's age is 60 years given the conditions from the problem -/
theorem arun_age_is_60 (arun_age madan_age gokul_age : ℕ) 
  (h1 : (arun_age - 6) / 18 = gokul_age)
  (h2 : gokul_age = madan_age - 2)
  (h3 : madan_age = 5) : 
  arun_age = 60 := by
  sorry

end NUMINAMATH_CALUDE_arun_age_is_60_l1688_168818


namespace NUMINAMATH_CALUDE_factory_output_increase_l1688_168829

theorem factory_output_increase (P : ℝ) : 
  (1 + P / 100) * (1 + 20 / 100) * (1 - 24.242424242424242 / 100) = 1 → P = 10 := by
sorry

end NUMINAMATH_CALUDE_factory_output_increase_l1688_168829


namespace NUMINAMATH_CALUDE_pascal_triangle_row20_element7_l1688_168892

theorem pascal_triangle_row20_element7 : Nat.choose 20 6 = 38760 := by
  sorry

end NUMINAMATH_CALUDE_pascal_triangle_row20_element7_l1688_168892


namespace NUMINAMATH_CALUDE_library_visitors_l1688_168887

/-- Proves that the average number of visitors on Sundays is 540 given the specified conditions --/
theorem library_visitors (total_days : Nat) (non_sunday_visitors : Nat) (avg_visitors : Nat) :
  total_days = 30 ∧
  non_sunday_visitors = 240 ∧
  avg_visitors = 290 →
  (5 * (((avg_visitors * total_days) - (25 * non_sunday_visitors)) / 5) + 25 * non_sunday_visitors) / total_days = avg_visitors :=
by sorry

end NUMINAMATH_CALUDE_library_visitors_l1688_168887


namespace NUMINAMATH_CALUDE_equation_solutions_l1688_168809

theorem equation_solutions :
  (∃ x : ℝ, 2 * (x - 2) = 3 * x - 7 ∧ x = 3) ∧
  (∃ x : ℝ, (x - 1) / 2 - (2 * x + 3) / 6 = 1 ∧ x = 12) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l1688_168809


namespace NUMINAMATH_CALUDE_sum_product_inequality_l1688_168886

theorem sum_product_inequality (a b c : ℝ) (h : a + b + c = 0) : a * b + b * c + c * a ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_product_inequality_l1688_168886


namespace NUMINAMATH_CALUDE_function_satisfying_equation_l1688_168810

theorem function_satisfying_equation (f : ℝ → ℝ) :
  (∀ x : ℝ, 2 * f x - f (-x) = 3 * x + 1) → (∀ x : ℝ, f x = x + 1) :=
by sorry

end NUMINAMATH_CALUDE_function_satisfying_equation_l1688_168810


namespace NUMINAMATH_CALUDE_total_pepper_weight_l1688_168834

theorem total_pepper_weight :
  let green_peppers : ℝ := 3.25
  let red_peppers : ℝ := 2.5
  let yellow_peppers : ℝ := 1.75
  let orange_peppers : ℝ := 4.6
  green_peppers + red_peppers + yellow_peppers + orange_peppers = 12.1 := by
  sorry

end NUMINAMATH_CALUDE_total_pepper_weight_l1688_168834


namespace NUMINAMATH_CALUDE_min_abs_z_l1688_168873

theorem min_abs_z (z : ℂ) (h : Complex.abs (z - 2*I) + Complex.abs (z - 5) = 7) :
  Complex.abs z ≥ 10 / Real.sqrt 29 ∧ ∃ w : ℂ, Complex.abs (w - 2*I) + Complex.abs (w - 5) = 7 ∧ Complex.abs w = 10 / Real.sqrt 29 :=
by sorry

end NUMINAMATH_CALUDE_min_abs_z_l1688_168873


namespace NUMINAMATH_CALUDE_intersection_chord_length_l1688_168862

theorem intersection_chord_length :
  let line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = p.1}
  let circle : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 8}
  let intersection := line ∩ circle
  ∃ (A B : ℝ × ℝ), A ∈ intersection ∧ B ∈ intersection ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt 30 :=
by sorry

end NUMINAMATH_CALUDE_intersection_chord_length_l1688_168862


namespace NUMINAMATH_CALUDE_exponent_division_l1688_168831

theorem exponent_division (a : ℝ) : a^3 / a^2 = a := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1688_168831


namespace NUMINAMATH_CALUDE_smaller_angle_at_4_oclock_l1688_168845

/-- The number of hours on a clock face -/
def clock_hours : ℕ := 12

/-- The number of degrees in a full circle -/
def full_circle_degrees : ℕ := 360

/-- The hour we're considering -/
def target_hour : ℕ := 4

/-- Calculates the angle between clock hands at a given hour -/
def clock_angle (hour : ℕ) : ℕ := 
  (hour * full_circle_degrees) / clock_hours

theorem smaller_angle_at_4_oclock : 
  min (clock_angle target_hour) (full_circle_degrees - clock_angle target_hour) = 120 := by
  sorry

end NUMINAMATH_CALUDE_smaller_angle_at_4_oclock_l1688_168845


namespace NUMINAMATH_CALUDE_state_returns_sold_l1688_168820

-- Define the prices and quantities
def federal_price : ℕ := 50
def state_price : ℕ := 30
def quarterly_price : ℕ := 80
def federal_quantity : ℕ := 60
def quarterly_quantity : ℕ := 10
def total_revenue : ℕ := 4400

-- Define the function to calculate total revenue
def calculate_revenue (state_quantity : ℕ) : ℕ :=
  federal_price * federal_quantity +
  state_price * state_quantity +
  quarterly_price * quarterly_quantity

-- Theorem statement
theorem state_returns_sold : 
  ∃ (state_quantity : ℕ), calculate_revenue state_quantity = total_revenue ∧ state_quantity = 20 := by
  sorry

end NUMINAMATH_CALUDE_state_returns_sold_l1688_168820


namespace NUMINAMATH_CALUDE_soccer_league_games_l1688_168805

theorem soccer_league_games (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1688_168805


namespace NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l1688_168844

def is_positive_even (n : ℕ) : Prop := n > 0 ∧ n % 2 = 0

def has_positive_units_digit (n : ℕ) : Prop := n % 10 > 0

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem units_digit_of_p_plus_two (p q : ℕ) (x : ℕ+) :
  is_positive_even p →
  is_positive_even q →
  has_positive_units_digit p →
  has_positive_units_digit q →
  units_digit (p^3) - units_digit (p^2) = 0 →
  sum_of_digits p % q = 0 →
  p^(x : ℕ) = q →
  units_digit (p + 2) = 8 :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_p_plus_two_l1688_168844


namespace NUMINAMATH_CALUDE_circle_to_rectangle_length_l1688_168856

/-- Given a circle with radius R, when divided into equal parts and rearranged to form
    an approximate rectangle with perimeter 20.7 cm, the length of this rectangle is π * R. -/
theorem circle_to_rectangle_length (R : ℝ) (h : (2 * R + 2 * π * R / 2) = 20.7) :
  π * R = (20.7 : ℝ) / 2 - R := by
  sorry

end NUMINAMATH_CALUDE_circle_to_rectangle_length_l1688_168856


namespace NUMINAMATH_CALUDE_prob_6_or_less_l1688_168891

/-- The probability of an archer hitting 9 rings or more in one shot. -/
def p_9_or_more : ℝ := 0.5

/-- The probability of an archer hitting exactly 8 rings in one shot. -/
def p_8 : ℝ := 0.2

/-- The probability of an archer hitting exactly 7 rings in one shot. -/
def p_7 : ℝ := 0.1

/-- Theorem: The probability of an archer hitting 6 rings or less in one shot is 0.2. -/
theorem prob_6_or_less : 1 - (p_9_or_more + p_8 + p_7) = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_prob_6_or_less_l1688_168891


namespace NUMINAMATH_CALUDE_ratio_proof_l1688_168876

theorem ratio_proof (x y : ℝ) (h : (3 * x^2 - 2 * y^2) / (x^2 + 4 * y^2) = 5 / 7) :
  x / y = Real.sqrt (17 / 8) :=
by sorry

end NUMINAMATH_CALUDE_ratio_proof_l1688_168876


namespace NUMINAMATH_CALUDE_solution_of_functional_equation_l1688_168806

def f (x : ℝ) := x^2 + 2*x - 5

theorem solution_of_functional_equation :
  let s1 := (-1 + Real.sqrt 21) / 2
  let s2 := (-1 - Real.sqrt 21) / 2
  let s3 := (-3 + Real.sqrt 17) / 2
  let s4 := (-3 - Real.sqrt 17) / 2
  (∀ x : ℝ, f (f x) = x ↔ x = s1 ∨ x = s2 ∨ x = s3 ∨ x = s4) :=
by sorry

end NUMINAMATH_CALUDE_solution_of_functional_equation_l1688_168806


namespace NUMINAMATH_CALUDE_product_as_sum_of_tens_l1688_168869

theorem product_as_sum_of_tens :
  ∃ n : ℕ, n * 10 = 100 * 100 ∧ n = 1000 := by
  sorry

end NUMINAMATH_CALUDE_product_as_sum_of_tens_l1688_168869


namespace NUMINAMATH_CALUDE_technician_round_trip_l1688_168836

theorem technician_round_trip 
  (D : ℝ) 
  (P : ℝ) 
  (h1 : D > 0) -- Ensure distance is positive
  (h2 : 0 ≤ P ∧ P ≤ 100) -- Ensure percentage is between 0 and 100
  (h3 : D + (P / 100) * D = 0.7 * (2 * D)) -- Total distance traveled equals 70% of round-trip
  : P = 40 := by
sorry

end NUMINAMATH_CALUDE_technician_round_trip_l1688_168836


namespace NUMINAMATH_CALUDE_smallest_base_is_five_l1688_168812

/-- Representation of a number in base b -/
def BaseRepresentation (digits : List Nat) (b : Nat) : Nat :=
  digits.foldl (fun acc d => acc * b + d) 0

/-- Condition: In base b, 12_b squared equals 144_b -/
def SquareCondition (b : Nat) : Prop :=
  (BaseRepresentation [1, 2] b) ^ 2 = BaseRepresentation [1, 4, 4] b

/-- The smallest base b greater than 4 for which 12_b squared equals 144_b is 5 -/
theorem smallest_base_is_five :
  ∃ (b : Nat), b > 4 ∧ SquareCondition b ∧ ∀ (k : Nat), k > 4 ∧ k < b → ¬SquareCondition k :=
by sorry

end NUMINAMATH_CALUDE_smallest_base_is_five_l1688_168812


namespace NUMINAMATH_CALUDE_inequality_proof_l1688_168870

theorem inequality_proof (x y z : ℝ) (h_nonneg_x : x ≥ 0) (h_nonneg_y : y ≥ 0) (h_nonneg_z : z ≥ 0)
  (h_condition : x^2 + y^2 + z^2 = x + y + z) :
  (x + 1) / Real.sqrt (x^5 + x + 1) + (y + 1) / Real.sqrt (y^5 + y + 1) + 
  (z + 1) / Real.sqrt (z^5 + z + 1) ≥ 3 ∧
  ((x + 1) / Real.sqrt (x^5 + x + 1) + (y + 1) / Real.sqrt (y^5 + y + 1) + 
   (z + 1) / Real.sqrt (z^5 + z + 1) = 3 ↔ x = 0 ∧ y = 0 ∧ z = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l1688_168870


namespace NUMINAMATH_CALUDE_no_valid_labeling_l1688_168824

/-- A labeling function that assigns one of four labels to each point in the integer lattice. -/
def Labeling := ℤ × ℤ → Fin 4

/-- Predicate that checks if a given labeling satisfies the constraints on a unit square. -/
def valid_square (f : Labeling) (x y : ℤ) : Prop :=
  f (x, y) ≠ f (x + 1, y) ∧
  f (x, y) ≠ f (x, y + 1) ∧
  f (x, y) ≠ f (x + 1, y + 1) ∧
  f (x + 1, y) ≠ f (x, y + 1) ∧
  f (x + 1, y) ≠ f (x + 1, y + 1) ∧
  f (x, y + 1) ≠ f (x + 1, y + 1)

/-- Predicate that checks if a given labeling satisfies the constraints on a row. -/
def valid_row (f : Labeling) (y : ℤ) : Prop :=
  ∀ x : ℤ, ∃ i j k l : ℤ, i < j ∧ j < k ∧ k < l ∧
    f (i, y) ≠ f (j, y) ∧ f (j, y) ≠ f (k, y) ∧ f (k, y) ≠ f (l, y) ∧
    f (i, y) ≠ f (k, y) ∧ f (i, y) ≠ f (l, y) ∧ f (j, y) ≠ f (l, y)

/-- Predicate that checks if a given labeling satisfies the constraints on a column. -/
def valid_column (f : Labeling) (x : ℤ) : Prop :=
  ∀ y : ℤ, ∃ i j k l : ℤ, i < j ∧ j < k ∧ k < l ∧
    f (x, i) ≠ f (x, j) ∧ f (x, j) ≠ f (x, k) ∧ f (x, k) ≠ f (x, l) ∧
    f (x, i) ≠ f (x, k) ∧ f (x, i) ≠ f (x, l) ∧ f (x, j) ≠ f (x, l)

/-- Theorem stating that no labeling can satisfy all the given constraints. -/
theorem no_valid_labeling : ¬∃ f : Labeling, 
  (∀ x y : ℤ, valid_square f x y) ∧ 
  (∀ y : ℤ, valid_row f y) ∧ 
  (∀ x : ℤ, valid_column f x) := by
  sorry

end NUMINAMATH_CALUDE_no_valid_labeling_l1688_168824
