import Mathlib

namespace NUMINAMATH_CALUDE_lcm_problem_l608_60897

theorem lcm_problem (a b : ℕ) (h : Nat.gcd a b = 47) (ha : a = 210) (hb : b = 517) :
  Nat.lcm a b = 2310 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l608_60897


namespace NUMINAMATH_CALUDE_factorization_x_squared_minus_xy_l608_60826

theorem factorization_x_squared_minus_xy (x y : ℝ) : x^2 - x*y = x*(x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_x_squared_minus_xy_l608_60826


namespace NUMINAMATH_CALUDE_carl_accident_cost_l608_60892

/-- Carl's car accident cost calculation -/
theorem carl_accident_cost (property_damage medical_bills : ℕ) 
  (h1 : property_damage = 40000)
  (h2 : medical_bills = 70000)
  (carl_percentage : ℚ)
  (h3 : carl_percentage = 1/5) :
  carl_percentage * (property_damage + medical_bills : ℚ) = 22000 := by
sorry

end NUMINAMATH_CALUDE_carl_accident_cost_l608_60892


namespace NUMINAMATH_CALUDE_f_intersects_x_axis_min_distance_between_roots_range_of_a_l608_60843

/-- The quadratic function f(x) = x^2 - 2ax - 2(a + 1) -/
def f (a x : ℝ) : ℝ := x^2 - 2*a*x - 2*(a + 1)

theorem f_intersects_x_axis (a : ℝ) : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 := by
  sorry

theorem min_distance_between_roots (a : ℝ) :
  ∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧ |x₁ - x₂| ≥ 2 ∧ (∀ y₁ y₂ : ℝ, f a y₁ = 0 → f a y₂ = 0 → |y₁ - y₂| ≥ 2) := by
  sorry

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x > -1 → f a x + 3 ≥ 0) → a ≤ Real.sqrt 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_f_intersects_x_axis_min_distance_between_roots_range_of_a_l608_60843


namespace NUMINAMATH_CALUDE_decimal_places_product_specific_case_l608_60856

/-- Given two real numbers a and b, this function returns the number of decimal places in their product. -/
def decimal_places_in_product (a b : ℝ) : ℕ :=
  sorry

/-- This function returns the number of decimal places in a real number. -/
def count_decimal_places (x : ℝ) : ℕ :=
  sorry

theorem decimal_places_product (a b : ℝ) :
  decimal_places_in_product a b = count_decimal_places a + count_decimal_places b :=
sorry

theorem specific_case : 
  decimal_places_in_product 0.38 0.26 = 4 :=
sorry

end NUMINAMATH_CALUDE_decimal_places_product_specific_case_l608_60856


namespace NUMINAMATH_CALUDE_system_solution_range_l608_60803

theorem system_solution_range (x y a : ℝ) : 
  x + 3*y = 2 + a → 
  3*x + y = -4*a → 
  x + y > 2 → 
  a < -2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_range_l608_60803


namespace NUMINAMATH_CALUDE_drama_club_ticket_sales_l608_60806

theorem drama_club_ticket_sales 
  (total_tickets : ℕ) 
  (adult_price student_price : ℚ) 
  (total_amount : ℚ) 
  (h1 : total_tickets = 1500)
  (h2 : adult_price = 12)
  (h3 : student_price = 6)
  (h4 : total_amount = 16200) :
  ∃ (adult_tickets student_tickets : ℕ),
    adult_tickets + student_tickets = total_tickets ∧
    adult_price * adult_tickets + student_price * student_tickets = total_amount ∧
    student_tickets = 300 := by
  sorry

end NUMINAMATH_CALUDE_drama_club_ticket_sales_l608_60806


namespace NUMINAMATH_CALUDE_pancake_flour_calculation_l608_60817

/-- Given a recipe for 20 pancakes requiring 3 cups of flour,
    prove that 27 cups of flour are needed for 180 pancakes. -/
theorem pancake_flour_calculation
  (original_pancakes : ℕ)
  (original_flour : ℕ)
  (desired_pancakes : ℕ)
  (h1 : original_pancakes = 20)
  (h2 : original_flour = 3)
  (h3 : desired_pancakes = 180) :
  (desired_pancakes / original_pancakes) * original_flour = 27 :=
by sorry

end NUMINAMATH_CALUDE_pancake_flour_calculation_l608_60817


namespace NUMINAMATH_CALUDE_polynomial_simplification_l608_60800

/-- The given polynomial is equal to its simplified form for all x. -/
theorem polynomial_simplification :
  ∀ x : ℝ, 3 - 5*x - 7*x^2 + 9 + 11*x - 13*x^2 - 15 + 17*x + 19*x^2 + 2*x^3 - 4*x^3 =
            -2*x^3 - x^2 + 23*x - 3 :=
by
  sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l608_60800


namespace NUMINAMATH_CALUDE_cube_plane_intersection_l608_60879

-- Define a cube
def Cube : Type := Unit

-- Define a plane
def Plane : Type := Unit

-- Define the set of faces of a cube
def faces (Q : Cube) : Set Unit := sorry

-- Define the union of faces
def S (Q : Cube) : Set Unit := faces Q

-- Define the set of planes intersecting the cube
def intersecting_planes (Q : Cube) (k : ℕ) : Set Plane := sorry

-- Define the union of intersecting planes
def P (Q : Cube) (k : ℕ) : Set Unit := sorry

-- Define the set of one-third points on the edges of a cube face
def one_third_points (face : Unit) : Set Unit := sorry

-- Define the set of segments joining one-third points on the same face
def one_third_segments (Q : Cube) : Set Unit := sorry

-- State the theorem
theorem cube_plane_intersection (Q : Cube) :
  ∃ k : ℕ, 
    (∀ k' : ℕ, k' ≥ k → 
      (P Q k' ∩ S Q = one_third_segments Q) → 
      k' = k) ∧
    (∀ k' : ℕ, k' ≤ k → 
      (P Q k' ∩ S Q = one_third_segments Q) → 
      k' = k) :=
sorry

end NUMINAMATH_CALUDE_cube_plane_intersection_l608_60879


namespace NUMINAMATH_CALUDE_minimize_square_root_difference_l608_60825

theorem minimize_square_root_difference (p : ℕ) (h_p : Nat.Prime p) (h_p_odd : Odd p) :
  ∃ (x y : ℕ), 
    x > 0 ∧ y > 0 ∧ x ≤ y ∧
    (Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≥ 0) ∧
    (∀ (a b : ℕ), a > 0 → b > 0 → a ≤ b → 
      Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b ≥ 0 →
      Real.sqrt (2 * p) - Real.sqrt x - Real.sqrt y ≤ Real.sqrt (2 * p) - Real.sqrt a - Real.sqrt b) ∧
    x = (p - 1) / 2 ∧ y = (p + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_minimize_square_root_difference_l608_60825


namespace NUMINAMATH_CALUDE_determinant_evaluation_l608_60812

theorem determinant_evaluation (x z : ℝ) : 
  Matrix.det !![1, x, z; 1, x + z, z; 1, x, x + z] = x * z + 2 * z^2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_evaluation_l608_60812


namespace NUMINAMATH_CALUDE_hyperbola_equation_l608_60852

/-- The standard equation of a hyperbola with one focus at (2,0) and an asymptote
    with inclination angle of 60° is x^2 - (y^2/3) = 1 -/
theorem hyperbola_equation (C : Set (ℝ × ℝ)) (F : ℝ × ℝ) (θ : ℝ) :
  F = (2, 0) →
  θ = π/3 →
  (∃ (a b : ℝ), ∀ (x y : ℝ),
    (x, y) ∈ C ↔ x^2 / a^2 - y^2 / b^2 = 1 ∧
    b / a = Real.sqrt 3 ∧
    2^2 = a^2 + b^2) →
  (∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 - y^2 / 3 = 1) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l608_60852


namespace NUMINAMATH_CALUDE_same_color_probability_l608_60898

/-- The number of red marbles in the bag -/
def red_marbles : ℕ := 6

/-- The number of white marbles in the bag -/
def white_marbles : ℕ := 7

/-- The number of blue marbles in the bag -/
def blue_marbles : ℕ := 8

/-- The total number of marbles in the bag -/
def total_marbles : ℕ := red_marbles + white_marbles + blue_marbles

/-- The number of marbles drawn -/
def drawn_marbles : ℕ := 4

/-- The probability of drawing four marbles of the same color -/
theorem same_color_probability : 
  (Nat.choose red_marbles drawn_marbles + 
   Nat.choose white_marbles drawn_marbles + 
   Nat.choose blue_marbles drawn_marbles) / 
  Nat.choose total_marbles drawn_marbles = 8 / 399 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l608_60898


namespace NUMINAMATH_CALUDE_expression_evaluation_l608_60884

theorem expression_evaluation :
  3 + 2 * Real.sqrt 3 + (3 + 2 * Real.sqrt 3)⁻¹ + (2 * Real.sqrt 3 - 3)⁻¹ = 3 + (16 * Real.sqrt 3) / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l608_60884


namespace NUMINAMATH_CALUDE_root_difference_implies_k_value_l608_60805

theorem root_difference_implies_k_value (k : ℝ) :
  (∃ r s : ℝ, r^2 + k*r + 10 = 0 ∧ s^2 + k*s + 10 = 0 ∧
   (r+3)^2 - k*(r+3) + 10 = 0 ∧ (s+3)^2 - k*(s+3) + 10 = 0) →
  k = 3 := by
sorry

end NUMINAMATH_CALUDE_root_difference_implies_k_value_l608_60805


namespace NUMINAMATH_CALUDE_mike_fred_salary_ratio_l608_60807

/-- Proves that Mike earned 11 times more money than Fred five months ago -/
theorem mike_fred_salary_ratio :
  ∀ (fred_salary mike_salary_now : ℕ),
    fred_salary = 1000 →
    mike_salary_now = 15400 →
    ∃ (mike_salary_before : ℕ),
      mike_salary_now = (140 * mike_salary_before) / 100 ∧
      mike_salary_before = 11 * fred_salary :=
by sorry

end NUMINAMATH_CALUDE_mike_fred_salary_ratio_l608_60807


namespace NUMINAMATH_CALUDE_chess_tournament_games_l608_60866

theorem chess_tournament_games (n : ℕ) (h : n = 20) : 
  (n * (n - 1)) = 380 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l608_60866


namespace NUMINAMATH_CALUDE_prob_three_green_apples_l608_60874

/-- The probability of picking 3 green apples out of 10 apples, where 4 are green -/
theorem prob_three_green_apples (total : ℕ) (green : ℕ) (pick : ℕ)
  (h1 : total = 10) (h2 : green = 4) (h3 : pick = 3) :
  (Nat.choose green pick : ℚ) / (Nat.choose total pick) = 1 / 30 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_green_apples_l608_60874


namespace NUMINAMATH_CALUDE_set_A_is_empty_l608_60880

def set_A : Set ℝ := {x : ℝ | x^2 + 2 = 0}
def set_B : Set ℝ := {0}
def set_C : Set ℝ := {x : ℝ | x > 8 ∨ x < 4}
def set_D : Set (Set ℝ) := {∅}

theorem set_A_is_empty : set_A = ∅ := by
  sorry

end NUMINAMATH_CALUDE_set_A_is_empty_l608_60880


namespace NUMINAMATH_CALUDE_intersection_empty_iff_a_lt_neg_one_l608_60887

/-- Define set A as {x | -1 ≤ x < 2} -/
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}

/-- Define set B as {x | x ≤ a} -/
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

/-- Theorem: The intersection of A and B is empty if and only if a < -1 -/
theorem intersection_empty_iff_a_lt_neg_one (a : ℝ) :
  A ∩ B a = ∅ ↔ a < -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_iff_a_lt_neg_one_l608_60887


namespace NUMINAMATH_CALUDE_ladder_distance_l608_60876

theorem ladder_distance (ladder_length height : ℝ) (h1 : ladder_length = 15) (h2 : height = 12) :
  ∃ (distance : ℝ), distance^2 + height^2 = ladder_length^2 ∧ distance = 9 := by
  sorry

end NUMINAMATH_CALUDE_ladder_distance_l608_60876


namespace NUMINAMATH_CALUDE_magnitude_of_complex_number_l608_60891

theorem magnitude_of_complex_number (i : ℂ) (h : i^2 = -1) :
  let z : ℂ := (3 + 2*i) * i
  Complex.abs z = Real.sqrt 13 := by sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_number_l608_60891


namespace NUMINAMATH_CALUDE_negation_of_odd_sum_even_l608_60859

theorem negation_of_odd_sum_even (a b : ℤ) :
  ¬(((Odd a ∧ Odd b) → Even (a + b))) ↔ (¬(Odd a ∧ Odd b) → ¬Even (a + b)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_odd_sum_even_l608_60859


namespace NUMINAMATH_CALUDE_union_M_complement_N_equals_U_l608_60864

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set M as the domain of ln(1-x)
def M : Set ℝ := {x | x < 1}

-- Define set N as {x | x²-x < 0}
def N : Set ℝ := {x | x^2 - x < 0}

-- Theorem statement
theorem union_M_complement_N_equals_U : M ∪ (U \ N) = U := by sorry

end NUMINAMATH_CALUDE_union_M_complement_N_equals_U_l608_60864


namespace NUMINAMATH_CALUDE_chess_tournament_players_l608_60899

/-- Chess tournament with specific conditions -/
structure ChessTournament where
  n : ℕ
  total_score : ℕ
  two_player_score : ℕ
  avg_score_others : ℕ
  odd_players : Odd n
  two_player_score_eq : two_player_score = 16
  even_avg_score : Even avg_score_others
  total_score_eq : total_score = n * (n - 1)

/-- Theorem stating that under given conditions, the number of players is 9 -/
theorem chess_tournament_players (t : ChessTournament) : t.n = 9 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l608_60899


namespace NUMINAMATH_CALUDE_locus_of_point_in_cube_l608_60869

/-- The locus of a point M in a unit cube, where the sum of squares of distances 
    from M to the faces of the cube is constant, is a sphere centered at (1/2, 1/2, 1/2). -/
theorem locus_of_point_in_cube (x y z : ℝ) (k : ℝ) : 
  (0 ≤ x ∧ x ≤ 1) ∧ (0 ≤ y ∧ y ≤ 1) ∧ (0 ≤ z ∧ z ≤ 1) →
  x^2 + (1 - x)^2 + y^2 + (1 - y)^2 + z^2 + (1 - z)^2 = k →
  ∃ r : ℝ, (x - 1/2)^2 + (y - 1/2)^2 + (z - 1/2)^2 = r^2 :=
by sorry


end NUMINAMATH_CALUDE_locus_of_point_in_cube_l608_60869


namespace NUMINAMATH_CALUDE_probability_divisor_of_12_on_8_sided_die_l608_60861

def is_divisor_of_12 (n : ℕ) : Prop := 12 % n = 0

def die_sides : ℕ := 8

def favorable_outcomes : Finset ℕ := {1, 2, 3, 4, 6}

theorem probability_divisor_of_12_on_8_sided_die :
  (favorable_outcomes.card : ℚ) / die_sides = 5 / 8 :=
sorry

end NUMINAMATH_CALUDE_probability_divisor_of_12_on_8_sided_die_l608_60861


namespace NUMINAMATH_CALUDE_A_intersect_CᵣB_equals_zero_one_l608_60814

-- Define the universal set
def 𝕌 : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {-1, 0, 1, 5}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

-- Define the complement of B in ℝ
def CᵣB : Set ℝ := 𝕌 \ B

-- Theorem statement
theorem A_intersect_CᵣB_equals_zero_one : A ∩ CᵣB = {0, 1} := by sorry

end NUMINAMATH_CALUDE_A_intersect_CᵣB_equals_zero_one_l608_60814


namespace NUMINAMATH_CALUDE_rectangle_fold_trapezoid_l608_60862

/-- 
Given a rectangle with sides a and b, if folding it along its diagonal 
creates an isosceles trapezoid with three equal sides and the fourth side 
of length 10√3, then a = 15 and b = 5√3.
-/
theorem rectangle_fold_trapezoid (a b : ℝ) 
  (h_rect : a > 0 ∧ b > 0)
  (h_fold : ∃ (x y z : ℝ), x = y ∧ y = z ∧ 
    x^2 + y^2 = a^2 + b^2 ∧ 
    z^2 + (10 * Real.sqrt 3)^2 = a^2 + b^2) : 
  a = 15 ∧ b = 5 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_rectangle_fold_trapezoid_l608_60862


namespace NUMINAMATH_CALUDE_bianca_birthday_money_l608_60882

/-- The amount of money Bianca received for her birthday -/
def birthday_money (num_friends : ℕ) (amount_per_friend : ℕ) : ℕ :=
  num_friends * amount_per_friend

/-- Theorem: Bianca received 120 dollars for her birthday -/
theorem bianca_birthday_money :
  birthday_money 8 15 = 120 := by
  sorry

end NUMINAMATH_CALUDE_bianca_birthday_money_l608_60882


namespace NUMINAMATH_CALUDE_roots_of_polynomials_l608_60822

theorem roots_of_polynomials (r : ℝ) : 
  r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end NUMINAMATH_CALUDE_roots_of_polynomials_l608_60822


namespace NUMINAMATH_CALUDE_inequality_solution_set_l608_60896

def inequality (a x : ℝ) : Prop := (a + 1) * x - 3 < x - 1

def solution_set (a : ℝ) : Set ℝ :=
  if a < 0 then {x | x < 2/a ∨ x > 1}
  else if a = 0 then {x | x > 1}
  else if 0 < a ∧ a < 2 then {x | 1 < x ∧ x < 2/a}
  else if a = 2 then ∅
  else {x | 2/a < x ∧ x < 1}

theorem inequality_solution_set (a : ℝ) :
  {x : ℝ | inequality a x} = solution_set a :=
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l608_60896


namespace NUMINAMATH_CALUDE_sqrt_11_simplest_l608_60886

def is_simplest_sqrt (n : ℕ) (others : List ℕ) : Prop :=
  ∀ m ∈ others, ¬ (∃ k : ℕ, k > 1 ∧ k * k ∣ n) ∧ (∃ k : ℕ, k > 1 ∧ k * k ∣ m)

theorem sqrt_11_simplest : is_simplest_sqrt 11 [8, 12, 36] := by
  sorry

end NUMINAMATH_CALUDE_sqrt_11_simplest_l608_60886


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l608_60831

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}

theorem complement_of_M_in_U :
  (U \ M) = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l608_60831


namespace NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l608_60844

theorem maximize_x_cubed_y_fourth (x y : ℝ) (hx : x > 0) (hy : y > 0) (h_sum : x + 2 * y = 35) :
  x^3 * y^4 ≤ 21^3 * 7^4 ∧ 
  (x^3 * y^4 = 21^3 * 7^4 ↔ x = 21 ∧ y = 7) :=
sorry

end NUMINAMATH_CALUDE_maximize_x_cubed_y_fourth_l608_60844


namespace NUMINAMATH_CALUDE_non_sophomore_musicians_count_l608_60875

/-- Represents the number of students who play a musical instrument in a college -/
structure MusicianCount where
  total : ℕ
  sophomore_play_percent : ℚ
  non_sophomore_not_play_percent : ℚ
  total_not_play_percent : ℚ

/-- Calculates the number of non-sophomores who play a musical instrument -/
def non_sophomore_musicians (mc : MusicianCount) : ℕ :=
  sorry

/-- Theorem stating the number of non-sophomores who play a musical instrument -/
theorem non_sophomore_musicians_count (mc : MusicianCount) 
  (h1 : mc.total = 400)
  (h2 : mc.sophomore_play_percent = 1/2)
  (h3 : mc.non_sophomore_not_play_percent = 2/5)
  (h4 : mc.total_not_play_percent = 11/25) :
  non_sophomore_musicians mc = 144 := by
  sorry

end NUMINAMATH_CALUDE_non_sophomore_musicians_count_l608_60875


namespace NUMINAMATH_CALUDE_gas_purchase_cost_l608_60868

/-- Calculates the total cost of gas purchases given a price rollback and two separate purchases. -/
theorem gas_purchase_cost 
  (rollback : ℝ) 
  (initial_price : ℝ) 
  (liters_today : ℝ) 
  (liters_friday : ℝ) 
  (h1 : rollback = 0.4) 
  (h2 : initial_price = 1.4) 
  (h3 : liters_today = 10) 
  (h4 : liters_friday = 25) : 
  initial_price * liters_today + (initial_price - rollback) * liters_friday = 39 := by
sorry

end NUMINAMATH_CALUDE_gas_purchase_cost_l608_60868


namespace NUMINAMATH_CALUDE_intersection_line_slope_l608_60870

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 8*x + 6*y + 9 = 0

-- Define the intersection points
def intersection (C D : ℝ × ℝ) : Prop :=
  circle1 C.1 C.2 ∧ circle1 D.1 D.2 ∧ circle2 C.1 C.2 ∧ circle2 D.1 D.2 ∧ C ≠ D

-- Theorem statement
theorem intersection_line_slope (C D : ℝ × ℝ) (h : intersection C D) :
  (D.2 - C.2) / (D.1 - C.1) = -1 := by sorry

end NUMINAMATH_CALUDE_intersection_line_slope_l608_60870


namespace NUMINAMATH_CALUDE_dogs_not_doing_anything_l608_60821

theorem dogs_not_doing_anything (total : ℕ) (running : ℕ) (playing : ℕ) (barking : ℕ) : 
  total = 88 → 
  running = 12 → 
  playing = total / 2 → 
  barking = total / 4 → 
  total - (running + playing + barking) = 10 := by
  sorry

end NUMINAMATH_CALUDE_dogs_not_doing_anything_l608_60821


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l608_60834

theorem repeating_decimal_sum (a b c : Nat) : 
  a < 10 ∧ b < 10 ∧ c < 10 →
  (10 * a + b) / 99 + (100 * a + 10 * b) / 9900 + (10 * b + c) / 99 = 25 / 99 →
  100 * a + 10 * b + c = 23 :=
by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l608_60834


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l608_60850

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | x^2 - (1 + a) * x + a > 0}
  (a > 1 → S = {x : ℝ | x > a ∨ x < 1}) ∧
  (a = 1 → S = {x : ℝ | x ≠ 1}) ∧
  (a < 1 → S = {x : ℝ | x > 1 ∨ x < a}) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l608_60850


namespace NUMINAMATH_CALUDE_baseball_cap_production_l608_60810

/-- Proves that given the conditions of the baseball cap factory problem, 
    the number of caps made in the third week is 300. -/
theorem baseball_cap_production : 
  ∀ (x : ℕ), 
    (320 + 400 + x + (320 + 400 + x) / 3 = 1360) → 
    x = 300 := by
  sorry

end NUMINAMATH_CALUDE_baseball_cap_production_l608_60810


namespace NUMINAMATH_CALUDE_compact_connected_preserving_implies_continuous_l608_60845

/-- A function that maps compact sets to compact sets and connected sets to connected sets -/
def CompactConnectedPreserving (n m : ℕ) :=
  {f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin m) |
    (∀ S : Set (EuclideanSpace ℝ (Fin n)), IsCompact S → IsCompact (f '' S)) ∧
    (∀ S : Set (EuclideanSpace ℝ (Fin n)), IsConnected S → IsConnected (f '' S))}

/-- Theorem: A function preserving compactness and connectedness is continuous -/
theorem compact_connected_preserving_implies_continuous
  {n m : ℕ} (f : CompactConnectedPreserving n m) :
  Continuous (f : EuclideanSpace ℝ (Fin n) → EuclideanSpace ℝ (Fin m)) :=
by sorry

end NUMINAMATH_CALUDE_compact_connected_preserving_implies_continuous_l608_60845


namespace NUMINAMATH_CALUDE_lcm_of_105_and_360_l608_60871

theorem lcm_of_105_and_360 : Nat.lcm 105 360 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_105_and_360_l608_60871


namespace NUMINAMATH_CALUDE_grace_lee_calculation_difference_l608_60857

theorem grace_lee_calculation_difference : 
  (12 - (3 * 4 - 2)) - (12 - 3 * 4 - 2) = -32 := by
  sorry

end NUMINAMATH_CALUDE_grace_lee_calculation_difference_l608_60857


namespace NUMINAMATH_CALUDE_mermaid_seashell_age_l608_60828

/-- Converts a base-9 number to base-10 --/
def base9_to_base10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 9^2 + tens * 9^1 + ones * 9^0

/-- The mermaid's seashell collection age conversion theorem --/
theorem mermaid_seashell_age :
  base9_to_base10 3 6 2 = 299 := by
  sorry

end NUMINAMATH_CALUDE_mermaid_seashell_age_l608_60828


namespace NUMINAMATH_CALUDE_probability_ratio_l608_60820

/-- The number of slips in the hat -/
def total_slips : ℕ := 50

/-- The number of different numbers on the slips -/
def num_numbers : ℕ := 10

/-- The number of slips for each number from 1 to 5 -/
def slips_per_low_number : ℕ := 5

/-- The number of slips for each number from 6 to 10 -/
def slips_per_high_number : ℕ := 3

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability that all four drawn slips bear the same number (only possible for numbers 1 to 5) -/
def r : ℚ := (slips_per_low_number.choose drawn_slips * 5 : ℚ) / total_slips.choose drawn_slips

/-- The probability that two slips bear a number c (1 to 5) and two slips bear a number d ≠ c (6 to 10) -/
def s : ℚ := (5 * 5 * slips_per_low_number.choose 2 * slips_per_high_number.choose 2 : ℚ) / total_slips.choose drawn_slips

theorem probability_ratio : s / r = 30 := by
  sorry

end NUMINAMATH_CALUDE_probability_ratio_l608_60820


namespace NUMINAMATH_CALUDE_derivative_at_zero_does_not_exist_l608_60893

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  if x = 0 then 0 else Real.sin x * Real.cos (5 / x)

-- State the theorem
theorem derivative_at_zero_does_not_exist :
  ¬ ∃ (L : ℝ), HasDerivAt f L 0 := by sorry

end NUMINAMATH_CALUDE_derivative_at_zero_does_not_exist_l608_60893


namespace NUMINAMATH_CALUDE_all_groups_have_access_l608_60853

-- Define the type for house groups
inductive HouseGroup : Type
  | a | b | c | d | e | f | g | h | i | j | k | l | m

-- Define the adjacency relation
def adjacent : HouseGroup → HouseGroup → Prop
  | HouseGroup.a, HouseGroup.b => True
  | HouseGroup.a, HouseGroup.d => True
  | HouseGroup.b, HouseGroup.a => True
  | HouseGroup.b, HouseGroup.c => True
  | HouseGroup.b, HouseGroup.d => True
  | HouseGroup.c, HouseGroup.b => True
  | HouseGroup.d, HouseGroup.a => True
  | HouseGroup.d, HouseGroup.b => True
  | HouseGroup.d, HouseGroup.f => True
  | HouseGroup.d, HouseGroup.e => True
  | HouseGroup.e, HouseGroup.d => True
  | HouseGroup.e, HouseGroup.f => True
  | HouseGroup.e, HouseGroup.j => True
  | HouseGroup.e, HouseGroup.l => True
  | HouseGroup.f, HouseGroup.d => True
  | HouseGroup.f, HouseGroup.e => True
  | HouseGroup.f, HouseGroup.j => True
  | HouseGroup.f, HouseGroup.i => True
  | HouseGroup.f, HouseGroup.g => True
  | HouseGroup.g, HouseGroup.f => True
  | HouseGroup.g, HouseGroup.i => True
  | HouseGroup.g, HouseGroup.h => True
  | HouseGroup.h, HouseGroup.g => True
  | HouseGroup.h, HouseGroup.i => True
  | HouseGroup.i, HouseGroup.j => True
  | HouseGroup.i, HouseGroup.f => True
  | HouseGroup.i, HouseGroup.g => True
  | HouseGroup.i, HouseGroup.h => True
  | HouseGroup.j, HouseGroup.k => True
  | HouseGroup.j, HouseGroup.e => True
  | HouseGroup.j, HouseGroup.f => True
  | HouseGroup.j, HouseGroup.i => True
  | HouseGroup.k, HouseGroup.l => True
  | HouseGroup.k, HouseGroup.j => True
  | HouseGroup.l, HouseGroup.k => True
  | HouseGroup.l, HouseGroup.e => True
  | _, _ => False

-- Define the set of pharmacy locations
def pharmacyLocations : Set HouseGroup :=
  {HouseGroup.b, HouseGroup.i, HouseGroup.l, HouseGroup.m}

-- Define the property of having access to a pharmacy
def hasAccessToPharmacy (g : HouseGroup) : Prop :=
  g ∈ pharmacyLocations ∨ ∃ h ∈ pharmacyLocations, adjacent g h

-- Theorem statement
theorem all_groups_have_access :
  ∀ g : HouseGroup, hasAccessToPharmacy g :=
by sorry

end NUMINAMATH_CALUDE_all_groups_have_access_l608_60853


namespace NUMINAMATH_CALUDE_inspection_sample_size_l608_60889

/-- Represents a batch of leather shoes -/
structure ShoeBatch where
  total : ℕ

/-- Represents a quality inspection of shoes -/
structure QualityInspection where
  batch : ShoeBatch
  drawn : ℕ

/-- Definition of sample size for a quality inspection -/
def sampleSize (inspection : QualityInspection) : ℕ :=
  inspection.drawn

theorem inspection_sample_size (batch : ShoeBatch) :
  let inspection := QualityInspection.mk batch 50
  sampleSize inspection = 50 := by
  sorry

end NUMINAMATH_CALUDE_inspection_sample_size_l608_60889


namespace NUMINAMATH_CALUDE_binomial_20_19_l608_60836

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_l608_60836


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l608_60829

theorem prime_sum_theorem (p q r s : ℕ) : 
  Prime p ∧ Prime q ∧ Prime r ∧ Prime s ∧
  p < q ∧ q < r ∧ r < s ∧
  1 - 1/p - 1/q - 1/r - 1/s = 1/(p*q*r*s) →
  p + q + r + s = 55 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l608_60829


namespace NUMINAMATH_CALUDE_map_area_calculation_l608_60858

/-- Proves that given a map scale of 1:50000 and an area of 100 cm² on the map, 
    the actual area of the land is 2.5 × 10^7 m². -/
theorem map_area_calculation (scale : ℚ) (map_area : ℝ) (actual_area : ℝ) : 
  scale = 1 / 50000 → 
  map_area = 100 → 
  actual_area = 2.5 * 10^7 → 
  map_area / actual_area = scale^2 := by
  sorry

#check map_area_calculation

end NUMINAMATH_CALUDE_map_area_calculation_l608_60858


namespace NUMINAMATH_CALUDE_initial_pc_cost_l608_60873

/-- Proves that the initial cost of a gaming PC is $1200, given the conditions of the video card upgrade and total spent. -/
theorem initial_pc_cost (old_card_sale : ℕ) (new_card_cost : ℕ) (total_spent : ℕ) 
  (h1 : old_card_sale = 300)
  (h2 : new_card_cost = 500)
  (h3 : total_spent = 1400) :
  total_spent - (new_card_cost - old_card_sale) = 1200 := by
  sorry

#check initial_pc_cost

end NUMINAMATH_CALUDE_initial_pc_cost_l608_60873


namespace NUMINAMATH_CALUDE_floor_sum_equals_negative_one_l608_60819

theorem floor_sum_equals_negative_one : ⌊(19.7 : ℝ)⌋ + ⌊(-19.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_equals_negative_one_l608_60819


namespace NUMINAMATH_CALUDE_highest_divisible_digit_l608_60863

theorem highest_divisible_digit : 
  ∃ (a : ℕ), a ≤ 9 ∧ 
  (43752 * 1000 + a * 100 + 539) % 8 = 0 ∧
  (43752 * 1000 + a * 100 + 539) % 9 = 0 ∧
  (43752 * 1000 + a * 100 + 539) % 12 = 0 ∧
  ∀ (b : ℕ), b > a → b ≤ 9 → 
    (43752 * 1000 + b * 100 + 539) % 8 ≠ 0 ∨
    (43752 * 1000 + b * 100 + 539) % 9 ≠ 0 ∨
    (43752 * 1000 + b * 100 + 539) % 12 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_highest_divisible_digit_l608_60863


namespace NUMINAMATH_CALUDE_polynomial_equality_l608_60816

/-- Two distinct quadratic polynomials with leading coefficient 1 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c
def g (d e : ℝ) (x : ℝ) : ℝ := x^2 + d*x + e

/-- The theorem stating the point of equality for the polynomials -/
theorem polynomial_equality (b c d e : ℝ) 
  (h_distinct : b ≠ d ∨ c ≠ e)
  (h_sum_eq : f b c 1 + f b c 10 + f b c 100 = g d e 1 + g d e 10 + g d e 100) :
  f b c 37 = g d e 37 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l608_60816


namespace NUMINAMATH_CALUDE_spending_difference_l608_60890

/-- The cost of the computer table in dollars -/
def table_cost : ℚ := 140

/-- The cost of the computer chair in dollars -/
def chair_cost : ℚ := 100

/-- The cost of the joystick in dollars -/
def joystick_cost : ℚ := 20

/-- Frank's share of the joystick cost -/
def frank_joystick_share : ℚ := 1/4

/-- Eman's share of the joystick cost -/
def eman_joystick_share : ℚ := 1 - frank_joystick_share

/-- Frank's total spending -/
def frank_total : ℚ := table_cost + frank_joystick_share * joystick_cost

/-- Eman's total spending -/
def eman_total : ℚ := chair_cost + eman_joystick_share * joystick_cost

theorem spending_difference : frank_total - eman_total = 30 := by
  sorry

end NUMINAMATH_CALUDE_spending_difference_l608_60890


namespace NUMINAMATH_CALUDE_train_length_l608_60849

/-- Given a train traveling at constant speed through a tunnel, this theorem
    proves the length of the train based on the given conditions. -/
theorem train_length
  (tunnel_length : ℝ)
  (total_time : ℝ)
  (light_time : ℝ)
  (h1 : tunnel_length = 310)
  (h2 : total_time = 18)
  (h3 : light_time = 8)
  (h4 : total_time > 0)
  (h5 : light_time > 0)
  (h6 : light_time < total_time) :
  ∃ (train_length : ℝ),
    train_length = 248 ∧
    (tunnel_length + train_length) / total_time = train_length / light_time :=
by sorry


end NUMINAMATH_CALUDE_train_length_l608_60849


namespace NUMINAMATH_CALUDE_chess_tournament_score_change_l608_60809

/-- Represents a chess tournament with 2n players -/
structure ChessTournament (n : ℕ) where
  players : Fin (2 * n)
  score : Fin (2 * n) → ℝ
  score_change : Fin (2 * n) → ℝ

/-- The theorem to be proved -/
theorem chess_tournament_score_change (n : ℕ) (tournament : ChessTournament n) :
  (∀ p, tournament.score_change p ≥ n) →
  (∀ p, tournament.score_change p = n) :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_score_change_l608_60809


namespace NUMINAMATH_CALUDE_min_set_size_with_mean_constraints_l608_60801

theorem min_set_size_with_mean_constraints (n : ℕ) (S : Finset ℕ) : 
  n > 0 ∧ 
  S.card = n ∧ 
  (∃ m L P : ℕ, 
    L ∈ S ∧ 
    P ∈ S ∧ 
    (∀ x ∈ S, x ≤ L ∧ x ≥ P) ∧
    (S.sum id) / n = m ∧
    m = (2 * L) / 5 ∧ 
    m = (7 * P) / 4) →
  n ≥ 5 :=
by sorry

end NUMINAMATH_CALUDE_min_set_size_with_mean_constraints_l608_60801


namespace NUMINAMATH_CALUDE_triangle_count_is_sixteen_l608_60878

/-- Represents a rectangle with diagonals and internal rectangle --/
structure ConfiguredRectangle where
  vertices : Fin 4 → Point
  diagonals : List (Point × Point)
  midpoints : Fin 4 → Point
  internal_rectangle : List (Point × Point)

/-- Counts the number of triangles in the configured rectangle --/
def count_triangles (rect : ConfiguredRectangle) : ℕ :=
  sorry

/-- Theorem stating that the number of triangles is 16 --/
theorem triangle_count_is_sixteen (rect : ConfiguredRectangle) : 
  count_triangles rect = 16 :=
sorry

end NUMINAMATH_CALUDE_triangle_count_is_sixteen_l608_60878


namespace NUMINAMATH_CALUDE_k_range_l608_60872

theorem k_range (x y k : ℝ) : 
  3 * x + y = k + 1 →
  x + 3 * y = 3 →
  0 < x + y →
  x + y < 1 →
  -4 < k ∧ k < 0 := by
sorry

end NUMINAMATH_CALUDE_k_range_l608_60872


namespace NUMINAMATH_CALUDE_sin_cos_identity_l608_60888

theorem sin_cos_identity : 
  Real.sin (20 * π / 180) * Real.cos (10 * π / 180) - 
  Real.cos (160 * π / 180) * Real.sin (10 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_identity_l608_60888


namespace NUMINAMATH_CALUDE_solution_exists_l608_60847

def f (x : ℝ) := x^3 + x - 3

theorem solution_exists : ∃ c ∈ Set.Icc 1 2, f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_exists_l608_60847


namespace NUMINAMATH_CALUDE_only_newborn_babies_is_set_l608_60860

-- Define a type for statements
inductive Statement
| NewbornBabies
| VerySmallNumbers
| HealthyStudents
| CutePandas

-- Define a function to check if a statement satisfies definiteness
def satisfiesDefiniteness (s : Statement) : Prop :=
  match s with
  | Statement.NewbornBabies => true
  | _ => false

-- Theorem: Only NewbornBabies satisfies definiteness
theorem only_newborn_babies_is_set :
  ∀ s : Statement, satisfiesDefiniteness s ↔ s = Statement.NewbornBabies :=
by
  sorry


end NUMINAMATH_CALUDE_only_newborn_babies_is_set_l608_60860


namespace NUMINAMATH_CALUDE_largest_sum_is_994_l608_60877

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents the sum of XXX + YY + Z -/
def sum (X Y Z : Digit) : ℕ := 111 * X.val + 11 * Y.val + Z.val

theorem largest_sum_is_994 :
  ∃ (X Y Z : Digit), X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z ∧
    sum X Y Z ≤ 999 ∧
    (∀ (A B C : Digit), A ≠ B ∧ A ≠ C ∧ B ≠ C → sum A B C ≤ sum X Y Z) ∧
    sum X Y Z = 994 ∧
    X = Y ∧ Y ≠ Z :=
by sorry

end NUMINAMATH_CALUDE_largest_sum_is_994_l608_60877


namespace NUMINAMATH_CALUDE_newspaper_pieces_l608_60855

theorem newspaper_pieces (petya_tears : ℕ) (vasya_tears : ℕ) (found_pieces : ℕ) :
  petya_tears = 5 →
  vasya_tears = 9 →
  found_pieces = 1988 →
  ∃ n : ℕ, (1 + n * (petya_tears - 1) + m * (vasya_tears - 1)) ≠ found_pieces :=
by sorry

end NUMINAMATH_CALUDE_newspaper_pieces_l608_60855


namespace NUMINAMATH_CALUDE_mona_monday_distance_l608_60833

/-- Represents the distance biked on a given day -/
structure DailyBike where
  distance : ℝ
  time : ℝ
  speed : ℝ

/-- Represents Mona's weekly biking schedule -/
structure WeeklyBike where
  monday : DailyBike
  wednesday : DailyBike
  saturday : DailyBike
  total_distance : ℝ

theorem mona_monday_distance (w : WeeklyBike) :
  w.total_distance = 30 ∧
  w.wednesday.distance = 12 ∧
  w.wednesday.time = 2 ∧
  w.saturday.distance = 2 * w.monday.distance ∧
  w.monday.speed = 15 ∧
  w.monday.time = 1.5 ∧
  w.saturday.speed = 0.8 * w.monday.speed →
  w.monday.distance = 6 := by
  sorry

end NUMINAMATH_CALUDE_mona_monday_distance_l608_60833


namespace NUMINAMATH_CALUDE_valid_seating_arrangements_count_l608_60883

/-- Represents a seating arrangement for two people -/
structure SeatingArrangement :=
  (front : Fin 4 → Bool)
  (back : Fin 5 → Bool)

/-- Checks if a seating arrangement is valid (two people not adjacent) -/
def is_valid (s : SeatingArrangement) : Bool :=
  sorry

/-- Counts the number of valid seating arrangements -/
def count_valid_arrangements : Nat :=
  sorry

/-- Theorem stating that the number of valid seating arrangements is 58 -/
theorem valid_seating_arrangements_count :
  count_valid_arrangements = 58 := by sorry

end NUMINAMATH_CALUDE_valid_seating_arrangements_count_l608_60883


namespace NUMINAMATH_CALUDE_train_length_l608_60841

/-- The length of a train given specific conditions -/
theorem train_length (jogger_speed : ℝ) (train_speed : ℝ) (initial_distance : ℝ) (passing_time : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  initial_distance = 230 →
  passing_time = 35 →
  (train_speed - jogger_speed) * passing_time + initial_distance = 580 :=
by sorry

end NUMINAMATH_CALUDE_train_length_l608_60841


namespace NUMINAMATH_CALUDE_small_tub_cost_l608_60824

theorem small_tub_cost (large_tubs small_tubs : ℕ) (total_cost large_tub_cost : ℚ) :
  large_tubs = 3 →
  small_tubs = 6 →
  total_cost = 48 →
  large_tub_cost = 6 →
  (total_cost - large_tubs * large_tub_cost) / small_tubs = 5 := by
  sorry

end NUMINAMATH_CALUDE_small_tub_cost_l608_60824


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l608_60865

theorem complex_number_quadrant : ∃ (z : ℂ), z = Complex.I * (1 - Complex.I) ∧ z.re > 0 ∧ z.im > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l608_60865


namespace NUMINAMATH_CALUDE_min_area_enclosed_l608_60837

/-- The function f(x) = 3 - x^2 --/
def f (x : ℝ) : ℝ := 3 - x^2

/-- A point on the graph of f --/
structure PointOnGraph where
  x : ℝ
  y : ℝ
  on_graph : y = f x

/-- The area enclosed by tangents and x-axis --/
def enclosed_area (A B : PointOnGraph) : ℝ :=
  sorry -- Definition of the area calculation

/-- Theorem: Minimum area enclosed by tangents and x-axis --/
theorem min_area_enclosed (A B : PointOnGraph) 
    (h_opposite : A.x * B.x < 0) : -- A and B are on opposite sides of y-axis
  ∃ (min_area : ℝ), min_area = 8 ∧ ∀ (P Q : PointOnGraph), 
    P.x * Q.x < 0 → enclosed_area P Q ≥ min_area := by
  sorry

end NUMINAMATH_CALUDE_min_area_enclosed_l608_60837


namespace NUMINAMATH_CALUDE_darwin_money_left_l608_60848

theorem darwin_money_left (initial_amount : ℝ) (gas_fraction : ℝ) (food_fraction : ℝ) : 
  initial_amount = 600 →
  gas_fraction = 1/3 →
  food_fraction = 1/4 →
  initial_amount - (gas_fraction * initial_amount) - (food_fraction * (initial_amount - gas_fraction * initial_amount)) = 300 := by
sorry

end NUMINAMATH_CALUDE_darwin_money_left_l608_60848


namespace NUMINAMATH_CALUDE_percentage_problem_l608_60802

theorem percentage_problem (P : ℝ) : P = 50 → 30 = (P / 100) * 40 + 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l608_60802


namespace NUMINAMATH_CALUDE_percentage_of_value_in_quarters_l608_60895

theorem percentage_of_value_in_quarters : 
  let num_dimes : ℕ := 40
  let num_quarters : ℕ := 30
  let num_nickels : ℕ := 10
  let value_dime : ℕ := 10
  let value_quarter : ℕ := 25
  let value_nickel : ℕ := 5
  let total_value := num_dimes * value_dime + num_quarters * value_quarter + num_nickels * value_nickel
  let value_in_quarters := num_quarters * value_quarter
  (value_in_quarters : ℚ) / total_value * 100 = 62.5 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_value_in_quarters_l608_60895


namespace NUMINAMATH_CALUDE_dice_sum_product_l608_60840

theorem dice_sum_product (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 ∧
  1 ≤ b ∧ b ≤ 6 ∧
  1 ≤ c ∧ c ≤ 6 ∧
  1 ≤ d ∧ d ≤ 6 ∧
  a * b * c * d = 180 →
  a + b + c + d ≠ 19 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_product_l608_60840


namespace NUMINAMATH_CALUDE_railway_graph_theorem_l608_60839

/-- A graph representing the railway network --/
structure RailwayGraph where
  V : Finset Nat
  E : Finset (Nat × Nat)
  edge_in_V : ∀ (e : Nat × Nat), e ∈ E → e.1 ∈ V ∧ e.2 ∈ V

/-- The degree of a vertex in the graph --/
def degree (G : RailwayGraph) (v : Nat) : Nat :=
  (G.E.filter (fun e => e.1 = v ∨ e.2 = v)).card

/-- The theorem statement --/
theorem railway_graph_theorem (G : RailwayGraph) 
  (hV : G.V.card = 9)
  (hM : degree G 1 = 7)
  (hSP : degree G 2 = 5)
  (hT : degree G 3 = 4)
  (hY : degree G 4 = 2)
  (hB : degree G 5 = 2)
  (hS : degree G 6 = 2)
  (hZ : degree G 7 = 1)
  (hEven : Even (G.E.card * 2))
  (hVV : G.V.card = 9 → ∃ v ∈ G.V, v ≠ 1 ∧ v ≠ 2 ∧ v ≠ 3 ∧ v ≠ 4 ∧ v ≠ 5 ∧ v ≠ 6 ∧ v ≠ 7 ∧ v ≠ 8) :
  ∃ v ∈ G.V, v ≠ 1 ∧ v ≠ 2 ∧ v ≠ 3 ∧ v ≠ 4 ∧ v ≠ 5 ∧ v ≠ 6 ∧ v ≠ 7 ∧ v ≠ 8 ∧ 
    (degree G v = 2 ∨ degree G v = 3 ∨ degree G v = 4 ∨ degree G v = 5) :=
by sorry

end NUMINAMATH_CALUDE_railway_graph_theorem_l608_60839


namespace NUMINAMATH_CALUDE_approximation_equality_l608_60823

/-- For any function f, f(69.28 × 0.004) / 0.03 = f(9.237333...) -/
theorem approximation_equality (f : ℝ → ℝ) : f (69.28 * 0.004) / 0.03 = f 9.237333333333333 := by
  sorry

end NUMINAMATH_CALUDE_approximation_equality_l608_60823


namespace NUMINAMATH_CALUDE_cannot_form_flipped_shape_asymmetrical_shape_requires_flipping_asymmetrical_shape_cannot_be_formed_l608_60885

/-- Represents a rhombus with two colored triangles -/
structure ColoredRhombus :=
  (orientation : ℕ)  -- Represents the rotation (0, 90, 180, 270 degrees)

/-- Represents a larger shape composed of multiple rhombuses -/
structure LargerShape :=
  (rhombuses : List ColoredRhombus)

/-- Represents whether a shape requires flipping to be formed -/
def requiresFlipping (shape : LargerShape) : Prop :=
  sorry  -- Definition of when a shape requires flipping

/-- Represents whether a shape can be formed by rotation only -/
def canFormByRotationOnly (shape : LargerShape) : Prop :=
  sorry  -- Definition of when a shape can be formed by rotation only

/-- Theorem: A shape that requires flipping cannot be formed by rotation only -/
theorem cannot_form_flipped_shape
  (shape : LargerShape) :
  requiresFlipping shape → ¬(canFormByRotationOnly shape) :=
by sorry

/-- The asymmetrical shape that cannot be formed -/
def asymmetricalShape : LargerShape :=
  sorry  -- Definition of the specific asymmetrical shape

/-- Theorem: The asymmetrical shape requires flipping -/
theorem asymmetrical_shape_requires_flipping :
  requiresFlipping asymmetricalShape :=
by sorry

/-- Main theorem: The asymmetrical shape cannot be formed by rotation only -/
theorem asymmetrical_shape_cannot_be_formed :
  ¬(canFormByRotationOnly asymmetricalShape) :=
by sorry

end NUMINAMATH_CALUDE_cannot_form_flipped_shape_asymmetrical_shape_requires_flipping_asymmetrical_shape_cannot_be_formed_l608_60885


namespace NUMINAMATH_CALUDE_hyperbola_satisfies_conditions_l608_60881

/-- A hyperbola is defined by its equation and properties -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop
  passes_through : ℝ × ℝ
  asymptotes : ℝ → ℝ → Prop

/-- The given hyperbola with equation x²/2 - y² = 1 -/
def given_hyperbola : Hyperbola where
  equation := fun x y => x^2 / 2 - y^2 = 1
  passes_through := (2, -2)
  asymptotes := fun x y => x^2 / 2 - y^2 = 0

/-- The hyperbola we need to prove -/
def our_hyperbola : Hyperbola where
  equation := fun x y => y^2 / 2 - x^2 / 4 = 1
  passes_through := (2, -2)
  asymptotes := fun x y => x^2 / 2 - y^2 = 0

/-- Theorem stating that our_hyperbola satisfies the required conditions -/
theorem hyperbola_satisfies_conditions :
  (our_hyperbola.equation our_hyperbola.passes_through.1 our_hyperbola.passes_through.2) ∧
  (∀ x y, our_hyperbola.asymptotes x y ↔ given_hyperbola.asymptotes x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_satisfies_conditions_l608_60881


namespace NUMINAMATH_CALUDE_point_d_and_k_value_l608_60894

/-- Given four points in a plane, prove the coordinates of D and the value of k. -/
theorem point_d_and_k_value 
  (A B C D : ℝ × ℝ)
  (hA : A = (1, 3))
  (hB : B = (2, -2))
  (hC : C = (4, 1))
  (h_AB_CD : B - A = D - C)
  (a b : ℝ × ℝ)
  (ha : a = B - A)
  (hb : b = C - B)
  (h_parallel : ∃ (t : ℝ), t ≠ 0 ∧ t • (k • a - b) = a + 3 • b) :
  D = (5, -4) ∧ k = -1/3 := by sorry

end NUMINAMATH_CALUDE_point_d_and_k_value_l608_60894


namespace NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l608_60838

theorem max_value_of_4x_plus_3y (x y : ℝ) : 
  x^2 + y^2 = 10*x + 8*y + 10 → (4*x + 3*y ≤ 70) ∧ ∃ x y, x^2 + y^2 = 10*x + 8*y + 10 ∧ 4*x + 3*y = 70 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_4x_plus_3y_l608_60838


namespace NUMINAMATH_CALUDE_inequality_proof_l608_60818

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Define the set T
def T : Set ℝ := {a | -Real.sqrt 3 < a ∧ a < Real.sqrt 3}

-- Theorem statement
theorem inequality_proof (h : ∀ x : ℝ, ∀ a ∈ T, f x > a^2) (m n : ℝ) (hm : m ∈ T) (hn : n ∈ T) :
  Real.sqrt 3 * |m + n| < |m * n + 3| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l608_60818


namespace NUMINAMATH_CALUDE_final_payment_calculation_final_payment_is_861_90_l608_60842

/-- Calculates the final payment amount for a product purchase given specific deposit and discount conditions --/
theorem final_payment_calculation (total_cost : ℝ) (first_deposit : ℝ) (second_deposit : ℝ) 
  (promotional_discount_rate : ℝ) (interest_rate : ℝ) : ℝ :=
  let remaining_balance_before_discount := total_cost - (first_deposit + second_deposit)
  let promotional_discount := total_cost * promotional_discount_rate
  let remaining_balance_after_discount := remaining_balance_before_discount - promotional_discount
  let interest := remaining_balance_after_discount * interest_rate
  remaining_balance_after_discount + interest

/-- Proves that the final payment amount is $861.90 given the specific conditions of the problem --/
theorem final_payment_is_861_90 : 
  let total_cost := 1300
  let first_deposit := 130
  let second_deposit := 260
  let promotional_discount_rate := 0.05
  let interest_rate := 0.02
  (final_payment_calculation total_cost first_deposit second_deposit promotional_discount_rate interest_rate) = 861.90 := by
  sorry

end NUMINAMATH_CALUDE_final_payment_calculation_final_payment_is_861_90_l608_60842


namespace NUMINAMATH_CALUDE_eves_diner_purchase_l608_60804

/-- The cost of a sandwich at Eve's Diner -/
def sandwich_cost : ℕ := 4

/-- The cost of a soda at Eve's Diner -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 7

/-- The number of sodas purchased -/
def num_sodas : ℕ := 12

/-- The total cost of the purchase at Eve's Diner -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem eves_diner_purchase :
  total_cost = 64 := by sorry

end NUMINAMATH_CALUDE_eves_diner_purchase_l608_60804


namespace NUMINAMATH_CALUDE_fibonacci_sum_quadruples_l608_60811

/-- The n-th Fibonacci number -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- A predicate that checks if a quadruple (a, b, c, d) satisfies the Fibonacci sum equation -/
def is_valid_quadruple (a b c d : ℕ) : Prop :=
  a ≥ 2 ∧ b ≥ 2 ∧ c ≥ 2 ∧ d ≥ 2 ∧ fib a + fib b = fib c + fib d

/-- The set of all valid quadruples -/
def valid_quadruples : Set (ℕ × ℕ × ℕ × ℕ) :=
  {q | ∃ a b c d, q = (a, b, c, d) ∧ is_valid_quadruple a b c d}

/-- The set of solution quadruples -/
def solution_quadruples : Set (ℕ × ℕ × ℕ × ℕ) :=
  {q | ∃ a b,
    (q = (a, b, a, b) ∨ q = (a, b, b, a) ∨
     q = (a, a-3, a-1, a-1) ∨ q = (a-3, a, a-1, a-1) ∨
     q = (a-1, a-1, a, a-3) ∨ q = (a-1, a-1, a-3, a)) ∧
    a ≥ 2 ∧ b ≥ 2}

/-- The main theorem stating that the valid quadruples are exactly the solution quadruples -/
theorem fibonacci_sum_quadruples : valid_quadruples = solution_quadruples := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_sum_quadruples_l608_60811


namespace NUMINAMATH_CALUDE_truck_count_l608_60827

theorem truck_count (tanks trucks : ℕ) : 
  tanks = 5 * trucks →
  tanks + trucks = 140 →
  trucks = 23 := by
sorry

end NUMINAMATH_CALUDE_truck_count_l608_60827


namespace NUMINAMATH_CALUDE_stock_price_calculation_stock_price_problem_l608_60835

theorem stock_price_calculation (initial_price : ℝ) 
  (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price

theorem stock_price_problem : 
  stock_price_calculation 100 0.5 0.3 = 105 := by
  sorry

end NUMINAMATH_CALUDE_stock_price_calculation_stock_price_problem_l608_60835


namespace NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l608_60830

/-- The number of tiles in a square with side length n -/
def tilesInSquare (n : ℕ) : ℕ := n * n

/-- The difference in tiles between two consecutive squares in the sequence -/
def tileDifference (n : ℕ) : ℕ :=
  tilesInSquare (n + 1) - tilesInSquare n

theorem ninth_minus_eighth_square_tiles : tileDifference 8 = 17 := by
  sorry

end NUMINAMATH_CALUDE_ninth_minus_eighth_square_tiles_l608_60830


namespace NUMINAMATH_CALUDE_plane_contains_line_and_parallel_to_intersection_l608_60854

-- Define the line L
def L : Set (ℝ × ℝ × ℝ) :=
  {(x, y, z) | (x - 1) / 2 = -y / 3 ∧ (x - 1) / 2 = 3 - z}

-- Define the two planes
def plane1 : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 4*x + 5*z - 3 = 0}
def plane2 : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 2*x + y + 2*z = 0}

-- Define the plane P we want to prove
def P : Set (ℝ × ℝ × ℝ) := {(x, y, z) | 2*x - y + 7*z - 23 = 0}

-- Theorem statement
theorem plane_contains_line_and_parallel_to_intersection :
  (∀ p ∈ L, p ∈ P) ∧
  (∃ v : ℝ × ℝ × ℝ, v ≠ 0 ∧
    (∀ p q : ℝ × ℝ × ℝ, p ∈ plane1 ∧ q ∈ plane1 ∧ p ∈ plane2 ∧ q ∈ plane2 → 
      ∃ t : ℝ, q - p = t • v) ∧
    (∀ p q : ℝ × ℝ × ℝ, p ∈ P ∧ q ∈ P → 
      ∃ u : ℝ × ℝ × ℝ, u ≠ 0 ∧ q - p = u • v)) :=
by
  sorry

end NUMINAMATH_CALUDE_plane_contains_line_and_parallel_to_intersection_l608_60854


namespace NUMINAMATH_CALUDE_valid_range_for_square_root_fraction_l608_60808

theorem valid_range_for_square_root_fraction (x : ℝ) :
  (∃ y : ℝ, y = 1 / Real.sqrt (x - 1)) ↔ x > 1 := by
  sorry

end NUMINAMATH_CALUDE_valid_range_for_square_root_fraction_l608_60808


namespace NUMINAMATH_CALUDE_x_value_proof_l608_60832

theorem x_value_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 3)
  (h2 : y^2 / z = 4)
  (h3 : z^2 / x = 5) :
  x = (6480 : ℝ)^(1/7) := by
sorry

end NUMINAMATH_CALUDE_x_value_proof_l608_60832


namespace NUMINAMATH_CALUDE_rectangular_prism_diagonal_l608_60846

/-- Given a rectangular prism with sides a, b, and c, if its surface area is 11
    and the sum of its edges is 24, then the length of its diagonal is 5. -/
theorem rectangular_prism_diagonal 
  (a b c : ℝ) 
  (h_surface : 2 * (a * b + a * c + b * c) = 11) 
  (h_edges : 4 * (a + b + c) = 24) : 
  Real.sqrt (a^2 + b^2 + c^2) = 5 := by
sorry

end NUMINAMATH_CALUDE_rectangular_prism_diagonal_l608_60846


namespace NUMINAMATH_CALUDE_pink_highlighters_l608_60815

theorem pink_highlighters (total : ℕ) (yellow : ℕ) (blue : ℕ) (h1 : yellow = 2) (h2 : blue = 5) (h3 : total = 11) :
  total - yellow - blue = 4 := by
  sorry

end NUMINAMATH_CALUDE_pink_highlighters_l608_60815


namespace NUMINAMATH_CALUDE_dogsled_race_speed_l608_60851

/-- Given two teams racing on a 300-mile course, where one team finishes 3 hours faster
    and has an average speed 5 mph greater than the other, prove that the slower team's
    average speed is 20 mph. -/
theorem dogsled_race_speed (course_length : ℝ) (time_difference : ℝ) (speed_difference : ℝ)
  (h1 : course_length = 300)
  (h2 : time_difference = 3)
  (h3 : speed_difference = 5) :
  let speed_B := (course_length / (course_length / (20 + speed_difference) + time_difference))
  speed_B = 20 := by sorry

end NUMINAMATH_CALUDE_dogsled_race_speed_l608_60851


namespace NUMINAMATH_CALUDE_cement_warehouse_distribution_l608_60867

theorem cement_warehouse_distribution (total : ℕ) (extra : ℕ) (multiplier : ℕ) 
  (warehouseA : ℕ) (warehouseB : ℕ) : 
  total = 462 → 
  extra = 32 → 
  multiplier = 4 →
  total = warehouseA + warehouseB → 
  warehouseA = multiplier * warehouseB + extra →
  warehouseA = 376 ∧ warehouseB = 86 := by
  sorry

end NUMINAMATH_CALUDE_cement_warehouse_distribution_l608_60867


namespace NUMINAMATH_CALUDE_gcd_eight_factorial_six_factorial_l608_60813

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem gcd_eight_factorial_six_factorial :
  Nat.gcd (factorial 8) (factorial 6) = 720 := by
  sorry

end NUMINAMATH_CALUDE_gcd_eight_factorial_six_factorial_l608_60813
