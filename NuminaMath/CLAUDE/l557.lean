import Mathlib

namespace dice_sum_divisibility_probability_l557_55714

theorem dice_sum_divisibility_probability (n : ℕ) (p q r : ℝ) : 
  p ≥ 0 → q ≥ 0 → r ≥ 0 → p + q + r = 1 → 
  p^3 + q^3 + r^3 + 6*p*q*r ≥ 1/4 := by sorry

end dice_sum_divisibility_probability_l557_55714


namespace kids_staying_home_l557_55703

def total_kids : ℕ := 1059955
def camp_kids : ℕ := 564237

theorem kids_staying_home : total_kids - camp_kids = 495718 := by
  sorry

end kids_staying_home_l557_55703


namespace least_number_of_cans_l557_55785

def maaza_volume : ℕ := 10
def pepsi_volume : ℕ := 144
def sprite_volume : ℕ := 368

theorem least_number_of_cans (can_volume : ℕ) 
  (h1 : can_volume > 0)
  (h2 : can_volume ∣ maaza_volume)
  (h3 : can_volume ∣ pepsi_volume)
  (h4 : can_volume ∣ sprite_volume)
  (h5 : ∀ x : ℕ, x > can_volume → ¬(x ∣ maaza_volume ∧ x ∣ pepsi_volume ∧ x ∣ sprite_volume)) :
  maaza_volume / can_volume + pepsi_volume / can_volume + sprite_volume / can_volume = 261 :=
sorry

end least_number_of_cans_l557_55785


namespace equations_truth_l557_55772

-- Define the theorem
theorem equations_truth :
  -- Equation 1
  (∀ a : ℝ, Real.sqrt ((a^2 + 1)^2) = a^2 + 1) ∧
  -- Equation 2
  (∀ a : ℝ, Real.sqrt (a^2) = abs a) ∧
  -- Equation 4
  (∀ x : ℝ, x ≥ 1 → Real.sqrt ((x + 1) * (x - 1)) = Real.sqrt (x + 1) * Real.sqrt (x - 1)) ∧
  -- Equation 3 (counterexample)
  (∃ a b : ℝ, Real.sqrt (a * b) ≠ Real.sqrt a * Real.sqrt b) :=
by
  sorry

end equations_truth_l557_55772


namespace inner_circle_radius_l557_55725

/-- The radius of the inner tangent circle in a rectangle with semicircles --/
theorem inner_circle_radius (length width : ℝ) (h_length : length = 4) (h_width : width = 2) :
  let semicircle_radius := length / 8
  let center_to_semicircle := (3 * length / 8)^2 + (width / 2)^2
  (Real.sqrt center_to_semicircle / 2) - semicircle_radius = (Real.sqrt 10 - 1) / 2 :=
sorry

end inner_circle_radius_l557_55725


namespace line_projection_onto_plane_line_projection_onto_plane_ratio_form_l557_55701

/-- Given a line in 3D space defined by two equations and a plane, 
    this theorem states the equation of the projection of the line onto the plane. -/
theorem line_projection_onto_plane :
  ∀ (x y z : ℝ),
  (3*x - 2*y - z + 4 = 0 ∧ x - 4*y - 3*z - 2 = 0) →  -- Line equations
  (5*x + 2*y + 2*z - 7 = 0) →                        -- Plane equation
  ∃ (t : ℝ), 
    x = -2*t + 1 ∧                                   -- Parametric form of
    y = -14*t + 1 ∧                                  -- the projected line
    z = 19*t :=
by sorry

/-- An alternative formulation of the projection theorem using ratios. -/
theorem line_projection_onto_plane_ratio_form :
  ∀ (x y z : ℝ),
  (3*x - 2*y - z + 4 = 0 ∧ x - 4*y - 3*z - 2 = 0) →  -- Line equations
  (5*x + 2*y + 2*z - 7 = 0) →                        -- Plane equation
  (x - 1) / (-2) = (y - 1) / (-14) ∧ (x - 1) / (-2) = z / 19 :=
by sorry

end line_projection_onto_plane_line_projection_onto_plane_ratio_form_l557_55701


namespace carriage_problem_representation_l557_55762

/-- Represents the problem of people sharing carriages -/
structure CarriageProblem where
  x : ℕ  -- Total number of people
  y : ℕ  -- Total number of carriages

/-- The conditions of the carriage problem are satisfied -/
def satisfies_conditions (p : CarriageProblem) : Prop :=
  (p.x / 3 = p.y + 2) ∧ ((p.x - 9) / 2 = p.y)

/-- The system of equations correctly represents the carriage problem -/
theorem carriage_problem_representation (p : CarriageProblem) :
  satisfies_conditions p ↔ 
    (p.x / 3 = p.y + 2) ∧ ((p.x - 9) / 2 = p.y) :=
by sorry


end carriage_problem_representation_l557_55762


namespace area_of_triangle_abc_l557_55794

noncomputable def triangle_area (a b c : ℝ) (A B C : ℝ) : ℝ :=
  (1/2) * b * c * Real.sin A

theorem area_of_triangle_abc (a b c : ℝ) (A B C : ℝ) 
  (h1 : A = π/4)
  (h2 : b^2 * Real.sin C = 4 * Real.sqrt 2 * Real.sin B) :
  triangle_area a b c A B C = 2 := by
  sorry

end area_of_triangle_abc_l557_55794


namespace distance_circle_center_to_point_l557_55780

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4*x + 6*y - 4

-- Define the center of the circle
def circle_center : ℝ × ℝ := (2, 3)

-- Define the given point
def given_point : ℝ × ℝ := (10, 8)

-- Theorem statement
theorem distance_circle_center_to_point :
  let (cx, cy) := circle_center
  let (px, py) := given_point
  (cx - px)^2 + (cy - py)^2 = 89 :=
sorry

end distance_circle_center_to_point_l557_55780


namespace find_m_l557_55786

theorem find_m (U A : Set ℤ) (m : ℤ) : 
  U = {2, 3, m^2 + m - 4} →
  A = {m, 2} →
  U \ A = {3} →
  m = -2 := by sorry

end find_m_l557_55786


namespace money_difference_proof_l557_55776

/-- The number of nickels in a quarter -/
def nickels_per_quarter : ℕ := 5

/-- Charles' quarters as a function of q -/
def charles_quarters (q : ℤ) : ℤ := 7 * q + 2

/-- Richard's quarters as a function of q -/
def richard_quarters (q : ℤ) : ℤ := 3 * q + 8

/-- The difference in money between Charles and Richard, expressed in nickels -/
def money_difference_in_nickels (q : ℤ) : ℤ :=
  nickels_per_quarter * (charles_quarters q - richard_quarters q)

theorem money_difference_proof (q : ℤ) :
  money_difference_in_nickels q = 20 * q - 30 := by
  sorry

end money_difference_proof_l557_55776


namespace sqrt_36_divided_by_itself_is_one_l557_55732

theorem sqrt_36_divided_by_itself_is_one : 
  (Real.sqrt 36) / (Real.sqrt 36) = 1 := by
  sorry

end sqrt_36_divided_by_itself_is_one_l557_55732


namespace valid_distribution_exists_l557_55767

/-- Represents a part of the city -/
structure CityPart where
  id : Nat

/-- Represents a currency exchange point -/
structure ExchangePoint where
  id : Nat

/-- A distribution of exchange points across city parts -/
def Distribution := CityPart → Finset ExchangePoint

/-- The property that each city part contains exactly two exchange points -/
def ValidDistribution (d : Distribution) (cityParts : Finset CityPart) (exchangePoints : Finset ExchangePoint) : Prop :=
  ∀ cp ∈ cityParts, (d cp).card = 2

/-- The main theorem stating that a valid distribution exists -/
theorem valid_distribution_exists (cityParts : Finset CityPart) (exchangePoints : Finset ExchangePoint)
    (h1 : cityParts.card = 4) (h2 : exchangePoints.card = 4) :
    ∃ d : Distribution, ValidDistribution d cityParts exchangePoints := by
  sorry

end valid_distribution_exists_l557_55767


namespace gcf_294_108_l557_55723

theorem gcf_294_108 : Nat.gcd 294 108 = 6 := by
  sorry

end gcf_294_108_l557_55723


namespace common_roots_imply_c_d_l557_55789

/-- Two cubic polynomials with two distinct common roots -/
def has_two_common_roots (c d : ℝ) : Prop :=
  ∃ r s : ℝ, r ≠ s ∧
    (r^3 + c*r^2 + 12*r + 7 = 0) ∧ 
    (r^3 + d*r^2 + 15*r + 9 = 0) ∧
    (s^3 + c*s^2 + 12*s + 7 = 0) ∧ 
    (s^3 + d*s^2 + 15*s + 9 = 0)

/-- The theorem stating that if the polynomials have two distinct common roots, then c = 5 and d = 4 -/
theorem common_roots_imply_c_d (c d : ℝ) :
  has_two_common_roots c d → c = 5 ∧ d = 4 := by
  sorry

end common_roots_imply_c_d_l557_55789


namespace tan_product_from_cos_sum_diff_l557_55718

theorem tan_product_from_cos_sum_diff (α β : Real) 
  (h1 : Real.cos (α + β) = 1/5)
  (h2 : Real.cos (α - β) = 3/5) : 
  Real.tan α * Real.tan β = 1/2 := by
  sorry

end tan_product_from_cos_sum_diff_l557_55718


namespace coffee_decaf_percentage_l557_55716

def initial_stock : ℝ := 800
def type_a_percent : ℝ := 0.40
def type_b_percent : ℝ := 0.35
def type_c_percent : ℝ := 0.25
def type_a_decaf : ℝ := 0.20
def type_b_decaf : ℝ := 0.50
def type_c_decaf : ℝ := 0

def additional_purchase : ℝ := 300
def additional_type_a_percent : ℝ := 0.50
def additional_type_b_percent : ℝ := 0.30
def additional_type_c_percent : ℝ := 0.20

theorem coffee_decaf_percentage :
  let total_stock := initial_stock + additional_purchase
  let initial_decaf := 
    initial_stock * (type_a_percent * type_a_decaf + 
                     type_b_percent * type_b_decaf + 
                     type_c_percent * type_c_decaf)
  let additional_decaf := 
    additional_purchase * (additional_type_a_percent * type_a_decaf + 
                           additional_type_b_percent * type_b_decaf + 
                           additional_type_c_percent * type_c_decaf)
  let total_decaf := initial_decaf + additional_decaf
  (total_decaf / total_stock) * 100 = (279 / 1100) * 100 := by
sorry

end coffee_decaf_percentage_l557_55716


namespace quadratic_monotonicity_l557_55726

-- Define the quadratic function
def f (t : ℝ) (x : ℝ) : ℝ := x^2 - 2*t*x + 1

-- Define monotonicity in an open interval
def MonotonicIn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

-- Theorem statement
theorem quadratic_monotonicity (t : ℝ) :
  MonotonicIn (f t) 1 3 → t ≤ 1 ∨ t ≥ 3 :=
sorry

end quadratic_monotonicity_l557_55726


namespace chessboard_ratio_l557_55763

/-- The number of squares on an n x n chessboard -/
def num_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- The number of rectangles on a chessboard with m horizontal and n vertical lines -/
def num_rectangles (m n : ℕ) : ℕ := (m.choose 2) * (n.choose 2)

theorem chessboard_ratio :
  (num_squares 9 : ℚ) / (num_rectangles 10 10 : ℚ) = 19 / 135 := by sorry

end chessboard_ratio_l557_55763


namespace sum_of_factors_360_l557_55779

/-- The sum of positive factors of a natural number n -/
def sum_of_factors (n : ℕ) : ℕ := sorry

/-- Theorem: The sum of the positive factors of 360 is 1170 -/
theorem sum_of_factors_360 : sum_of_factors 360 = 1170 := by sorry

end sum_of_factors_360_l557_55779


namespace complex_magnitude_product_l557_55766

theorem complex_magnitude_product : Complex.abs ((7 - 4*I) * (3 + 11*I)) = Real.sqrt 8450 := by
  sorry

end complex_magnitude_product_l557_55766


namespace boat_speed_in_still_water_l557_55758

/-- The speed of a boat in still water, given its downstream travel time and distance, and the speed of the stream. -/
theorem boat_speed_in_still_water 
  (stream_speed : ℝ) 
  (downstream_time : ℝ) 
  (downstream_distance : ℝ) 
  (h1 : stream_speed = 5)
  (h2 : downstream_time = 5)
  (h3 : downstream_distance = 125) :
  downstream_distance = (boat_speed + stream_speed) * downstream_time →
  boat_speed = 20 :=
by
  sorry


end boat_speed_in_still_water_l557_55758


namespace abc_inequality_l557_55750

theorem abc_inequality (a b c : ℝ) (sum_eq_one : a + b + c = 1) (prod_pos : a * b * c > 0) :
  a * b + b * c + c * a < Real.sqrt (a * b * c) / 2 + 1 / 4 := by
  sorry

end abc_inequality_l557_55750


namespace paper_length_proof_l557_55768

theorem paper_length_proof (cube_volume : ℝ) (paper_width : ℝ) (inches_per_foot : ℝ) :
  cube_volume = 8 →
  paper_width = 72 →
  inches_per_foot = 12 →
  ∃ (paper_length : ℝ),
    paper_length * paper_width = (cube_volume^(1/3) * inches_per_foot)^2 ∧
    paper_length = 8 :=
by
  sorry

end paper_length_proof_l557_55768


namespace digit_125_of_1_17_l557_55742

/-- The decimal representation of 1/17 -/
def decimal_rep_1_17 : List ℕ := [0, 5, 8, 8, 2, 3, 5, 2, 9, 4, 1, 1, 7, 6, 4, 7]

/-- The length of the repeating sequence in the decimal representation of 1/17 -/
def repeat_length : ℕ := 16

/-- The 125th digit after the decimal point in the decimal representation of 1/17 is 4 -/
theorem digit_125_of_1_17 : 
  (decimal_rep_1_17[(125 - 1) % repeat_length]) = 4 := by sorry

end digit_125_of_1_17_l557_55742


namespace cistern_fill_time_with_leak_l557_55745

/-- The additional time required to fill a cistern with a leak -/
theorem cistern_fill_time_with_leak 
  (normal_fill_time : ℝ) 
  (leak_empty_time : ℝ) 
  (h1 : normal_fill_time = 8) 
  (h2 : leak_empty_time = 40.00000000000001) : 
  (1 / (1 / normal_fill_time - 1 / leak_empty_time)) - normal_fill_time = 2.000000000000003 := by
  sorry

end cistern_fill_time_with_leak_l557_55745


namespace equation_solution_l557_55788

/-- Floor function: greatest integer less than or equal to x -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Ceiling function: smallest integer greater than or equal to x -/
noncomputable def ceil (x : ℝ) : ℤ :=
  -Int.floor (-x)

/-- Nearest integer function: integer closest to x (x ≠ n + 0.5 for any integer n) -/
noncomputable def nearest (x : ℝ) : ℤ :=
  if x - Int.floor x < 0.5 then Int.floor x else Int.floor x + 1

/-- Theorem: The equation 3⌊x⌋ + 2⌈x⌉ + ⟨x⟩ = 8 is satisfied if and only if 1 < x < 1.5 -/
theorem equation_solution (x : ℝ) :
  3 * (floor x) + 2 * (ceil x) + (nearest x) = 8 ↔ 1 < x ∧ x < 1.5 := by
  sorry

end equation_solution_l557_55788


namespace multiply_subtract_difference_l557_55769

theorem multiply_subtract_difference (x : ℝ) (h : x = 13) : 3 * x - (36 - x) = 16 := by
  sorry

end multiply_subtract_difference_l557_55769


namespace chess_tournament_rounds_l557_55715

/-- The number of rounds needed for a chess tournament -/
theorem chess_tournament_rounds (n : ℕ) (games_per_round : ℕ) 
  (h1 : n = 20) 
  (h2 : games_per_round = 10) : 
  (n * (n - 1)) / games_per_round = 38 := by
  sorry

#check chess_tournament_rounds

end chess_tournament_rounds_l557_55715


namespace jean_calories_consumed_l557_55746

/-- Calculates the total calories consumed by Jean while writing her paper. -/
def total_calories (pages : ℕ) (pages_per_donut : ℕ) (calories_per_donut : ℕ) : ℕ :=
  let donuts := (pages + pages_per_donut - 1) / pages_per_donut
  donuts * calories_per_donut

/-- Proves that Jean consumes 1260 calories while writing her paper. -/
theorem jean_calories_consumed :
  total_calories 20 3 180 = 1260 := by
  sorry

end jean_calories_consumed_l557_55746


namespace x_range_given_quadratic_inequality_l557_55775

theorem x_range_given_quadratic_inequality (x : ℝ) :
  4 - x^2 ≤ 0 → x ≤ -2 ∨ x ≥ 2 := by
  sorry

end x_range_given_quadratic_inequality_l557_55775


namespace second_square_area_l557_55704

/-- An isosceles right triangle with two inscribed squares -/
structure IsoscelesRightTriangleWithSquares where
  /-- Side length of the first inscribed square -/
  s : ℝ
  /-- Area of the first inscribed square is 484 -/
  h_area : s^2 = 484
  /-- Side length of the second inscribed square -/
  S : ℝ
  /-- The second square shares one side with the hypotenuse and its opposite vertex touches the midpoint of the hypotenuse -/
  h_S : S = (2 * s * Real.sqrt 2) / 3

/-- The area of the second inscribed square is 3872/9 -/
theorem second_square_area (triangle : IsoscelesRightTriangleWithSquares) : 
  triangle.S^2 = 3872 / 9 := by
  sorry

end second_square_area_l557_55704


namespace eventually_monotonic_sequence_l557_55730

/-- An infinite sequence of real numbers where no two members are equal -/
def UniqueMemberSequence (a : ℕ → ℝ) : Prop :=
  ∀ i j, i ≠ j → a i ≠ a j

/-- A monotonic segment of length m starting at index i -/
def MonotonicSegment (a : ℕ → ℝ) (i m : ℕ) : Prop :=
  (∀ k, k < m - 1 → a (i + k) < a (i + k + 1)) ∨
  (∀ k, k < m - 1 → a (i + k) > a (i + k + 1))

/-- For each natural k, the term aₖ is contained in some monotonic segment of length k + 1 -/
def ContainedInMonotonicSegment (a : ℕ → ℝ) : Prop :=
  ∀ k, ∃ i, k ∈ Finset.range (k + 1) ∧ MonotonicSegment a i (k + 1)

/-- The sequence is eventually monotonic -/
def EventuallyMonotonic (a : ℕ → ℝ) : Prop :=
  ∃ N, (∀ n ≥ N, a n < a (n + 1)) ∨ (∀ n ≥ N, a n > a (n + 1))

theorem eventually_monotonic_sequence
  (a : ℕ → ℝ)
  (h1 : UniqueMemberSequence a)
  (h2 : ContainedInMonotonicSegment a) :
  EventuallyMonotonic a :=
sorry

end eventually_monotonic_sequence_l557_55730


namespace total_cans_is_twelve_l557_55752

/-- Represents the ratio of chili beans to tomato soup -/
def chili_to_tomato_ratio : ℚ := 2

/-- Represents the number of chili bean cans ordered -/
def chili_beans_ordered : ℕ := 8

/-- Calculates the total number of cans ordered -/
def total_cans_ordered : ℕ :=
  chili_beans_ordered + (chili_beans_ordered / chili_to_tomato_ratio.num).toNat

/-- Proves that the total number of cans ordered is 12 -/
theorem total_cans_is_twelve : total_cans_ordered = 12 := by
  sorry

end total_cans_is_twelve_l557_55752


namespace hotel_rooms_theorem_l557_55799

/-- The minimum number of rooms needed for 100 tourists with k rooms under renovation -/
def min_rooms (k : ℕ) : ℕ :=
  let m := k / 2
  if k % 2 = 0 then
    100 * (m + 1)
  else
    100 * (m + 1) + 1

/-- Theorem stating the minimum number of rooms needed for 100 tourists -/
theorem hotel_rooms_theorem (k : ℕ) :
  ∀ n : ℕ, n ≥ min_rooms k →
  ∃ strategy : (Fin 100 → Fin n → Option (Fin n)),
  (∀ i : Fin 100, ∃ room : Fin n, strategy i room = some room) ∧
  (∀ i j : Fin 100, i ≠ j →
    ∀ room : Fin n, strategy i room ≠ none → strategy j room = none) :=
by
  sorry

#check hotel_rooms_theorem

end hotel_rooms_theorem_l557_55799


namespace min_value_expression_l557_55765

theorem min_value_expression (x y : ℝ) : 4 + x^2 * y^4 + x^4 * y^2 - 3 * x^2 * y^2 ≥ 3 := by
  sorry

end min_value_expression_l557_55765


namespace angle_2_measure_l557_55700

-- Define complementary angles
def complementary (a1 a2 : ℝ) : Prop := a1 + a2 = 90

-- Theorem statement
theorem angle_2_measure (angle1 angle2 : ℝ) 
  (h1 : complementary angle1 angle2) (h2 : angle1 = 25) : 
  angle2 = 65 := by
  sorry

end angle_2_measure_l557_55700


namespace truncated_cone_radius_theorem_l557_55743

/-- Represents a cone with its base radius -/
structure Cone where
  baseRadius : ℝ

/-- Represents a truncated cone with its smaller base radius -/
structure TruncatedCone where
  smallerBaseRadius : ℝ

/-- Given three cones touching each other and a truncated cone sharing
    a common generatrix with each, compute the smaller base radius of the truncated cone -/
def computeTruncatedConeRadius (c1 c2 c3 : Cone) : ℝ :=
  6

theorem truncated_cone_radius_theorem (c1 c2 c3 : Cone) (tc : TruncatedCone) 
    (h1 : c1.baseRadius = 23)
    (h2 : c2.baseRadius = 46)
    (h3 : c3.baseRadius = 69)
    (h4 : tc.smallerBaseRadius = computeTruncatedConeRadius c1 c2 c3) :
  tc.smallerBaseRadius = 6 := by
  sorry

end truncated_cone_radius_theorem_l557_55743


namespace optimal_pole_minimizes_time_l557_55790

/-- The optimal pole number for tying Bolivar -/
def optimal_pole : ℕ := 21

/-- The total number of poles -/
def total_poles : ℕ := 27

/-- The total number of segments -/
def total_segments : ℕ := 28

/-- Dotson's walking speed in units per minute -/
def dotson_speed : ℚ := 1 / 9

/-- Williams' walking speed in units per minute -/
def williams_speed : ℚ := 1 / 11

/-- Bolivar's riding speed in units per minute -/
def bolivar_speed : ℚ := 1 / 3

/-- The time taken by Dotson to complete the journey -/
def dotson_time (k : ℕ) : ℚ := 9 - (6 * k : ℚ) / 28

/-- The time taken by Williams to complete the journey -/
def williams_time (k : ℕ) : ℚ := 3 + (2 * k : ℚ) / 7

theorem optimal_pole_minimizes_time :
  ∀ k : ℕ, k ≤ total_poles →
    max (dotson_time k) (williams_time k) ≥ dotson_time optimal_pole :=
by sorry


end optimal_pole_minimizes_time_l557_55790


namespace polynomial_coefficients_l557_55748

theorem polynomial_coefficients (a b c : ℚ) :
  let f : ℚ → ℚ := λ x => c * x^4 + a * x^3 - 3 * x^2 + b * x - 8
  (f 2 = -8) ∧ (f (-3) = -68) → (a = 5 ∧ b = 7 ∧ c = 1) := by
  sorry

end polynomial_coefficients_l557_55748


namespace pentagonal_prism_sum_l557_55712

/-- A pentagonal prism is a three-dimensional geometric shape with specific properties. -/
structure PentagonalPrism where
  /-- The number of faces in a pentagonal prism -/
  faces : ℕ
  /-- The number of edges in a pentagonal prism -/
  edges : ℕ
  /-- The number of vertices in a pentagonal prism -/
  vertices : ℕ
  /-- A pentagonal prism has 7 faces (2 pentagonal bases + 5 rectangular lateral faces) -/
  faces_count : faces = 7
  /-- A pentagonal prism has 15 edges (5 for each base + 5 connecting edges) -/
  edges_count : edges = 15
  /-- A pentagonal prism has 10 vertices (5 for each base) -/
  vertices_count : vertices = 10

/-- The sum of faces, edges, and vertices of a pentagonal prism is 32. -/
theorem pentagonal_prism_sum (p : PentagonalPrism) : p.faces + p.edges + p.vertices = 32 := by
  sorry

end pentagonal_prism_sum_l557_55712


namespace gmat_test_problem_l557_55781

theorem gmat_test_problem (first_correct : Real) (second_correct : Real) (neither_correct : Real) :
  first_correct = 85 / 100 →
  second_correct = 65 / 100 →
  neither_correct = 5 / 100 →
  first_correct + second_correct - (1 - neither_correct) = 55 / 100 := by
  sorry

end gmat_test_problem_l557_55781


namespace faulty_engine_sampling_l557_55783

/-- Given a set of 33 items where 8 are faulty, this theorem proves:
    1. The probability of identifying all faulty items by sampling 32 items
    2. The expected number of samplings required to identify all faulty items -/
theorem faulty_engine_sampling (n : Nat) (k : Nat) (h1 : n = 33) (h2 : k = 8) :
  let p := Nat.choose (n - 1) (k - 1) / Nat.choose n k
  let e := (n * k) / (n - k + 1)
  (p = 25 / 132) ∧ (e = 272 / 9) := by
  sorry

#check faulty_engine_sampling

end faulty_engine_sampling_l557_55783


namespace bridge_length_l557_55722

/-- The length of a bridge that a train can cross, given the train's length, speed, and time to cross. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (time_to_cross : ℝ) :
  train_length = 256 →
  train_speed_kmh = 72 →
  time_to_cross = 20 →
  (train_speed_kmh * 1000 / 3600 * time_to_cross) - train_length = 144 := by
  sorry

#check bridge_length

end bridge_length_l557_55722


namespace abs_w_equals_3_fourth_root_2_l557_55736

-- Define w as a complex number
variable (w : ℂ)

-- State the theorem
theorem abs_w_equals_3_fourth_root_2 (h : w^2 = -18 + 18*I) : 
  Complex.abs w = 3 * (2 : ℝ)^(1/4) := by
  sorry

end abs_w_equals_3_fourth_root_2_l557_55736


namespace variance_describes_dispersion_l557_55759

-- Define the type for statistical measures
inductive StatMeasure
  | Mean
  | Variance
  | Median
  | Mode

-- Define a property for measures that describe dispersion
def describes_dispersion (m : StatMeasure) : Prop :=
  match m with
  | StatMeasure.Variance => True
  | _ => False

-- Theorem statement
theorem variance_describes_dispersion :
  ∀ m : StatMeasure, describes_dispersion m ↔ m = StatMeasure.Variance :=
by sorry

end variance_describes_dispersion_l557_55759


namespace bike_retail_price_l557_55797

/-- The retail price of a bike, given Maria's savings, her mother's offer, and the additional amount needed. -/
theorem bike_retail_price
  (maria_savings : ℕ)
  (mother_offer : ℕ)
  (additional_needed : ℕ)
  (h1 : maria_savings = 120)
  (h2 : mother_offer = 250)
  (h3 : additional_needed = 230) :
  maria_savings + mother_offer + additional_needed = 600 :=
by sorry

end bike_retail_price_l557_55797


namespace rationality_of_given_numbers_l557_55753

theorem rationality_of_given_numbers :
  (∃ (a b : ℚ), a^2 = 4 ∧ b ≠ 0) ∧  -- √4 is rational
  (∀ (a b : ℚ), a^3 ≠ 0.5 * b^3) ∧  -- ∛0.5 is irrational
  (∃ (a b : ℚ), a^4 = 0.0625 * b^4 ∧ b ≠ 0) ∧  -- ∜0.0625 is rational
  (∃ (a b : ℚ), a^3 = -8 ∧ b^2 = 4 ∧ b ≠ 0) :=  -- ∛(-8) * √((0.25)^(-1)) is rational
by sorry

end rationality_of_given_numbers_l557_55753


namespace cubic_equation_with_double_root_l557_55792

theorem cubic_equation_with_double_root (k : ℝ) :
  (∃ a b : ℝ, (3 * a^3 + 9 * a^2 - 150 * a + k = 0) ∧
              (3 * b^3 + 9 * b^2 - 150 * b + k = 0) ∧
              (a ≠ b)) ∧
  (∃ x : ℝ, (3 * x^3 + 9 * x^2 - 150 * x + k = 0) ∧
            (∃ y : ℝ, y ≠ x ∧ 3 * y^3 + 9 * y^2 - 150 * y + k = 0)) ∧
  (k > 0) →
  k = 84 :=
by sorry

end cubic_equation_with_double_root_l557_55792


namespace man_rowing_speed_l557_55751

/-- Calculates the speed of a man rowing upstream given his speed in still water and downstream speed -/
def speed_upstream (speed_still : ℝ) (speed_downstream : ℝ) : ℝ :=
  2 * speed_still - speed_downstream

/-- Theorem stating that given a man's speed in still water is 32 kmph and his speed downstream is 42 kmph, his speed upstream is 22 kmph -/
theorem man_rowing_speed 
  (speed_still : ℝ) 
  (speed_downstream : ℝ) 
  (h1 : speed_still = 32) 
  (h2 : speed_downstream = 42) : 
  speed_upstream speed_still speed_downstream = 22 := by
  sorry

#eval speed_upstream 32 42

end man_rowing_speed_l557_55751


namespace expression_simplification_l557_55702

theorem expression_simplification (x : ℝ) (h : x = 3) :
  (x^2 / (x - 2) - x - 2) / (4 * x / (x^2 - 4)) = 5 / 3 := by
  sorry

end expression_simplification_l557_55702


namespace contest_end_time_l557_55771

def contest_start : Nat := 12 * 60  -- noon in minutes since midnight
def contest_duration : Nat := 1000  -- duration in minutes

theorem contest_end_time :
  (contest_start + contest_duration) % (24 * 60) = 4 * 60 + 40 :=
sorry

end contest_end_time_l557_55771


namespace smallest_valid_number_l557_55784

def starts_with_19 (n : ℕ) : Prop :=
  ∃ k : ℕ, n ≥ 19 * 10^k ∧ n < 20 * 10^k

def ends_with_89 (n : ℕ) : Prop :=
  n % 100 = 89

def is_valid_number (n : ℕ) : Prop :=
  starts_with_19 (n^2) ∧ ends_with_89 (n^2)

theorem smallest_valid_number :
  is_valid_number 1383 ∧ ∀ m : ℕ, m < 1383 → ¬(is_valid_number m) :=
by sorry

end smallest_valid_number_l557_55784


namespace part_one_part_two_l557_55707

-- Part 1
theorem part_one (α : Real) (h : Real.tan α = 2) :
  (Real.cos α + Real.sin α) / (Real.cos α - Real.sin α) = -3 := by sorry

-- Part 2
theorem part_two (α : Real) :
  (Real.sin (α - π/2) * Real.cos (π/2 - α) * Real.tan (π - α)) / 
  (Real.tan (π + α) * Real.sin (π + α)) = -Real.cos α := by sorry

end part_one_part_two_l557_55707


namespace employee_payment_proof_l557_55713

/-- The weekly payment for employee B -/
def payment_B : ℝ := 180

/-- The weekly payment for employee A -/
def payment_A : ℝ := 1.5 * payment_B

/-- The total weekly payment for both employees -/
def total_payment : ℝ := 450

theorem employee_payment_proof :
  payment_A + payment_B = total_payment :=
by sorry

end employee_payment_proof_l557_55713


namespace unique_divisor_square_sum_l557_55798

theorem unique_divisor_square_sum (p n : ℕ) (hp : Prime p) (hn : n > 0) (hodd : Odd p) :
  ∃! d : ℕ, d > 0 ∧ d ∣ (p * n^2) ∧ ∃ m : ℕ, d + n^2 = m^2 ↔ ∃ k : ℕ, n = k * ((p - 1) / 2) :=
sorry

end unique_divisor_square_sum_l557_55798


namespace unattainable_y_l557_55747

theorem unattainable_y (x : ℝ) (y : ℝ) (h1 : x ≠ -3/2) (h2 : y = (1-x)/(2*x+3)) :
  y ≠ -1/2 :=
by sorry

end unattainable_y_l557_55747


namespace limit_expression_l557_55737

/-- The limit of (2 - e^(arcsin²(√x)))^(3/x) as x approaches 0 is e^(-3) -/
theorem limit_expression : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → 
    |(2 - Real.exp (Real.arcsin (Real.sqrt x))^2)^(3/x) - Real.exp (-3)| < ε := by
  sorry

end limit_expression_l557_55737


namespace number_of_women_at_event_l557_55708

/-- Proves that the number of women at an event is 20, given the specified dancing conditions -/
theorem number_of_women_at_event (num_men : ℕ) (men_dance_count : ℕ) (women_dance_count : ℕ) 
  (h1 : num_men = 15)
  (h2 : men_dance_count = 4)
  (h3 : women_dance_count = 3)
  : (num_men * men_dance_count) / women_dance_count = 20 := by
  sorry

#check number_of_women_at_event

end number_of_women_at_event_l557_55708


namespace sphere_packing_radius_l557_55711

/-- A sphere in a unit cube -/
structure SpherePacking where
  radius : ℝ
  center_at_vertex : Bool
  touches_three_faces : Bool
  tangent_to_six_neighbors : Bool

/-- The theorem stating the radius of spheres in the specific packing -/
theorem sphere_packing_radius (s : SpherePacking) :
  s.center_at_vertex ∧ 
  s.touches_three_faces ∧ 
  s.tangent_to_six_neighbors →
  s.radius = (Real.sqrt 3 * (Real.sqrt 3 - 1)) / 4 :=
sorry

end sphere_packing_radius_l557_55711


namespace triangle_shape_l557_55705

/-- Given a triangle ABC where BC⋅cos A = AC⋅cos B, prove that the triangle is either isosceles or right-angled -/
theorem triangle_shape (A B C : Real) (BC AC : Real) 
  (h : BC * Real.cos A = AC * Real.cos B) :
  (A = B) ∨ (A + B = Real.pi / 2) := by
  sorry

end triangle_shape_l557_55705


namespace chosen_number_proof_l557_55778

theorem chosen_number_proof (x : ℝ) : (x / 6) - 189 = 3 → x = 1152 := by
  sorry

end chosen_number_proof_l557_55778


namespace line_through_point_and_circle_center_l557_55757

/-- The equation of the line passing through (2,1) and the center of the circle (x-1)^2 + (y+2)^2 = 5 is 3x - y - 5 = 0 -/
theorem line_through_point_and_circle_center :
  let point : ℝ × ℝ := (2, 1)
  let circle_center : ℝ × ℝ := (1, -2)
  let line_equation (x y : ℝ) := 3 * x - y - 5 = 0
  ∀ x y : ℝ, line_equation x y ↔ (y - point.2) / (x - point.1) = (circle_center.2 - point.2) / (circle_center.1 - point.1) :=
by sorry

end line_through_point_and_circle_center_l557_55757


namespace larger_number_proof_l557_55796

theorem larger_number_proof (x y : ℕ) 
  (h1 : y - x = 480) 
  (h2 : y = 4 * x + 30) : 
  y = 630 := by
sorry

end larger_number_proof_l557_55796


namespace reflection_line_l557_55733

/-- Given a line y = mx + b, if the reflection of point (2, 3) across this line is (10, 9), then m + b = 38/3 -/
theorem reflection_line (m b : ℝ) : 
  (∃ (x y : ℝ), x = 10 ∧ y = 9 ∧ 
    (x - 2)^2 + (y - 3)^2 = ((x - 2) * m + (y - 3))^2 / (m^2 + 1) ∧
    y - 3 = -m * (x - 2)) →
  m + b = 38/3 :=
by sorry

end reflection_line_l557_55733


namespace arithmetic_geometric_sequence_l557_55738

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  (∀ n, a (n + 1) = a n + d) →  -- arithmetic sequence
  d ≠ 0 →                       -- non-zero common difference
  a 1 = 1 →                     -- a_1 = 1
  (a 3)^2 = a 1 * a 13 →        -- a_1, a_3, a_13 form a geometric sequence
  d = 2 := by sorry

end arithmetic_geometric_sequence_l557_55738


namespace ramsey_type_theorem_l557_55764

theorem ramsey_type_theorem (n r : ℕ) (hn : n > 0) (hr : r > 0) :
  ∃ m : ℕ, m > 0 ∧
  (∀ (A : Fin r → Set ℕ),
    (∀ i j : Fin r, i ≠ j → A i ∩ A j = ∅) →
    (⋃ i : Fin r, A i) = Finset.range m →
    ∃ (i : Fin r) (a b : ℕ), a ∈ A i ∧ b ∈ A i ∧ b < a ∧ a ≤ (n + 1) * b / n) ∧
  (∀ k : ℕ, 0 < k → k < m →
    ∃ (A : Fin r → Set ℕ),
      (∀ i j : Fin r, i ≠ j → A i ∩ A j = ∅) ∧
      (⋃ i : Fin r, A i) = Finset.range k ∧
      ∀ (i : Fin r) (a b : ℕ), a ∈ A i → b ∈ A i → b < a → a > (n + 1) * b / n) ∧
  m = (n + 1) * r :=
by sorry

end ramsey_type_theorem_l557_55764


namespace same_solution_value_l557_55720

theorem same_solution_value (c : ℝ) : 
  (∃ x : ℝ, 3 * x + 5 = 2 ∧ c * x + 4 = 1) ↔ c = 3 :=
by sorry

end same_solution_value_l557_55720


namespace remainder_of_polynomial_l557_55721

theorem remainder_of_polynomial (r : ℝ) : 
  (r^15 - r^3 + 1) % (r - 1) = 1 := by
sorry

end remainder_of_polynomial_l557_55721


namespace max_sum_abc_l557_55719

def An (a n : ℕ) : ℕ := a * (8^n - 1) / 7
def Bn (b n : ℕ) : ℕ := b * (6^n - 1) / 5
def Cn (c n : ℕ) : ℕ := c * (10^(3*n) - 1) / 9

theorem max_sum_abc :
  ∃ (a b c : ℕ),
    (0 < a ∧ a ≤ 9) ∧
    (0 < b ∧ b ≤ 9) ∧
    (0 < c ∧ c ≤ 9) ∧
    (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ 
      Cn c n₁ - Bn b n₁ = (An a n₁)^3 ∧
      Cn c n₂ - Bn b n₂ = (An a n₂)^3) ∧
    (∀ (a' b' c' : ℕ),
      (0 < a' ∧ a' ≤ 9) →
      (0 < b' ∧ b' ≤ 9) →
      (0 < c' ∧ c' ≤ 9) →
      (∃ (n₁ n₂ : ℕ), n₁ ≠ n₂ ∧ 
        Cn c' n₁ - Bn b' n₁ = (An a' n₁)^3 ∧
        Cn c' n₂ - Bn b' n₂ = (An a' n₂)^3) →
      a + b + c ≥ a' + b' + c') ∧
    a + b + c = 21 := by
  sorry

end max_sum_abc_l557_55719


namespace largest_non_expressible_l557_55770

/-- A function that checks if a number is composite -/
def IsComposite (n : ℕ) : Prop :=
  ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

/-- A function that checks if a number can be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
def CanBeExpressed (n : ℕ) : Prop :=
  ∃ k m, k > 0 ∧ IsComposite m ∧ n = 30 * k + m

/-- Theorem stating that 211 is the largest positive integer that cannot be expressed as the sum of a positive integral multiple of 30 and a positive composite integer -/
theorem largest_non_expressible : ∀ n : ℕ, n > 211 → CanBeExpressed n ∧ ¬CanBeExpressed 211 :=
sorry

end largest_non_expressible_l557_55770


namespace a_10_has_1000_nines_l557_55791

def sequence_a : ℕ → ℕ
  | 0 => 9
  | (k + 1) => 3 * (sequence_a k)^4 + 4 * (sequence_a k)^3

def has_consecutive_nines (n : ℕ) (count : ℕ) : Prop :=
  ∃ m : ℕ, n = m * 10^count + (10^count - 1)

theorem a_10_has_1000_nines : 
  has_consecutive_nines (sequence_a 10) 1000 := by sorry

end a_10_has_1000_nines_l557_55791


namespace max_blocks_fit_l557_55741

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a rectangular object given its dimensions -/
def volume (d : Dimensions) : ℕ := d.length * d.width * d.height

/-- Represents the box and block dimensions -/
def box : Dimensions := ⟨40, 60, 80⟩
def block : Dimensions := ⟨20, 30, 40⟩

/-- Calculates the maximum number of blocks that can fit in the box based on volume -/
def max_blocks_by_volume : ℕ := volume box / volume block

/-- Checks if the blocks can be arranged to fit in the box -/
def can_arrange (n : ℕ) : Prop :=
  ∃ (l w h : ℕ), l * block.length ≤ box.length ∧
                 w * block.width ≤ box.width ∧
                 h * block.height ≤ box.height ∧
                 l * w * h = n

/-- The main theorem to prove -/
theorem max_blocks_fit :
  max_blocks_by_volume = 8 ∧ can_arrange 8 ∧
  ∀ n > 8, ¬can_arrange n :=
sorry

end max_blocks_fit_l557_55741


namespace parabola_with_directrix_neg_seven_l557_55761

/-- Represents a parabola with a vertical axis of symmetry -/
structure Parabola where
  /-- The distance from the vertex to the focus or directrix -/
  p : ℝ
  /-- Indicates whether the parabola opens to the right (true) or left (false) -/
  opensRight : Bool

/-- The standard equation of a parabola -/
def standardEquation (par : Parabola) : ℝ → ℝ → Prop :=
  if par.opensRight then
    fun y x => y^2 = 4 * par.p * x
  else
    fun y x => y^2 = -4 * par.p * x

/-- The equation of the directrix of a parabola -/
def directrixEquation (par : Parabola) : ℝ → Prop :=
  if par.opensRight then
    fun x => x = -par.p
  else
    fun x => x = par.p

theorem parabola_with_directrix_neg_seven (par : Parabola) :
  directrixEquation par = fun x => x = -7 →
  standardEquation par = fun y x => y^2 = 28 * x :=
by
  sorry


end parabola_with_directrix_neg_seven_l557_55761


namespace acme_cheaper_at_min_shirts_l557_55735

/-- Acme's pricing function -/
def acme_price (x : ℕ) : ℚ := 30 + 7 * x

/-- Gamma's pricing function -/
def gamma_price (x : ℕ) : ℚ := 11 * x

/-- The minimum number of shirts for which Acme is cheaper -/
def min_shirts_acme_cheaper : ℕ := 8

theorem acme_cheaper_at_min_shirts :
  acme_price min_shirts_acme_cheaper < gamma_price min_shirts_acme_cheaper ∧
  ∀ n : ℕ, n < min_shirts_acme_cheaper → acme_price n ≥ gamma_price n :=
by sorry

end acme_cheaper_at_min_shirts_l557_55735


namespace final_fish_count_l557_55717

/- Define the initial number of fish -/
variable (F : ℚ)

/- Define the number of fish after each day's operations -/
def fish_count (day : ℕ) : ℚ :=
  match day with
  | 0 => F
  | 1 => 2 * F
  | 2 => 4 * F * (2/3)
  | 3 => 8 * F * (2/3)
  | 4 => 16 * F * (2/3) * (3/4)
  | 5 => 32 * F * (2/3) * (3/4)
  | 6 => 64 * F * (2/3) * (3/4)
  | _ => 128 * F * (2/3) * (3/4) + 15

/- Theorem stating that the final count is 207 if and only if F = 6 -/
theorem final_fish_count (F : ℚ) : fish_count F 7 = 207 ↔ F = 6 := by
  sorry

end final_fish_count_l557_55717


namespace five_balls_two_boxes_l557_55734

/-- The number of ways to distribute distinguishable objects into distinguishable containers -/
def distribute_objects (n : ℕ) (k : ℕ) : ℕ := k^n

/-- Theorem: There are 32 ways to distribute 5 distinguishable balls into 2 distinguishable boxes -/
theorem five_balls_two_boxes : distribute_objects 5 2 = 32 := by
  sorry

end five_balls_two_boxes_l557_55734


namespace square_region_area_l557_55777

/-- A region consisting of equal squares inscribed in a rectangle -/
structure SquareRegion where
  num_squares : ℕ
  rect_width : ℝ
  rect_height : ℝ

/-- Calculate the area of a SquareRegion -/
def area (r : SquareRegion) : ℝ :=
  sorry

/-- The main theorem to prove -/
theorem square_region_area (r : SquareRegion) 
  (h1 : r.num_squares = 13)
  (h2 : r.rect_width = 28)
  (h3 : r.rect_height = 26) : 
  area r = 338 := by
  sorry

end square_region_area_l557_55777


namespace problem_solution_l557_55795

def A : Set ℝ := {-2, 3, 4, 6}
def B (a : ℝ) : Set ℝ := {3, a, a^2}

theorem problem_solution (a : ℝ) :
  (B a ⊆ A → a = 2) ∧
  (A ∩ B a = {3, 4} → a = 2 ∨ a = 4) :=
by sorry

end problem_solution_l557_55795


namespace nested_fraction_evaluation_l557_55749

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by
  sorry

end nested_fraction_evaluation_l557_55749


namespace triangle_inequality_violation_l557_55724

/-- Theorem: A triangle cannot be formed with side lengths 9, 4, and 3. -/
theorem triangle_inequality_violation (a b c : ℝ) 
  (ha : a = 9) (hb : b = 4) (hc : c = 3) : 
  ¬(a + b > c ∧ a + c > b ∧ b + c > a) := by
  sorry

end triangle_inequality_violation_l557_55724


namespace part_one_part_two_l557_55782

-- Define set A
def A : Set ℝ := {x | (x + 1) / (x - 3) ≤ 0}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - (m - 1) * x + m - 2 ≤ 0}

-- Statement for part (1)
theorem part_one (a b : ℝ) : A ∪ Set.Icc a b = Set.Icc (-1) 4 → b = 4 ∧ -1 ≤ a ∧ a < 3 := by
  sorry

-- Statement for part (2)
theorem part_two (m : ℝ) : A ∪ B m = A → 1 ≤ m ∧ m < 5 := by
  sorry

end part_one_part_two_l557_55782


namespace quadratic_inequality_solution_set_l557_55787

-- Define the quadratic function
def f (a b c x : ℝ) := a * x^2 + b * x + c

-- Define the solution set of the original inequality
def S := {x : ℝ | -1 < x ∧ x < 2}

-- Theorem statement
theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : ∀ x, f a b c x > 0 ↔ x ∈ S) :
  ∀ x, f c b a x > 0 ↔ x < -1 ∨ x > (1/2) :=
sorry

end quadratic_inequality_solution_set_l557_55787


namespace cowboy_shortest_path_l557_55739

/-- The shortest path for a cowboy to travel from his position to a stream and then to his cabin -/
theorem cowboy_shortest_path (cowboy_pos cabin_pos : ℝ × ℝ) (stream_y : ℝ) :
  cowboy_pos = (0, -5) →
  cabin_pos = (6, 4) →
  stream_y = 0 →
  let dist_to_stream := |cowboy_pos.2 - stream_y|
  let dist_stream_to_cabin := Real.sqrt ((cabin_pos.1 - cowboy_pos.1)^2 + (cabin_pos.2 - stream_y)^2)
  dist_to_stream + dist_stream_to_cabin = 5 + 2 * Real.sqrt 58 :=
by sorry

end cowboy_shortest_path_l557_55739


namespace taco_cost_is_90_cents_l557_55793

-- Define the cost of a taco and an enchilada
variable (taco_cost enchilada_cost : ℚ)

-- Define the two orders
def order1_cost := 2 * taco_cost + 3 * enchilada_cost
def order2_cost := 3 * taco_cost + 5 * enchilada_cost

-- State the theorem
theorem taco_cost_is_90_cents 
  (h1 : order1_cost = 780/100)
  (h2 : order2_cost = 1270/100) :
  taco_cost = 90/100 := by
  sorry

end taco_cost_is_90_cents_l557_55793


namespace cats_remaining_after_sale_l557_55754

/-- The number of cats remaining after a sale at a pet store -/
theorem cats_remaining_after_sale 
  (siamese_cats : ℕ) 
  (house_cats : ℕ) 
  (cats_sold : ℕ) 
  (h1 : siamese_cats = 13)
  (h2 : house_cats = 5)
  (h3 : cats_sold = 10) :
  siamese_cats + house_cats - cats_sold = 8 := by
  sorry

end cats_remaining_after_sale_l557_55754


namespace roots_quadratic_equation_l557_55731

theorem roots_quadratic_equation (m n : ℝ) : 
  (m^2 + 5*m + 3 = 0) → 
  (n^2 + 5*n + 3 = 0) → 
  m * Real.sqrt (n / m) + n * Real.sqrt (m / n) = -2 * Real.sqrt 3 :=
by sorry

end roots_quadratic_equation_l557_55731


namespace smallest_prime_after_six_nonprimes_l557_55755

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def consecutive_nonprimes (start : ℕ) : Prop :=
  ∀ k : ℕ, k ≥ start ∧ k < start + 6 → ¬(is_prime k)

theorem smallest_prime_after_six_nonprimes :
  ∃ n : ℕ, consecutive_nonprimes n ∧ is_prime (n + 6) ∧
  ∀ m : ℕ, m < n → ¬(consecutive_nonprimes m ∧ is_prime (m + 6)) :=
sorry

end smallest_prime_after_six_nonprimes_l557_55755


namespace linear_equation_solution_l557_55709

theorem linear_equation_solution (V : ℝ → ℝ) (p q : ℝ) 
  (h1 : ∀ t, V t = p * t + q)
  (h2 : V 0 = 100)
  (h3 : V 10 = 103.5) :
  p = 0.35 := by
sorry

end linear_equation_solution_l557_55709


namespace polynomial_equality_l557_55744

theorem polynomial_equality : 102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 := by
  sorry

end polynomial_equality_l557_55744


namespace parallel_vectors_m_value_l557_55774

/-- Given two vectors a and b in R², prove that if they are parallel and have the given components, then m equals either -√2 or √2. -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (m, 2)
  (∃ (k : ℝ), a = k • b) → (m = -Real.sqrt 2 ∨ m = Real.sqrt 2) := by
  sorry

end parallel_vectors_m_value_l557_55774


namespace fraction_problem_l557_55727

theorem fraction_problem (x : ℚ) (h : 75 * x = 37.5) : x = 1/2 := by
  sorry

end fraction_problem_l557_55727


namespace prob_white_same_color_five_balls_l557_55760

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The probability of drawing two white balls given that the drawn balls are of the same color -/
def prob_white_given_same_color (total white black : ℕ) : ℚ :=
  let white_ways := choose white 2
  let black_ways := choose black 2
  white_ways / (white_ways + black_ways)

theorem prob_white_same_color_five_balls :
  prob_white_given_same_color 5 3 2 = 3/4 := by sorry

end prob_white_same_color_five_balls_l557_55760


namespace arcsin_equation_solution_l557_55710

theorem arcsin_equation_solution :
  let f (x : ℝ) := Real.arcsin (x * Real.sqrt 5 / 3) + Real.arcsin (x * Real.sqrt 5 / 6) - Real.arcsin (7 * x * Real.sqrt 5 / 18)
  ∀ x : ℝ, 
    (abs x ≤ 18 / (7 * Real.sqrt 5)) →
    (f x = 0 ↔ x = 0 ∨ x = 8/7 ∨ x = -8/7) :=
by sorry

end arcsin_equation_solution_l557_55710


namespace x_sixth_minus_six_x_squared_l557_55706

theorem x_sixth_minus_six_x_squared (x : ℝ) (h : x = 3) : x^6 - 6*x^2 = 675 := by
  sorry

end x_sixth_minus_six_x_squared_l557_55706


namespace flower_cost_minimization_l557_55729

/-- The cost of one lily in dollars -/
def lily_cost : ℝ := 5

/-- The cost of one carnation in dollars -/
def carnation_cost : ℝ := 6

/-- The total number of flowers to be bought -/
def total_flowers : ℕ := 12

/-- The minimum number of carnations to be bought -/
def min_carnations : ℕ := 5

/-- The cost function for buying x lilies -/
def cost_function (x : ℝ) : ℝ := -x + 72

theorem flower_cost_minimization :
  ∃ (x : ℝ),
    x ≤ total_flowers - min_carnations ∧
    x ≥ 0 ∧
    ∀ (y : ℝ),
      y ≤ total_flowers - min_carnations ∧
      y ≥ 0 →
      cost_function x ≤ cost_function y ∧
      cost_function x = 65 :=
sorry

end flower_cost_minimization_l557_55729


namespace function_difference_theorem_l557_55740

theorem function_difference_theorem (m : ℚ) : 
  let f : ℚ → ℚ := λ x => 4 * x^2 - 3 * x + 5
  let g : ℚ → ℚ := λ x => 2 * x^2 - m * x + 8
  (f 5 - g 5 = 15) → m = -17/5 := by
sorry

end function_difference_theorem_l557_55740


namespace sarah_apples_to_teachers_l557_55773

/-- The number of apples Sarah gives to teachers -/
def apples_to_teachers (initial : ℕ) (locker : ℕ) (classmates : ℕ) (friends : ℕ) (eaten : ℕ) (left : ℕ) : ℕ :=
  initial - locker - classmates - friends - eaten - left

theorem sarah_apples_to_teachers :
  apples_to_teachers 50 10 8 5 1 4 = 22 := by
  sorry

#eval apples_to_teachers 50 10 8 5 1 4

end sarah_apples_to_teachers_l557_55773


namespace arithmetic_sequence_smallest_negative_smallest_n_is_minimal_l557_55728

/-- The smallest positive integer n such that 2009 - 7n < 0 -/
def smallest_n : ℕ := 288

theorem arithmetic_sequence_smallest_negative (n : ℕ) :
  n ≥ smallest_n ↔ 2009 - 7 * n < 0 :=
by
  sorry

theorem smallest_n_is_minimal :
  ∀ k : ℕ, k < smallest_n → 2009 - 7 * k ≥ 0 :=
by
  sorry

end arithmetic_sequence_smallest_negative_smallest_n_is_minimal_l557_55728


namespace percentage_increase_l557_55756

theorem percentage_increase (x : ℝ) (h : x = 78.4) : 
  (x - 70) / 70 * 100 = 12 := by
  sorry

end percentage_increase_l557_55756
