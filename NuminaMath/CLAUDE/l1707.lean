import Mathlib

namespace equation_solution_l1707_170799

theorem equation_solution : ∃ (x : ℝ), x > 0 ∧ x = Real.sqrt (x - 1/x) + Real.sqrt (1 - 1/x) ∧ x = (1 + Real.sqrt 5) / 2 := by
  sorry

end equation_solution_l1707_170799


namespace password_probability_l1707_170796

/-- Represents the composition of a password --/
structure Password :=
  (first_letter : Char)
  (middle_digit : Nat)
  (last_letter : Char)

/-- Defines the set of vowels --/
def vowels : Set Char := {'A', 'E', 'I', 'O', 'U'}

/-- Defines the set of even single-digit numbers --/
def even_single_digits : Set Nat := {0, 2, 4, 6, 8}

/-- The total number of letters in the alphabet --/
def alphabet_size : Nat := 26

/-- The number of vowels --/
def vowel_count : Nat := 5

/-- The number of single-digit numbers --/
def single_digit_count : Nat := 10

/-- The number of even single-digit numbers --/
def even_single_digit_count : Nat := 5

/-- Theorem stating the probability of a specific password pattern --/
theorem password_probability :
  (((vowel_count : ℚ) / alphabet_size) *
   ((even_single_digit_count : ℚ) / single_digit_count) *
   ((alphabet_size - vowel_count : ℚ) / alphabet_size)) =
  (105 : ℚ) / 1352 := by
  sorry

end password_probability_l1707_170796


namespace rightmost_three_digits_of_7_to_1994_l1707_170766

theorem rightmost_three_digits_of_7_to_1994 : 7^1994 % 1000 = 49 := by
  sorry

end rightmost_three_digits_of_7_to_1994_l1707_170766


namespace clothes_washer_discount_l1707_170707

theorem clothes_washer_discount (original_price : ℝ) 
  (discount1 discount2 discount3 : ℝ) : 
  original_price = 500 →
  discount1 = 0.1 →
  discount2 = 0.2 →
  discount3 = 0.05 →
  (original_price * (1 - discount1) * (1 - discount2) * (1 - discount3)) / original_price = 0.684 := by
sorry

end clothes_washer_discount_l1707_170707


namespace closest_integer_to_cube_root_l1707_170712

theorem closest_integer_to_cube_root (x : ℝ := (7^3 + 9^3) ^ (1/3)) : 
  ∃ (n : ℤ), ∀ (m : ℤ), |x - n| ≤ |x - m| ∧ n = 10 := by
sorry

end closest_integer_to_cube_root_l1707_170712


namespace jackies_pushup_count_l1707_170713

/-- Calculates the number of push-ups Jackie can do in one minute with breaks -/
def jackies_pushups (pushups_per_ten_seconds : ℕ) (total_time : ℕ) (break_duration : ℕ) (num_breaks : ℕ) : ℕ :=
  let total_break_time := break_duration * num_breaks
  let pushup_time := total_time - total_break_time
  let pushups_per_second := pushups_per_ten_seconds / 10
  pushup_time * pushups_per_second

/-- Proves that Jackie can do 22 push-ups in one minute with two 8-second breaks -/
theorem jackies_pushup_count : jackies_pushups 5 60 8 2 = 22 := by
  sorry

#eval jackies_pushups 5 60 8 2

end jackies_pushup_count_l1707_170713


namespace max_sum_composite_shape_l1707_170784

/-- Represents a composite shape formed by adding a pyramid to a pentagonal prism --/
structure CompositePrismPyramid where
  prism_faces : Nat
  prism_edges : Nat
  prism_vertices : Nat
  pyramid_faces : Nat
  pyramid_edges : Nat
  pyramid_vertices : Nat

/-- The total number of faces in the composite shape --/
def total_faces (shape : CompositePrismPyramid) : Nat :=
  shape.prism_faces + shape.pyramid_faces - 1

/-- The total number of edges in the composite shape --/
def total_edges (shape : CompositePrismPyramid) : Nat :=
  shape.prism_edges + shape.pyramid_edges

/-- The total number of vertices in the composite shape --/
def total_vertices (shape : CompositePrismPyramid) : Nat :=
  shape.prism_vertices + shape.pyramid_vertices

/-- The sum of faces, edges, and vertices in the composite shape --/
def total_sum (shape : CompositePrismPyramid) : Nat :=
  total_faces shape + total_edges shape + total_vertices shape

/-- Theorem stating the maximum sum of faces, edges, and vertices --/
theorem max_sum_composite_shape :
  ∃ (shape : CompositePrismPyramid),
    shape.prism_faces = 7 ∧
    shape.prism_edges = 15 ∧
    shape.prism_vertices = 10 ∧
    shape.pyramid_faces = 5 ∧
    shape.pyramid_edges = 5 ∧
    shape.pyramid_vertices = 1 ∧
    total_sum shape = 42 ∧
    ∀ (other : CompositePrismPyramid), total_sum other ≤ 42 :=
by
  sorry

end max_sum_composite_shape_l1707_170784


namespace special_point_is_zero_l1707_170790

/-- Definition of the polynomial p(x,y) -/
def p (b : Fin 14 → ℝ) (x y : ℝ) : ℝ :=
  b 0 + b 1 * x + b 2 * y + b 3 * x^2 + b 4 * x * y + b 5 * y^2 + 
  b 6 * x^3 + b 7 * x^2 * y + b 8 * x * y^2 + b 9 * y^3 + 
  b 10 * x^4 + b 11 * y^4 + b 12 * x^3 * y^2 + b 13 * y^3 * x^2

/-- The theorem stating that (5/19, 16/19) is a zero of all polynomials p satisfying the given conditions -/
theorem special_point_is_zero (b : Fin 14 → ℝ) : 
  (p b 0 0 = 0 ∧ p b 1 0 = 0 ∧ p b (-1) 0 = 0 ∧ p b 0 1 = 0 ∧ 
   p b 0 (-1) = 0 ∧ p b 1 1 = 0 ∧ p b (-1) (-1) = 0 ∧ 
   p b 2 2 = 0 ∧ p b 2 (-2) = 0 ∧ p b (-2) 2 = 0) → 
  p b (5/19) (16/19) = 0 := by
  sorry

end special_point_is_zero_l1707_170790


namespace expand_and_simplify_l1707_170771

theorem expand_and_simplify (y : ℝ) (h : y ≠ 0) :
  (3 / 4) * (8 / y + 6 * y^3) = 6 / y + (9 * y^3) / 2 := by
  sorry

end expand_and_simplify_l1707_170771


namespace sum_of_cubes_equals_fourth_power_l1707_170704

theorem sum_of_cubes_equals_fourth_power : 5^3 + 5^3 + 5^3 + 5^3 = 5^4 := by
  sorry

end sum_of_cubes_equals_fourth_power_l1707_170704


namespace binomial_coefficient_22_15_l1707_170764

theorem binomial_coefficient_22_15 (h1 : Nat.choose 21 13 = 20349)
                                   (h2 : Nat.choose 21 14 = 11628)
                                   (h3 : Nat.choose 23 15 = 490314) :
  Nat.choose 22 15 = 458337 := by
  sorry

end binomial_coefficient_22_15_l1707_170764


namespace exponent_sum_l1707_170774

theorem exponent_sum (a x y : ℝ) (hx : a^x = 2) (hy : a^y = 3) : a^(x + y) = 6 := by
  sorry

end exponent_sum_l1707_170774


namespace right_triangle_max_area_right_triangle_max_area_achieved_right_triangle_max_area_is_nine_l1707_170714

theorem right_triangle_max_area (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_hypotenuse : c = 6) :
  a * b ≤ 18 := by
  sorry

theorem right_triangle_max_area_achieved (a b : ℝ) (h_right : a^2 + b^2 = 36) (h_equal : a = b) :
  a * b = 18 := by
  sorry

theorem right_triangle_max_area_is_nine :
  ∃ (a b : ℝ), a^2 + b^2 = 36 ∧ a * b / 2 = 9 := by
  sorry

end right_triangle_max_area_right_triangle_max_area_achieved_right_triangle_max_area_is_nine_l1707_170714


namespace cubic_roots_expression_l1707_170748

theorem cubic_roots_expression (α β γ : ℂ) : 
  (α^3 - 3*α - 2 = 0) → 
  (β^3 - 3*β - 2 = 0) → 
  (γ^3 - 3*γ - 2 = 0) → 
  α*(β - γ)^2 + β*(γ - α)^2 + γ*(α - β)^2 = -18 := by
sorry

end cubic_roots_expression_l1707_170748


namespace inequality_proof_l1707_170788

theorem inequality_proof (x m : ℝ) (hx : x ≥ 1) (hm : m ≥ 1/2) :
  x * Real.log x ≤ m * (x^2 - 1) := by
  sorry

end inequality_proof_l1707_170788


namespace b_share_is_1540_l1707_170719

/-- Represents the share of profits for a partner in a partnership. -/
structure PartnerShare where
  investment : ℕ
  share : ℕ

/-- Calculates the share of a partner given the total profit and the investment ratios. -/
def calculateShare (totalProfit : ℕ) (investmentRatios : List ℕ) (partnerRatio : ℕ) : ℕ :=
  (totalProfit * partnerRatio) / (investmentRatios.sum)

/-- Theorem stating that given the investments and a's share, b's share is $1540. -/
theorem b_share_is_1540 (a b c : PartnerShare) 
  (h1 : a.investment = 15000)
  (h2 : b.investment = 21000)
  (h3 : c.investment = 27000)
  (h4 : a.share = 1100) : 
  b.share = 1540 := by
  sorry


end b_share_is_1540_l1707_170719


namespace sum_of_cubes_equals_hundred_l1707_170756

theorem sum_of_cubes_equals_hundred : (1 : ℕ)^3 + 2^3 + 3^3 + 4^3 = 10^2 := by
  sorry

end sum_of_cubes_equals_hundred_l1707_170756


namespace modular_inverse_89_mod_90_l1707_170730

theorem modular_inverse_89_mod_90 : ∃ x : ℕ, 0 ≤ x ∧ x < 90 ∧ (89 * x) % 90 = 1 :=
by
  sorry

end modular_inverse_89_mod_90_l1707_170730


namespace sequence_equality_l1707_170718

def A : ℕ → ℚ
  | 0 => 1
  | n + 1 => (A n + 2) / (A n + 1)

def B : ℕ → ℚ
  | 0 => 1
  | n + 1 => (B n ^ 2 + 2) / (2 * B n)

theorem sequence_equality (n : ℕ) : B (n + 1) = A (2 ^ n) := by
  sorry

end sequence_equality_l1707_170718


namespace negation_of_universal_proposition_l1707_170753

theorem negation_of_universal_proposition :
  (¬ ∀ (m : ℝ) (x : ℝ), m ∈ Set.Icc 0 1 → x + 1/x ≥ 2^m) ↔
  (∃ (m : ℝ) (x : ℝ), m ∈ Set.Icc 0 1 ∧ x + 1/x < 2^m) := by sorry

end negation_of_universal_proposition_l1707_170753


namespace complex_number_in_quadrant_III_l1707_170758

def complex_number : ℂ := (-2 + Complex.I) * Complex.I^5

theorem complex_number_in_quadrant_III : 
  Real.sign (complex_number.re) = -1 ∧ Real.sign (complex_number.im) = -1 :=
sorry

end complex_number_in_quadrant_III_l1707_170758


namespace count_box_triples_l1707_170759

/-- The number of ordered triples (a, b, c) satisfying the box conditions -/
def box_triples : ℕ := 3

/-- Predicate defining the conditions for a valid box triple -/
def is_valid_box_triple (a b c : ℕ) : Prop :=
  2 ≤ a ∧ a ≤ b ∧ b ≤ c ∧ (a * b * c : ℚ) = (2 / 3) * (a * b + b * c + c * a)

/-- Theorem stating that there are exactly 3 ordered triples satisfying the box conditions -/
theorem count_box_triples :
  (∃ (S : Finset (ℕ × ℕ × ℕ)), S.card = box_triples ∧
    (∀ (t : ℕ × ℕ × ℕ), t ∈ S ↔ is_valid_box_triple t.1 t.2.1 t.2.2)) :=
sorry

end count_box_triples_l1707_170759


namespace inequality_proof_l1707_170785

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end inequality_proof_l1707_170785


namespace heptagon_diagonals_l1707_170727

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A heptagon is a polygon with 7 sides --/
def is_heptagon (n : ℕ) : Prop := n = 7

theorem heptagon_diagonals :
  ∀ n : ℕ, is_heptagon n → num_diagonals n = 14 := by
  sorry

end heptagon_diagonals_l1707_170727


namespace journey_average_speed_l1707_170717

/-- Prove that the average speed of a journey with four equal-length segments,
    traveled at speeds of 3, 2, 6, and 3 km/h respectively, is 3 km/h. -/
theorem journey_average_speed (x : ℝ) (hx : x > 0) : 
  let total_distance := 4 * x
  let total_time := x / 3 + x / 2 + x / 6 + x / 3
  total_distance / total_time = 3 := by
  sorry

end journey_average_speed_l1707_170717


namespace sum_equals_200_l1707_170767

theorem sum_equals_200 : 139 + 27 + 23 + 11 = 200 := by
  sorry

end sum_equals_200_l1707_170767


namespace trader_pens_sold_l1707_170794

/-- Calculates the number of pens sold given the gain and gain percentage -/
def pens_sold (gain_in_pens : ℕ) (gain_percentage : ℕ) : ℕ :=
  (gain_in_pens * 100) / gain_percentage

theorem trader_pens_sold : pens_sold 40 40 = 100 := by
  sorry

end trader_pens_sold_l1707_170794


namespace seven_people_round_table_l1707_170775

/-- The number of distinct arrangements of n people around a round table. -/
def roundTableArrangements (n : ℕ) : ℕ := (n - 1).factorial

/-- Theorem: There are 720 distinct ways to arrange 7 people around a round table. -/
theorem seven_people_round_table : roundTableArrangements 7 = 720 := by
  sorry

end seven_people_round_table_l1707_170775


namespace quadI_area_less_than_quadII_area_l1707_170752

/-- Calculates the area of a quadrilateral given its vertices -/
def quadrilateralArea (v1 v2 v3 v4 : ℝ × ℝ) : ℝ := sorry

/-- Quadrilateral I with vertices (0,0), (2,0), (2,2), and (0,1) -/
def quadI : List (ℝ × ℝ) := [(0,0), (2,0), (2,2), (0,1)]

/-- Quadrilateral II with vertices (0,0), (3,0), (3,1), and (0,2) -/
def quadII : List (ℝ × ℝ) := [(0,0), (3,0), (3,1), (0,2)]

theorem quadI_area_less_than_quadII_area :
  quadrilateralArea quadI.head! quadI.tail!.head! quadI.tail!.tail!.head! quadI.tail!.tail!.tail!.head! <
  quadrilateralArea quadII.head! quadII.tail!.head! quadII.tail!.tail!.head! quadII.tail!.tail!.tail!.head! :=
by sorry

end quadI_area_less_than_quadII_area_l1707_170752


namespace expand_x_plus_y_seventh_third_to_fourth_term_ratio_p_plus_q_equals_three_p_and_q_positive_prove_p_value_l1707_170726

/-- The value of p in the expansion of (x+y)^7 -/
def p : ℚ :=
  30/13

/-- The value of q in the expansion of (x+y)^7 -/
def q : ℚ :=
  9/13

/-- The ratio of the third to fourth term in the expansion of (x+y)^7 when x=p and y=q -/
def ratio : ℚ :=
  2/1

theorem expand_x_plus_y_seventh (x y : ℚ) :
  (x + y)^7 = x^7 + 7*x^6*y + 21*x^5*y^2 + 35*x^4*y^3 + 35*x^3*y^4 + 21*x^2*y^5 + 7*x*y^6 + y^7 :=
sorry

theorem third_to_fourth_term_ratio :
  (21 * p^5 * q^2) / (35 * p^4 * q^3) = ratio :=
sorry

theorem p_plus_q_equals_three :
  p + q = 3 :=
sorry

theorem p_and_q_positive :
  p > 0 ∧ q > 0 :=
sorry

theorem prove_p_value :
  p = 30/13 :=
sorry

end expand_x_plus_y_seventh_third_to_fourth_term_ratio_p_plus_q_equals_three_p_and_q_positive_prove_p_value_l1707_170726


namespace weekday_hours_are_six_l1707_170783

/-- Represents the daily weekday operation hours of Jean's business -/
def weekday_hours : ℝ := sorry

/-- The number of weekdays the business operates -/
def weekdays : ℕ := 5

/-- The number of hours the business operates each day on weekends -/
def weekend_daily_hours : ℕ := 4

/-- The number of weekend days -/
def weekend_days : ℕ := 2

/-- The total weekly operation hours -/
def total_weekly_hours : ℕ := 38

/-- Theorem stating that the daily weekday operation hours are 6 -/
theorem weekday_hours_are_six : weekday_hours = 6 := by sorry

end weekday_hours_are_six_l1707_170783


namespace bowling_ball_surface_area_l1707_170757

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem bowling_ball_surface_area :
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 81 * Real.pi := by sorry

end bowling_ball_surface_area_l1707_170757


namespace connie_calculation_l1707_170791

theorem connie_calculation (x : ℝ) : 200 - x = 100 → 200 + x = 300 := by
  sorry

end connie_calculation_l1707_170791


namespace fred_initial_balloons_l1707_170740

/-- The number of green balloons Fred gave to Sandy -/
def balloons_given : ℕ := 221

/-- The number of green balloons Fred has left -/
def balloons_left : ℕ := 488

/-- The initial number of green balloons Fred had -/
def initial_balloons : ℕ := balloons_given + balloons_left

theorem fred_initial_balloons : initial_balloons = 709 := by
  sorry

end fred_initial_balloons_l1707_170740


namespace truck_to_car_ratio_l1707_170728

/-- The number of people needed to lift a car -/
def people_per_car : ℕ := 5

/-- The total number of people needed to lift 6 cars and 3 trucks -/
def total_people : ℕ := 60

/-- The number of cars that can be lifted -/
def num_cars : ℕ := 6

/-- The number of trucks that can be lifted -/
def num_trucks : ℕ := 3

/-- The number of people needed to lift a truck -/
def people_per_truck : ℕ := (total_people - num_cars * people_per_car) / num_trucks

theorem truck_to_car_ratio :
  (people_per_truck : ℚ) / people_per_car = 2 / 1 := by sorry

end truck_to_car_ratio_l1707_170728


namespace regular_star_polygon_points_l1707_170716

/-- A regular star polygon with n points -/
structure RegularStarPolygon where
  n : ℕ
  edges : Fin (2 * n) → ℝ
  angles_A : Fin n → ℝ
  angles_B : Fin n → ℝ
  edges_equal : ∀ i j, edges i = edges j
  angles_A_equal : ∀ i j, angles_A i = angles_A j
  angles_B_equal : ∀ i j, angles_B i = angles_B j
  angle_difference : ∀ i, angles_B i - angles_A i = 15

/-- The theorem stating that for a regular star polygon with the given conditions, n must be 24 -/
theorem regular_star_polygon_points (star : RegularStarPolygon) :
  (∀ i, star.angles_B i - star.angles_A i = 15) → star.n = 24 :=
by sorry

end regular_star_polygon_points_l1707_170716


namespace smallest_part_of_proportional_division_l1707_170762

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 90 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ (x : ℚ), x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 18 :=
by sorry

end smallest_part_of_proportional_division_l1707_170762


namespace triangle_side_ratio_bounds_l1707_170749

theorem triangle_side_ratio_bounds (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_geom_seq : b^2 = a*c) :
  2 ≤ (b/a + a/b) ∧ (b/a + a/b) < Real.sqrt 5 := by
  sorry

end triangle_side_ratio_bounds_l1707_170749


namespace range_of_z_l1707_170715

theorem range_of_z (x y : ℝ) (h : x^2 + y^2 = 1) :
  ∃ (z_min z_max : ℝ), z_min = -4/3 ∧ z_max = 0 ∧
  ∀ z, z = (y - 1) / (x + 2) → z_min ≤ z ∧ z ≤ z_max :=
sorry

end range_of_z_l1707_170715


namespace inscribed_square_area_l1707_170760

theorem inscribed_square_area (x y : ℝ) :
  (∃ s : ℝ, s > 0 ∧ x^2 / 4 + y^2 / 8 = 1 ∧ x = s ∧ y = s) →
  (4 * s^2 = 32 / 3) := by
  sorry

end inscribed_square_area_l1707_170760


namespace modified_bowling_tournament_distributions_l1707_170786

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Represents the number of matches in the tournament -/
def num_matches : ℕ := 5

/-- Theorem: The number of different prize distributions in the modified bowling tournament -/
theorem modified_bowling_tournament_distributions :
  (outcomes_per_match ^ num_matches : ℕ) = 32 :=
sorry

end modified_bowling_tournament_distributions_l1707_170786


namespace ratio_p_to_q_l1707_170706

theorem ratio_p_to_q (p q : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) 
  (h3 : (p + q) / (p - q) = 4 / 3) : p / q = 7 := by
  sorry

end ratio_p_to_q_l1707_170706


namespace johns_initial_speed_johns_initial_speed_proof_l1707_170773

theorem johns_initial_speed 
  (initial_time : ℝ) 
  (time_increase_percent : ℝ) 
  (speed_increase : ℝ) 
  (final_distance : ℝ) : ℝ :=
  let final_time := initial_time * (1 + time_increase_percent / 100)
  let initial_speed := (final_distance / final_time) - speed_increase
  initial_speed

#check johns_initial_speed 8 75 4 168 = 8

theorem johns_initial_speed_proof 
  (initial_time : ℝ) 
  (time_increase_percent : ℝ) 
  (speed_increase : ℝ) 
  (final_distance : ℝ) :
  johns_initial_speed initial_time time_increase_percent speed_increase final_distance = 8 :=
by sorry

end johns_initial_speed_johns_initial_speed_proof_l1707_170773


namespace fixed_point_on_graph_l1707_170746

theorem fixed_point_on_graph (m : ℝ) : 
  let f : ℝ → ℝ := fun x ↦ 9 * x^2 + m * x - 5 * m
  f 5 = 225 := by
  sorry

end fixed_point_on_graph_l1707_170746


namespace winnie_repetitions_l1707_170744

/-- Calculates the number of repetitions completed today given yesterday's
    repetitions and the difference in performance. -/
def repetitions_today (yesterday : ℕ) (difference : ℕ) : ℕ :=
  yesterday - difference

/-- Proves that Winnie completed 73 repetitions today given the conditions. -/
theorem winnie_repetitions :
  repetitions_today 86 13 = 73 := by
  sorry

end winnie_repetitions_l1707_170744


namespace lee_fruit_loading_l1707_170702

/-- Represents the capacity of a large truck in tons -/
def large_truck_capacity : ℕ := 15

/-- Represents the number of large trucks used -/
def num_large_trucks : ℕ := 8

/-- Represents the total amount of fruits to be loaded in tons -/
def total_fruits : ℕ := num_large_trucks * large_truck_capacity

theorem lee_fruit_loading :
  total_fruits = 120 :=
by sorry

end lee_fruit_loading_l1707_170702


namespace polynomial_divisibility_l1707_170700

theorem polynomial_divisibility (m n : ℕ+) :
  ∃ q : Polynomial ℚ, (X^2 + X + 1) * q = X^(3*m.val + 1) + X^(3*n.val + 2) + 1 := by
  sorry

end polynomial_divisibility_l1707_170700


namespace sin_2017pi_over_3_l1707_170703

theorem sin_2017pi_over_3 : Real.sin (2017 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_2017pi_over_3_l1707_170703


namespace race_outcomes_l1707_170708

/-- The number of permutations of n items taken k at a time -/
def permutations (n k : ℕ) : ℕ := 
  if k ≤ n then Nat.factorial n / Nat.factorial (n - k) else 0

/-- The number of participants in the race -/
def num_participants : ℕ := 6

/-- The number of positions to be filled -/
def positions_to_fill : ℕ := 4

theorem race_outcomes : permutations num_participants positions_to_fill = 360 := by
  sorry

end race_outcomes_l1707_170708


namespace isosceles_trapezoid_base_ratio_l1707_170743

structure IsoscelesTrapezoid where
  smaller_base : ℝ
  larger_base : ℝ
  diagonal : ℝ
  altitude : ℝ
  is_isosceles : True
  smaller_base_half_diagonal : smaller_base = diagonal / 2
  altitude_half_larger_base : altitude = larger_base / 2

theorem isosceles_trapezoid_base_ratio 
  (t : IsoscelesTrapezoid) : t.smaller_base / t.larger_base = 3 / 8 := by
  sorry

end isosceles_trapezoid_base_ratio_l1707_170743


namespace quadratic_equation_conversion_l1707_170729

theorem quadratic_equation_conversion :
  ∀ x : ℝ, (x - 8)^2 = 5 ↔ x^2 - 16*x + 59 = 0 :=
by
  sorry

end quadratic_equation_conversion_l1707_170729


namespace equation_solution_l1707_170778

theorem equation_solution : ∃ x : ℚ, (5 * x + 9 * x = 570 - 12 * (x - 5)) ∧ (x = 315 / 13) := by
  sorry

end equation_solution_l1707_170778


namespace no_inscribed_parallelepiped_l1707_170769

theorem no_inscribed_parallelepiped (π : ℝ) (h_π : π = Real.pi) :
  ¬ ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
    x * y * z = 2 * π / 3 ∧
    x * y + y * z + z * x = π ∧
    x^2 + y^2 + z^2 = 4 := by
  sorry

end no_inscribed_parallelepiped_l1707_170769


namespace max_perimeter_special_triangle_l1707_170739

theorem max_perimeter_special_triangle :
  ∀ x : ℕ,
    x > 0 →
    x ≤ 20 →
    x + 4*x > 20 →
    x + 20 > 4*x →
    4*x + 20 > x →
    (∀ y : ℕ, 
      y > 0 →
      y ≤ 20 →
      y + 4*y > 20 →
      y + 20 > 4*y →
      4*y + 20 > y →
      x + 4*x + 20 ≥ y + 4*y + 20) →
    x + 4*x + 20 = 50 :=
by sorry

end max_perimeter_special_triangle_l1707_170739


namespace problem_statements_l1707_170737

theorem problem_statements :
  (∀ (x : ℝ), x ≥ 3 → 2*x - 10 ≥ 0) ↔ ¬(∃ (x : ℝ), x ≥ 3 ∧ 2*x - 10 < 0) ∧
  (∀ (a b c : ℝ), c > a ∧ a > b ∧ b > 0 → a / (c - a) > b / (c - b)) ∧
  (∀ (a b m : ℝ), a > b ∧ b > 0 ∧ m > 0 → a / b > (a + m) / (b + m)) :=
by sorry

end problem_statements_l1707_170737


namespace tetrahedron_volume_ratio_l1707_170738

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A tetrahedron defined by four points -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Check if a point is inside a triangle -/
def isInside (p : Point3D) (t : Tetrahedron) : Prop :=
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + b + c = 1 ∧
    p = Point3D.mk (a * t.A.x + b * t.B.x + c * t.C.x)
                   (a * t.A.y + b * t.B.y + c * t.C.y)
                   (a * t.A.z + b * t.B.z + c * t.C.z)

/-- Calculate the volume of a tetrahedron -/
noncomputable def volume (t : Tetrahedron) : ℝ := sorry

/-- Find the intersection point of a line parallel to DD₁ passing through a vertex -/
noncomputable def intersectionPoint (t : Tetrahedron) (D₁ : Point3D) (vertex : Point3D) : Point3D := sorry

/-- The main theorem -/
theorem tetrahedron_volume_ratio (t : Tetrahedron) (D₁ : Point3D) :
  isInside D₁ t →
  let A₁ := intersectionPoint t D₁ t.A
  let B₁ := intersectionPoint t D₁ t.B
  let C₁ := intersectionPoint t D₁ t.C
  let t₁ := Tetrahedron.mk A₁ B₁ C₁ D₁
  volume t = (1/3) * volume t₁ := by sorry

end tetrahedron_volume_ratio_l1707_170738


namespace simplify_fraction_l1707_170795

theorem simplify_fraction : (180 : ℚ) / 270 = 2 / 3 := by sorry

end simplify_fraction_l1707_170795


namespace square_area_from_adjacent_points_l1707_170763

/-- Given two adjacent points (1,2) and (2,5) on a square in a Cartesian coordinate plane,
    the area of the square is 10. -/
theorem square_area_from_adjacent_points :
  let p1 : ℝ × ℝ := (1, 2)
  let p2 : ℝ × ℝ := (2, 5)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 10 := by sorry

end square_area_from_adjacent_points_l1707_170763


namespace inequality_solution_set_l1707_170736

-- Define the inequality
def inequality (x : ℝ) : Prop := 4 * x - 5 < 3

-- Define the solution set
def solution_set : Set ℝ := {x | x < 2}

-- Theorem statement
theorem inequality_solution_set : 
  ∀ x : ℝ, inequality x ↔ x ∈ solution_set := by sorry

end inequality_solution_set_l1707_170736


namespace smallest_multiple_of_11_23_37_l1707_170745

theorem smallest_multiple_of_11_23_37 : ∃ (n : ℕ), n > 0 ∧ 11 ∣ n ∧ 23 ∣ n ∧ 37 ∣ n ∧ ∀ (m : ℕ), (m > 0 ∧ 11 ∣ m ∧ 23 ∣ m ∧ 37 ∣ m) → n ≤ m := by
  sorry

end smallest_multiple_of_11_23_37_l1707_170745


namespace music_club_ratio_l1707_170777

theorem music_club_ratio :
  ∀ (total girls boys : ℕ) (p_girl p_boy : ℝ),
    total = girls + boys →
    total > 0 →
    p_girl + p_boy = 1 →
    p_girl = (3 / 5 : ℝ) * p_boy →
    (girls : ℝ) / total = 3 / 8 := by
  sorry

end music_club_ratio_l1707_170777


namespace reunion_attendance_overlap_l1707_170755

theorem reunion_attendance_overlap (total_guests : ℕ) (oates_attendees : ℕ) (hall_attendees : ℕ) (brown_attendees : ℕ)
  (h_total : total_guests = 200)
  (h_oates : oates_attendees = 60)
  (h_hall : hall_attendees = 90)
  (h_brown : brown_attendees = 80)
  (h_all_attend : total_guests ≤ oates_attendees + hall_attendees + brown_attendees) :
  let min_overlap := oates_attendees + hall_attendees + brown_attendees - total_guests
  let max_overlap := min oates_attendees (min hall_attendees brown_attendees)
  (min_overlap = 30 ∧ max_overlap = 60) :=
by sorry

end reunion_attendance_overlap_l1707_170755


namespace triangle_problem_l1707_170779

theorem triangle_problem (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π →
  -- Sides a, b, c are opposite to angles A, B, C respectively
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Given equation
  (2 * c - b) / (Real.sqrt 3 * Real.sin C - Real.cos C) = a →
  -- b = 1
  b = 1 →
  -- Area condition
  (1 / 2) * b * c * Real.sin A = (3 / 4) * Real.tan A →
  -- Prove A = π/3 and a = √7
  A = π / 3 ∧ a = Real.sqrt 7 := by
sorry

end triangle_problem_l1707_170779


namespace exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary_l1707_170772

/-- Represents the color of a ball -/
inductive BallColor
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The set of all possible outcomes when drawing two balls -/
def sampleSpace : Set DrawOutcome := sorry

/-- The event of drawing exactly one black ball -/
def exactlyOneBlack : Set DrawOutcome := sorry

/-- The event of drawing exactly two black balls -/
def exactlyTwoBlack : Set DrawOutcome := sorry

/-- Definition of mutually exclusive events -/
def mutuallyExclusive (A B : Set DrawOutcome) : Prop :=
  A ∩ B = ∅

/-- Definition of complementary events -/
def complementary (A B : Set DrawOutcome) : Prop :=
  A ∪ B = sampleSpace ∧ A ∩ B = ∅

/-- Main theorem: exactlyOneBlack and exactlyTwoBlack are mutually exclusive but not complementary -/
theorem exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary :
  mutuallyExclusive exactlyOneBlack exactlyTwoBlack ∧
  ¬complementary exactlyOneBlack exactlyTwoBlack := by
  sorry

end exactlyOneBlack_exactlyTwoBlack_mutually_exclusive_not_complementary_l1707_170772


namespace first_nonzero_digit_of_one_over_129_l1707_170734

theorem first_nonzero_digit_of_one_over_129 :
  ∃ (n : ℕ) (r : ℚ), (1 : ℚ) / 129 = (n : ℚ) / 10^(n+1) + r ∧ 0 ≤ r ∧ r < 1 / 10^(n+1) ∧ n = 7 :=
sorry

end first_nonzero_digit_of_one_over_129_l1707_170734


namespace clock_tower_rings_per_year_l1707_170741

/-- The number of times a clock tower bell rings in a year -/
def bell_rings_per_year (rings_per_hour : ℕ) (hours_per_day : ℕ) (days_per_year : ℕ) : ℕ :=
  rings_per_hour * hours_per_day * days_per_year

/-- Theorem: The clock tower bell rings 8760 times in a year -/
theorem clock_tower_rings_per_year :
  bell_rings_per_year 1 24 365 = 8760 := by
  sorry

end clock_tower_rings_per_year_l1707_170741


namespace set_operation_result_l1707_170792

open Set

def U : Set ℕ := {0, 1, 2, 3, 4, 5}
def M : Set ℕ := {0, 3, 5}
def N : Set ℕ := {1, 4, 5}

theorem set_operation_result :
  (M ∩ N) ∪ (U \ N) = {0, 1, 3, 4, 5} := by sorry

end set_operation_result_l1707_170792


namespace some_number_value_l1707_170735

theorem some_number_value (a : ℕ) (some_number : ℕ) 
  (h1 : a = 105)
  (h2 : a^3 = 21 * 25 * some_number * 7) :
  some_number = 105 := by
  sorry

end some_number_value_l1707_170735


namespace triangle_perimeter_32_l1707_170747

/-- Given a triangle ABC with vertices A(-3, 5), B(3, -3), and M(6, 1) as the midpoint of BC,
    prove that the perimeter of the triangle is 32. -/
theorem triangle_perimeter_32 :
  let A : ℝ × ℝ := (-3, 5)
  let B : ℝ × ℝ := (3, -3)
  let M : ℝ × ℝ := (6, 1)
  let C : ℝ × ℝ := (2 * M.1 - B.1, 2 * M.2 - B.2)  -- Derived from midpoint formula
  let AB : ℝ := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)
  let BC : ℝ := Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2)
  let AC : ℝ := Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2)
  AB + BC + AC = 32 := by
  sorry

end triangle_perimeter_32_l1707_170747


namespace sqrt_6_irrational_l1707_170710

/-- A number is rational if it can be expressed as a ratio of two integers -/
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

/-- A number is irrational if it is not rational -/
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

/-- √6 is irrational -/
theorem sqrt_6_irrational : IsIrrational (Real.sqrt 6) := by sorry

end sqrt_6_irrational_l1707_170710


namespace theatre_distance_is_340_l1707_170720

/-- Represents the problem of Julia's drive to the theatre. -/
structure JuliaDrive where
  initial_speed : ℝ
  speed_increase : ℝ
  initial_time : ℝ
  late_time : ℝ
  early_time : ℝ

/-- Calculates the total distance to the theatre based on the given conditions. -/
def calculate_distance (drive : JuliaDrive) : ℝ :=
  let total_time := drive.initial_time + (drive.late_time + drive.early_time)
  let remaining_time := total_time - drive.initial_time
  let remaining_distance := (drive.initial_speed + drive.speed_increase) * remaining_time
  drive.initial_speed * drive.initial_time + remaining_distance

/-- Theorem stating that the distance to the theatre is 340 miles. -/
theorem theatre_distance_is_340 (drive : JuliaDrive)
  (h1 : drive.initial_speed = 40)
  (h2 : drive.speed_increase = 20)
  (h3 : drive.initial_time = 1)
  (h4 : drive.late_time = 1.5)
  (h5 : drive.early_time = 1) :
  calculate_distance drive = 340 := by
  sorry

end theatre_distance_is_340_l1707_170720


namespace three_cards_different_suits_l1707_170776

/-- The number of suits in a standard deck of cards -/
def num_suits : ℕ := 4

/-- The number of cards in each suit -/
def cards_per_suit : ℕ := 13

/-- The number of cards to choose -/
def cards_to_choose : ℕ := 3

/-- The total number of ways to choose 3 cards from a standard deck of 52 cards,
    where all three cards are of different suits and the order doesn't matter -/
def ways_to_choose : ℕ := num_suits.choose cards_to_choose * cards_per_suit ^ cards_to_choose

theorem three_cards_different_suits :
  ways_to_choose = 8788 := by sorry

end three_cards_different_suits_l1707_170776


namespace pencils_misplaced_l1707_170701

theorem pencils_misplaced (initial : ℕ) (broken found bought final : ℕ) : 
  initial = 20 →
  broken = 3 →
  found = 4 →
  bought = 2 →
  final = 16 →
  initial - broken + found + bought - final = 7 := by
  sorry

end pencils_misplaced_l1707_170701


namespace tammy_haircuts_needed_l1707_170722

/-- Represents the haircut system for Tammy -/
structure HaircutSystem where
  total_haircuts : ℕ
  free_haircuts : ℕ
  haircuts_until_next_free : ℕ

/-- Calculates the number of haircuts needed for the next free one -/
def haircuts_needed (system : HaircutSystem) : ℕ :=
  system.haircuts_until_next_free

/-- Theorem stating that Tammy needs 5 more haircuts for her next free one -/
theorem tammy_haircuts_needed (system : HaircutSystem) 
  (h1 : system.total_haircuts = 79)
  (h2 : system.free_haircuts = 5)
  (h3 : system.haircuts_until_next_free = 5) :
  haircuts_needed system = 5 := by
  sorry

#eval haircuts_needed { total_haircuts := 79, free_haircuts := 5, haircuts_until_next_free := 5 }

end tammy_haircuts_needed_l1707_170722


namespace system_solution_l1707_170723

theorem system_solution (x y k : ℝ) : 
  4 * x + 3 * y = 1 → 
  k * x + (k - 1) * y = 3 → 
  x = y → 
  k = 11 := by
sorry

end system_solution_l1707_170723


namespace triangle_angle_calculation_l1707_170754

theorem triangle_angle_calculation (y : ℝ) : 
  y > 0 ∧ y < 60 ∧ 45 + 3 * y + y = 180 → y = 33.75 := by
  sorry

end triangle_angle_calculation_l1707_170754


namespace smallest_k_inequality_k_is_smallest_l1707_170711

theorem smallest_k_inequality (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) :
  (x * y) ^ (1/3 : ℝ) + (3/8 : ℝ) * (x - y)^2 ≥ (3/8 : ℝ) * (x + y) :=
sorry

theorem k_is_smallest :
  ∀ k : ℝ, k > 0 → k < 3/8 →
  ∃ x y : ℝ, x ≥ 0 ∧ y ≥ 0 ∧ (x * y) ^ (1/3 : ℝ) + k * (x - y)^2 < (3/8 : ℝ) * (x + y) :=
sorry

end smallest_k_inequality_k_is_smallest_l1707_170711


namespace quadruplet_equation_equivalence_l1707_170782

theorem quadruplet_equation_equivalence (x y z w : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hw : w ≠ 0) :
  (1 + 1/x + 2*(x+1)/(x*y) + 3*(x+1)*(y+2)/(x*y*z) + 4*(x+1)*(y+2)*(z+3)/(x*y*z*w) = 0) ↔
  ((x+1)*(y+2)*(z+3)*(w+4) = 0) := by
sorry

end quadruplet_equation_equivalence_l1707_170782


namespace pride_and_prejudice_watching_time_l1707_170724

/-- Calculates the total hours spent watching a TV series given the number of episodes and minutes per episode -/
def total_watching_hours (num_episodes : ℕ) (minutes_per_episode : ℕ) : ℚ :=
  (num_episodes * minutes_per_episode : ℚ) / 60

/-- Proves that watching 6 episodes of 50 minutes each takes 5 hours -/
theorem pride_and_prejudice_watching_time :
  total_watching_hours 6 50 = 5 := by
  sorry

#eval total_watching_hours 6 50

end pride_and_prejudice_watching_time_l1707_170724


namespace unique_solution_for_b_l1707_170768

def base_75_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (75 ^ i)) 0

theorem unique_solution_for_b : ∃! b : ℕ, 
  0 ≤ b ∧ b ≤ 19 ∧ 
  (base_75_to_decimal [9, 2, 4, 6, 1, 8, 7, 2, 5] - b) % 17 = 0 ∧
  b = 8 := by sorry

end unique_solution_for_b_l1707_170768


namespace product_of_roots_l1707_170705

theorem product_of_roots (x : ℂ) : 
  (x^3 - 15*x^2 + 75*x - 50 = 0) → 
  (∃ p q r : ℂ, x^3 - 15*x^2 + 75*x - 50 = (x - p) * (x - q) * (x - r) ∧ p * q * r = 50) :=
by sorry

end product_of_roots_l1707_170705


namespace brothers_age_difference_l1707_170787

theorem brothers_age_difference (michael_age younger_brother_age older_brother_age : ℕ) : 
  younger_brother_age = 5 →
  older_brother_age = 3 * younger_brother_age →
  michael_age + older_brother_age + younger_brother_age = 28 →
  older_brother_age - 2 * (michael_age - 1) = 1 := by
  sorry

end brothers_age_difference_l1707_170787


namespace f_composition_of_i_l1707_170750

noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then 2 * z^2 + 1 else -z^2 - 1

theorem f_composition_of_i : f (f (f (f Complex.I))) = -26 := by sorry

end f_composition_of_i_l1707_170750


namespace jackson_vacuum_count_l1707_170798

def chore_pay_rate : ℝ := 5
def vacuum_time : ℝ := 2
def dish_washing_time : ℝ := 0.5
def total_earnings : ℝ := 30

def total_chore_time (vacuum_count : ℝ) : ℝ :=
  vacuum_count * vacuum_time + dish_washing_time + 3 * dish_washing_time

theorem jackson_vacuum_count :
  ∃ (vacuum_count : ℝ), 
    total_chore_time vacuum_count * chore_pay_rate = total_earnings ∧ 
    vacuum_count = 2 := by
  sorry

end jackson_vacuum_count_l1707_170798


namespace quadrilateral_area_l1707_170789

/-- Represents a triangle partitioned into three triangles and a quadrilateral -/
structure PartitionedTriangle where
  /-- Area of the first smaller triangle -/
  area1 : ℝ
  /-- Area of the second smaller triangle -/
  area2 : ℝ
  /-- Area of the third smaller triangle -/
  area3 : ℝ
  /-- Area of the quadrilateral -/
  areaQuad : ℝ
  /-- The sum of all areas equals the area of the original triangle -/
  area_sum : area1 + area2 + area3 + areaQuad > 0

/-- The main theorem about the area of the quadrilateral -/
theorem quadrilateral_area (t : PartitionedTriangle) 
  (h1 : t.area1 = 5) (h2 : t.area2 = 9) (h3 : t.area3 = 9) : 
  t.areaQuad = 45 := by
  sorry


end quadrilateral_area_l1707_170789


namespace greatest_x_4a_value_l1707_170761

theorem greatest_x_4a_value : 
  ∀ (x a b c : ℕ), 
    (100 ≤ x) ∧ (x < 1000) →  -- x is a 3-digit integer
    (x = 100*a + 10*b + c) →  -- a, b, c are hundreds, tens, and units digits
    (4*a = 2*b) ∧ (2*b = c) → -- 4a = 2b = c
    (a > 0) →                 -- a > 0
    (∃ (x₁ x₂ : ℕ), (100 ≤ x₁) ∧ (x₁ < 1000) ∧ (100 ≤ x₂) ∧ (x₂ < 1000) ∧ 
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) → y ≤ x₁) ∧ 
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) ∧ (y ≠ x₁) → y ≤ x₂) ∧
      (x₁ - x₂ = 124)) →     -- difference between two greatest values is 124
    (∃ (a_max : ℕ), (100 ≤ 100*a_max + 10*(2*a_max) + 4*a_max) ∧ 
      (100*a_max + 10*(2*a_max) + 4*a_max < 1000) ∧
      (∀ (y : ℕ), (100 ≤ y) ∧ (y < 1000) → y ≤ 100*a_max + 10*(2*a_max) + 4*a_max) ∧
      (4*a_max = 8)) :=
by sorry

end greatest_x_4a_value_l1707_170761


namespace direction_vector_proof_l1707_170797

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A vector in 2D space -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Check if a vector is a direction vector of a line -/
def isDirectionVector (l : Line2D) (v : Vector2D) : Prop :=
  v.x * l.b = -v.y * l.a

/-- The given line 4x - 3y + m = 0 -/
def givenLine : Line2D :=
  { a := 4, b := -3, c := 0 }  -- We set c to 0 as 'm' is arbitrary

/-- The vector (3, 4) -/
def givenVector : Vector2D :=
  { x := 3, y := 4 }

/-- Theorem: (3, 4) is a direction vector of the line 4x - 3y + m = 0 -/
theorem direction_vector_proof : 
  isDirectionVector givenLine givenVector := by
  sorry

end direction_vector_proof_l1707_170797


namespace dark_tile_fraction_for_given_floor_l1707_170781

/-- Represents a tiled floor with a repeating pattern -/
structure TiledFloor :=
  (pattern_size : Nat)
  (dark_tiles_in_quarter : Nat)

/-- Calculates the fraction of dark tiles in the entire floor -/
def dark_tile_fraction (floor : TiledFloor) : Rat :=
  sorry

/-- Theorem stating the fraction of dark tiles in the given floor configuration -/
theorem dark_tile_fraction_for_given_floor :
  let floor := TiledFloor.mk 8 10
  dark_tile_fraction floor = 5 / 16 := by
  sorry

end dark_tile_fraction_for_given_floor_l1707_170781


namespace regular_polygon_sides_l1707_170733

theorem regular_polygon_sides (n : ℕ) (interior_angle : ℝ) : 
  n ≥ 3 → 
  interior_angle = 135 → 
  (n - 2) * 180 = n * interior_angle → 
  n = 8 :=
by
  sorry

end regular_polygon_sides_l1707_170733


namespace perpendicular_lines_from_perpendicular_planes_l1707_170732

/-- Two planes are mutually perpendicular -/
def mutually_perpendicular (α β : Plane) : Prop := sorry

/-- A line is parallel to a plane -/
def line_parallel_plane (m : Line) (α : Plane) : Prop := sorry

/-- A line is perpendicular to a plane -/
def line_perp_plane (n : Line) (β : Plane) : Prop := sorry

/-- Two planes intersect at a line -/
def planes_intersect_at (α β : Plane) (l : Line) : Prop := sorry

/-- A line is perpendicular to another line -/
def line_perp_line (n l : Line) : Prop := sorry

/-- Main theorem -/
theorem perpendicular_lines_from_perpendicular_planes 
  (α β : Plane) (l m n : Line) 
  (h1 : mutually_perpendicular α β)
  (h2 : planes_intersect_at α β l)
  (h3 : line_parallel_plane m α)
  (h4 : line_perp_plane n β) :
  line_perp_line n l := by sorry

end perpendicular_lines_from_perpendicular_planes_l1707_170732


namespace polygon_35_sides_5_restricted_l1707_170751

/-- The number of diagonals in a convex polygon with restricted vertices -/
def diagonals_with_restrictions (n : ℕ) (r : ℕ) : ℕ :=
  let effective_vertices := n - r
  (effective_vertices * (effective_vertices - 3)) / 2

/-- Theorem: A convex polygon with 35 sides and 5 restricted vertices has 405 diagonals -/
theorem polygon_35_sides_5_restricted : diagonals_with_restrictions 35 5 = 405 := by
  sorry

end polygon_35_sides_5_restricted_l1707_170751


namespace min_constant_for_sqrt_inequality_l1707_170742

theorem min_constant_for_sqrt_inequality :
  (∃ (a : ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ∧
  (∀ (a : ℝ), (∀ (x y : ℝ), x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) → a ≥ Real.sqrt 2) :=
by sorry


end min_constant_for_sqrt_inequality_l1707_170742


namespace smallest_possible_d_l1707_170709

theorem smallest_possible_d : 
  let f : ℝ → ℝ := λ d => (5 * Real.sqrt 3)^2 + (2 * d + 6)^2 - (4 * d)^2
  ∃ d : ℝ, f d = 0 ∧ ∀ d' : ℝ, f d' = 0 → d ≤ d' ∧ d = 1 + Real.sqrt 41 / 2 :=
by sorry

end smallest_possible_d_l1707_170709


namespace complex_number_opposite_parts_l1707_170770

theorem complex_number_opposite_parts (a : ℝ) : 
  (∃ z : ℂ, z = (2 + a * Complex.I) * Complex.I ∧ 
   z.re = -z.im) → a = 2 := by
sorry

end complex_number_opposite_parts_l1707_170770


namespace apple_pie_calculation_l1707_170731

theorem apple_pie_calculation (total_apples : ℕ) (unripe_apples : ℕ) (apples_per_pie : ℕ) :
  total_apples = 34 →
  unripe_apples = 6 →
  apples_per_pie = 4 →
  (total_apples - unripe_apples) / apples_per_pie = 7 :=
by sorry

end apple_pie_calculation_l1707_170731


namespace angle_three_times_complement_l1707_170721

theorem angle_three_times_complement (x : ℝ) : 
  (x = 3 * (180 - x)) → x = 135 := by
  sorry

end angle_three_times_complement_l1707_170721


namespace prob_6_to_7_l1707_170765

-- Define a normally distributed random variable
def X : Real → Real := sorry

-- Define the probability density function for X
def pdf (x : Real) : Real := sorry

-- Define the cumulative distribution function for X
def cdf (x : Real) : Real := sorry

-- Given probabilities
axiom prob_1sigma : (cdf 6 - cdf 4) = 0.6826
axiom prob_2sigma : (cdf 7 - cdf 3) = 0.9544
axiom prob_3sigma : (cdf 8 - cdf 2) = 0.9974

-- The statement to prove
theorem prob_6_to_7 : (cdf 7 - cdf 6) = 0.1359 := by sorry

end prob_6_to_7_l1707_170765


namespace square_of_one_plus_i_l1707_170725

theorem square_of_one_plus_i :
  let z : ℂ := 1 + Complex.I
  z^2 = 2 * Complex.I := by sorry

end square_of_one_plus_i_l1707_170725


namespace arithmetic_sequence_sum_l1707_170793

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end arithmetic_sequence_sum_l1707_170793


namespace b_range_l1707_170780

/-- A cubic function with a parameter b -/
def f (b : ℝ) (x : ℝ) : ℝ := -x^3 + b*x

/-- Theorem stating the range of b given the conditions -/
theorem b_range :
  ∀ b : ℝ,
  (∀ x : ℝ, f b x = 0 → x ∈ Set.Icc (-2) 2) →
  (∀ x y : ℝ, x ∈ Set.Ioo 0 1 → y ∈ Set.Ioo 0 1 → x < y → f b x < f b y) →
  b ∈ Set.Icc 3 4 := by
sorry

end b_range_l1707_170780
