import Mathlib

namespace min_value_of_expression_l935_93515

theorem min_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1) : x^2 + (1 / 4) * y^2 ≥ 1 / 8 :=
sorry

end min_value_of_expression_l935_93515


namespace sin_half_alpha_l935_93524

theorem sin_half_alpha (α : ℝ) (hα : 0 < α ∧ α < π / 2) (hcos : Real.cos α = (1 + Real.sqrt 5) / 4) : 
  Real.sin (α / 2) = (Real.sqrt 5 - 1) / 4 := 
by 
  sorry

end sin_half_alpha_l935_93524


namespace abs_neg_six_l935_93526

theorem abs_neg_six : abs (-6) = 6 :=
by
  -- Proof goes here
  sorry

end abs_neg_six_l935_93526


namespace find_A_l935_93533

theorem find_A (A B C : ℕ) (h1 : A = B * C + 8) (h2 : A + B + C = 2994) : A = 8 ∨ A = 2864 :=
by
  sorry

end find_A_l935_93533


namespace miss_davis_sticks_left_l935_93525

theorem miss_davis_sticks_left (initial_sticks groups_per_class sticks_per_group : ℕ) 
(h1 : initial_sticks = 170) (h2 : groups_per_class = 10) (h3 : sticks_per_group = 15) : 
initial_sticks - (groups_per_class * sticks_per_group) = 20 :=
by sorry

end miss_davis_sticks_left_l935_93525


namespace problem1_problem2_problem3_problem4_l935_93539

-- Proof statement for problem 1
theorem problem1 : (1 : ℤ) * (-8) + 10 + 2 + (-1) = 3 := sorry

-- Proof statement for problem 2
theorem problem2 : (-21.6 : ℝ) - (-3) - |(-7.4)| + (-2 / 5) = -26.4 := sorry

-- Proof statement for problem 3
theorem problem3 : (-12 / 5) / (-1 / 10) * (-5 / 6) * (-0.4 : ℝ) = 8 := sorry

-- Proof statement for problem 4
theorem problem4 : ((5 / 8) - (1 / 6) + (7 / 12)) * (-24 : ℝ) = -25 := sorry

end problem1_problem2_problem3_problem4_l935_93539


namespace top_quality_soccer_balls_l935_93595

theorem top_quality_soccer_balls (N : ℕ) (f : ℝ) (hN : N = 10000) (hf : f = 0.975) : N * f = 9750 := by
  sorry

end top_quality_soccer_balls_l935_93595


namespace count_right_triangles_with_given_conditions_l935_93523

-- Define the type of our points as a pair of integers
def Point := (ℤ × ℤ)

-- Define the orthocenter being a specific point
def isOrthocenter (P : Point) := P = (-1, 7)

-- Define that a given triangle has a right angle at the origin
def rightAngledAtOrigin (O A B : Point) :=
  O = (0, 0) ∧
  (A.fst = 0 ∨ A.snd = 0) ∧
  (B.fst = 0 ∨ B.snd = 0) ∧
  (A.fst ≠ 0 ∨ A.snd ≠ 0) ∧
  (B.fst ≠ 0 ∨ B.snd ≠ 0)

-- Define that the points are lattice points
def areLatticePoints (O A B : Point) :=
  ∃ t k : ℤ, (A = (3 * t, 4 * t) ∧ B = (-4 * k, 3 * k)) ∨
            (B = (3 * t, 4 * t) ∧ A = (-4 * k, 3 * k))

-- Define the number of right triangles given the constraints
def numberOfRightTriangles : ℕ := 2

-- Statement of the problem
theorem count_right_triangles_with_given_conditions :
  ∃ (O A B : Point),
    rightAngledAtOrigin O A B ∧
    isOrthocenter (-1, 7) ∧
    areLatticePoints O A B ∧
    numberOfRightTriangles = 2 :=
  sorry

end count_right_triangles_with_given_conditions_l935_93523


namespace ratio_of_areas_l935_93556

-- Definitions of perimeter in Lean terms
def P_A : ℕ := 16
def P_B : ℕ := 32

-- Ratio of the area of region A to region C
theorem ratio_of_areas (s_A s_C : ℕ) (h₀ : 4 * s_A = P_A)
  (h₁ : 4 * s_C = 12) : s_A^2 / s_C^2 = 1 / 9 :=
by 
  sorry

end ratio_of_areas_l935_93556


namespace gcd_459_357_l935_93555

theorem gcd_459_357 : Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l935_93555


namespace granger_cisco_combined_spots_l935_93580

theorem granger_cisco_combined_spots :
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  G + C = 108 := by 
  let R := 46
  let C := (R / 2) - 5
  let G := 5 * C
  sorry

end granger_cisco_combined_spots_l935_93580


namespace poplar_more_than_pine_l935_93543

theorem poplar_more_than_pine (pine poplar : ℕ) (h1 : pine = 180) (h2 : poplar = 4 * pine) : poplar - pine = 540 :=
by
  -- Proof will be filled here
  sorry

end poplar_more_than_pine_l935_93543


namespace new_average_weight_l935_93504

theorem new_average_weight 
  (average_weight_19 : ℕ → ℝ)
  (weight_new_student : ℕ → ℝ)
  (new_student_count : ℕ)
  (old_student_count : ℕ)
  (h1 : average_weight_19 old_student_count = 15.0)
  (h2 : weight_new_student new_student_count = 11.0)
  : (average_weight_19 (old_student_count + new_student_count) = 14.8) :=
by
  sorry

end new_average_weight_l935_93504


namespace regular_polygon_sides_l935_93502

-- Define the main theorem statement
theorem regular_polygon_sides (n : ℕ) : 
  (n > 2) ∧ 
  ((n - 2) * 180 / n - 360 / n = 90) → 
  n = 8 := by
  sorry

end regular_polygon_sides_l935_93502


namespace function_monotonically_increasing_range_l935_93516

theorem function_monotonically_increasing_range (a : ℝ) :
  (∀ x y : ℝ, x ≤ 1 ∧ y ≤ 1 ∧ x ≤ y → ((4 - a / 2) * x + 2) ≤ ((4 - a / 2) * y + 2)) ∧
  (∀ x y : ℝ, x > 1 ∧ y > 1 ∧ x ≤ y → a^x ≤ a^y) ∧
  (∀ x : ℝ, if x = 1 then a^1 ≥ (4 - a / 2) * 1 + 2 else true) ↔
  4 ≤ a ∧ a < 8 :=
sorry

end function_monotonically_increasing_range_l935_93516


namespace largest_value_l935_93596

def X := (2010 / 2009) + (2010 / 2011)
def Y := (2010 / 2011) + (2012 / 2011)
def Z := (2011 / 2010) + (2011 / 2012)

theorem largest_value : X > Y ∧ X > Z := 
by
  sorry

end largest_value_l935_93596


namespace inclination_angle_of_line_l935_93593

theorem inclination_angle_of_line 
  (l : ℝ) (h : l = Real.tan (-π / 6)) : 
  ∀ θ, θ = Real.pi / 2 :=
by
  -- Placeholder proof
  sorry

end inclination_angle_of_line_l935_93593


namespace positive_difference_l935_93500

theorem positive_difference (a b : ℝ) (h₁ : a + b = 10) (h₂ : a^2 - b^2 = 40) : |a - b| = 4 :=
by
  sorry

end positive_difference_l935_93500


namespace simplify_sqrt_450_l935_93592

theorem simplify_sqrt_450 : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_450_l935_93592


namespace problem_condition_relationship_l935_93540

theorem problem_condition_relationship (x : ℝ) :
  (x^2 - x - 2 > 0) → (|x - 1| > 1) := 
sorry

end problem_condition_relationship_l935_93540


namespace cupcake_cookie_price_ratio_l935_93567

theorem cupcake_cookie_price_ratio
  (c k : ℚ)
  (h1 : 5 * c + 3 * k = 23)
  (h2 : 4 * c + 4 * k = 21) :
  k / c = 13 / 29 :=
  sorry

end cupcake_cookie_price_ratio_l935_93567


namespace stratified_sampling_second_grade_l935_93566

theorem stratified_sampling_second_grade (r1 r2 r3 : ℕ) (total_sample : ℕ) (total_ratio : ℕ):
  r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ total_sample = 50 ∧ total_ratio = r1 + r2 + r3 →
  (r2 * total_sample) / total_ratio = 15 :=
by
  sorry

end stratified_sampling_second_grade_l935_93566


namespace total_distance_travelled_eight_boys_on_circle_l935_93565

noncomputable def distance_travelled_by_boys (radius : ℝ) : ℝ :=
  let n := 8
  let angle := 2 * Real.pi / n
  let distance_to_non_adjacent := 2 * radius * Real.sin (2 * angle / 2)
  n * (100 + 3 * distance_to_non_adjacent)

theorem total_distance_travelled_eight_boys_on_circle :
  distance_travelled_by_boys 50 = 800 + 1200 * Real.sqrt 2 :=
  by
    sorry

end total_distance_travelled_eight_boys_on_circle_l935_93565


namespace daily_wage_c_l935_93530

theorem daily_wage_c (a_days b_days c_days total_earnings : ℕ)
  (ratio_a_b ratio_b_c : ℚ)
  (a_wage b_wage c_wage : ℚ) :
  a_days = 6 →
  b_days = 9 →
  c_days = 4 →
  total_earnings = 1480 →
  ratio_a_b = 3 / 4 →
  ratio_b_c = 4 / 5 →
  b_wage = ratio_a_b * a_wage → 
  c_wage = ratio_b_c * b_wage → 
  a_days * a_wage + b_days * b_wage + c_days * c_wage = total_earnings →
  c_wage = 100 / 3 :=
by
  intros
  sorry

end daily_wage_c_l935_93530


namespace equation_solution_l935_93535

def solve_equation (x : ℝ) : Prop :=
  ((3 * x + 6) / (x^2 + 5 * x - 6) = (4 - x) / (x - 2)) ↔ 
  x = -3 ∨ x = (1 + Real.sqrt 17) / 2 ∨ x = (1 - Real.sqrt 17) / 2

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ -6) (h3 : x ≠ 2) : solve_equation x :=
by
  sorry

end equation_solution_l935_93535


namespace election_winner_votes_l935_93538

theorem election_winner_votes :
  ∃ V W : ℝ, (V = (71.42857142857143 / 100) * V + 3000 + 5000) ∧
            (W = (71.42857142857143 / 100) * V) ∧
            W = 20000 := by
  sorry

end election_winner_votes_l935_93538


namespace jeff_average_skips_is_14_l935_93513

-- Definitions of the given conditions in the problem
def sam_skips_per_round : ℕ := 16
def rounds : ℕ := 4

-- Number of skips by Jeff in each round based on the conditions
def jeff_first_round_skips : ℕ := sam_skips_per_round - 1
def jeff_second_round_skips : ℕ := sam_skips_per_round - 3
def jeff_third_round_skips : ℕ := sam_skips_per_round + 4
def jeff_fourth_round_skips : ℕ := sam_skips_per_round / 2

-- Total skips by Jeff in all rounds
def jeff_total_skips : ℕ := jeff_first_round_skips + 
                           jeff_second_round_skips + 
                           jeff_third_round_skips + 
                           jeff_fourth_round_skips

-- Average skips per round by Jeff
def jeff_average_skips : ℕ := jeff_total_skips / rounds

-- Theorem statement
theorem jeff_average_skips_is_14 : jeff_average_skips = 14 := 
by 
    sorry

end jeff_average_skips_is_14_l935_93513


namespace amin_probability_four_attempts_before_three_hits_amin_probability_not_qualified_stops_after_two_consecutive_misses_l935_93531

/-- Prove that the probability Amin makes 4 attempts before hitting 3 times (given the probability of each hit is 1/2) is 3/16. -/
theorem amin_probability_four_attempts_before_three_hits (p_hit : ℚ := 1 / 2) : 
  (∃ (P : ℚ), P = 3/16) :=
sorry

/-- Prove that the probability Amin stops shooting after missing two consecutive shots and not qualifying as level B or A player is 25/32, given the probability of each hit is 1/2. -/
theorem amin_probability_not_qualified_stops_after_two_consecutive_misses (p_hit : ℚ := 1 / 2) : 
  (∃ (P : ℚ), P = 25/32) :=
sorry

end amin_probability_four_attempts_before_three_hits_amin_probability_not_qualified_stops_after_two_consecutive_misses_l935_93531


namespace find_value_of_m_l935_93569

theorem find_value_of_m :
  (∃ y : ℝ, y = 20 - (0.5 * -6.7)) →
  (m : ℝ) = 3 * -6.7 + (20 - (0.5 * -6.7)) :=
by {
  sorry
}

end find_value_of_m_l935_93569


namespace assign_grades_l935_93563

-- Definitions based on the conditions:
def num_students : ℕ := 12
def num_grades : ℕ := 4

-- Statement of the theorem
theorem assign_grades : num_grades ^ num_students = 16777216 := by
  sorry

end assign_grades_l935_93563


namespace natural_numbers_satisfying_condition_l935_93546

open Nat

theorem natural_numbers_satisfying_condition (r : ℕ) :
  ∃ k : Set ℕ, k = { k | ∃ s t : ℕ, k = 2^(r + s) * t ∧ 2 ∣ t ∧ 2 ∣ s } :=
by
  sorry

end natural_numbers_satisfying_condition_l935_93546


namespace sum_possible_x_coordinates_l935_93503

-- Define the vertices of the parallelogram
def A := (1, 2)
def B := (3, 8)
def C := (4, 1)

-- Definition of what it means to be a fourth vertex that forms a parallelogram
def is_fourth_vertex (D : ℤ × ℤ) : Prop :=
  (D = (6, 7)) ∨ (D = (2, -5)) ∨ (D = (0, 9))

-- The sum of possible x-coordinates for the fourth vertex
def sum_x_coordinates : ℤ :=
  6 + 2 + 0

theorem sum_possible_x_coordinates :
  (∃ D, is_fourth_vertex D) → sum_x_coordinates = 8 :=
by
  -- Sorry is used to skip the detailed proof steps
  sorry

end sum_possible_x_coordinates_l935_93503


namespace tom_and_elizabeth_climb_ratio_l935_93512

theorem tom_and_elizabeth_climb_ratio :
  let elizabeth_time := 30
  let tom_time_hours := 2
  let tom_time_minutes := tom_time_hours * 60
  (tom_time_minutes / elizabeth_time) = 4 :=
by sorry

end tom_and_elizabeth_climb_ratio_l935_93512


namespace prove_perpendicular_planes_l935_93572

-- Defining the non-coincident lines m and n
variables {m n : Set Point} {α β : Set Point}

-- Lines and plane relationship definitions
def parallel (x y : Set Point) : Prop := sorry
def perpendicular (x y : Set Point) : Prop := sorry
def subset (x y : Set Point) : Prop := sorry

-- Given conditions
axiom h1 : parallel m n
axiom h2 : subset m α
axiom h3 : perpendicular n β

-- Prove that α is perpendicular to β
theorem prove_perpendicular_planes :
  perpendicular α β :=
  sorry

end prove_perpendicular_planes_l935_93572


namespace expression_never_prime_l935_93511

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem expression_never_prime (p : ℕ) (hp : is_prime p) : ¬ is_prime (p^2 + 20) := sorry

end expression_never_prime_l935_93511


namespace find_multiple_l935_93573

theorem find_multiple (m : ℤ) (h : 38 + m * 43 = 124) : m = 2 := by
    sorry

end find_multiple_l935_93573


namespace chips_left_uneaten_l935_93558

theorem chips_left_uneaten 
    (chips_per_cookie : ℕ)
    (cookies_per_dozen : ℕ)
    (dozens_of_cookies : ℕ)
    (cookies_eaten_ratio : ℕ) 
    (h_chips : chips_per_cookie = 7)
    (h_cookies_dozen : cookies_per_dozen = 12)
    (h_dozens : dozens_of_cookies = 4)
    (h_eaten_ratio : cookies_eaten_ratio = 2) : 
  (cookies_per_dozen * dozens_of_cookies / cookies_eaten_ratio) * chips_per_cookie = 168 :=
by 
  sorry

end chips_left_uneaten_l935_93558


namespace gasoline_tank_capacity_l935_93518

theorem gasoline_tank_capacity
  (initial_fill : ℝ) (final_fill : ℝ) (gallons_used : ℝ) (x : ℝ)
  (h1 : initial_fill = 3 / 4)
  (h2 : final_fill = 1 / 3)
  (h3 : gallons_used = 18)
  (h4 : initial_fill * x - final_fill * x = gallons_used) :
  x = 43 :=
by
  -- Skipping the proof
  sorry

end gasoline_tank_capacity_l935_93518


namespace meghan_total_money_l935_93514

theorem meghan_total_money :
  let num_100_bills := 2
  let num_50_bills := 5
  let num_10_bills := 10
  let value_100_bills := num_100_bills * 100
  let value_50_bills := num_50_bills * 50
  let value_10_bills := num_10_bills * 10
  let total_money := value_100_bills + value_50_bills + value_10_bills
  total_money = 550 := by sorry

end meghan_total_money_l935_93514


namespace greatest_number_of_cool_cells_l935_93510

noncomputable def greatest_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) : ℕ :=
n^2 - 2 * n + 1

theorem greatest_number_of_cool_cells (n : ℕ) (grid : Matrix (Fin n) (Fin n) ℝ) (h : 0 < n) :
  ∃ m, m = (n - 1)^2 ∧ m = greatest_cool_cells n grid :=
sorry

end greatest_number_of_cool_cells_l935_93510


namespace function_evaluation_l935_93536

theorem function_evaluation (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = 2 * x ^ 2 + 1) : 
  ∀ x : ℝ, f x = 2 * x ^ 2 - 4 * x + 3 :=
sorry

end function_evaluation_l935_93536


namespace pyramid_apex_angle_l935_93517

theorem pyramid_apex_angle (A B C D E O : Type) 
  (square_base : Π (P Q : Type), Prop) 
  (isosceles_triangle : Π (R S T : Type), Prop)
  (AEB_angle : Π (X Y Z : Type), Prop) 
  (angle_AOB : ℝ)
  (angle_AEB : ℝ)
  (square_base_conditions : square_base A B ∧ square_base B C ∧ square_base C D ∧ square_base D A)
  (isosceles_triangle_conditions : isosceles_triangle A E B ∧ isosceles_triangle B E C ∧ isosceles_triangle C E D ∧ isosceles_triangle D E A)
  (center : O)
  (diagonals_intersect_at_right_angle : angle_AOB = 90)
  (measured_angle_at_apex : angle_AEB = 100) :
False :=
sorry

end pyramid_apex_angle_l935_93517


namespace nonincreasing_7_digit_integers_l935_93574

theorem nonincreasing_7_digit_integers : 
  ∃ n : ℕ, n = 11439 ∧ (∀ x : ℕ, (10^6 ≤ x ∧ x < 10^7) → 
    (∀ i j : ℕ, 1 ≤ i ∧ i < j ∧ j ≤ 7 → (x / 10^(7 - i) % 10) ≥ (x / 10^(7 - j) % 10))) :=
by
  sorry

end nonincreasing_7_digit_integers_l935_93574


namespace part_a_part_b_l935_93585

open Nat

theorem part_a (n: ℕ) (h_pos: 0 < n) : (2^n - 1) % 7 = 0 ↔ ∃ k : ℕ, k > 0 ∧ n = 3 * k :=
sorry

theorem part_b (n: ℕ) (h_pos: 0 < n) : (2^n + 1) % 7 ≠ 0 :=
sorry

end part_a_part_b_l935_93585


namespace cos_seven_pi_over_four_l935_93564

theorem cos_seven_pi_over_four :
  Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_seven_pi_over_four_l935_93564


namespace escalator_length_l935_93575

theorem escalator_length :
  ∃ L : ℝ, L = 150 ∧ 
    (∀ t : ℝ, t = 10 → ∀ v_p : ℝ, v_p = 3 → ∀ v_e : ℝ, v_e = 12 → L = (v_p + v_e) * t) :=
by sorry

end escalator_length_l935_93575


namespace player_holds_seven_black_cards_l935_93509

theorem player_holds_seven_black_cards
    (total_cards : ℕ := 13)
    (num_red_cards : ℕ := 6)
    (S D H C : ℕ)
    (h1 : D = 2 * S)
    (h2 : H = 2 * D)
    (h3 : C = 6)
    (h4 : S + D + H + C = total_cards) :
    S + C = 7 := 
by
  sorry

end player_holds_seven_black_cards_l935_93509


namespace trapezium_distance_l935_93547

theorem trapezium_distance (a b area : ℝ) (h : ℝ) :
  a = 20 ∧ b = 18 ∧ area = 266 ∧
  area = (1/2) * (a + b) * h -> h = 14 :=
by
  sorry

end trapezium_distance_l935_93547


namespace max_additional_hours_l935_93597

/-- Define the additional hours of studying given the investments in dorms, food, and parties -/
def additional_hours (a b c : ℝ) : ℝ :=
  5 * a + 3 * b + (11 * c - c^2)

/-- Define the total investment constraint -/
def investment_constraint (a b c : ℝ) : Prop :=
  a + b + c = 5

/-- Prove the maximal additional hours of studying -/
theorem max_additional_hours : ∃ (a b c : ℝ), investment_constraint a b c ∧ additional_hours a b c = 34 :=
by
  sorry

end max_additional_hours_l935_93597


namespace smallest_three_digit_in_pascals_triangle_l935_93577

theorem smallest_three_digit_in_pascals_triangle : ∃ k n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ ∀ m, ((m <= n) ∧ (m >= 100)) → m ≥ n :=
by
  sorry

end smallest_three_digit_in_pascals_triangle_l935_93577


namespace billboard_shorter_side_length_l935_93590

theorem billboard_shorter_side_length
  (L W : ℝ)
  (h1 : L * W = 91)
  (h2 : 2 * L + 2 * W = 40) :
  L = 7 ∨ W = 7 :=
by sorry

end billboard_shorter_side_length_l935_93590


namespace find_positive_solutions_l935_93576

theorem find_positive_solutions (x₁ x₂ x₃ x₄ x₅ : ℝ) (h_pos : 0 < x₁ ∧ 0 < x₂ ∧ 0 < x₃ ∧ 0 < x₄ ∧ 0 < x₅)
    (h1 : x₁ + x₂ = x₃^2)
    (h2 : x₂ + x₃ = x₄^2)
    (h3 : x₃ + x₄ = x₅^2)
    (h4 : x₄ + x₅ = x₁^2)
    (h5 : x₅ + x₁ = x₂^2) :
    x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
    by {
        -- Proof goes here
        sorry
    }

end find_positive_solutions_l935_93576


namespace isosceles_triangle_perimeter_l935_93545

/-- 
  Given an isosceles triangle with two sides of length 6 and the third side of length 2,
  prove that the perimeter of the triangle is 14.
-/
theorem isosceles_triangle_perimeter (a b c : ℕ) (h1 : a = 6) (h2 : b = 6) (h3 : c = 2) 
  (triangle_ineq1 : a + b > c) (triangle_ineq2 : a + c > b) (triangle_ineq3 : b + c > a) :
  a + b + c = 14 :=
  sorry

end isosceles_triangle_perimeter_l935_93545


namespace a_1994_is_7_l935_93529

def f (m : ℕ) : ℕ := m % 10

def a (n : ℕ) : ℕ := f (2^(n + 1) - 1)

theorem a_1994_is_7 : a 1994 = 7 :=
by
  sorry

end a_1994_is_7_l935_93529


namespace quadratic_inequality_iff_l935_93570

noncomputable def quadratic_inequality_solution (x : ℝ) : Prop := x^2 + 4*x - 96 > abs x

theorem quadratic_inequality_iff (x : ℝ) : quadratic_inequality_solution x ↔ x < -12 ∨ x > 8 := by
  sorry

end quadratic_inequality_iff_l935_93570


namespace find_overall_mean_score_l935_93519

variable (M N E : ℝ)
variable (m n e : ℝ)

theorem find_overall_mean_score :
  M = 85 → N = 75 → E = 65 →
  m / n = 4 / 5 → n / e = 3 / 2 →
  ((85 * m) + (75 * n) + (65 * e)) / (m + n + e) = 82 :=
by
  sorry

end find_overall_mean_score_l935_93519


namespace fraction_of_garden_occupied_by_triangle_beds_l935_93549

theorem fraction_of_garden_occupied_by_triangle_beds :
  ∀ (rect_height rect_width trapezoid_short_base trapezoid_long_base : ℝ) 
    (num_triangles : ℕ) 
    (triangle_leg_length : ℝ)
    (total_area_triangles : ℝ)
    (total_garden_area : ℝ)
    (fraction : ℝ),
  rect_height = 10 → rect_width = 30 →
  trapezoid_short_base = 20 → trapezoid_long_base = 30 → num_triangles = 3 →
  triangle_leg_length = 10 / 3 →
  total_area_triangles = 3 * (1 / 2 * (triangle_leg_length ^ 2)) →
  total_garden_area = rect_height * rect_width →
  fraction = total_area_triangles / total_garden_area →
  fraction = 1 / 18 := by
  intros rect_height rect_width trapezoid_short_base trapezoid_long_base
         num_triangles triangle_leg_length total_area_triangles
         total_garden_area fraction
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end fraction_of_garden_occupied_by_triangle_beds_l935_93549


namespace general_term_sequence_sum_of_cn_l935_93507

theorem general_term_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n) :
  ∀ n, a n = n :=
by
  sorry

theorem sum_of_cn (S : ℕ → ℕ) (a : ℕ → ℕ) (c : ℕ → ℕ) (T : ℕ → ℕ)
  (hS : S 2 = 3)
  (hS_eq : ∀ n, 2 * S n = n + n * a n)
  (ha : ∀ n, a n = n)
  (hc_odd : ∀ n, c (2 * n - 1) = a (2 * n))
  (hc_even : ∀ n, c (2 * n) = 3 * 2^(a (2 * n - 1)) + 1) :
  ∀ n, T (2 * n) = 2^(2 * n + 1) + n^2 + 2 * n - 2 :=
by
  sorry

end general_term_sequence_sum_of_cn_l935_93507


namespace contradiction_proof_l935_93542

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : ¬ (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :=
by 
  sorry

end contradiction_proof_l935_93542


namespace original_weight_of_marble_l935_93591

theorem original_weight_of_marble (W : ℝ) (h1 : W * 0.75 * 0.85 * 0.90 = 109.0125) : W = 190 :=
by
  sorry

end original_weight_of_marble_l935_93591


namespace surface_area_increase_l935_93550

def cube_dimensions : ℝ × ℝ × ℝ := (10, 10, 10)

def number_of_cuts := 3

def initial_surface_area (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  6 * (length * width)

def increase_in_surface_area (cuts : ℕ) (length : ℝ) (width : ℝ) : ℝ :=
  cuts * 2 * (length * width)

theorem surface_area_increase : 
  initial_surface_area 10 10 10 + increase_in_surface_area 3 10 10 = 
  initial_surface_area 10 10 10 + 600 :=
by
  sorry

end surface_area_increase_l935_93550


namespace work_completion_l935_93589

theorem work_completion (Rp Rq Dp W : ℕ) 
  (h1 : Rq = W / 12) 
  (h2 : W = 4*Rp + 6*(Rp + Rq)) 
  (h3 : Rp = W / Dp) 
  : Dp = 20 :=
by
  sorry

end work_completion_l935_93589


namespace solve_for_x_l935_93579

theorem solve_for_x (x : ℝ) (h : 5 * x + 3 = 10 * x - 22) : x = 5 :=
sorry

end solve_for_x_l935_93579


namespace gcd_of_12012_and_18018_l935_93598

theorem gcd_of_12012_and_18018 : Int.gcd 12012 18018 = 6006 := 
by
  -- Here we are assuming the factorization given in the conditions
  have h₁ : 12012 = 12 * 1001 := sorry
  have h₂ : 18018 = 18 * 1001 := sorry
  have gcd_12_18 : Int.gcd 12 18 = 6 := sorry
  -- This sorry will be replaced by the actual proof involving the above conditions to conclude the stated theorem
  sorry

end gcd_of_12012_and_18018_l935_93598


namespace mistake_position_is_34_l935_93588

def arithmetic_sequence_sum (n : ℕ) (a_1 : ℕ) (d : ℕ) : ℕ :=
  n * (2 * a_1 + (n - 1) * d) / 2

def modified_sequence_sum (n : ℕ) (a_1 : ℕ) (d : ℕ) (mistake_index : ℕ) : ℕ :=
  let correct_sum := arithmetic_sequence_sum n a_1 d
  correct_sum - 2 * d

theorem mistake_position_is_34 :
  ∃ mistake_index : ℕ, mistake_index = 34 ∧ 
    modified_sequence_sum 37 1 3 mistake_index = 2011 :=
by
  sorry

end mistake_position_is_34_l935_93588


namespace equal_costs_l935_93520

noncomputable def cost_scheme_1 (x : ℕ) : ℝ := 350 + 5 * x

noncomputable def cost_scheme_2 (x : ℕ) : ℝ := 360 + 4.5 * x

theorem equal_costs (x : ℕ) : cost_scheme_1 x = cost_scheme_2 x ↔ x = 20 := by
  sorry

end equal_costs_l935_93520


namespace qudrilateral_diagonal_length_l935_93578

theorem qudrilateral_diagonal_length (A h1 h2 d : ℝ) 
  (h_area : A = 140) (h_offsets : h1 = 8) (h_offsets2 : h2 = 2) 
  (h_formula : A = 1 / 2 * d * (h1 + h2)) : 
  d = 28 :=
by
  sorry

end qudrilateral_diagonal_length_l935_93578


namespace unique_function_l935_93587

theorem unique_function (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x + 1) ≥ f x + 1) 
  (h2 : ∀ x y : ℝ, f (x * y) ≥ f x * f y) : 
  ∀ x : ℝ, f x = x := 
sorry

end unique_function_l935_93587


namespace find_cos_value_l935_93561

open Real

noncomputable def cos_value (α : ℝ) : ℝ :=
  cos (2 * π / 3 + 2 * α)

theorem find_cos_value (α : ℝ) (h : sin (π / 6 - α) = 1 / 4) :
  cos_value α = -7 / 8 :=
sorry

end find_cos_value_l935_93561


namespace chord_length_intercepted_l935_93506

theorem chord_length_intercepted 
  (line_eq : ∀ x y : ℝ, 3 * x - 4 * y = 0)
  (circle_eq : ∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 2) : 
  ∃ l : ℝ, l = 2 :=
by 
  sorry

end chord_length_intercepted_l935_93506


namespace sum_of_valid_b_values_l935_93501

/-- Given a quadratic equation 3x² + 7x + b = 0, where b is a positive integer,
and the requirement that the equation must have rational roots, the sum of all
possible positive integer values of b is 6. -/
theorem sum_of_valid_b_values : 
  ∃ (b_values : List ℕ), 
    (∀ b ∈ b_values, 0 < b ∧ ∃ n : ℤ, 49 - 12 * b = n^2) ∧ b_values.sum = 6 :=
by sorry

end sum_of_valid_b_values_l935_93501


namespace smallest_root_of_g_l935_93527

noncomputable def g (x : ℝ) : ℝ := 21 * x^4 - 20 * x^2 + 3

theorem smallest_root_of_g : ∀ x : ℝ, g x = 0 → x = - Real.sqrt (3 / 7) :=
by
  sorry

end smallest_root_of_g_l935_93527


namespace percentage_fewer_than_50000_l935_93505

def percentage_lt_20000 : ℝ := 35
def percentage_20000_to_49999 : ℝ := 45
def percentage_lt_50000 : ℝ := 80

theorem percentage_fewer_than_50000 :
  percentage_lt_20000 + percentage_20000_to_49999 = percentage_lt_50000 := 
by
  sorry

end percentage_fewer_than_50000_l935_93505


namespace find_x_l935_93553

-- Declaration for the custom operation
def star (a b : ℝ) : ℝ := a * b + 3 * b - 2 * a

-- Theorem statement
theorem find_x (x : ℝ) (h : star 3 x = 23) : x = 29 / 6 :=
by {
    sorry -- The proof steps are to be filled here.
}

end find_x_l935_93553


namespace team_leads_per_supervisor_l935_93557

def num_workers : ℕ := 390
def num_supervisors : ℕ := 13
def leads_per_worker_ratio : ℕ := 10

theorem team_leads_per_supervisor : (num_workers / leads_per_worker_ratio) / num_supervisors = 3 :=
by
  sorry

end team_leads_per_supervisor_l935_93557


namespace inequality_solution_l935_93537

theorem inequality_solution (x : ℝ) : 
  (x < 2 ∨ x = 3) ↔ (x - 3) / ((x - 2) * (x - 3)) ≤ 0 := 
by {
  sorry
}

end inequality_solution_l935_93537


namespace cylinder_volume_l935_93521

theorem cylinder_volume (short_side long_side : ℝ) (h_short_side : short_side = 12) (h_long_side : long_side = 18) : 
  ∀ (r h : ℝ) (h_radius : r = short_side / 2) (h_height : h = long_side), 
    volume = π * r^2 * h := 
by
  sorry

end cylinder_volume_l935_93521


namespace average_price_per_book_l935_93571

theorem average_price_per_book
  (spent1 spent2 spent3 spent4 : ℝ) (books1 books2 books3 books4 : ℕ)
  (h1 : spent1 = 1080) (h2 : spent2 = 840) (h3 : spent3 = 765) (h4 : spent4 = 630)
  (hb1 : books1 = 65) (hb2 : books2 = 55) (hb3 : books3 = 45) (hb4 : books4 = 35) :
  (spent1 + spent2 + spent3 + spent4) / (books1 + books2 + books3 + books4) = 16.575 :=
by {
  sorry
}

end average_price_per_book_l935_93571


namespace total_amount_earned_is_90_l935_93554

variable (W : ℕ)

-- Define conditions
def work_capacity_condition : Prop :=
  5 = W ∧ W = 8

-- Define wage per man in Rs.
def wage_per_man : ℕ := 6

-- Define total amount earned by 5 men
def total_earned_by_5_men : ℕ := 5 * wage_per_man

-- Define total amount for the problem
def total_earned (W : ℕ) : ℕ :=
  3 * total_earned_by_5_men

-- The final proof statement
theorem total_amount_earned_is_90 (W : ℕ) (h : work_capacity_condition W) : total_earned W = 90 := by
  sorry

end total_amount_earned_is_90_l935_93554


namespace total_length_of_ribbon_l935_93599

-- Define the conditions
def length_per_piece : ℕ := 73
def number_of_pieces : ℕ := 51

-- The theorem to prove
theorem total_length_of_ribbon : length_per_piece * number_of_pieces = 3723 :=
by
  sorry

end total_length_of_ribbon_l935_93599


namespace prob_neither_prime_nor_composite_l935_93559

theorem prob_neither_prime_nor_composite :
  (1 / 95 : ℚ) = 1 / 95 := by
  sorry

end prob_neither_prime_nor_composite_l935_93559


namespace volume_and_surface_area_of_inscribed_sphere_l935_93522

theorem volume_and_surface_area_of_inscribed_sphere (edge_length : ℝ) (h_edge : edge_length = 10) :
    let r := edge_length / 2
    let V := (4 / 3) * π * r^3
    let A := 4 * π * r^2
    V = (500 / 3) * π ∧ A = 100 * π := 
by
  sorry

end volume_and_surface_area_of_inscribed_sphere_l935_93522


namespace exists_c_d_in_set_of_13_reals_l935_93551

theorem exists_c_d_in_set_of_13_reals (a : Fin 13 → ℝ) :
  ∃ (c d : ℝ), c ∈ Set.range a ∧ d ∈ Set.range a ∧ 0 < (c - d) / (1 + c * d) ∧ (c - d) / (1 + c * d) < 2 - Real.sqrt 3 := 
by
  sorry

end exists_c_d_in_set_of_13_reals_l935_93551


namespace carnations_count_l935_93568

theorem carnations_count (c : ℕ) : 
  (9 * 6 = 54) ∧ (47 ≤ c + 47) ∧ (c + 47 = 54) → c = 7 := 
by
  sorry

end carnations_count_l935_93568


namespace find_divisor_l935_93562

theorem find_divisor (D N : ℕ) (h₁ : N = 265) (h₂ : N / D + 8 = 61) : D = 5 :=
by
  sorry

end find_divisor_l935_93562


namespace binom_sub_floor_divisible_by_prime_l935_93581

theorem binom_sub_floor_divisible_by_prime (p n : ℕ) (hp : Nat.Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - (n / p)) :=
sorry

end binom_sub_floor_divisible_by_prime_l935_93581


namespace find_a_l935_93582

theorem find_a (a : ℝ) (h : (2:ℝ)^2 + 2 * a - 3 * a = 0) : a = 4 :=
sorry

end find_a_l935_93582


namespace find_extrema_l935_93586

noncomputable def f (x : ℝ) : ℝ := x^3 + (-3/2) * x^2 + (-3) * x + 1
noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 + 2 * (-3/2) * x + (-3)
noncomputable def g (x : ℝ) : ℝ := f' x * Real.exp x

theorem find_extrema :
  (a = -3/2 ∧ b = -3 ∧ f' (1) = (3 * (1:ℝ)^2 - 3/2 * (1:ℝ) - 3) ) ∧
  (g 1 = -3 * Real.exp 1 ∧ g (-2) = 15 * Real.exp (-2)) := 
by
  -- Sorry for skipping the proof
  sorry

end find_extrema_l935_93586


namespace greatest_num_fruit_in_each_basket_l935_93552

theorem greatest_num_fruit_in_each_basket : 
  let oranges := 15
  let peaches := 9
  let pears := 18
  let gcd := Nat.gcd (Nat.gcd oranges peaches) pears
  gcd = 3 :=
by
  sorry

end greatest_num_fruit_in_each_basket_l935_93552


namespace amount_in_paise_l935_93532

theorem amount_in_paise (a : ℝ) (h_a : a = 170) (percentage_value : ℝ) (h_percentage : percentage_value = 0.5 / 100) : 
  (percentage_value * a * 100) = 85 := 
by
  sorry

end amount_in_paise_l935_93532


namespace remainder_of_n_mod_1000_l935_93560

-- Definition of the set S
def S : Set ℕ := { n | 1 ≤ n ∧ n ≤ 15 }

-- Define the number of sets of three non-empty disjoint subsets of S
def num_sets_of_three_non_empty_disjoint_subsets (S : Set ℕ) : ℕ :=
  let total_partitions := 4^15
  let single_empty_partition := 3 * 3^15
  let double_empty_partition := 3 * 2^15
  let all_empty_partition := 1
  total_partitions - single_empty_partition + double_empty_partition - all_empty_partition

-- Compute the result of the number modulo 1000
def result := (num_sets_of_three_non_empty_disjoint_subsets S) % 1000

-- Theorem that states the remainder when n is divided by 1000
theorem remainder_of_n_mod_1000 : result = 406 := by
  sorry

end remainder_of_n_mod_1000_l935_93560


namespace delivery_driver_stops_l935_93541

theorem delivery_driver_stops (total_boxes : ℕ) (boxes_per_stop : ℕ) (stops : ℕ) :
  total_boxes = 27 → boxes_per_stop = 9 → stops = total_boxes / boxes_per_stop → stops = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end delivery_driver_stops_l935_93541


namespace range_of_f_l935_93534

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2

theorem range_of_f : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 3 → f x ∈ Set.Icc (-18 : ℝ) (2 : ℝ) :=
sorry

end range_of_f_l935_93534


namespace sqrt_sixteen_is_four_l935_93584

theorem sqrt_sixteen_is_four : Real.sqrt 16 = 4 := 
by 
  sorry

end sqrt_sixteen_is_four_l935_93584


namespace movie_marathon_first_movie_length_l935_93544

theorem movie_marathon_first_movie_length 
  (x : ℝ)
  (h2 : 1.5 * x = second_movie)
  (h3 : second_movie + x - 1 = last_movie)
  (h4 : (x + second_movie + last_movie) = 9)
  (h5 : last_movie = 2.5 * x - 1) :
  x = 2 :=
by
  sorry

end movie_marathon_first_movie_length_l935_93544


namespace ratio_of_running_speed_l935_93508

theorem ratio_of_running_speed (distance : ℝ) (time_jack : ℝ) (time_jill : ℝ) 
  (h_distance_eq : distance = 42) (h_time_jack_eq : time_jack = 6) 
  (h_time_jill_eq : time_jill = 4.2) :
  (distance / time_jack) / (distance / time_jill) = 7 / 10 := by 
  sorry

end ratio_of_running_speed_l935_93508


namespace chess_group_players_l935_93548

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 1225) : n = 50 :=
sorry

end chess_group_players_l935_93548


namespace g_value_l935_93583

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_value (h : ∀ x ≠ 0, g x - 3 * g (1 / x) = 3^x) :
  g 3 = -(27 + 3 * (3:ℝ)^(1/3)) / 8 :=
sorry

end g_value_l935_93583


namespace ratio_of_football_to_hockey_l935_93594

variables (B F H s : ℕ)

-- Definitions from conditions
def condition1 : Prop := B = F - 50
def condition2 : Prop := F = s * H
def condition3 : Prop := H = 200
def condition4 : Prop := B + F + H = 1750

-- Proof statement
theorem ratio_of_football_to_hockey (B F H s : ℕ) 
  (h1 : condition1 B F)
  (h2 : condition2 F s H)
  (h3 : condition3 H)
  (h4 : condition4 B F H) : F / H = 4 :=
sorry

end ratio_of_football_to_hockey_l935_93594


namespace inequality_solution_l935_93528

theorem inequality_solution : { x : ℝ | (x - 1) / (x + 3) < 0 } = { x : ℝ | -3 < x ∧ x < 1 } :=
sorry

end inequality_solution_l935_93528
