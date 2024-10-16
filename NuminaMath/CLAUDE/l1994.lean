import Mathlib

namespace NUMINAMATH_CALUDE_root_product_plus_one_l1994_199447

theorem root_product_plus_one (r s t : ℂ) : 
  r^3 - 15*r^2 + 26*r - 8 = 0 →
  s^3 - 15*s^2 + 26*s - 8 = 0 →
  t^3 - 15*t^2 + 26*t - 8 = 0 →
  (1 + r) * (1 + s) * (1 + t) = 50 := by
sorry

end NUMINAMATH_CALUDE_root_product_plus_one_l1994_199447


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1994_199469

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geo : GeometricSequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_a2 : a 2 = 3) 
  (h_a6 : a 6 = 48) : 
  ∃ q : ℝ, q = 2 ∧ ∀ n : ℕ, a (n + 1) = a n * q :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l1994_199469


namespace NUMINAMATH_CALUDE_hyperbola_focus_a_value_l1994_199451

-- Define the hyperbola
def hyperbola (x y a : ℝ) : Prop := x^2 / 9 - y^2 / a = 1

-- Define the right focus
def right_focus (x y : ℝ) : Prop := x = Real.sqrt 13 ∧ y = 0

-- Theorem statement
theorem hyperbola_focus_a_value :
  ∀ a : ℝ, (∀ x y : ℝ, hyperbola x y a → right_focus x y) → a = 4 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_focus_a_value_l1994_199451


namespace NUMINAMATH_CALUDE_new_person_weight_l1994_199461

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (initial_count : ℕ) (average_increase : ℚ) (replaced_weight : ℚ) : ℚ :=
  replaced_weight + (initial_count : ℚ) * average_increase

/-- Theorem stating that the weight of the new person is 87 kg -/
theorem new_person_weight :
  weight_of_new_person 8 (5/2) 67 = 87 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l1994_199461


namespace NUMINAMATH_CALUDE_martian_right_angle_theorem_l1994_199485

/-- The number of clerts in a full circle in the Martian system -/
def full_circle_clerts : ℕ := 600

/-- The fraction of a full circle that represents a Martian right angle -/
def martian_right_angle_fraction : ℚ := 1/3

/-- The number of clerts in a Martian right angle -/
def martian_right_angle_clerts : ℕ := 200

/-- Theorem stating that the number of clerts in a Martian right angle is 200 -/
theorem martian_right_angle_theorem : 
  (↑full_circle_clerts : ℚ) * martian_right_angle_fraction = martian_right_angle_clerts := by
  sorry

end NUMINAMATH_CALUDE_martian_right_angle_theorem_l1994_199485


namespace NUMINAMATH_CALUDE_extreme_value_and_maximum_l1994_199440

-- Define the function f and its derivative
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.sin (2 * x) + a * Real.cos x + x
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (2 * x) - a * Real.sin x + 1

theorem extreme_value_and_maximum (a : ℝ) :
  f' a (π / 6) = 0 →
  a = 4 ∧
  ∀ x ∈ Set.Icc (-π / 6) (7 * π / 6), f 4 x ≤ (5 * Real.sqrt 3) / 2 + π / 6 :=
by sorry

end NUMINAMATH_CALUDE_extreme_value_and_maximum_l1994_199440


namespace NUMINAMATH_CALUDE_quadratic_solution_product_l1994_199446

theorem quadratic_solution_product (p q : ℝ) : 
  (3 * p^2 - 2 * p - 8 = 0) → 
  (3 * q^2 - 2 * q - 8 = 0) → 
  p ≠ q →
  (p - 1) * (q - 1) = -7/3 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_product_l1994_199446


namespace NUMINAMATH_CALUDE_number_of_factors_60_l1994_199438

/-- The number of positive factors of 60 is 12 -/
theorem number_of_factors_60 : Finset.card (Nat.divisors 60) = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_factors_60_l1994_199438


namespace NUMINAMATH_CALUDE_cone_lateral_surface_area_l1994_199499

theorem cone_lateral_surface_area 
  (r : ℝ) (h : ℝ) (l : ℝ) (A : ℝ) 
  (h_r : r = 2) 
  (h_h : h = Real.sqrt 5) 
  (h_l : l = Real.sqrt (r^2 + h^2)) 
  (h_A : A = π * r * l) : A = 6 * π := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_area_l1994_199499


namespace NUMINAMATH_CALUDE_tangent_line_equation_l1994_199405

/-- The circle C -/
def C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 5

/-- The point P -/
def P : ℝ × ℝ := (2, 4)

/-- The tangent line -/
def tangent_line (x y : ℝ) : Prop := x + 2*y - 10 = 0

/-- Theorem: The tangent line to circle C passing through point P has the equation x + 2y - 10 = 0 -/
theorem tangent_line_equation :
  ∀ x y : ℝ, C x y → (x = P.1 ∧ y = P.2) → tangent_line x y :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l1994_199405


namespace NUMINAMATH_CALUDE_apples_given_to_neighbor_l1994_199409

theorem apples_given_to_neighbor (initial_apples remaining_apples : ℕ) 
  (h1 : initial_apples = 127)
  (h2 : remaining_apples = 39) :
  initial_apples - remaining_apples = 88 := by
sorry

end NUMINAMATH_CALUDE_apples_given_to_neighbor_l1994_199409


namespace NUMINAMATH_CALUDE_min_product_of_three_l1994_199472

def S : Finset Int := {-8, -6, -4, 0, 3, 5, 7}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hdiff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∀ x y z, x ∈ S → y ∈ S → z ∈ S → x ≠ y ∧ y ≠ z ∧ x ≠ z → a * b * c ≥ x * y * z → a * b * c ≥ -280 :=
sorry

end NUMINAMATH_CALUDE_min_product_of_three_l1994_199472


namespace NUMINAMATH_CALUDE_bounded_sequence_with_recurrence_is_constant_two_l1994_199429

def is_bounded_sequence (a : ℕ → ℕ) : Prop :=
  ∃ M : ℕ, ∀ n, a n ≤ M

def satisfies_recurrence (a : ℕ → ℕ) : Prop :=
  ∀ n ≥ 3, a n = (a (n - 1) + a (n - 2)) / Nat.gcd (a (n - 1)) (a (n - 2))

theorem bounded_sequence_with_recurrence_is_constant_two (a : ℕ → ℕ) 
  (h_bounded : is_bounded_sequence a)
  (h_recurrence : satisfies_recurrence a) :
  ∀ n, a n = 2 :=
by sorry

end NUMINAMATH_CALUDE_bounded_sequence_with_recurrence_is_constant_two_l1994_199429


namespace NUMINAMATH_CALUDE_count_of_six_from_100_to_999_l1994_199462

/-- Count of digit 6 in a specific place (units, tens, or hundreds) for numbers from 100 to 999 -/
def count_digit_in_place (place : Nat) : Nat :=
  if place = 2 then 100 else 90

/-- Total count of digit 6 in all places for numbers from 100 to 999 -/
def total_count_of_six : Nat :=
  count_digit_in_place 0 + count_digit_in_place 1 + count_digit_in_place 2

/-- Theorem: The digit 6 appears 280 times when writing integers from 100 through 999 inclusive -/
theorem count_of_six_from_100_to_999 : total_count_of_six = 280 := by
  sorry

end NUMINAMATH_CALUDE_count_of_six_from_100_to_999_l1994_199462


namespace NUMINAMATH_CALUDE_g_at_negative_two_l1994_199416

-- Define the function g
def g (x : ℚ) : ℚ := (2 * x - 3) / (4 * x + 5)

-- State the theorem
theorem g_at_negative_two : g (-2) = 7/3 := by sorry

end NUMINAMATH_CALUDE_g_at_negative_two_l1994_199416


namespace NUMINAMATH_CALUDE_solve_for_c_l1994_199444

theorem solve_for_c (m b d c : ℝ) (h : m = (b * d * c) / (d + c)) :
  c = (m * d) / (b * d - m) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_c_l1994_199444


namespace NUMINAMATH_CALUDE_grocery_bill_calculation_l1994_199435

/-- Calculates the new total bill for a grocery delivery order with item substitutions -/
theorem grocery_bill_calculation
  (original_order : ℝ)
  (tomatoes_old tomatoes_new : ℝ)
  (lettuce_old lettuce_new : ℝ)
  (celery_old celery_new : ℝ)
  (delivery_and_tip : ℝ)
  (h1 : original_order = 25)
  (h2 : tomatoes_old = 0.99)
  (h3 : tomatoes_new = 2.20)
  (h4 : lettuce_old = 1.00)
  (h5 : lettuce_new = 1.75)
  (h6 : celery_old = 1.96)
  (h7 : celery_new = 2.00)
  (h8 : delivery_and_tip = 8.00) :
  original_order + (tomatoes_new - tomatoes_old) + (lettuce_new - lettuce_old) +
  (celery_new - celery_old) + delivery_and_tip = 35 :=
by sorry

end NUMINAMATH_CALUDE_grocery_bill_calculation_l1994_199435


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1994_199408

/-- The number of chess players in the tournament. -/
def n : ℕ := 21

/-- The score of the winner. -/
def winner_score (n : ℕ) : ℚ := 3/4 * (n - 1)

/-- The total score of all games in the tournament. -/
def total_score (n : ℕ) : ℚ := 1/2 * n * (n - 1)

/-- The main theorem stating the conditions and the result of the chess tournament. -/
theorem chess_tournament_players :
  (∀ (m : ℕ), m > 1 →
    (winner_score m = 1/13 * (total_score m - winner_score m)) →
    m = n) ∧
  n > 1 := by sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1994_199408


namespace NUMINAMATH_CALUDE_medicine_percentage_l1994_199487

/-- Proves that the percentage of income spent on medicines is 15% --/
theorem medicine_percentage (income : ℕ) (household_percent : ℚ) (clothes_percent : ℚ) (savings : ℕ)
  (h1 : income = 90000)
  (h2 : household_percent = 50 / 100)
  (h3 : clothes_percent = 25 / 100)
  (h4 : savings = 9000) :
  (income - (household_percent * income + clothes_percent * income + savings)) / income = 15 / 100 := by
  sorry

end NUMINAMATH_CALUDE_medicine_percentage_l1994_199487


namespace NUMINAMATH_CALUDE_negation_even_prime_l1994_199457

theorem negation_even_prime :
  (¬ ∃ n : ℕ, Even n ∧ Prime n) ↔ (∀ n : ℕ, Even n → ¬ Prime n) :=
by sorry

end NUMINAMATH_CALUDE_negation_even_prime_l1994_199457


namespace NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l1994_199450

def q (x : ℚ) : ℚ := (17 * x^3 - 30 * x^2 + x + 12) / 6

theorem cubic_polynomial_satisfies_conditions :
  q (-1) = -6 ∧ q 2 = 5 ∧ q 0 = 2 ∧ q 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomial_satisfies_conditions_l1994_199450


namespace NUMINAMATH_CALUDE_roberto_outfits_l1994_199478

/-- The number of different outfits Roberto can assemble -/
def number_of_outfits (trousers shirts jackets : ℕ) : ℕ := trousers * shirts * jackets

/-- Theorem stating the number of outfits Roberto can assemble -/
theorem roberto_outfits : number_of_outfits 5 5 3 = 75 := by
  sorry

end NUMINAMATH_CALUDE_roberto_outfits_l1994_199478


namespace NUMINAMATH_CALUDE_function_max_min_difference_l1994_199481

theorem function_max_min_difference (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^x
  (∃ (max min : ℝ), (∀ x ∈ Set.Icc 1 2, f x ≤ max ∧ f x ≥ min) ∧ max - min = a / 2) →
  a = 3/2 ∨ a = 1/2 := by
sorry

end NUMINAMATH_CALUDE_function_max_min_difference_l1994_199481


namespace NUMINAMATH_CALUDE_sharons_drive_distance_l1994_199424

theorem sharons_drive_distance (usual_time : ℝ) (actual_time : ℝ) 
  (h1 : usual_time = 180)
  (h2 : actual_time = 300)
  (h3 : ∃ (usual_speed : ℝ), 
    actual_time = 
      (1/3 * usual_time) + 
      (1/3 * usual_time * usual_speed / (usual_speed - 25)) + 
      (1/3 * usual_time * usual_speed / (usual_speed + 10)))
  : ∃ (distance : ℝ), distance = 135 := by
  sorry

end NUMINAMATH_CALUDE_sharons_drive_distance_l1994_199424


namespace NUMINAMATH_CALUDE_janelle_gave_six_green_marbles_l1994_199445

/-- Represents the number of marbles Janelle has and gives away. -/
structure MarbleCount where
  initialGreen : Nat
  blueBags : Nat
  marblesPerBag : Nat
  giftBlue : Nat
  finalTotal : Nat

/-- Calculates the number of green marbles Janelle gave to her friend. -/
def greenMarblesGiven (m : MarbleCount) : Nat :=
  m.initialGreen - (m.finalTotal - (m.blueBags * m.marblesPerBag - m.giftBlue))

/-- Theorem stating that Janelle gave 6 green marbles to her friend. -/
theorem janelle_gave_six_green_marbles (m : MarbleCount) 
    (h1 : m.initialGreen = 26)
    (h2 : m.blueBags = 6)
    (h3 : m.marblesPerBag = 10)
    (h4 : m.giftBlue = 8)
    (h5 : m.finalTotal = 72) :
  greenMarblesGiven m = 6 := by
  sorry

#eval greenMarblesGiven { initialGreen := 26, blueBags := 6, marblesPerBag := 10, giftBlue := 8, finalTotal := 72 }

end NUMINAMATH_CALUDE_janelle_gave_six_green_marbles_l1994_199445


namespace NUMINAMATH_CALUDE_student_count_l1994_199453

theorem student_count (total_pencils : ℕ) (pencils_per_student : ℕ) 
  (h1 : total_pencils = 18) 
  (h2 : pencils_per_student = 9) : 
  total_pencils / pencils_per_student = 2 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l1994_199453


namespace NUMINAMATH_CALUDE_no_primes_in_range_l1994_199414

theorem no_primes_in_range (n : ℕ) (hn : n > 1) : 
  ∀ k ∈ Set.Ioo (n.factorial) (n.factorial + n + 1), ¬ Nat.Prime k := by
sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l1994_199414


namespace NUMINAMATH_CALUDE_sum_of_areas_equals_100_l1994_199494

-- Define the circle radius
def circle_radius : ℝ := 5

-- Define the maximum rectangle inscribed in the circle
def max_rectangle_area (r : ℝ) : ℝ := 2 * r^2

-- Define the maximum parallelogram circumscribed around the circle
def max_parallelogram_area (r : ℝ) : ℝ := 4 * r^2

-- Theorem statement
theorem sum_of_areas_equals_100 :
  max_rectangle_area circle_radius + max_parallelogram_area circle_radius = 100 := by
  sorry

#eval max_rectangle_area circle_radius + max_parallelogram_area circle_radius

end NUMINAMATH_CALUDE_sum_of_areas_equals_100_l1994_199494


namespace NUMINAMATH_CALUDE_original_paint_intensity_l1994_199443

theorem original_paint_intensity 
  (f : ℝ) 
  (h1 : f = 2/3)
  (h2 : (1 - f) * I + f * 0.3 = 0.4) : 
  I = 0.6 :=
by sorry

end NUMINAMATH_CALUDE_original_paint_intensity_l1994_199443


namespace NUMINAMATH_CALUDE_grade_assignment_count_l1994_199497

theorem grade_assignment_count : 
  (Nat.choose 12 2) * (3^10) = 3906234 := by sorry

end NUMINAMATH_CALUDE_grade_assignment_count_l1994_199497


namespace NUMINAMATH_CALUDE_jumping_contest_l1994_199417

/-- The jumping contest problem -/
theorem jumping_contest (grasshopper_jump : ℕ) (frog_extra : ℕ) (mouse_extra : ℕ) 
  (h1 : grasshopper_jump = 19)
  (h2 : frog_extra = 10)
  (h3 : mouse_extra = 20) :
  (grasshopper_jump + frog_extra + mouse_extra) - grasshopper_jump = 30 := by
  sorry


end NUMINAMATH_CALUDE_jumping_contest_l1994_199417


namespace NUMINAMATH_CALUDE_plane_equation_correct_l1994_199404

/-- A plane equation represented by integers A, B, C, and D -/
structure PlaneEquation where
  A : Int
  B : Int
  C : Int
  D : Int
  A_pos : A > 0
  gcd_one : Nat.gcd (Nat.gcd (Int.natAbs A) (Int.natAbs B)) (Nat.gcd (Int.natAbs C) (Int.natAbs D)) = 1

/-- Check if a point (x, y, z) lies on a plane -/
def lies_on_plane (p : PlaneEquation) (x y z : ℝ) : Prop :=
  p.A * x + p.B * y + p.C * z + p.D = 0

/-- Check if two planes are perpendicular -/
def perpendicular_planes (p1 p2 : PlaneEquation) : Prop :=
  p1.A * p2.A + p1.B * p2.B + p1.C * p2.C = 0

theorem plane_equation_correct (p : PlaneEquation) 
  (h1 : p.A = 2 ∧ p.B = -2 ∧ p.C = 1 ∧ p.D = 1) 
  (h2 : lies_on_plane p 0 2 3) 
  (h3 : lies_on_plane p 2 0 3) 
  (h4 : perpendicular_planes p { A := 1, B := -1, C := 4, D := -7, A_pos := by norm_num, gcd_one := by norm_num }) : 
  p.A = 2 ∧ p.B = -2 ∧ p.C = 1 ∧ p.D = 1 := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_correct_l1994_199404


namespace NUMINAMATH_CALUDE_cube_face_sum_l1994_199420

theorem cube_face_sum (a b c d e f : ℕ+) : 
  (a * b * c + a * e * c + a * b * f + a * e * f + 
   d * b * c + d * e * c + d * b * f + d * e * f = 1089) → 
  (a + b + c + d + e + f = 31) := by
sorry

end NUMINAMATH_CALUDE_cube_face_sum_l1994_199420


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1994_199452

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 2*x + m - 2 = 0 ∧ y^2 - 2*y + m - 2 = 0) → m < 3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1994_199452


namespace NUMINAMATH_CALUDE_hexagon_segment_probability_l1994_199418

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The total number of segments (sides and diagonals) in a regular hexagon -/
def total_segments : ℕ := num_sides + num_diagonals

/-- The number of different diagonal lengths in a regular hexagon -/
def num_diagonal_lengths : ℕ := 3

/-- The number of diagonals of each length in a regular hexagon -/
def diagonals_per_length : ℕ := num_diagonals / num_diagonal_lengths

theorem hexagon_segment_probability : 
  (num_sides * (num_sides - 1) + num_diagonals * (diagonals_per_length - 1)) / 
  (total_segments * (total_segments - 1)) = 11 / 35 := by
sorry

end NUMINAMATH_CALUDE_hexagon_segment_probability_l1994_199418


namespace NUMINAMATH_CALUDE_percentage_of_green_caps_l1994_199490

def total_caps : ℕ := 125
def red_caps : ℕ := 50

theorem percentage_of_green_caps :
  (total_caps - red_caps : ℚ) / total_caps * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_green_caps_l1994_199490


namespace NUMINAMATH_CALUDE_rap_song_requests_l1994_199465

/-- Represents the number of song requests for different genres --/
structure SongRequests where
  total : ℕ
  electropop : ℕ
  dance : ℕ
  rock : ℕ
  oldies : ℕ
  dj_choice : ℕ
  rap : ℕ

/-- Theorem stating the number of rap song requests --/
theorem rap_song_requests (req : SongRequests) : req.rap = 2 :=
  by
  have h1 : req.total = 30 := by sorry
  have h2 : req.electropop = req.total / 2 := by sorry
  have h3 : req.dance = req.electropop / 3 := by sorry
  have h4 : req.rock = 5 := by sorry
  have h5 : req.oldies = req.rock - 3 := by sorry
  have h6 : req.dj_choice = req.oldies / 2 := by sorry
  have h7 : req.total = req.electropop + req.dance + req.rock + req.oldies + req.dj_choice + req.rap := by sorry
  sorry

end NUMINAMATH_CALUDE_rap_song_requests_l1994_199465


namespace NUMINAMATH_CALUDE_class_average_problem_l1994_199437

/-- Given a class where:
  - 20% of students average 80% on a test
  - 50% of students average X% on a test
  - 30% of students average 40% on a test
  - The overall class average is 58%
  Prove that X = 60 -/
theorem class_average_problem (X : ℝ) : 
  0.2 * 80 + 0.5 * X + 0.3 * 40 = 58 → X = 60 := by
  sorry

end NUMINAMATH_CALUDE_class_average_problem_l1994_199437


namespace NUMINAMATH_CALUDE_parabola_point_distances_l1994_199406

theorem parabola_point_distances (a c : ℝ) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) :
  a > 0 →
  y₂ = -9 * a + c →
  y₁ = a * x₁^2 - 6 * a * x₁ + c →
  y₂ = a * x₂^2 - 6 * a * x₂ + c →
  y₃ = a * x₃^2 - 6 * a * x₃ + c →
  y₁ > y₃ →
  y₃ ≥ y₂ →
  |x₁ - x₂| > |x₂ - x₃| :=
by sorry

end NUMINAMATH_CALUDE_parabola_point_distances_l1994_199406


namespace NUMINAMATH_CALUDE_charles_earnings_correct_l1994_199419

/-- Calculates Charles' earnings after tax deduction based on his housesitting and dog walking activities. -/
def charles_earnings : ℝ :=
  let housesitting_rate : ℝ := 15
  let labrador_rate : ℝ := 22
  let golden_retriever_rate : ℝ := 25
  let german_shepherd_rate : ℝ := 30
  let housesitting_hours : ℝ := 10
  let labrador_hours : ℝ := 3
  let golden_retriever_hours : ℝ := 2
  let german_shepherd_hours : ℝ := 1.5
  let tax_rate : ℝ := 0.1

  let total_before_tax : ℝ := 
    housesitting_rate * housesitting_hours +
    labrador_rate * labrador_hours * 2 +
    golden_retriever_rate * golden_retriever_hours +
    german_shepherd_rate * german_shepherd_hours

  total_before_tax * (1 - tax_rate)

theorem charles_earnings_correct : charles_earnings = 339.30 := by
  sorry

end NUMINAMATH_CALUDE_charles_earnings_correct_l1994_199419


namespace NUMINAMATH_CALUDE_a_1_greater_than_500_l1994_199492

theorem a_1_greater_than_500 (a : Fin 10000 → ℕ)
  (h1 : ∀ i j, i < j → a i < a j)
  (h2 : a 0 > 0)
  (h3 : a 9999 < 20000)
  (h4 : ∀ i j, i < j → Nat.gcd (a i) (a j) < a i) :
  500 < a 0 := by
  sorry

end NUMINAMATH_CALUDE_a_1_greater_than_500_l1994_199492


namespace NUMINAMATH_CALUDE_tangent_line_cubic_l1994_199432

/-- The equation of the tangent line to y = x³ at (1, 1) is 3x - y - 2 = 0 -/
theorem tangent_line_cubic (x y : ℝ) : 
  (y = x^3) → -- Curve equation
  (∃ (m b : ℝ), ∀ (x' y' : ℝ), 
    (y' = m * (x' - 1) + 1) ∧ -- Point-slope form of tangent line
    (y' = x'^3 → x' = 1) → -- Tangent point (1, 1)
    (3 * x' - y' - 2 = 0)) -- Equation of the tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_l1994_199432


namespace NUMINAMATH_CALUDE_eating_time_theorem_l1994_199410

/-- Represents the eating rate of a character in jars per minute -/
structure EatingRate :=
  (condensed_milk : ℚ)
  (honey : ℚ)

/-- Calculates the time taken to eat a certain amount of food given the eating rate -/
def time_to_eat (rate : EatingRate) (condensed_milk : ℚ) (honey : ℚ) : ℚ :=
  (condensed_milk / rate.condensed_milk) + (honey / rate.honey)

/-- Calculates the combined eating rate of two characters -/
def combined_rate (rate1 rate2 : EatingRate) : EatingRate :=
  { condensed_milk := rate1.condensed_milk + rate2.condensed_milk,
    honey := rate1.honey + rate2.honey }

theorem eating_time_theorem (pooh_rate piglet_rate : EatingRate) : 
  (time_to_eat pooh_rate 3 1 = 25) →
  (time_to_eat piglet_rate 3 1 = 55) →
  (time_to_eat pooh_rate 1 3 = 35) →
  (time_to_eat piglet_rate 1 3 = 85) →
  (time_to_eat (combined_rate pooh_rate piglet_rate) 6 0 = 20) := by
  sorry

end NUMINAMATH_CALUDE_eating_time_theorem_l1994_199410


namespace NUMINAMATH_CALUDE_inequality_proof_l1994_199431

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (1 / (a^3 * (b + c))) + (1 / (b^3 * (c + a))) + (1 / (c^3 * (a + b))) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1994_199431


namespace NUMINAMATH_CALUDE_inverse_matrices_sum_l1994_199474

def A (a b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![a, 2, b; 3, 3, 4; c, 6, d]

def B (e f g h : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![-4, e, -12; f, -14, g; 3, h, 5]

theorem inverse_matrices_sum (a b c d e f g h : ℝ) :
  (A a b c d) * (B e f g h) = 1 →
  a + b + c + d + e + f + g + h = 47 := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrices_sum_l1994_199474


namespace NUMINAMATH_CALUDE_exactly_one_correct_proposition_l1994_199460

theorem exactly_one_correct_proposition : 
  let prop1 := ∀ (a b : ℝ), a < b → a^2 < b^2
  let prop2 := ∀ (a : ℝ), (∀ (x : ℝ), |x+1| + |x-1| ≥ a) ↔ a ≤ 2
  let prop3 := (¬ ∃ (x : ℝ), x^2 - x > 0) ↔ (∀ (x : ℝ), x^2 - x < 0)
  (¬prop1 ∧ prop2 ∧ ¬prop3) := by sorry

end NUMINAMATH_CALUDE_exactly_one_correct_proposition_l1994_199460


namespace NUMINAMATH_CALUDE_mrs_hilt_initial_money_l1994_199434

/-- Mrs. Hilt's shopping problem -/
theorem mrs_hilt_initial_money :
  ∀ (initial_money toy_truck_cost pencil_case_cost money_left : ℕ),
  toy_truck_cost = 3 →
  pencil_case_cost = 2 →
  money_left = 5 →
  initial_money = toy_truck_cost + pencil_case_cost + money_left →
  initial_money = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_mrs_hilt_initial_money_l1994_199434


namespace NUMINAMATH_CALUDE_parabola_translation_l1994_199413

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h v : ℝ) : Parabola :=
  { a := p.a
    b := -2 * p.a * h + p.b
    c := p.a * h^2 - p.b * h + p.c + v }

theorem parabola_translation (x y : ℝ) :
  let p := Parabola.mk 2 0 0  -- y = 2x^2
  let p_translated := translate p 1 3  -- Translate 1 right, 3 up
  y = 2 * x^2 → y = 2 * (x - 1)^2 + 3 :=
by
  sorry

#check parabola_translation

end NUMINAMATH_CALUDE_parabola_translation_l1994_199413


namespace NUMINAMATH_CALUDE_sphere_surface_area_l1994_199412

/-- The surface area of a sphere with diameter 9 inches is 81π square inches. -/
theorem sphere_surface_area (π : ℝ) (h : π > 0) : 
  let diameter : ℝ := 9
  let radius : ℝ := diameter / 2
  let surface_area : ℝ := 4 * π * radius^2
  surface_area = 81 * π :=
by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l1994_199412


namespace NUMINAMATH_CALUDE_dice_sum_probabilities_l1994_199459

/-- The probability of rolling a sum of 15 with 3 dice -/
def p_sum_15 : ℚ := 10 / 216

/-- The probability of rolling a sum of at least 15 with 3 dice -/
def p_sum_at_least_15 : ℚ := 20 / 216

/-- The minimum number of trials to roll a sum of 15 exactly once with probability > 1/2 -/
def min_trials_sum_15 : ℕ := 15

/-- The minimum number of trials to roll a sum of at least 15 exactly once with probability > 1/2 -/
def min_trials_sum_at_least_15 : ℕ := 8

theorem dice_sum_probabilities :
  (1 - (1 - p_sum_15) ^ min_trials_sum_15 > 1/2) ∧
  (∀ n : ℕ, n < min_trials_sum_15 → 1 - (1 - p_sum_15) ^ n ≤ 1/2) ∧
  (1 - (1 - p_sum_at_least_15) ^ min_trials_sum_at_least_15 > 1/2) ∧
  (∀ n : ℕ, n < min_trials_sum_at_least_15 → 1 - (1 - p_sum_at_least_15) ^ n ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_dice_sum_probabilities_l1994_199459


namespace NUMINAMATH_CALUDE_range_of_a_inequality_proof_l1994_199463

-- Define the function f
def f (x : ℝ) : ℝ := 3 * |x - 1| + |3 * x + 7|

-- Part 1
theorem range_of_a (a : ℝ) : 
  (∀ x, f x ≥ a^2 - 3*a) → -2 ≤ a ∧ a ≤ 5 := by sorry

-- Part 2
theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 3) :
  ∀ x, Real.sqrt (a + 1) + Real.sqrt (b + 1) ≤ Real.sqrt (f x) := by sorry

end NUMINAMATH_CALUDE_range_of_a_inequality_proof_l1994_199463


namespace NUMINAMATH_CALUDE_no_four_digit_numbers_divisible_by_5_ending_47_l1994_199442

theorem no_four_digit_numbers_divisible_by_5_ending_47 :
  ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 →
  n % 100 = 47 →
  ¬(n % 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_four_digit_numbers_divisible_by_5_ending_47_l1994_199442


namespace NUMINAMATH_CALUDE_museum_artifact_count_l1994_199401

/-- Represents a museum with paintings and artifacts --/
structure Museum where
  total_wings : ℕ
  painting_wings : ℕ
  large_painting_count : ℕ
  small_paintings_per_wing : ℕ
  artifact_multiplier : ℕ

/-- Calculates the number of artifacts in each artifact wing --/
def artifacts_per_wing (m : Museum) : ℕ :=
  let total_paintings := m.large_painting_count + (m.painting_wings - 1) * m.small_paintings_per_wing
  let total_artifacts := ((m.artifact_multiplier * total_paintings) / 8) * 8
  let artifact_wings := m.total_wings - m.painting_wings
  total_artifacts / artifact_wings

/-- Theorem: In the given museum setup, each artifact wing contains 34 artifacts --/
theorem museum_artifact_count (m : Museum) 
  (h1 : m.total_wings = 12)
  (h2 : m.painting_wings = 4)
  (h3 : m.large_painting_count = 1)
  (h4 : m.small_paintings_per_wing = 15)
  (h5 : m.artifact_multiplier = 6) :
  artifacts_per_wing m = 34 := by
  sorry

end NUMINAMATH_CALUDE_museum_artifact_count_l1994_199401


namespace NUMINAMATH_CALUDE_function_simplification_l1994_199479

/-- Given f(x) = (2x + 1)^5 - 5(2x + 1)^4 + 10(2x + 1)^3 - 10(2x + 1)^2 + 5(2x + 1) - 1,
    prove that f(x) = 32x^5 for all real x -/
theorem function_simplification (x : ℝ) : 
  let f : ℝ → ℝ := λ x => (2*x + 1)^5 - 5*(2*x + 1)^4 + 10*(2*x + 1)^3 - 10*(2*x + 1)^2 + 5*(2*x + 1) - 1
  f x = 32*x^5 := by
  sorry

end NUMINAMATH_CALUDE_function_simplification_l1994_199479


namespace NUMINAMATH_CALUDE_one_fourth_of_eight_x_plus_two_l1994_199433

theorem one_fourth_of_eight_x_plus_two (x : ℝ) : (1 / 4) * (8 * x + 2) = 2 * x + 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_fourth_of_eight_x_plus_two_l1994_199433


namespace NUMINAMATH_CALUDE_inequality_proof_l1994_199480

theorem inequality_proof (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) 
  (h_sum : x * y + y * z + z * x = 1) : 
  x * y * z * (x + y + z) ≤ 1 / 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1994_199480


namespace NUMINAMATH_CALUDE_fraction_value_l1994_199482

theorem fraction_value : (12345 : ℕ) / (1 + 2 + 3 + 4 + 5) = 823 := by
  sorry

end NUMINAMATH_CALUDE_fraction_value_l1994_199482


namespace NUMINAMATH_CALUDE_lcm_gcd_problem_l1994_199489

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2400 →
  Nat.gcd a b = 30 →
  a = 150 →
  b = 480 := by
sorry

end NUMINAMATH_CALUDE_lcm_gcd_problem_l1994_199489


namespace NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_equation_solutions_l1994_199449

theorem quadratic_equation_solutions (x : ℝ) :
  (x^2 + 5*x - 1 = 0) ↔ (x = (-5 + Real.sqrt 29) / 2 ∨ x = (-5 - Real.sqrt 29) / 2) :=
sorry

theorem linear_equation_solutions (x : ℝ) :
  (7*x*(5*x + 2) = 6*(5*x + 2)) ↔ (x = -2/5 ∨ x = 6/7) :=
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solutions_linear_equation_solutions_l1994_199449


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1994_199403

theorem imaginary_part_of_z (x y : ℝ) (h : (x - Complex.I) * Complex.I = y + 2 * Complex.I) :
  (x + y * Complex.I).im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1994_199403


namespace NUMINAMATH_CALUDE_election_winning_margin_l1994_199493

/-- Represents an election with two candidates -/
structure Election :=
  (total_votes : ℕ)
  (winner_votes : ℕ)
  (winner_percentage : ℚ)

/-- Calculates the number of votes the winner won by -/
def winning_margin (e : Election) : ℕ :=
  e.winner_votes - (e.total_votes - e.winner_votes)

/-- Theorem stating the winning margin for the given election scenario -/
theorem election_winning_margin :
  ∃ (e : Election),
    e.winner_percentage = 62 / 100 ∧
    e.winner_votes = 899 ∧
    winning_margin e = 348 := by
  sorry

end NUMINAMATH_CALUDE_election_winning_margin_l1994_199493


namespace NUMINAMATH_CALUDE_inequality_proof_l1994_199427

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y + y * z + z * x = 3) : 
  (x + 3) / (y + z) + (y + 3) / (z + x) + (z + 3) / (x + y) + 3 ≥ 
  27 * ((Real.sqrt x + Real.sqrt y + Real.sqrt z)^2) / ((x + y + z)^3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1994_199427


namespace NUMINAMATH_CALUDE_percentage_to_decimal_decimal_representation_of_208_percent_l1994_199484

/-- The decimal representation of a percentage is equal to the percentage divided by 100. -/
theorem percentage_to_decimal (p : ℝ) : p / 100 = p * (1 / 100) := by sorry

/-- The decimal representation of 208% is 2.08. -/
theorem decimal_representation_of_208_percent : (208 : ℝ) / 100 = 2.08 := by sorry

end NUMINAMATH_CALUDE_percentage_to_decimal_decimal_representation_of_208_percent_l1994_199484


namespace NUMINAMATH_CALUDE_f_extrema_l1994_199475

noncomputable def f (x : ℝ) : ℝ := 1 + 3*x - x^3

theorem f_extrema :
  (∃ x : ℝ, f x = -1 ∧ ∀ y : ℝ, f y ≥ -1) ∧
  (∃ x : ℝ, f x = 3 ∧ ∀ y : ℝ, f y ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_f_extrema_l1994_199475


namespace NUMINAMATH_CALUDE_second_stack_height_difference_l1994_199448

def stack_problem (h : ℕ) : Prop :=
  let first_stack := 7
  let second_stack := h
  let third_stack := h + 7
  let fallen_blocks := first_stack + (second_stack - 2) + (third_stack - 3)
  (fallen_blocks = 33) ∧ (second_stack > first_stack)

theorem second_stack_height_difference : ∃ h : ℕ, stack_problem h ∧ (h - 7 = 5) :=
sorry

end NUMINAMATH_CALUDE_second_stack_height_difference_l1994_199448


namespace NUMINAMATH_CALUDE_sock_pairs_count_l1994_199421

/-- Given a number of sock pairs, calculate the number of ways to select two socks
    from different pairs. -/
def nonMatchingSelections (n : ℕ) : ℕ := n * (2 * n - 1)

/-- The problem statement -/
theorem sock_pairs_count : ∃ (n : ℕ), n > 0 ∧ nonMatchingSelections n = 112 :=
  sorry

end NUMINAMATH_CALUDE_sock_pairs_count_l1994_199421


namespace NUMINAMATH_CALUDE_science_books_in_large_box_probability_l1994_199466

def total_textbooks : ℕ := 16
def science_textbooks : ℕ := 4
def box_capacities : List ℕ := [2, 4, 5, 5]

theorem science_books_in_large_box_probability :
  let total_ways := (total_textbooks.choose box_capacities[2]) * 
                    ((total_textbooks - box_capacities[2]).choose box_capacities[2]) * 
                    ((total_textbooks - box_capacities[2] - box_capacities[2]).choose box_capacities[1]) * 
                    1
  let favorable_ways := 2 * (box_capacities[2].choose science_textbooks) * 
                        (total_textbooks - science_textbooks).choose 1 * 
                        ((total_textbooks - box_capacities[2]).choose box_capacities[2]) * 
                        ((total_textbooks - box_capacities[2] - box_capacities[2]).choose box_capacities[1])
  (favorable_ways : ℚ) / total_ways = 5 / 182 := by
  sorry

end NUMINAMATH_CALUDE_science_books_in_large_box_probability_l1994_199466


namespace NUMINAMATH_CALUDE_julies_salary_l1994_199477

/-- Calculates the monthly salary for a worker given specific conditions -/
def monthlySalary (hourlyRate : ℕ) (hoursPerDay : ℕ) (daysPerWeek : ℕ) (missedDays : ℕ) : ℕ :=
  let dailyEarnings := hourlyRate * hoursPerDay
  let weeklyEarnings := dailyEarnings * daysPerWeek
  let monthlyEarnings := weeklyEarnings * 4
  monthlyEarnings - (dailyEarnings * missedDays)

/-- Proves that given the specific conditions, the monthly salary is $920 -/
theorem julies_salary : 
  monthlySalary 5 8 6 1 = 920 := by
  sorry

#eval monthlySalary 5 8 6 1

end NUMINAMATH_CALUDE_julies_salary_l1994_199477


namespace NUMINAMATH_CALUDE_subset_iff_range_l1994_199422

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x : ℝ | x^2 - a*x - 4 ≤ 0}

-- State the theorem
theorem subset_iff_range (a : ℝ) : B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
sorry

end NUMINAMATH_CALUDE_subset_iff_range_l1994_199422


namespace NUMINAMATH_CALUDE_inequality_for_real_numbers_l1994_199498

theorem inequality_for_real_numbers (a b : ℝ) : a * b ≤ ((a + b) / 2) ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_for_real_numbers_l1994_199498


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1994_199407

theorem arithmetic_sequence_terms (a₁ aₙ : ℤ) (d : ℤ) (n : ℕ) : 
  a₁ = 1 ∧ aₙ = -89 ∧ d = -2 ∧ aₙ = a₁ + (n - 1) * d → n = 46 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l1994_199407


namespace NUMINAMATH_CALUDE_equal_intercepts_iff_area_two_iff_l1994_199423

/-- The line equation type -/
structure LineEquation where
  a : ℝ
  eq : ℝ → ℝ → Prop := fun x y => y = -(a + 1) * x + (2 - a)

/-- Equal intercepts condition -/
def has_equal_intercepts (l : LineEquation) : Prop :=
  ∃ x y : ℝ, l.eq x 0 ∧ l.eq 0 y ∧ x = y

/-- Triangle area condition -/
def has_area_two (l : LineEquation) : Prop :=
  abs ((2 - l.a) * (2 - l.a)) / (2 * abs (l.a + 1)) = 2

/-- Theorem for equal intercepts -/
theorem equal_intercepts_iff (l : LineEquation) :
  has_equal_intercepts l ↔ l.a = 2 ∨ l.a = 0 :=
sorry

/-- Theorem for area of 2 -/
theorem area_two_iff (l : LineEquation) :
  has_area_two l ↔ l.a = 8 ∨ l.a = 0 :=
sorry

end NUMINAMATH_CALUDE_equal_intercepts_iff_area_two_iff_l1994_199423


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1994_199488

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 6 ∧ x₂ = -1 - Real.sqrt 6 ∧ 
    x₁^2 + 2*x₁ = 5 ∧ x₂^2 + 2*x₂ = 5) ∧
  (∃ x₁ x₂ : ℝ, x₁ = 1 + Real.sqrt 2 ∧ x₂ = 1 - Real.sqrt 2 ∧ 
    x₁^2 - 2*x₁ - 1 = 0 ∧ x₂^2 - 2*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -5/2 ∧ x₂ = 1 ∧ 
    2*x₁^2 + 3*x₁ - 5 = 0 ∧ 2*x₂^2 + 3*x₂ - 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1994_199488


namespace NUMINAMATH_CALUDE_erica_safari_elephants_l1994_199468

/-- The number of elephants Erica saw on her safari --/
def elephants_seen (total_animals : ℕ) (lions_saturday : ℕ) (animals_sunday_monday : ℕ) : ℕ :=
  total_animals - lions_saturday - animals_sunday_monday

/-- Theorem stating the number of elephants Erica saw on Saturday --/
theorem erica_safari_elephants :
  elephants_seen 20 3 15 = 2 := by
  sorry

end NUMINAMATH_CALUDE_erica_safari_elephants_l1994_199468


namespace NUMINAMATH_CALUDE_fraction_calculation_l1994_199491

theorem fraction_calculation : (16 : ℚ) / 42 * 18 / 27 - 4 / 21 = 4 / 63 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l1994_199491


namespace NUMINAMATH_CALUDE_proportional_segment_length_l1994_199496

/-- Triangle ABC with sides a, b, c, and an interior point P creating parallel segments of length d -/
structure ProportionalTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- The condition that the parallel segments split the sides proportionally -/
def is_proportional (t : ProportionalTriangle) : Prop :=
  t.d / t.c * t.b + t.d / t.a * t.b = t.b

/-- The theorem stating that for the given triangle, the proportional segments have length 28.25 -/
theorem proportional_segment_length 
  (t : ProportionalTriangle) 
  (h1 : t.a = 500) 
  (h2 : t.b = 550) 
  (h3 : t.c = 650) 
  (h4 : is_proportional t) : 
  t.d = 28.25 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segment_length_l1994_199496


namespace NUMINAMATH_CALUDE_system_solution_l1994_199458

theorem system_solution (a b c : ℂ) : 
  (a^2 + a*b + c = 0 ∧ b^2 + b*c + a = 0 ∧ c^2 + c*a + b = 0) → 
  ((a = 0 ∧ b = 0 ∧ c = 0) ∨ (a = -1/2 ∧ b = -1/2 ∧ c = -1/2)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1994_199458


namespace NUMINAMATH_CALUDE_angle_around_point_l1994_199436

theorem angle_around_point (a b : ℝ) (h1 : a + b + 200 = 360) (h2 : a = b) : a = 80 := by
  sorry

end NUMINAMATH_CALUDE_angle_around_point_l1994_199436


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1994_199455

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 → 45 ∣ n → 45 ≤ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1994_199455


namespace NUMINAMATH_CALUDE_count_numbers_with_three_between_100_and_499_l1994_199425

def count_numbers_with_three (lower_bound upper_bound : ℕ) : ℕ :=
  let first_digit_three := 100
  let second_digit_three := 40
  let third_digit_three := 40
  let all_digits_three := 1
  first_digit_three + second_digit_three + third_digit_three - all_digits_three

theorem count_numbers_with_three_between_100_and_499 :
  count_numbers_with_three 100 499 = 181 := by
  sorry

#eval count_numbers_with_three 100 499

end NUMINAMATH_CALUDE_count_numbers_with_three_between_100_and_499_l1994_199425


namespace NUMINAMATH_CALUDE_abs_diff_given_prod_and_sum_l1994_199454

theorem abs_diff_given_prod_and_sum (a b : ℝ) (h1 : a * b = 6) (h2 : a + b = 7) : |a - b| = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_diff_given_prod_and_sum_l1994_199454


namespace NUMINAMATH_CALUDE_tangent_line_properties_l1994_199441

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^3 - x
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 1

-- Define the tangent line condition
def tangent_condition (x₁ : ℝ) (a : ℝ) : Prop :=
  ∃ x₂ : ℝ, f' x₁ * (x₂ - x₁) + f x₁ = g a x₂ ∧ f' x₁ = 2 * x₂

-- State the theorem
theorem tangent_line_properties :
  (∀ x₁ a : ℝ, tangent_condition x₁ a → (x₁ = -1 → a = 3)) ∧
  (∀ a : ℝ, (∃ x₁ : ℝ, tangent_condition x₁ a) → a ≥ -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_properties_l1994_199441


namespace NUMINAMATH_CALUDE_brick_width_is_four_l1994_199486

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: A brick with length 8 cm, height 2 cm, and surface area 112 cm² has a width of 4 cm -/
theorem brick_width_is_four :
  ∀ w : ℝ, surface_area 8 w 2 = 112 → w = 4 := by
  sorry

end NUMINAMATH_CALUDE_brick_width_is_four_l1994_199486


namespace NUMINAMATH_CALUDE_average_value_of_series_l1994_199415

theorem average_value_of_series (z : ℝ) :
  let series := [4*z, 6*z, 9*z, 13.5*z, 20.25*z]
  (series.sum / series.length : ℝ) = 10.55 * z :=
by sorry

end NUMINAMATH_CALUDE_average_value_of_series_l1994_199415


namespace NUMINAMATH_CALUDE_product_remainder_by_ten_l1994_199400

theorem product_remainder_by_ten (a b c : ℕ) (ha : a = 1824) (hb : b = 5435) (hc : c = 80525) : 
  (a * b * c) % 10 = 0 := by
sorry

end NUMINAMATH_CALUDE_product_remainder_by_ten_l1994_199400


namespace NUMINAMATH_CALUDE_billy_caught_three_fish_l1994_199467

/-- Represents the number of fish caught by each family member and other relevant information --/
structure FishingTrip where
  ben_fish : ℕ
  judy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ
  filets_per_fish : ℕ

/-- Calculates the number of fish Billy caught given the fishing trip information --/
def billy_fish_count (trip : FishingTrip) : ℕ :=
  let total_kept := trip.total_filets / trip.filets_per_fish
  let total_caught := total_kept + trip.thrown_back
  total_caught - trip.ben_fish - trip.judy_fish - trip.jim_fish - trip.susie_fish

/-- Theorem stating that Billy caught 3 fish given the specific conditions of the fishing trip --/
theorem billy_caught_three_fish (trip : FishingTrip) 
  (h1 : trip.ben_fish = 4)
  (h2 : trip.judy_fish = 1)
  (h3 : trip.jim_fish = 2)
  (h4 : trip.susie_fish = 5)
  (h5 : trip.thrown_back = 3)
  (h6 : trip.total_filets = 24)
  (h7 : trip.filets_per_fish = 2) :
  billy_fish_count trip = 3 := by
  sorry

#eval billy_fish_count ⟨4, 1, 2, 5, 3, 24, 2⟩

end NUMINAMATH_CALUDE_billy_caught_three_fish_l1994_199467


namespace NUMINAMATH_CALUDE_bucket_weight_l1994_199428

theorem bucket_weight (p q : ℝ) 
  (h1 : ∃ x y : ℝ, x + 3/4 * y = p ∧ x + 1/3 * y = q) : 
  ∃ w : ℝ, w = (5*q - p)/5 ∧ 
  ∀ x y : ℝ, (x + 3/4 * y = p ∧ x + 1/3 * y = q) → x + 1/4 * y = w :=
sorry

end NUMINAMATH_CALUDE_bucket_weight_l1994_199428


namespace NUMINAMATH_CALUDE_janet_time_saved_l1994_199473

/-- The number of minutes Janet spends looking for her keys daily -/
def looking_time : ℕ := 8

/-- The number of minutes Janet spends complaining after finding her keys daily -/
def complaining_time : ℕ := 3

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total time Janet saves in a week if she stops losing her keys -/
def time_saved : ℕ := (looking_time + complaining_time) * days_in_week

theorem janet_time_saved : time_saved = 77 := by
  sorry

end NUMINAMATH_CALUDE_janet_time_saved_l1994_199473


namespace NUMINAMATH_CALUDE_hongfu_supermarket_salt_purchase_l1994_199476

/-- The number of bags of salt initially purchased by Hongfu Supermarket -/
def initial_salt : ℕ := 1200

/-- The fraction of salt sold in the first month -/
def first_month_sold : ℚ := 2/5

/-- The number of bags of salt sold in the second month -/
def second_month_sold : ℕ := 420

/-- The ratio of sold salt to remaining salt after the second month -/
def sold_to_remaining_ratio : ℚ := 3

theorem hongfu_supermarket_salt_purchase :
  initial_salt = 1200 ∧
  (initial_salt : ℚ) * first_month_sold + second_month_sold =
    sold_to_remaining_ratio * (initial_salt - (initial_salt : ℚ) * first_month_sold - second_month_sold) :=
by sorry

end NUMINAMATH_CALUDE_hongfu_supermarket_salt_purchase_l1994_199476


namespace NUMINAMATH_CALUDE_percentage_difference_l1994_199483

theorem percentage_difference (a b : ℝ) 
  (ha : 3 = 0.15 * a) 
  (hb : 3 = 0.25 * b) : 
  a - b = 8 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l1994_199483


namespace NUMINAMATH_CALUDE_power_of_three_expression_l1994_199471

theorem power_of_three_expression : 3^3 - 3^2 + 3^1 - 3^0 = 20 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_expression_l1994_199471


namespace NUMINAMATH_CALUDE_range_of_f_l1994_199464

def f (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem range_of_f :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 0 ∧ f x = y) ↔ y ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1994_199464


namespace NUMINAMATH_CALUDE_eight_mile_taxi_cost_l1994_199402

/-- Calculates the cost of a taxi ride given the fixed cost, cost per mile, and distance traveled. -/
def taxi_cost (fixed_cost : ℚ) (cost_per_mile : ℚ) (distance : ℚ) : ℚ :=
  fixed_cost + cost_per_mile * distance

/-- Theorem: The cost of an 8-mile taxi ride with a $2.00 fixed cost and $0.30 per mile is $4.40. -/
theorem eight_mile_taxi_cost :
  taxi_cost 2 (3/10) 8 = 44/10 := by
  sorry

end NUMINAMATH_CALUDE_eight_mile_taxi_cost_l1994_199402


namespace NUMINAMATH_CALUDE_equal_commissions_l1994_199495

/-- The list price of an item that satisfies the given conditions -/
def list_price : ℝ := 40

/-- Alice's selling price -/
def alice_price (x : ℝ) : ℝ := x - 15

/-- Bob's selling price -/
def bob_price (x : ℝ) : ℝ := x - 25

/-- Charlie's selling price -/
def charlie_price (x : ℝ) : ℝ := x - 20

/-- Alice's commission -/
def alice_commission (x : ℝ) : ℝ := 0.15 * alice_price x

/-- Bob's commission -/
def bob_commission (x : ℝ) : ℝ := 0.25 * bob_price x

/-- Charlie's commission -/
def charlie_commission (x : ℝ) : ℝ := 0.20 * charlie_price x

theorem equal_commissions :
  alice_commission list_price = bob_commission list_price ∧
  bob_commission list_price = charlie_commission list_price := by
  sorry

end NUMINAMATH_CALUDE_equal_commissions_l1994_199495


namespace NUMINAMATH_CALUDE_sum_nth_from_both_ends_l1994_199411

/-- A set of consecutive integers -/
structure ConsecutiveIntegerSet where
  first : ℤ
  last : ℤ
  h_consecutive : last ≥ first

/-- The median of a set of consecutive integers -/
def median (s : ConsecutiveIntegerSet) : ℚ :=
  (s.first + s.last : ℚ) / 2

/-- The nth number from the beginning of the set -/
def nth_from_beginning (s : ConsecutiveIntegerSet) (n : ℕ) : ℤ :=
  s.first + n - 1

/-- The nth number from the end of the set -/
def nth_from_end (s : ConsecutiveIntegerSet) (n : ℕ) : ℤ :=
  s.last - n + 1

theorem sum_nth_from_both_ends (s : ConsecutiveIntegerSet) (n : ℕ) 
  (h_median : median s = 60) :
  nth_from_beginning s n + nth_from_end s n = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_nth_from_both_ends_l1994_199411


namespace NUMINAMATH_CALUDE_complex_sum_eighth_power_l1994_199430

open Complex

theorem complex_sum_eighth_power :
  ((-1 + I) / 2) ^ 8 + ((-1 - I) / 2) ^ 8 = (1 : ℂ) / 8 := by sorry

end NUMINAMATH_CALUDE_complex_sum_eighth_power_l1994_199430


namespace NUMINAMATH_CALUDE_total_weight_is_120_pounds_l1994_199426

/-- The weight of a single dumbbell in pounds -/
def dumbbell_weight : ℕ := 20

/-- The number of dumbbells initially set up -/
def initial_dumbbells : ℕ := 4

/-- The number of additional dumbbells Parker adds -/
def added_dumbbells : ℕ := 2

/-- The total number of dumbbells Parker uses -/
def total_dumbbells : ℕ := initial_dumbbells + added_dumbbells

/-- Theorem: The total weight of dumbbells Parker is using is 120 pounds -/
theorem total_weight_is_120_pounds :
  total_dumbbells * dumbbell_weight = 120 := by
  sorry


end NUMINAMATH_CALUDE_total_weight_is_120_pounds_l1994_199426


namespace NUMINAMATH_CALUDE_quadratic_coefficient_bound_l1994_199439

theorem quadratic_coefficient_bound (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c
  (∀ x ∈ Set.Icc 0 1, f x ∈ Set.Icc (-1) 1) →
  |f 1| ≤ 1 →
  |f (1/2)| ≤ 1 →
  |f 0| ≤ 1 →
  |a| + |b| + |c| ≤ 17 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_bound_l1994_199439


namespace NUMINAMATH_CALUDE_unique_positive_solution_l1994_199456

theorem unique_positive_solution : 
  ∃! x : ℝ, x > 0 ∧ (1/3) * (4*x^2 - 2) = (x^2 - 30*x - 5) * (x^2 + 15*x + 1) ∧ 
  x = 15 + Real.sqrt 8328 / 6 := by
sorry

end NUMINAMATH_CALUDE_unique_positive_solution_l1994_199456


namespace NUMINAMATH_CALUDE_product_of_factors_for_six_factor_number_l1994_199470

def has_six_factors (x : ℕ) : Prop :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).card = 6

def product_of_factors (x : ℕ) : ℕ :=
  (Finset.filter (· ∣ x) (Finset.range (x + 1))).prod id

theorem product_of_factors_for_six_factor_number (x : ℕ) 
  (h1 : x > 1) (h2 : has_six_factors x) : 
  product_of_factors x = x^3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_factors_for_six_factor_number_l1994_199470
