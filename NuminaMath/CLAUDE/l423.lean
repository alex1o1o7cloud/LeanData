import Mathlib

namespace NUMINAMATH_CALUDE_symmetric_axis_of_translated_sine_l423_42368

theorem symmetric_axis_of_translated_sine (f g : ℝ → ℝ) :
  (∀ x, f x = Real.sin (2 * x - π / 6)) →
  (∀ x, g x = f (x - π / 4)) →
  (∀ x, g x = Real.sin (2 * x - 2 * π / 3)) →
  (π / 12 : ℝ) ∈ {x | ∀ y, g (x + y) = g (x - y)} :=
by sorry

end NUMINAMATH_CALUDE_symmetric_axis_of_translated_sine_l423_42368


namespace NUMINAMATH_CALUDE_ali_and_leila_trip_cost_l423_42334

/-- The total cost of a trip for two people with a given original price and discount. -/
def trip_cost (original_price discount : ℕ) : ℕ :=
  2 * (original_price - discount)

/-- Theorem stating that the trip cost for Ali and Leila is $266. -/
theorem ali_and_leila_trip_cost :
  trip_cost 147 14 = 266 := by
  sorry

#eval trip_cost 147 14  -- This line is optional, for verification purposes

end NUMINAMATH_CALUDE_ali_and_leila_trip_cost_l423_42334


namespace NUMINAMATH_CALUDE_fraction_equality_implies_values_l423_42353

theorem fraction_equality_implies_values (a b : ℚ) : 
  (∀ n : ℕ, (1 : ℚ) / ((2 * n - 1) * (2 * n + 1)) = a / (2 * n - 1) + b / (2 * n + 1)) →
  a = 1 / 2 ∧ b = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_implies_values_l423_42353


namespace NUMINAMATH_CALUDE_equation_solution_l423_42358

theorem equation_solution :
  ∃ x : ℚ, 6 * (2 * x + 3) - 4 = -3 * (2 - 5 * x) + 3 * x ∧ x = 10 / 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l423_42358


namespace NUMINAMATH_CALUDE_complex_equation_solution_l423_42369

theorem complex_equation_solution (z : ℂ) : (1 + 3*I)*z = 10 → z = 1 - 3*I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l423_42369


namespace NUMINAMATH_CALUDE_joan_clothing_expenses_l423_42307

theorem joan_clothing_expenses : 
  15 + 14.82 + 12.51 = 42.33 := by sorry

end NUMINAMATH_CALUDE_joan_clothing_expenses_l423_42307


namespace NUMINAMATH_CALUDE_bottle_cap_division_l423_42316

theorem bottle_cap_division (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
    (h1 : total_caps = 560) 
    (h2 : num_groups = 80) 
    (h3 : caps_per_group * num_groups = total_caps) : 
  caps_per_group = 7 := by
  sorry

end NUMINAMATH_CALUDE_bottle_cap_division_l423_42316


namespace NUMINAMATH_CALUDE_reciprocal_power_2014_l423_42359

theorem reciprocal_power_2014 (a : ℚ) (h : a ≠ 0) : (a = a⁻¹) → a^2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_2014_l423_42359


namespace NUMINAMATH_CALUDE_expression_evaluation_l423_42347

theorem expression_evaluation : 4 * (9 - 6) / 2 - 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l423_42347


namespace NUMINAMATH_CALUDE_classroom_shirts_problem_l423_42391

theorem classroom_shirts_problem (total_students : ℕ) 
  (striped_ratio : ℚ) (shorts_difference : ℕ) : 
  total_students = 81 →
  striped_ratio = 2 / 3 →
  shorts_difference = 19 →
  let striped := (total_students : ℚ) * striped_ratio
  let checkered := total_students - striped.floor
  let shorts := checkered + shorts_difference
  striped.floor - shorts = 8 := by
sorry

end NUMINAMATH_CALUDE_classroom_shirts_problem_l423_42391


namespace NUMINAMATH_CALUDE_fencing_calculation_l423_42303

/-- The total fencing length for a square playground and a rectangular garden -/
def total_fencing (playground_side : ℝ) (garden_length garden_width : ℝ) : ℝ :=
  4 * playground_side + 2 * (garden_length + garden_width)

/-- Theorem: The total fencing for a square playground with side length 27 yards
    and a rectangular garden of 12 yards by 9 yards is equal to 150 yards -/
theorem fencing_calculation :
  total_fencing 27 12 9 = 150 := by
  sorry

end NUMINAMATH_CALUDE_fencing_calculation_l423_42303


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l423_42355

theorem quadratic_roots_sum_product (m n : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - m * x + n = 0 ∧ 3 * y^2 - m * y + n = 0 ∧ x + y = 10 ∧ x * y = 26) → 
  m + n = 108 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l423_42355


namespace NUMINAMATH_CALUDE_complement_of_union_l423_42357

def U : Finset Nat := {1,2,3,4,5,6,7,8}
def M : Finset Nat := {1,3,5,7}
def N : Finset Nat := {5,6,7}

theorem complement_of_union :
  (U \ (M ∪ N)) = {2,4,8} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l423_42357


namespace NUMINAMATH_CALUDE_solution_set_equals_interval_l423_42378

-- Define the set of real numbers satisfying the inequality
def solution_set : Set ℝ := {x : ℝ | 2 * x^2 + 8 * x ≤ -6}

-- State the theorem
theorem solution_set_equals_interval : 
  solution_set = Set.Icc (-3) (-1) := by sorry

end NUMINAMATH_CALUDE_solution_set_equals_interval_l423_42378


namespace NUMINAMATH_CALUDE_equal_area_necessary_not_sufficient_l423_42397

-- Define a triangle type
structure Triangle where
  -- You might add more specific properties here, but for this problem we only need area
  area : ℝ

-- Define congruence for triangles
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem statement
theorem equal_area_necessary_not_sufficient :
  (∀ t1 t2 : Triangle, congruent t1 t2 → t1.area = t2.area) ∧
  (∃ t1 t2 : Triangle, t1.area = t2.area ∧ ¬congruent t1 t2) := by
  sorry

end NUMINAMATH_CALUDE_equal_area_necessary_not_sufficient_l423_42397


namespace NUMINAMATH_CALUDE_remainder_3_pow_2023_mod_17_l423_42363

theorem remainder_3_pow_2023_mod_17 : 3^2023 % 17 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_pow_2023_mod_17_l423_42363


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l423_42360

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_properties
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_a2 : a 2 = 5)
  (h_sum : a 6 + a 8 = 30) :
  d = 2 ∧
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, 1 / ((a n)^2 - 1) = (1/4) * (1/n - 1/(n+1))) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l423_42360


namespace NUMINAMATH_CALUDE_problem_solution_l423_42389

theorem problem_solution (x y z : ℚ) 
  (h1 : 2*x + y + z = 14)
  (h2 : 2*x + y = 7)
  (h3 : x + 2*y = 10) :
  (x + y - z) / 3 = -4/9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l423_42389


namespace NUMINAMATH_CALUDE_carrots_rows_planted_l423_42367

/-- Calculates the number of rows of carrots planted given the planting conditions -/
theorem carrots_rows_planted (plants_per_row : ℕ) (planting_time : ℕ) (planting_rate : ℕ) : 
  plants_per_row > 0 →
  planting_time * planting_rate / plants_per_row = 400 :=
by
  intro h
  sorry

#check carrots_rows_planted 300 20 6000

end NUMINAMATH_CALUDE_carrots_rows_planted_l423_42367


namespace NUMINAMATH_CALUDE_square_difference_divided_l423_42373

theorem square_difference_divided : (111^2 - 102^2) / 9 = 213 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_divided_l423_42373


namespace NUMINAMATH_CALUDE_intersection_complement_equal_set_l423_42335

def M : Set ℝ := {-1, 0, 1, 3}
def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_complement_equal_set : M ∩ (Set.univ \ N) = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equal_set_l423_42335


namespace NUMINAMATH_CALUDE_f_simplification_f_third_quadrant_f_specific_angle_l423_42310

noncomputable def f (α : Real) : Real :=
  (Real.sin (α - 3 * Real.pi) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 / 2 * Real.pi)) /
  (Real.cos (-Real.pi - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) : f α = -Real.cos α := by sorry

theorem f_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3 * Real.pi / 2)  -- α is in the third quadrant
  (h2 : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := by sorry

theorem f_specific_angle : f (-31 * Real.pi / 3) = -1 / 2 := by sorry

end NUMINAMATH_CALUDE_f_simplification_f_third_quadrant_f_specific_angle_l423_42310


namespace NUMINAMATH_CALUDE_complex_square_sum_positive_l423_42387

theorem complex_square_sum_positive (z₁ z₂ z₃ : ℂ) :
  (z₁^2 + z₂^2 : ℂ).re > (-z₃^2 : ℂ).re → (z₁^2 + z₂^2 + z₃^2 : ℂ).re > 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_sum_positive_l423_42387


namespace NUMINAMATH_CALUDE_dexter_cards_count_l423_42309

/-- The number of boxes filled with basketball cards -/
def basketball_boxes : ℕ := 15

/-- The number of cards in each basketball box -/
def basketball_cards_per_box : ℕ := 20

/-- The difference in the number of boxes between basketball and football cards -/
def box_difference : ℕ := 7

/-- The number of cards in each football box -/
def football_cards_per_box : ℕ := 25

/-- The total number of cards Dexter has -/
def total_cards : ℕ := basketball_boxes * basketball_cards_per_box + 
  (basketball_boxes - box_difference) * football_cards_per_box

theorem dexter_cards_count : total_cards = 500 := by
  sorry

end NUMINAMATH_CALUDE_dexter_cards_count_l423_42309


namespace NUMINAMATH_CALUDE_art_gallery_theorem_l423_42361

theorem art_gallery_theorem (total_pieces : ℕ) : 
  (total_pieces : ℚ) * (1 / 3) * (1 / 6) + 
  (total_pieces : ℚ) * (2 / 3) * (2 / 3) = 800 →
  total_pieces = 1800 := by
  sorry

end NUMINAMATH_CALUDE_art_gallery_theorem_l423_42361


namespace NUMINAMATH_CALUDE_base_seven_65432_equals_16340_l423_42393

def base_seven_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ (digits.length - 1 - i))) 0

theorem base_seven_65432_equals_16340 :
  base_seven_to_decimal [6, 5, 4, 3, 2] = 16340 := by
  sorry

end NUMINAMATH_CALUDE_base_seven_65432_equals_16340_l423_42393


namespace NUMINAMATH_CALUDE_fraction_product_proof_l423_42337

theorem fraction_product_proof : 
  (7 : ℚ) / 4 * 8 / 14 * 14 / 8 * 16 / 40 * 35 / 20 * 18 / 45 * 49 / 28 * 32 / 64 = 49 / 200 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_proof_l423_42337


namespace NUMINAMATH_CALUDE_f_range_l423_42327

-- Define the function f
def f (x : ℝ) : ℝ := 3 * (x - 4)

-- State the theorem
theorem f_range :
  Set.range f = {y : ℝ | y < -27 ∨ y > -27} :=
by
  sorry

end NUMINAMATH_CALUDE_f_range_l423_42327


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l423_42317

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: A hexagon has 9 internal diagonals -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l423_42317


namespace NUMINAMATH_CALUDE_total_views_and_likes_theorem_l423_42352

def total_views_and_likes (
  initial_yt_views : ℕ) (initial_yt_likes : ℕ)
  (initial_other_views : ℕ) (initial_other_likes : ℕ)
  (yt_view_increase_factor : ℕ) (yt_like_increase_factor : ℕ)
  (other_view_increase_factor : ℕ) (other_like_increase_percent : ℕ)
  (additional_yt_views : ℕ) (additional_yt_likes : ℕ)
  (additional_other_views : ℕ) (additional_other_likes : ℕ) : ℕ × ℕ :=
  let yt_views_after_4_days := initial_yt_views + initial_yt_views * yt_view_increase_factor
  let yt_likes_after_4_days := initial_yt_likes + initial_yt_likes * (yt_like_increase_factor - 1)
  let other_views_after_4_days := initial_other_views + initial_other_views * (other_view_increase_factor - 1)
  let other_likes_after_4_days := initial_other_likes + initial_other_likes * other_like_increase_percent / 100
  let final_yt_views := yt_views_after_4_days + additional_yt_views
  let final_yt_likes := yt_likes_after_4_days + additional_yt_likes
  let final_other_views := other_views_after_4_days + additional_other_views
  let final_other_likes := other_likes_after_4_days + additional_other_likes
  (final_yt_views + final_other_views, final_yt_likes + final_other_likes)

theorem total_views_and_likes_theorem :
  total_views_and_likes 4000 500 2000 300 10 3 2 50 50000 2000 30000 500 = (130000, 5250) := by
  sorry

end NUMINAMATH_CALUDE_total_views_and_likes_theorem_l423_42352


namespace NUMINAMATH_CALUDE_theresa_julia_multiple_l423_42374

/-- The number of video games Tory has -/
def tory_games : ℕ := 6

/-- The number of video games Julia has -/
def julia_games : ℕ := tory_games / 3

/-- The number of video games Theresa has -/
def theresa_games : ℕ := 11

/-- The multiple of video games Theresa has compared to Julia -/
def multiple : ℕ := (theresa_games - 5) / julia_games

theorem theresa_julia_multiple :
  multiple = 3 :=
sorry

end NUMINAMATH_CALUDE_theresa_julia_multiple_l423_42374


namespace NUMINAMATH_CALUDE_nth_root_equation_l423_42301

theorem nth_root_equation (n : ℕ) : n = 3 →
  (((17 * Real.sqrt 5 + 38) ^ (1 / n : ℝ)) + ((17 * Real.sqrt 5 - 38) ^ (1 / n : ℝ))) = Real.sqrt 20 :=
by sorry

end NUMINAMATH_CALUDE_nth_root_equation_l423_42301


namespace NUMINAMATH_CALUDE_correct_remaining_money_l423_42366

def remaining_money (olivia_initial : ℕ) (nigel_initial : ℕ) (num_passes : ℕ) (cost_per_pass : ℕ) : ℕ :=
  olivia_initial + nigel_initial - num_passes * cost_per_pass

theorem correct_remaining_money :
  remaining_money 112 139 6 28 = 83 := by
  sorry

end NUMINAMATH_CALUDE_correct_remaining_money_l423_42366


namespace NUMINAMATH_CALUDE_janet_flight_cost_l423_42321

/-- The cost of flying between two cities -/
def flying_cost (distance : ℝ) (cost_per_km : ℝ) (booking_fee : ℝ) : ℝ :=
  distance * cost_per_km + booking_fee

/-- Theorem: The cost for Janet to fly from City D to City E is $720 -/
theorem janet_flight_cost : 
  flying_cost 4750 0.12 150 = 720 := by
  sorry

end NUMINAMATH_CALUDE_janet_flight_cost_l423_42321


namespace NUMINAMATH_CALUDE_square_side_length_range_l423_42390

theorem square_side_length_range (area : ℝ) (h : area = 15) :
  ∃ side : ℝ, side^2 = area ∧ 3 < side ∧ side < 4 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_range_l423_42390


namespace NUMINAMATH_CALUDE_middle_manager_sample_size_l423_42354

theorem middle_manager_sample_size
  (total_employees : ℕ) (total_middle_managers : ℕ) (sample_size : ℕ)
  (h1 : total_employees = 1000)
  (h2 : total_middle_managers = 150)
  (h3 : sample_size = 200) :
  (total_middle_managers : ℚ) / total_employees * sample_size = 30 :=
by sorry

end NUMINAMATH_CALUDE_middle_manager_sample_size_l423_42354


namespace NUMINAMATH_CALUDE_route_down_length_l423_42346

/-- Proves that the length of the route down the mountain is 12 miles given the specified conditions. -/
theorem route_down_length (time_up time_down : ℝ) (rate_up : ℝ) (rate_down_factor : ℝ) :
  time_up = time_down →
  rate_down_factor = 1.5 →
  rate_up = 4 →
  time_up = 2 →
  rate_up * time_up = 8 →
  rate_down_factor * rate_up * time_down = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_route_down_length_l423_42346


namespace NUMINAMATH_CALUDE_triangle_concurrency_l423_42341

-- Define the basic structures
structure Point : Type :=
  (x y : ℝ)

structure Line : Type :=
  (a b c : ℝ)

-- Define the triangle
def Triangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define perpendicular
def Perpendicular (l1 l2 : Line) : Prop := sorry

-- Define reflection
def Reflection (P Q : Point) (l : Line) : Prop := sorry

-- Define intersection
def Intersect (l1 l2 : Line) : Point := sorry

-- Define concurrency
def Concurrent (l1 l2 l3 : Line) : Prop := sorry

-- Theorem statement
theorem triangle_concurrency 
  (A B C D E F H E' F' X Y : Point) 
  (ABC : Triangle A B C)
  (not_right : ¬ Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0)) -- Assuming right angle is between x and y axes
  (D_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (E_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (F_perp : Perpendicular (Line.mk 1 0 0) (Line.mk 0 1 0))
  (H_ortho : H = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0))
  (E'_refl : Reflection E E' (Line.mk 1 0 0))
  (F'_refl : Reflection F F' (Line.mk 1 0 0))
  (X_def : X = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0))
  (Y_def : Y = Intersect (Line.mk 1 0 0) (Line.mk 0 1 0)) :
  Concurrent (Line.mk 1 0 0) (Line.mk 0 1 0) (Line.mk 1 1 0) := by
  sorry

end NUMINAMATH_CALUDE_triangle_concurrency_l423_42341


namespace NUMINAMATH_CALUDE_point_A_coordinates_l423_42330

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The line x + y + 3 = 0 -/
def line (p : Point) : Prop :=
  p.x + p.y + 3 = 0

/-- Two points are symmetric about a line if their midpoint lies on the line
    and the line is perpendicular to the line segment connecting the points -/
def symmetric_about (a b : Point) : Prop :=
  let midpoint : Point := ⟨(a.x + b.x) / 2, (a.y + b.y) / 2⟩
  line midpoint ∧ (a.y - b.y) = (a.x - b.x)

/-- The main theorem -/
theorem point_A_coordinates :
  ∀ (A : Point),
    symmetric_about A ⟨1, 2⟩ →
    A.x = -5 ∧ A.y = -4 := by
  sorry

end NUMINAMATH_CALUDE_point_A_coordinates_l423_42330


namespace NUMINAMATH_CALUDE_problem_statement_l423_42381

theorem problem_statement (a b : ℝ) (ha : a > 0) (heq : Real.exp a * (1 - Real.log b) = 1) :
  (1 < b ∧ b < Real.exp 1) ∧ (a > Real.log b) ∧ (b - a < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l423_42381


namespace NUMINAMATH_CALUDE_constant_remainder_implies_b_25_l423_42304

def dividend (b x : ℝ) : ℝ := 8 * x^3 - b * x^2 + 2 * x + 5
def divisor (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem constant_remainder_implies_b_25 :
  (∃ (q r : ℝ → ℝ) (c : ℝ), ∀ x, dividend b x = divisor x * q x + c) → b = 25 := by
  sorry

end NUMINAMATH_CALUDE_constant_remainder_implies_b_25_l423_42304


namespace NUMINAMATH_CALUDE_specific_quadratic_equation_l423_42350

/-- A quadratic equation with given roots and leading coefficient -/
def quadratic_equation (root1 root2 : ℝ) (leading_coeff : ℝ) : ℝ → ℝ :=
  fun x => leading_coeff * (x - root1) * (x - root2)

/-- Theorem: The quadratic equation with roots -3 and 7 and leading coefficient 1 is x^2 - 4x - 21 = 0 -/
theorem specific_quadratic_equation :
  quadratic_equation (-3) 7 1 = fun x => x^2 - 4*x - 21 := by sorry

end NUMINAMATH_CALUDE_specific_quadratic_equation_l423_42350


namespace NUMINAMATH_CALUDE_chess_team_size_l423_42319

theorem chess_team_size (total_students : ℕ) (percentage : ℚ) (team_size : ℕ) : 
  total_students = 160 → percentage = 1/10 → team_size = (total_students : ℚ) * percentage → team_size = 16 := by
  sorry

end NUMINAMATH_CALUDE_chess_team_size_l423_42319


namespace NUMINAMATH_CALUDE_integer_in_range_l423_42308

theorem integer_in_range : ∃ x : ℤ, -Real.sqrt 2 < x ∧ x < Real.sqrt 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_in_range_l423_42308


namespace NUMINAMATH_CALUDE_basketball_playoff_condition_l423_42362

/-- A basketball team's playoff qualification condition -/
theorem basketball_playoff_condition (x : ℕ) : 
  (∀ (game : ℕ), game ≤ 32 → (game = 32 - x ∨ game = x)) →  -- Each game is either won or lost
  (2 * x + (32 - x) ≥ 48) →                                  -- Points condition
  (x ≤ 32) →                                                 -- Cannot win more games than played
  (2 * x + (32 - x) ≥ 48) :=                                 -- Conclusion: same as second hypothesis
by sorry

end NUMINAMATH_CALUDE_basketball_playoff_condition_l423_42362


namespace NUMINAMATH_CALUDE_max_additional_plates_l423_42396

/-- Represents the number of elements in each set of characters for license plates -/
def initial_sets : Fin 3 → Nat
  | 0 => 4  -- {B, G, J, S}
  | 1 => 2  -- {E, U}
  | 2 => 3  -- {K, V, X}
  | _ => 0

/-- Calculates the total number of license plate combinations -/
def total_combinations (sets : Fin 3 → Nat) : Nat :=
  (sets 0) * (sets 1) * (sets 2)

/-- Represents the addition of two new letters to the sets -/
structure NewLetterAddition where
  set1 : Nat  -- Number of letters added to set 1
  set2 : Nat  -- Number of letters added to set 2
  set3 : Nat  -- Number of letters added to set 3

/-- The theorem to be proved -/
theorem max_additional_plates :
  ∃ (addition : NewLetterAddition),
    addition.set1 + addition.set2 + addition.set3 = 2 ∧
    ∀ (other : NewLetterAddition),
      other.set1 + other.set2 + other.set3 = 2 →
      total_combinations (λ i => initial_sets i + other.set1) -
      total_combinations initial_sets ≤
      total_combinations (λ i => initial_sets i + addition.set1) -
      total_combinations initial_sets ∧
      total_combinations (λ i => initial_sets i + addition.set1) -
      total_combinations initial_sets = 24 :=
sorry

end NUMINAMATH_CALUDE_max_additional_plates_l423_42396


namespace NUMINAMATH_CALUDE_log_product_equals_two_l423_42349

theorem log_product_equals_two (k x : ℝ) (h : k > 0) (h' : x > 0) :
  (Real.log x / Real.log k) * (Real.log k / Real.log 10) = 2 →
  x = 100 := by
sorry

end NUMINAMATH_CALUDE_log_product_equals_two_l423_42349


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_symmetry_l423_42306

/-- A circle in which the quadrilateral is inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the 2D plane -/
def Point := ℝ × ℝ

/-- A line in the 2D plane -/
structure Line where
  point1 : Point
  point2 : Point

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  circle : Circle
  A : Point
  B : Point
  C : Point
  D : Point

/-- Intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point := sorry

/-- Check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Check if two points are symmetrical with respect to a third point -/
def areSymmetrical (p1 p2 center : Point) : Prop := sorry

/-- The main theorem -/
theorem inscribed_quadrilateral_symmetry 
  (quad : InscribedQuadrilateral)
  (E : Point)
  (t : Line) :
  let AB := Line.mk quad.A quad.B
  let CD := Line.mk quad.C quad.D
  let AC := Line.mk quad.A quad.C
  let BD := Line.mk quad.B quad.D
  let BC := Line.mk quad.B quad.C
  let AD := Line.mk quad.A quad.D
  let O := quad.circle.center
  E = intersectionPoint AB CD →
  isPointOnLine E t →
  (∀ p : Point, isPointOnLine p (Line.mk O E) → isPointOnLine p t → p = E) →
  ∃ (P Q R S : Point),
    P = intersectionPoint AC t ∧
    Q = intersectionPoint BD t ∧
    R = intersectionPoint BC t ∧
    S = intersectionPoint AD t ∧
    areSymmetrical P Q E ∧
    areSymmetrical R S E :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_symmetry_l423_42306


namespace NUMINAMATH_CALUDE_real_roots_necessary_condition_l423_42342

theorem real_roots_necessary_condition (a : ℝ) :
  (∃ x : ℝ, x^2 - a*x + 1 = 0) →
  (a ≥ 1 ∨ a ≤ -2) :=
by sorry

end NUMINAMATH_CALUDE_real_roots_necessary_condition_l423_42342


namespace NUMINAMATH_CALUDE_digit_sum_puzzle_l423_42344

theorem digit_sum_puzzle (x y z w : ℕ) : 
  x ≤ 9 ∧ y ≤ 9 ∧ z ≤ 9 ∧ w ≤ 9 →
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w →
  x + z + x = 11 →
  y + z = 10 →
  x + w = 10 →
  x + y + z + w = 24 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_puzzle_l423_42344


namespace NUMINAMATH_CALUDE_divisibility_np_minus_n_l423_42392

theorem divisibility_np_minus_n (p : Nat) (n : Int) (h : p = 3 ∨ p = 7 ∨ p = 13) :
  ∃ k : Int, n^p - n = k * p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_np_minus_n_l423_42392


namespace NUMINAMATH_CALUDE_jellybeans_left_l423_42300

/-- Calculates the number of jellybeans left in a jar after a class party --/
theorem jellybeans_left (total_jellybeans : ℕ) 
  (kindergarteners first_graders second_graders : ℕ)
  (absent_kindergarteners absent_second_graders : ℕ)
  (present_kindergartener_rate first_grader_rate : ℕ)
  (absent_kindergartener_rate absent_second_grader_rate : ℕ)
  (h1 : total_jellybeans = 500)
  (h2 : kindergarteners = 10)
  (h3 : first_graders = 10)
  (h4 : second_graders = 10)
  (h5 : absent_kindergarteners = 2)
  (h6 : absent_second_graders = 3)
  (h7 : present_kindergartener_rate = 3)
  (h8 : first_grader_rate = 5)
  (h9 : absent_kindergartener_rate = 5)
  (h10 : absent_second_grader_rate = 10) :
  total_jellybeans - 
  ((kindergarteners - absent_kindergarteners) * present_kindergartener_rate +
   first_graders * first_grader_rate +
   (second_graders - absent_second_graders) * (first_graders * first_grader_rate / 2)) = 176 := by
sorry


end NUMINAMATH_CALUDE_jellybeans_left_l423_42300


namespace NUMINAMATH_CALUDE_systematic_sampling_calculation_l423_42375

def population_size : ℕ := 2005
def sample_size : ℕ := 50

theorem systematic_sampling_calculation :
  let sampling_interval := population_size / sample_size
  let discarded := population_size % sample_size
  sampling_interval = 40 ∧ discarded = 5 := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_calculation_l423_42375


namespace NUMINAMATH_CALUDE_polynomial_multiple_condition_l423_42399

/-- A polynomial f(x) of the form x^4 + p x^2 + q x + a^2 is a multiple of (x^2 - 1) 
    if and only if p = -(a^2 + 1), q = 0, and the other factor is (x^2 - a^2) -/
theorem polynomial_multiple_condition (a : ℝ) :
  ∃ (p q : ℝ), ∀ (x : ℝ), 
    (x^4 + p*x^2 + q*x + a^2 = (x^2 - 1) * (x^2 - a^2)) ↔ 
    (p = -(a^2 + 1) ∧ q = 0) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_multiple_condition_l423_42399


namespace NUMINAMATH_CALUDE_group_size_before_new_member_l423_42370

theorem group_size_before_new_member 
  (avg_after : ℚ) 
  (new_member_amount : ℚ) 
  (avg_before : ℚ) 
  (h1 : avg_after = 20)
  (h2 : new_member_amount = 56)
  (h3 : avg_before = 14) : 
  ∃ n : ℕ, n > 0 ∧ 
    (n : ℚ) * avg_before + new_member_amount = (n + 1 : ℚ) * avg_after ∧
    n = 6 :=
by sorry

end NUMINAMATH_CALUDE_group_size_before_new_member_l423_42370


namespace NUMINAMATH_CALUDE_cone_volume_l423_42388

/-- The volume of a cone with slant height 5 cm and height 4 cm is 12π cm³ -/
theorem cone_volume (l h : ℝ) (hl : l = 5) (hh : h = 4) :
  let r := Real.sqrt (l^2 - h^2)
  (1/3 : ℝ) * Real.pi * r^2 * h = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_volume_l423_42388


namespace NUMINAMATH_CALUDE_scarf_cost_proof_l423_42382

def sweater_cost : ℕ := 30
def num_items : ℕ := 6
def total_savings : ℕ := 500
def remaining_savings : ℕ := 200

theorem scarf_cost_proof :
  ∃ (scarf_cost : ℕ),
    scarf_cost * num_items = total_savings - remaining_savings - (sweater_cost * num_items) :=
by sorry

end NUMINAMATH_CALUDE_scarf_cost_proof_l423_42382


namespace NUMINAMATH_CALUDE_find_a_l423_42318

theorem find_a (A B : Set ℝ) (a : ℝ) :
  A = {-1, 1, 3} →
  B = {2, 2^a - 1} →
  A ∩ B = {1} →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l423_42318


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l423_42333

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 7 → x ≠ -2 →
  (5 * x - 4) / (x^2 - 5*x - 14) = (31/9) / (x - 7) + (14/9) / (x + 2) := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l423_42333


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l423_42364

/-- The equation of the fixed circle -/
def fixed_circle (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

/-- The equation of the y-axis -/
def y_axis (x : ℝ) : Prop := x = 0

/-- The trajectory of the center of the moving circle -/
def trajectory (x y : ℝ) : Prop := (x > 0 ∧ y^2 = 8*x) ∨ (x ≤ 0 ∧ y = 0)

/-- Theorem stating the trajectory of the center of the moving circle -/
theorem moving_circle_trajectory :
  ∀ (x y : ℝ), 
  (∃ (r : ℝ), r > 0 ∧ 
    (∃ (x₀ y₀ : ℝ), fixed_circle x₀ y₀ ∧ (x - x₀)^2 + (y - y₀)^2 = r^2) ∧
    (∃ (x₁ : ℝ), y_axis x₁ ∧ (x - x₁)^2 + y^2 = r^2)) →
  trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l423_42364


namespace NUMINAMATH_CALUDE_even_digits_in_base7_of_528_l423_42377

/-- Converts a natural number to its base-7 representation -/
def toBase7 (n : ℕ) : List ℕ :=
  sorry

/-- Counts the number of even digits in a list of natural numbers -/
def countEvenDigits (digits : List ℕ) : ℕ :=
  sorry

/-- Theorem: The number of even digits in the base-7 representation of 528₁₀ is 0 -/
theorem even_digits_in_base7_of_528 : 
  countEvenDigits (toBase7 528) = 0 := by
  sorry

end NUMINAMATH_CALUDE_even_digits_in_base7_of_528_l423_42377


namespace NUMINAMATH_CALUDE_lcm_12_35_l423_42338

theorem lcm_12_35 : Nat.lcm 12 35 = 420 := by
  sorry

end NUMINAMATH_CALUDE_lcm_12_35_l423_42338


namespace NUMINAMATH_CALUDE_craft_fair_ring_cost_l423_42339

/-- Given the sales data from a craft fair, prove the cost of each ring --/
theorem craft_fair_ring_cost :
  let total_sales : ℚ := 320
  let num_necklaces : ℕ := 4
  let num_rings : ℕ := 8
  let num_earrings : ℕ := 5
  let num_bracelets : ℕ := 6
  let cost_necklace : ℚ := 20
  let cost_earrings : ℚ := 15
  let cost_ring : ℚ := 8.25
  let cost_bracelet : ℚ := 2 * cost_ring
  total_sales = num_necklaces * cost_necklace + num_rings * cost_ring +
                num_earrings * cost_earrings + num_bracelets * cost_bracelet
  → cost_ring = 8.25 := by
  sorry


end NUMINAMATH_CALUDE_craft_fair_ring_cost_l423_42339


namespace NUMINAMATH_CALUDE_equation_solution_l423_42302

theorem equation_solution : ∀ x : ℝ, x^2 - 2*x - 3 = x + 7 → x = 5 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l423_42302


namespace NUMINAMATH_CALUDE_sqrt_inequality_l423_42376

theorem sqrt_inequality (x : ℝ) :
  x > 0 → (Real.sqrt x > 3 * x - 2 ↔ 4/9 < x ∧ x < 1) := by sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l423_42376


namespace NUMINAMATH_CALUDE_trigonometric_identity_l423_42395

theorem trigonometric_identity (α : ℝ) (h : (1 + Real.cos α) / Real.sin α = 1 / 2) :
  Real.cos α + 2 * Real.sin α = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l423_42395


namespace NUMINAMATH_CALUDE_exam_mean_score_l423_42314

theorem exam_mean_score (score_below mean standard_deviation : ℝ) 
  (h1 : score_below = mean - 2 * standard_deviation)
  (h2 : 98 = mean + 3 * standard_deviation)
  (h3 : score_below = 58) : mean = 74 := by
  sorry

end NUMINAMATH_CALUDE_exam_mean_score_l423_42314


namespace NUMINAMATH_CALUDE_senior_employee_bonus_l423_42322

/-- Proves that the senior employee receives $3,100 given the conditions of the bonus distribution -/
theorem senior_employee_bonus (total_bonus : ℕ) (difference : ℕ) (senior_share : ℕ) : 
  total_bonus = 5000 →
  difference = 1200 →
  senior_share = total_bonus - difference →
  2 * senior_share = total_bonus + difference →
  senior_share = 3100 := by
sorry

end NUMINAMATH_CALUDE_senior_employee_bonus_l423_42322


namespace NUMINAMATH_CALUDE_compare_negative_fractions_l423_42394

theorem compare_negative_fractions : -3/4 > -4/5 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_fractions_l423_42394


namespace NUMINAMATH_CALUDE_congruence_problem_l423_42305

theorem congruence_problem (x : ℤ) :
  (5 * x + 9) % 18 = 3 → (3 * x + 14) % 18 = 14 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l423_42305


namespace NUMINAMATH_CALUDE_min_value_of_expression_l423_42380

def S : Finset Int := {-8, -6, -4, -1, 3, 5, 7, 14}

theorem min_value_of_expression (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  ∃ (x : Int), 3 * (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 300 ∧
               3 * x^2 + (20 - x)^2 = 300 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l423_42380


namespace NUMINAMATH_CALUDE_degree_to_radian_conversion_l423_42372

theorem degree_to_radian_conversion (π : Real) :
  (60 : Real) * (π / 180) = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_degree_to_radian_conversion_l423_42372


namespace NUMINAMATH_CALUDE_salem_population_decrease_l423_42343

def salem_leesburg_ratio : ℕ := 15
def leesburg_population : ℕ := 58940
def salem_women_population : ℕ := 377050

def salem_original_population : ℕ := salem_leesburg_ratio * leesburg_population
def salem_current_population : ℕ := 2 * salem_women_population

theorem salem_population_decrease :
  salem_original_population - salem_current_population = 130000 :=
by sorry

end NUMINAMATH_CALUDE_salem_population_decrease_l423_42343


namespace NUMINAMATH_CALUDE_daily_earnings_of_c_l423_42313

theorem daily_earnings_of_c (a b c : ℕ) 
  (h1 : a + b + c = 600)
  (h2 : a + c = 400)
  (h3 : b + c = 300) :
  c = 100 := by
sorry

end NUMINAMATH_CALUDE_daily_earnings_of_c_l423_42313


namespace NUMINAMATH_CALUDE_gcd_polynomial_and_b_l423_42320

theorem gcd_polynomial_and_b (b : ℤ) (h : ∃ k : ℤ, b = 570 * k) :
  Int.gcd (5 * b^4 + 2 * b^3 + 5 * b^2 + 9 * b + 95) b = 95 := by
  sorry

end NUMINAMATH_CALUDE_gcd_polynomial_and_b_l423_42320


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l423_42325

/-- Proves that for an increasing geometric sequence with a_3 = 8 and S_3 = 14,
    the common ratio is 2. -/
theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)  -- The geometric sequence
  (h_incr : ∀ n, a n < a (n + 1))  -- The sequence is increasing
  (h_a3 : a 3 = 8)  -- Third term is 8
  (h_S3 : (a 1) + (a 2) + (a 3) = 14)  -- Sum of first 3 terms is 14
  : ∃ q : ℝ, (∀ n, a (n + 1) = q * a n) ∧ q = 2 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l423_42325


namespace NUMINAMATH_CALUDE_equation_solution_l423_42379

theorem equation_solution :
  ∃ (a b p q : ℝ),
    (∀ x : ℝ, (2*x - 1)^20 - (a*x + b)^20 = (x^2 + p*x + q)^10) ↔
    ((a = (2^20 - 1)^(1/20) ∧ b = -(2^20 - 1)^(1/20)/2) ∨
     (a = -(2^20 - 1)^(1/20) ∧ b = (2^20 - 1)^(1/20)/2)) ∧
    p = -1 ∧ q = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l423_42379


namespace NUMINAMATH_CALUDE_edward_money_theorem_l423_42356

def edward_money_problem (initial_amount spent1 spent2 remaining : ℕ) : Prop :=
  initial_amount = spent1 + spent2 + remaining

theorem edward_money_theorem :
  ∃ initial_amount : ℕ,
    edward_money_problem initial_amount 9 8 17 ∧ initial_amount = 34 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_theorem_l423_42356


namespace NUMINAMATH_CALUDE_calculate_expression_l423_42385

theorem calculate_expression : (1 + Real.pi) ^ 0 + 2 - |(-3)| + 2 * Real.sin (π / 4) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l423_42385


namespace NUMINAMATH_CALUDE_distance_against_current_14km_l423_42345

/-- Calculates the distance traveled against a current given swimming speed, current speed, and time. -/
def distanceAgainstCurrent (swimmingSpeed currentSpeed : ℝ) (time : ℝ) : ℝ :=
  (swimmingSpeed - currentSpeed) * time

/-- Proves that the distance traveled against the current is 14 km under given conditions. -/
theorem distance_against_current_14km
  (swimmingSpeed : ℝ)
  (currentSpeed : ℝ)
  (time : ℝ)
  (h1 : swimmingSpeed = 4)
  (h2 : currentSpeed = 2)
  (h3 : time = 7) :
  distanceAgainstCurrent swimmingSpeed currentSpeed time = 14 := by
  sorry

#eval distanceAgainstCurrent 4 2 7

end NUMINAMATH_CALUDE_distance_against_current_14km_l423_42345


namespace NUMINAMATH_CALUDE_sues_necklace_beads_l423_42324

/-- The number of beads in Sue's necklace -/
def total_beads (purple blue green : ℕ) : ℕ := purple + blue + green

/-- Theorem stating the total number of beads in Sue's necklace -/
theorem sues_necklace_beads : 
  ∀ (purple blue green : ℕ),
  purple = 7 →
  blue = 2 * purple →
  green = blue + 11 →
  total_beads purple blue green = 46 := by
sorry

end NUMINAMATH_CALUDE_sues_necklace_beads_l423_42324


namespace NUMINAMATH_CALUDE_subset_implies_C_C_complete_l423_42329

def A (a : ℝ) : Set ℝ := {x | a * x - 2 = 0}
def B : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def C : Set ℝ := {1, 2}

theorem subset_implies_C (a : ℝ) (h : A a ⊆ B) : a ∈ C := by
  sorry

theorem C_complete : ∀ a ∈ C, A a ⊆ B := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_C_C_complete_l423_42329


namespace NUMINAMATH_CALUDE_fractional_equation_m_range_l423_42351

theorem fractional_equation_m_range : 
  ∀ (x m : ℝ), 
    ((x + m) / (x - 2) - (2 * m) / (x - 2) = 3) →
    (x > 0) →
    (m < 6 ∧ m ≠ 2) := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_m_range_l423_42351


namespace NUMINAMATH_CALUDE_sqrt_equation_solutions_l423_42323

theorem sqrt_equation_solutions : 
  {x : ℝ | Real.sqrt (4 * x - 3) + 12 / Real.sqrt (4 * x - 3) = 8} = {39/4, 7/4} := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solutions_l423_42323


namespace NUMINAMATH_CALUDE_area_enclosed_by_cosine_and_lines_l423_42336

theorem area_enclosed_by_cosine_and_lines :
  let f (x : ℝ) := Real.cos x
  let a : ℝ := -π/3
  let b : ℝ := π/3
  ∫ x in a..b, f x = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_area_enclosed_by_cosine_and_lines_l423_42336


namespace NUMINAMATH_CALUDE_first_player_advantage_l423_42383

/-- Represents a chocolate bar game state -/
structure ChocolateBar :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a player in the game -/
inductive Player
  | First
  | Second

/-- The result of a game -/
structure GameResult :=
  (first_player_pieces : ℕ)
  (second_player_pieces : ℕ)

/-- A strategy for playing the game -/
def Strategy := ChocolateBar → Player → ChocolateBar

/-- Play the game with a given strategy -/
def play_game (initial : ChocolateBar) (strategy : Strategy) : GameResult :=
  sorry

/-- The optimal strategy for the first player -/
def optimal_strategy : Strategy :=
  sorry

/-- Theorem stating that the first player can get at least 6 more pieces -/
theorem first_player_advantage (initial : ChocolateBar) :
  initial.rows = 9 ∧ initial.cols = 6 →
  let result := play_game initial optimal_strategy
  result.first_player_pieces ≥ result.second_player_pieces + 6 :=
by sorry

end NUMINAMATH_CALUDE_first_player_advantage_l423_42383


namespace NUMINAMATH_CALUDE_swimming_speed_calculation_l423_42340

/-- Represents the swimming scenario with a stream -/
structure SwimmingScenario where
  stream_speed : ℝ
  upstream_time : ℝ
  downstream_time : ℝ
  swimming_speed : ℝ

/-- The conditions of the problem -/
def problem_conditions (s : SwimmingScenario) : Prop :=
  s.stream_speed = 3 ∧ s.upstream_time = 2 * s.downstream_time

/-- The theorem to be proved -/
theorem swimming_speed_calculation (s : SwimmingScenario) :
  problem_conditions s → s.swimming_speed = 9 := by
  sorry


end NUMINAMATH_CALUDE_swimming_speed_calculation_l423_42340


namespace NUMINAMATH_CALUDE_symmetric_angle_660_l423_42312

def is_symmetric_angle (θ : ℤ) : Prop :=
  ∃ k : ℤ, θ = -60 + 360 * k

theorem symmetric_angle_660 :
  is_symmetric_angle 660 ∧
  ¬ is_symmetric_angle (-660) ∧
  ¬ is_symmetric_angle 690 ∧
  ¬ is_symmetric_angle (-690) :=
sorry

end NUMINAMATH_CALUDE_symmetric_angle_660_l423_42312


namespace NUMINAMATH_CALUDE_pedro_skateboard_problem_l423_42365

/-- Pedro's skateboarding problem -/
theorem pedro_skateboard_problem 
  (pedro_speed : ℝ) 
  (pedro_time : ℝ) 
  (liam_time : ℝ) 
  (h1 : pedro_speed = 10)
  (h2 : pedro_time = 4)
  (h3 : liam_time = 5) :
  let pedro_distance := pedro_speed * pedro_time
  let liam_speed := pedro_distance / liam_time
  liam_speed = 8 := by
sorry

end NUMINAMATH_CALUDE_pedro_skateboard_problem_l423_42365


namespace NUMINAMATH_CALUDE_no_solution_cubic_inequality_l423_42315

theorem no_solution_cubic_inequality :
  ∀ x : ℝ, x ≠ 2 → ¬((x^3 - 8*x^2 + 16*x - 32) / (x - 2) < 0) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_cubic_inequality_l423_42315


namespace NUMINAMATH_CALUDE_prob_at_least_one_expired_l423_42331

def total_bottles : ℕ := 10
def expired_bottles : ℕ := 3
def selected_bottles : ℕ := 3

def probability_at_least_one_expired : ℚ := 17/24

theorem prob_at_least_one_expired :
  (1 : ℚ) - (Nat.choose (total_bottles - expired_bottles) selected_bottles : ℚ) / 
  (Nat.choose total_bottles selected_bottles : ℚ) = probability_at_least_one_expired := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_expired_l423_42331


namespace NUMINAMATH_CALUDE_probability_of_math_page_l423_42348

def total_pages : ℕ := 12
def math_pages : ℕ := 2

theorem probability_of_math_page :
  (math_pages : ℚ) / total_pages = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_math_page_l423_42348


namespace NUMINAMATH_CALUDE_zero_points_of_f_l423_42328

def f (x : ℝ) := 2 * x^2 + 3 * x + 1

theorem zero_points_of_f :
  ∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0 ∧ 
  ∀ z : ℝ, f z = 0 → z = x ∨ z = y ∧
  x = -1/2 ∧ y = -1 :=
sorry

end NUMINAMATH_CALUDE_zero_points_of_f_l423_42328


namespace NUMINAMATH_CALUDE_square_division_l423_42384

theorem square_division (s : ℝ) (w : ℝ) (h : w = 5) :
  ∃ (a b c d e : ℝ),
    s = 20 ∧
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    a + b + w = s ∧
    c + d = s ∧
    a * c = b * d ∧
    a * c = w * e ∧
    a * c = (s - c) * (s - a - b) :=
by sorry

#check square_division

end NUMINAMATH_CALUDE_square_division_l423_42384


namespace NUMINAMATH_CALUDE_intersection_M_N_l423_42311

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l423_42311


namespace NUMINAMATH_CALUDE_thalassa_population_2050_l423_42398

/-- The population growth factor for Thalassa every 30 years -/
def growth_factor : ℕ := 3

/-- The initial population of Thalassa in 1990 -/
def initial_population : ℕ := 300

/-- The number of 30-year periods between 1990 and 2050 -/
def num_periods : ℕ := 2

/-- The population of Thalassa in 2050 -/
def population_2050 : ℕ := initial_population * growth_factor ^ num_periods

theorem thalassa_population_2050 : population_2050 = 2700 := by
  sorry

end NUMINAMATH_CALUDE_thalassa_population_2050_l423_42398


namespace NUMINAMATH_CALUDE_oil_price_reduction_l423_42332

/-- Represents the price reduction percentage of oil -/
def price_reduction : ℚ := 30 / 100

/-- Represents the additional amount of oil that can be bought after the price reduction -/
def additional_oil : ℚ := 9

/-- Represents the fixed amount spent on oil -/
def fixed_amount : ℚ := 900

/-- Represents the price increase percentage of oil compared to rice -/
def price_increase : ℚ := 50 / 100

/-- Represents the prime number that divides the reduced oil price -/
def prime_divisor : ℕ := 5

theorem oil_price_reduction (original_price reduced_price rice_price : ℚ) : 
  reduced_price = original_price * (1 - price_reduction) →
  fixed_amount / original_price - fixed_amount / reduced_price = additional_oil →
  ∃ (n : ℕ), reduced_price = n * prime_divisor →
  original_price = rice_price * (1 + price_increase) →
  original_price = 857142 / 20000 ∧ 
  reduced_price = 30 ∧
  rice_price = 571428 / 20000 := by
  sorry

#eval 857142 / 20000  -- Outputs 42.8571
#eval 571428 / 20000  -- Outputs 28.5714

end NUMINAMATH_CALUDE_oil_price_reduction_l423_42332


namespace NUMINAMATH_CALUDE_valid_numbers_l423_42371

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 
    a < b ∧ 
    b ≤ 9 ∧
    n = 10 * a + b ∧ 
    n = (b - a + 1) * (a + b) / 2

theorem valid_numbers : 
  {n : ℕ | is_valid_number n} = {14, 26, 37, 48, 59} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l423_42371


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l423_42386

theorem rectangle_dimension_change (L B : ℝ) (L' B' : ℝ) (h1 : L' = 1.35 * L) (h2 : (L' * B') / (L * B) = 1.0665) : B' = 0.79 * B := by
  sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l423_42386


namespace NUMINAMATH_CALUDE_billy_wednesday_apples_l423_42326

/-- The number of apples Billy ate in a week -/
def total_apples : ℕ := 20

/-- The number of apples Billy ate on Monday -/
def monday_apples : ℕ := 2

/-- The number of apples Billy ate on Tuesday -/
def tuesday_apples : ℕ := 2 * monday_apples

/-- The number of apples Billy ate on Friday -/
def friday_apples : ℕ := monday_apples / 2

/-- The number of apples Billy ate on Thursday -/
def thursday_apples : ℕ := 4 * friday_apples

/-- The number of apples Billy ate on Wednesday -/
def wednesday_apples : ℕ := total_apples - (monday_apples + tuesday_apples + thursday_apples + friday_apples)

theorem billy_wednesday_apples :
  wednesday_apples = 9 := by sorry

end NUMINAMATH_CALUDE_billy_wednesday_apples_l423_42326
