import Mathlib

namespace NUMINAMATH_CALUDE_jiangsu_population_scientific_notation_l2275_227560

/-- The population of Jiangsu Province in 2021 -/
def jiangsu_population : ℕ := 85000000

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ significand ∧ significand < 10

/-- Theorem: The population of Jiangsu Province expressed in scientific notation -/
theorem jiangsu_population_scientific_notation :
  ∃ (sn : ScientificNotation), (sn.significand * (10 : ℝ) ^ sn.exponent) = jiangsu_population := by
  sorry

end NUMINAMATH_CALUDE_jiangsu_population_scientific_notation_l2275_227560


namespace NUMINAMATH_CALUDE_junk_mail_distribution_l2275_227589

theorem junk_mail_distribution (blocks : ℕ) (houses_per_block : ℕ) (total_mail : ℕ) 
  (h1 : blocks = 16) 
  (h2 : houses_per_block = 17) 
  (h3 : total_mail = 1088) : 
  total_mail / (blocks * houses_per_block) = 4 := by
  sorry

end NUMINAMATH_CALUDE_junk_mail_distribution_l2275_227589


namespace NUMINAMATH_CALUDE_simplify_expression_l2275_227596

theorem simplify_expression (x : ℝ) : 4*x + 9*x^2 + 8 - (5 - 4*x - 9*x^2) = 18*x^2 + 8*x + 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2275_227596


namespace NUMINAMATH_CALUDE_new_range_theorem_l2275_227535

/-- Represents the number of mutual funds -/
def num_funds : ℕ := 150

/-- Represents the range of annual yield last year -/
def last_year_range : ℝ := 12500

/-- Represents the percentage increase for the first group of funds -/
def increase_group1 : ℝ := 0.12

/-- Represents the percentage increase for the second group of funds -/
def increase_group2 : ℝ := 0.17

/-- Represents the percentage increase for the third group of funds -/
def increase_group3 : ℝ := 0.22

/-- Represents the size of each group of funds -/
def group_size : ℕ := 50

/-- Theorem stating that the range of annual yield this year is $27,750 -/
theorem new_range_theorem : 
  ∃ (L H : ℝ), 
    H - L = last_year_range ∧ 
    (H * (1 + increase_group3)) - (L * (1 + increase_group1)) = 27750 :=
sorry

end NUMINAMATH_CALUDE_new_range_theorem_l2275_227535


namespace NUMINAMATH_CALUDE_radhika_total_games_l2275_227567

def christmas_games : ℕ := 12
def birthday_games : ℕ := 8
def original_games_ratio : ℚ := 1 / 2

theorem radhika_total_games :
  let total_gift_games := christmas_games + birthday_games
  let original_games := (total_gift_games : ℚ) * original_games_ratio
  (original_games + total_gift_games : ℚ) = 30 := by
  sorry

end NUMINAMATH_CALUDE_radhika_total_games_l2275_227567


namespace NUMINAMATH_CALUDE_average_book_price_l2275_227556

/-- The average price of books Sandy bought, given the number of books and total cost from two shops. -/
theorem average_book_price 
  (shop1_books : ℕ) 
  (shop1_cost : ℕ) 
  (shop2_books : ℕ) 
  (shop2_cost : ℕ) 
  (h1 : shop1_books = 65) 
  (h2 : shop1_cost = 1480) 
  (h3 : shop2_books = 55) 
  (h4 : shop2_cost = 920) : 
  (shop1_cost + shop2_cost) / (shop1_books + shop2_books) = 20 := by
sorry

end NUMINAMATH_CALUDE_average_book_price_l2275_227556


namespace NUMINAMATH_CALUDE_final_values_correct_l2275_227587

/-- Represents the state of variables a, b, and c -/
structure State where
  a : Int
  b : Int
  c : Int

/-- Applies the assignments a = b, b = c, c = a to a given state -/
def applyAssignments (s : State) : State :=
  { a := s.b, b := s.c, c := s.b }

/-- The theorem statement -/
theorem final_values_correct :
  let initialState : State := { a := 3, b := -5, c := 8 }
  let finalState := applyAssignments initialState
  finalState.a = -5 ∧ finalState.b = 8 ∧ finalState.c = -5 := by
  sorry


end NUMINAMATH_CALUDE_final_values_correct_l2275_227587


namespace NUMINAMATH_CALUDE_bread_duration_l2275_227553

-- Define the parameters
def household_members : ℕ := 4
def breakfast_slices : ℕ := 3
def snack_slices : ℕ := 2
def slices_per_loaf : ℕ := 12
def number_of_loaves : ℕ := 5

-- Define the theorem
theorem bread_duration : 
  let total_slices := number_of_loaves * slices_per_loaf
  let daily_consumption := household_members * (breakfast_slices + snack_slices)
  total_slices / daily_consumption = 3 := by
  sorry


end NUMINAMATH_CALUDE_bread_duration_l2275_227553


namespace NUMINAMATH_CALUDE_sum_of_altitudes_l2275_227565

-- Define the line equation
def line_equation (x y : ℝ) : Prop := 10 * x + 8 * y = 80

-- Define the triangle formed by the line and coordinate axes
def triangle_vertices : Set (ℝ × ℝ) :=
  {(0, 0), (8, 0), (0, 10)}

-- State the theorem
theorem sum_of_altitudes :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    (∀ (x y : ℝ), (x, y) ∈ triangle_vertices → line_equation x y) ∧
    a + b + c = 18 + (80 * Real.sqrt 164) / 164 :=
sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_l2275_227565


namespace NUMINAMATH_CALUDE_hemisphere_surface_area_l2275_227511

theorem hemisphere_surface_area (r : ℝ) (h : π * r^2 = 144 * π) :
  2 * π * r^2 + π * r^2 = 432 * π := by
  sorry

end NUMINAMATH_CALUDE_hemisphere_surface_area_l2275_227511


namespace NUMINAMATH_CALUDE_unique_solution_2a3b_7c_l2275_227543

theorem unique_solution_2a3b_7c : ∃! (a b c : ℕ+), 2^(a:ℕ) * 3^(b:ℕ) = 7^(c:ℕ) - 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_2a3b_7c_l2275_227543


namespace NUMINAMATH_CALUDE_different_color_probability_l2275_227538

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 3
def green_chips : ℕ := 2

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

theorem different_color_probability :
  let p_blue : ℚ := blue_chips / total_chips
  let p_red : ℚ := red_chips / total_chips
  let p_yellow : ℚ := yellow_chips / total_chips
  let p_green : ℚ := green_chips / total_chips
  
  p_blue * (1 - p_blue) + p_red * (1 - p_red) + 
  p_yellow * (1 - p_yellow) + p_green * (1 - p_green) = 91 / 128 := by
  sorry

end NUMINAMATH_CALUDE_different_color_probability_l2275_227538


namespace NUMINAMATH_CALUDE_expansion_max_coefficient_l2275_227562

/-- The coefficient of x^3 in the expansion of (x - a/x)^5 is -5 -/
def coefficient_condition (a : ℝ) : Prop :=
  (5 : ℝ) * a = 5

/-- The maximum coefficient in the expansion of (x - a/x)^5 -/
def max_coefficient (a : ℝ) : ℕ :=
  Nat.max (Nat.choose 5 0)
    (Nat.max (Nat.choose 5 1)
      (Nat.max (Nat.choose 5 2)
        (Nat.max (Nat.choose 5 3)
          (Nat.max (Nat.choose 5 4)
            (Nat.choose 5 5)))))

theorem expansion_max_coefficient :
  ∀ a : ℝ, coefficient_condition a → max_coefficient a = 10 := by
  sorry

end NUMINAMATH_CALUDE_expansion_max_coefficient_l2275_227562


namespace NUMINAMATH_CALUDE_four_valid_dimensions_l2275_227586

/-- The number of valid floor dimensions -/
def valid_floor_dimensions : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    let (a, b) := p
    a ≥ 5 ∧ b > a ∧ (a - 6) * (b - 6) = 36
  ) (Finset.product (Finset.range 100) (Finset.range 100))).card

/-- The theorem stating that there are exactly 4 valid floor dimensions -/
theorem four_valid_dimensions : valid_floor_dimensions = 4 := by
  sorry

end NUMINAMATH_CALUDE_four_valid_dimensions_l2275_227586


namespace NUMINAMATH_CALUDE_inequality_solution_l2275_227573

theorem inequality_solution (x : ℝ) : 2 * (x - 3) < 8 ↔ x < 7 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l2275_227573


namespace NUMINAMATH_CALUDE_problem_solution_l2275_227515

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n < 10000

theorem problem_solution (n : ℕ) (h1 : n = 1221) :
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n} ∧ d.card = 8) ∧
  (∃ (d1 d2 d3 : ℕ), d1 ∣ n ∧ d2 ∣ n ∧ d3 ∣ n ∧
    d1 < d2 ∧ d2 < d3 ∧
    ∀ (x : ℕ), x ∣ n → x ≤ d1 ∨ x = d2 ∨ x ≥ d3 ∧
    d1 + d2 + d3 = 15) ∧
  is_four_digit n ∧
  (∃ (p q r : ℕ), Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    n = p * q * r ∧ p - 5 * q = 2 * r) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2275_227515


namespace NUMINAMATH_CALUDE_shortest_chord_equation_l2275_227558

-- Define the line l
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 3 + t ∧ p.2 = 1 + a * t}

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 4}

-- Define the condition for shortest chord
def shortest_chord (l : Set (ℝ × ℝ)) (C : Set (ℝ × ℝ)) : Prop :=
  ∃ A B : ℝ × ℝ, A ∈ l ∧ A ∈ C ∧ B ∈ l ∧ B ∈ C ∧
  ∀ X Y : ℝ × ℝ, X ∈ l ∧ X ∈ C ∧ Y ∈ l ∧ Y ∈ C →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (X.1 - Y.1)^2 + (X.2 - Y.2)^2

-- Theorem statement
theorem shortest_chord_equation :
  ∃ a : ℝ, shortest_chord (line_l a) circle_C →
  ∀ p : ℝ × ℝ, p ∈ line_l a ↔ p.1 + p.2 = 4 :=
sorry

end NUMINAMATH_CALUDE_shortest_chord_equation_l2275_227558


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2275_227597

theorem inequality_and_equality_condition (a b : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_sum : a + b < 2) :
  (1 / (1 + a^2) + 1 / (1 + b^2) ≤ 2 / (1 + a*b)) ∧
  ((1 / (1 + a^2) + 1 / (1 + b^2) = 2 / (1 + a*b)) ↔ (a = b ∧ a < 1)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2275_227597


namespace NUMINAMATH_CALUDE_line_translation_slope_l2275_227590

/-- A line in the xy-plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Translate a line horizontally and vertically -/
def translate (l : Line) (dx dy : ℝ) : Line :=
  { slope := l.slope,
    intercept := l.intercept + dy - l.slope * dx }

theorem line_translation_slope (l : Line) :
  translate (translate l 3 0) 0 1 = l →
  l.slope = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_line_translation_slope_l2275_227590


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l2275_227576

def U : Finset Int := {-1, 0, 1, 2, 3}
def A : Finset Int := {-1, 0, 2}
def B : Finset Int := {0, 1}

theorem complement_intersection_theorem :
  (U \ A) ∩ (U \ B) = {3} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l2275_227576


namespace NUMINAMATH_CALUDE_area_increase_bound_l2275_227595

/-- A convex polygon -/
structure ConvexPolygon where
  -- Add necessary fields for a convex polygon
  perimeter : ℝ
  area : ℝ

/-- The result of moving all sides of a polygon outward by distance h -/
def moveOutward (poly : ConvexPolygon) (h : ℝ) : ConvexPolygon := sorry

theorem area_increase_bound (poly : ConvexPolygon) (h : ℝ) (h_pos : h > 0) :
  (moveOutward poly h).area - poly.area > poly.perimeter * h + π * h^2 := by
  sorry

end NUMINAMATH_CALUDE_area_increase_bound_l2275_227595


namespace NUMINAMATH_CALUDE_search_plans_count_l2275_227554

/-- Represents the number of children in the group -/
def total_children : ℕ := 8

/-- Represents whether Grace participates in the task -/
inductive GraceParticipation
| Participates
| DoesNotParticipate

/-- Calculates the number of ways to distribute children for the search task -/
def count_search_plans : ℕ :=
  let grace_participates := Nat.choose 7 3  -- Choose 3 out of 7 to go with Grace
  let grace_not_participates := 7 * Nat.choose 6 3  -- Choose 1 to stay, then distribute 6
  grace_participates + grace_not_participates

/-- Theorem stating that the number of different search plans is 175 -/
theorem search_plans_count :
  count_search_plans = 175 := by sorry

end NUMINAMATH_CALUDE_search_plans_count_l2275_227554


namespace NUMINAMATH_CALUDE_chess_game_probability_l2275_227539

theorem chess_game_probability (p_draw p_B_win p_A_win : ℚ) :
  p_draw = 1/2 →
  p_B_win = 1/3 →
  p_draw + p_B_win + p_A_win = 1 →
  p_A_win = 1/6 := by
sorry

end NUMINAMATH_CALUDE_chess_game_probability_l2275_227539


namespace NUMINAMATH_CALUDE_range_of_x_l2275_227583

theorem range_of_x (M : Set ℝ) (h : M = {x ^ 2 | x : ℝ} ∪ {1}) :
  {x : ℝ | x ≠ 1 ∧ x ≠ -1} = {x : ℝ | ∃ y ∈ M, y = x ^ 2} := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2275_227583


namespace NUMINAMATH_CALUDE_total_birds_l2275_227569

/-- Given 3 pairs of birds, prove that the total number of birds is 6. -/
theorem total_birds (pairs : ℕ) (h : pairs = 3) : pairs * 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_l2275_227569


namespace NUMINAMATH_CALUDE_solve_equation_l2275_227527

theorem solve_equation (x : ℚ) : 15 * x = 165 ↔ x = 11 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2275_227527


namespace NUMINAMATH_CALUDE_unique_cube_ending_in_nine_l2275_227571

theorem unique_cube_ending_in_nine :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧ 1000 ≤ n^3 ∧ n^3 < 10000 ∧ n^3 % 10 = 9 ∧ n = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_cube_ending_in_nine_l2275_227571


namespace NUMINAMATH_CALUDE_bowling_ball_surface_area_l2275_227591

theorem bowling_ball_surface_area :
  ∀ d r A : ℝ,
  d = 9 →
  r = d / 2 →
  A = 4 * Real.pi * r^2 →
  A = 81 * Real.pi :=
by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_surface_area_l2275_227591


namespace NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2275_227501

/-- Given a geometric sequence where the first term is 512 and the 6th term is 8,
    the 4th term is 64. -/
theorem geometric_sequence_fourth_term : ∀ (a : ℕ → ℝ),
  (∀ n, a (n + 1) = a n * (a 1)⁻¹ * a 0) →  -- Geometric sequence definition
  a 0 = 512 →                               -- First term is 512
  a 5 = 8 →                                 -- 6th term is 8
  a 3 = 64 :=                               -- 4th term is 64
by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fourth_term_l2275_227501


namespace NUMINAMATH_CALUDE_extreme_points_condition_one_zero_point_no_zero_points_l2275_227588

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (a * x) / (x + 1)

-- Define the derivative of f
def f_deriv (a : ℝ) (x : ℝ) : ℝ := 1 / x + a / ((x + 1) ^ 2)

-- Theorem for the number of extreme points
theorem extreme_points_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f_deriv a x₁ = 0 ∧ f_deriv a x₂ = 0) ↔ a < -4 :=
sorry

-- Theorem for the number of zero points when a ≥ -4
theorem one_zero_point (a : ℝ) (h : a ≥ -4) :
  ∃! x : ℝ, f a x = 0 :=
sorry

-- Theorem for the number of zero points when a < -4
theorem no_zero_points (a : ℝ) (h : a < -4) :
  ¬∃ x : ℝ, f a x = 0 :=
sorry

end NUMINAMATH_CALUDE_extreme_points_condition_one_zero_point_no_zero_points_l2275_227588


namespace NUMINAMATH_CALUDE_girls_combined_average_l2275_227559

structure School where
  boys_score : ℝ
  girls_score : ℝ
  combined_score : ℝ

def central : School := { boys_score := 68, girls_score := 72, combined_score := 70 }
def delta : School := { boys_score := 78, girls_score := 85, combined_score := 80 }

def combined_boys_score : ℝ := 74

theorem girls_combined_average (c d : ℝ) 
  (hc : c > 0) (hd : d > 0)
  (h_central : c * central.boys_score + c * central.girls_score = (c + c) * central.combined_score)
  (h_delta : d * delta.boys_score + d * delta.girls_score = (d + d) * delta.combined_score)
  (h_boys : (c * central.boys_score + d * delta.boys_score) / (c + d) = combined_boys_score) :
  (c * central.girls_score + d * delta.girls_score) / (c + d) = 79 := by
  sorry

end NUMINAMATH_CALUDE_girls_combined_average_l2275_227559


namespace NUMINAMATH_CALUDE_range_of_m_l2275_227503

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ (a b : ℝ), a > b ∧ b > 0 ∧ a = 8 - m ∧ b = 2*m - 1

def q (m : ℝ) : Prop := (m + 1) * (m - 2) < 0

-- Define the theorem
theorem range_of_m : 
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Set.Ioo (-1 : ℝ) (1/2) ∪ Set.Icc 2 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2275_227503


namespace NUMINAMATH_CALUDE_max_value_function_l2275_227526

theorem max_value_function (a : ℝ) (h : a > 0) :
  ∃ (max : ℝ), ∀ (x : ℝ), x > 0 → a > 2*x → x*(a - 2*x) ≤ max ∧
  ∃ (x₀ : ℝ), x₀ > 0 ∧ a > 2*x₀ ∧ x₀*(a - 2*x₀) = max :=
by sorry

end NUMINAMATH_CALUDE_max_value_function_l2275_227526


namespace NUMINAMATH_CALUDE_taller_tree_height_l2275_227580

/-- Given two trees with specific height relationships, prove the height of the taller tree -/
theorem taller_tree_height (h_shorter h_taller : ℝ) : 
  h_taller = h_shorter + 20 →  -- The top of one tree is 20 feet higher
  h_shorter / h_taller = 2 / 3 →  -- The heights are in the ratio 2:3
  h_shorter = 40 →  -- The shorter tree is 40 feet tall
  h_taller = 60 := by sorry

end NUMINAMATH_CALUDE_taller_tree_height_l2275_227580


namespace NUMINAMATH_CALUDE_power_product_cube_l2275_227563

theorem power_product_cube (x y : ℝ) : (x^2 * y)^3 = x^6 * y^3 := by
  sorry

end NUMINAMATH_CALUDE_power_product_cube_l2275_227563


namespace NUMINAMATH_CALUDE_man_speed_man_speed_approx_6kmh_l2275_227542

/-- Calculates the speed of a man given the parameters of a train passing him --/
theorem man_speed (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let relative_speed := train_length / passing_time
  let man_speed_ms := relative_speed - train_speed_ms
  let man_speed_kmh := man_speed_ms * 3600 / 1000
  man_speed_kmh

/-- The speed of the man is approximately 6 km/h --/
theorem man_speed_approx_6kmh :
  ∃ ε > 0, |man_speed 160 90 6 - 6| < ε :=
sorry

end NUMINAMATH_CALUDE_man_speed_man_speed_approx_6kmh_l2275_227542


namespace NUMINAMATH_CALUDE_sector_arc_length_l2275_227529

/-- Given a circular sector with circumference 4 and central angle 2 radians, 
    the arc length of the sector is 2. -/
theorem sector_arc_length (r : ℝ) (l : ℝ) : 
  l + 2 * r = 4 →  -- circumference of the sector
  l = 2 * r →      -- relationship between arc length and radius
  l = 2 :=         -- arc length is 2
by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l2275_227529


namespace NUMINAMATH_CALUDE_beta_value_l2275_227514

/-- Given α = 2023°, if β has the same terminal side as α and β ∈ (0, 2π), then β = 223π/180 -/
theorem beta_value (α β : Real) : 
  α = 2023 * (π / 180) →
  (∃ k : ℤ, β = α + k * 2 * π) →
  β ∈ Set.Ioo 0 (2 * π) →
  β = 223 * (π / 180) := by
  sorry

end NUMINAMATH_CALUDE_beta_value_l2275_227514


namespace NUMINAMATH_CALUDE_revenue_decrease_l2275_227545

theorem revenue_decrease (projected_increase : ℝ) (actual_vs_projected : ℝ) : 
  projected_increase = 0.30 →
  actual_vs_projected = 0.57692307692307686 →
  1 - actual_vs_projected * (1 + projected_increase) = 0.25 := by
sorry

end NUMINAMATH_CALUDE_revenue_decrease_l2275_227545


namespace NUMINAMATH_CALUDE_commodity_price_problem_l2275_227549

theorem commodity_price_problem (total_cost first_price second_price : ℕ) :
  total_cost = 827 →
  first_price = second_price + 127 →
  total_cost = first_price + second_price →
  first_price = 477 := by
  sorry

end NUMINAMATH_CALUDE_commodity_price_problem_l2275_227549


namespace NUMINAMATH_CALUDE_square_fraction_above_line_l2275_227512

-- Define the square
def square_vertices : List (ℝ × ℝ) := [(4, 1), (7, 1), (7, 4), (4, 4)]

-- Define the line passing through two points
def line_points : List (ℝ × ℝ) := [(4, 3), (7, 1)]

-- Function to calculate the fraction of square area above the line
def fraction_above_line (square : List (ℝ × ℝ)) (line : List (ℝ × ℝ)) : ℚ :=
  sorry

-- Theorem statement
theorem square_fraction_above_line :
  fraction_above_line square_vertices line_points = 1/2 :=
sorry

end NUMINAMATH_CALUDE_square_fraction_above_line_l2275_227512


namespace NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l2275_227533

/-- A decagon is a polygon with 10 sides -/
def Decagon : Type := Unit

/-- The number of vertices in a decagon -/
def num_vertices : Nat := 10

/-- The number of ways to choose 3 distinct vertices from a decagon -/
def total_choices : Nat := Nat.choose num_vertices 3

/-- The number of ways to choose 3 adjacent vertices in a decagon -/
def adjacent_choices : Nat := num_vertices

/-- The probability of choosing 3 adjacent vertices in a decagon -/
def prob_adjacent_vertices : Rat := adjacent_choices / total_choices

theorem decagon_adjacent_vertices_probability :
  prob_adjacent_vertices = 1 / 12 := by
  sorry

end NUMINAMATH_CALUDE_decagon_adjacent_vertices_probability_l2275_227533


namespace NUMINAMATH_CALUDE_circle_tangency_theorem_l2275_227598

/-- Represents a circle with a center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is internally tangent to another circle -/
def is_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c2.radius - c1.radius)^2

/-- Checks if three points are collinear -/
def are_collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

/-- Represents the length of a chord as m√n/p -/
structure ChordLength where
  m : ℕ
  n : ℕ
  p : ℕ

/-- Checks if two numbers are relatively prime -/
def are_relatively_prime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

/-- Checks if a number is not divisible by the square of any prime -/
def not_divisible_by_prime_square (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → ¬(p^2 ∣ n)

theorem circle_tangency_theorem (C1 C2 C3 : Circle) (chord : ChordLength) :
  are_externally_tangent C1 C2 ∧
  is_internally_tangent C1 C3 ∧
  is_internally_tangent C2 C3 ∧
  are_collinear C1.center C2.center C3.center ∧
  C1.radius = 5 ∧
  C2.radius = 13 ∧
  are_relatively_prime chord.m chord.p ∧
  not_divisible_by_prime_square chord.n →
  chord.m + chord.n + chord.p = 455 := by
  sorry


end NUMINAMATH_CALUDE_circle_tangency_theorem_l2275_227598


namespace NUMINAMATH_CALUDE_percentage_of_women_in_survey_l2275_227550

theorem percentage_of_women_in_survey (w : ℝ) (m : ℝ) : 
  w + m = 100 →
  (3/4 : ℝ) * w + (9/10 : ℝ) * m = 84 →
  w = 40 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_women_in_survey_l2275_227550


namespace NUMINAMATH_CALUDE_min_xy_value_l2275_227557

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4*x*y - x - 2*y = 4) :
  ∀ z, z = x*y → z ≥ 2 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l2275_227557


namespace NUMINAMATH_CALUDE_smallest_period_five_cycles_l2275_227518

def is_periodic (f : ℝ → ℝ) (T : ℝ) : Prop :=
  ∀ x, f (x + T) = f x

def completes_n_cycles (f : ℝ → ℝ) (n : ℕ) (a b : ℝ) : Prop :=
  ∃ T > 0, is_periodic f T ∧ n * T = b - a

theorem smallest_period_five_cycles (f : ℝ → ℝ) 
  (h : completes_n_cycles f 5 0 (2 * Real.pi)) :
  ∃ T > 0, is_periodic f T ∧ 
    (∀ T' > 0, is_periodic f T' → T ≤ T') ∧
    T = 2 * Real.pi / 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_period_five_cycles_l2275_227518


namespace NUMINAMATH_CALUDE_sequence_modulo_eight_property_l2275_227517

theorem sequence_modulo_eight_property (s : ℕ → ℕ) 
  (h : ∀ n : ℕ, s (n + 2) = s (n + 1) + s n) : 
  ∃ r : ℤ, ∀ n : ℕ, ¬ (8 ∣ (s n - r)) :=
sorry

end NUMINAMATH_CALUDE_sequence_modulo_eight_property_l2275_227517


namespace NUMINAMATH_CALUDE_perpendicular_lines_sum_l2275_227552

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1

/-- Definition of Line 1: ax + 4y - 2 = 0 -/
def line1 (a : ℝ) (x y : ℝ) : Prop := a * x + 4 * y - 2 = 0

/-- Definition of Line 2: 2x - 5y + b = 0 -/
def line2 (b : ℝ) (x y : ℝ) : Prop := 2 * x - 5 * y + b = 0

/-- The foot of the perpendicular (1, c) lies on both lines -/
def foot_on_lines (a b c : ℝ) : Prop := line1 a 1 c ∧ line2 b 1 c

theorem perpendicular_lines_sum (a b c : ℝ) : 
  perpendicular (-a/4) (2/5) → foot_on_lines a b c → a + b + c = -4 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_sum_l2275_227552


namespace NUMINAMATH_CALUDE_problem_1_l2275_227513

theorem problem_1 : 2^2 - 2023^0 + |3 - Real.pi| = Real.pi := by sorry

end NUMINAMATH_CALUDE_problem_1_l2275_227513


namespace NUMINAMATH_CALUDE_circle_diameter_l2275_227534

theorem circle_diameter (A : ℝ) (r : ℝ) (d : ℝ) : 
  A = 4 * Real.pi → A = Real.pi * r^2 → d = 2 * r → d = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_diameter_l2275_227534


namespace NUMINAMATH_CALUDE_decimal_addition_subtraction_l2275_227575

theorem decimal_addition_subtraction :
  (0.513 : ℝ) + (0.0067 : ℝ) - (0.048 : ℝ) = (0.4717 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_subtraction_l2275_227575


namespace NUMINAMATH_CALUDE_perfect_square_sequence_l2275_227508

theorem perfect_square_sequence (a b : ℤ) 
  (h : ∀ n : ℕ, ∃ x : ℤ, 2^n * a + b = x^2) : 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sequence_l2275_227508


namespace NUMINAMATH_CALUDE_max_value_P_l2275_227582

theorem max_value_P (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a^2 + b^2 + c^2 = 2) :
  let P := (Real.sqrt (b^2 + c^2)) / (3 - a) + (Real.sqrt (c^2 + a^2)) / (3 - b) + a + b - 2022 * c
  P ≤ 3 ∧ (P = 3 ↔ a = 1 ∧ b = 1 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_max_value_P_l2275_227582


namespace NUMINAMATH_CALUDE_movie_ticket_difference_l2275_227522

theorem movie_ticket_difference (romance_tickets horror_tickets : ℕ) : 
  romance_tickets = 25 → 
  horror_tickets = 93 → 
  horror_tickets - 3 * romance_tickets = 18 := by
sorry

end NUMINAMATH_CALUDE_movie_ticket_difference_l2275_227522


namespace NUMINAMATH_CALUDE_ab_value_l2275_227578

theorem ab_value (a b : ℤ) (h1 : |a| = 5) (h2 : b = -3) (h3 : a < b) : a * b = 15 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2275_227578


namespace NUMINAMATH_CALUDE_multiply_powers_of_y_l2275_227532

theorem multiply_powers_of_y (y : ℝ) : 5 * y^3 * (3 * y^2) = 15 * y^5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_powers_of_y_l2275_227532


namespace NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2275_227581

def f (x : ℝ) : ℝ := x^2 - 2*x + 4

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h1 : ∀ n, a (n + 1) - a n = d) 
  (h2 : a 1 = f (d - 1)) 
  (h3 : a 3 = f (d + 1)) :
  ∀ n, a n = 2 * n + 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_general_term_l2275_227581


namespace NUMINAMATH_CALUDE_chef_nuts_total_weight_l2275_227520

theorem chef_nuts_total_weight (almond_weight pecan_weight : Real) 
  (h1 : almond_weight = 0.14)
  (h2 : pecan_weight = 0.38) :
  almond_weight + pecan_weight = 0.52 := by
sorry

end NUMINAMATH_CALUDE_chef_nuts_total_weight_l2275_227520


namespace NUMINAMATH_CALUDE_solution_inequality1_solution_system_inequalities_l2275_227516

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := x + 1 > 2*x - 3

def inequality2 (x : ℝ) : Prop := 2*x - 1 > x

def inequality3 (x : ℝ) : Prop := (x + 5) / 2 - x ≥ 1

-- Theorem for the first inequality
theorem solution_inequality1 : 
  {x : ℝ | inequality1 x} = {x : ℝ | x < 4} :=
sorry

-- Theorem for the system of inequalities
theorem solution_system_inequalities :
  {x : ℝ | inequality2 x ∧ inequality3 x} = {x : ℝ | 1 < x ∧ x ≤ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_inequality1_solution_system_inequalities_l2275_227516


namespace NUMINAMATH_CALUDE_sum_of_specific_numbers_l2275_227585

theorem sum_of_specific_numbers : 3 + 33 + 333 + 33.3 = 402.3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_specific_numbers_l2275_227585


namespace NUMINAMATH_CALUDE_square_properties_l2275_227536

theorem square_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ a^2 - 10 = 0 ∧ 3 < a ∧ a < 4 := by
  sorry

end NUMINAMATH_CALUDE_square_properties_l2275_227536


namespace NUMINAMATH_CALUDE_initial_white_lights_correct_l2275_227541

/-- The number of white lights Malcolm had initially -/
def initial_white_lights : ℕ := 59

/-- The number of red lights Malcolm bought -/
def red_lights : ℕ := 12

/-- The number of blue lights Malcolm bought -/
def blue_lights : ℕ := 3 * red_lights

/-- The number of green lights Malcolm bought -/
def green_lights : ℕ := 6

/-- The number of colored lights Malcolm still needs to buy -/
def remaining_lights : ℕ := 5

/-- Theorem stating that the initial number of white lights is correct -/
theorem initial_white_lights_correct : 
  initial_white_lights = red_lights + blue_lights + green_lights + remaining_lights :=
sorry

end NUMINAMATH_CALUDE_initial_white_lights_correct_l2275_227541


namespace NUMINAMATH_CALUDE_square_sum_equals_twice_square_a_l2275_227537

theorem square_sum_equals_twice_square_a 
  (x y a θ : ℝ) 
  (h1 : x * Real.cos θ - y * Real.sin θ = a) 
  (h2 : (x - a * Real.sin θ)^2 + (y - a * Real.cos θ)^2 = a^2) : 
  x^2 + y^2 = 2 * a^2 := by
sorry

end NUMINAMATH_CALUDE_square_sum_equals_twice_square_a_l2275_227537


namespace NUMINAMATH_CALUDE_soccer_team_lineup_count_l2275_227546

theorem soccer_team_lineup_count :
  let total_players : ℕ := 15
  let roles : ℕ := 7
  total_players.factorial / (total_players - roles).factorial = 2541600 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_lineup_count_l2275_227546


namespace NUMINAMATH_CALUDE_raspberry_pies_count_l2275_227504

/-- The total number of pies -/
def total_pies : ℕ := 36

/-- The ratio of apple pies -/
def apple_ratio : ℕ := 1

/-- The ratio of blueberry pies -/
def blueberry_ratio : ℕ := 3

/-- The ratio of cherry pies -/
def cherry_ratio : ℕ := 2

/-- The ratio of raspberry pies -/
def raspberry_ratio : ℕ := 4

/-- The sum of all ratios -/
def total_ratio : ℕ := apple_ratio + blueberry_ratio + cherry_ratio + raspberry_ratio

/-- Theorem: The number of raspberry pies is 14.4 -/
theorem raspberry_pies_count : 
  (total_pies : ℚ) * raspberry_ratio / total_ratio = 14.4 := by
  sorry

end NUMINAMATH_CALUDE_raspberry_pies_count_l2275_227504


namespace NUMINAMATH_CALUDE_almost_perfect_is_odd_square_l2275_227577

/-- Sum of divisors function -/
def sigma (N : ℕ+) : ℕ := sorry

/-- Definition of almost perfect number -/
def is_almost_perfect (N : ℕ+) : Prop :=
  sigma N = 2 * N.val + 1

/-- Main theorem: Every almost perfect number is a square of an odd number -/
theorem almost_perfect_is_odd_square (N : ℕ+) (h : is_almost_perfect N) :
  ∃ M : ℕ, N.val = M^2 ∧ Odd M := by sorry

end NUMINAMATH_CALUDE_almost_perfect_is_odd_square_l2275_227577


namespace NUMINAMATH_CALUDE_balance_point_specific_rod_l2275_227570

/-- Represents the rod with attached weights -/
structure WeightedRod where
  length : Real
  weights : List (Real × Real)  -- List of (position, weight) pairs

/-- Calculates the balance point of a weighted rod -/
def balancePoint (rod : WeightedRod) : Real :=
  sorry

/-- Theorem stating the balance point for the specific rod configuration -/
theorem balance_point_specific_rod :
  let rod : WeightedRod := {
    length := 4,
    weights := [(0, 20), (1, 30), (2, 40), (3, 50), (4, 60)]
  }
  balancePoint rod = 2.5 := by sorry

end NUMINAMATH_CALUDE_balance_point_specific_rod_l2275_227570


namespace NUMINAMATH_CALUDE_q_invest_time_is_20_l2275_227528

/-- Represents a business partnership between two partners -/
structure Partnership where
  investment_ratio : ℚ × ℚ
  profit_ratio : ℚ × ℚ
  p_invest_time : ℕ

/-- Calculates the investment time for partner q given a Partnership -/
def q_invest_time (p : Partnership) : ℚ :=
  (p.profit_ratio.2 * p.investment_ratio.1 * p.p_invest_time : ℚ) / (p.profit_ratio.1 * p.investment_ratio.2)

theorem q_invest_time_is_20 (p : Partnership) 
  (h1 : p.investment_ratio = (7, 5))
  (h2 : p.profit_ratio = (7, 10))
  (h3 : p.p_invest_time = 10) :
  q_invest_time p = 20 := by
  sorry

end NUMINAMATH_CALUDE_q_invest_time_is_20_l2275_227528


namespace NUMINAMATH_CALUDE_percentage_of_eight_l2275_227509

theorem percentage_of_eight : ∃ p : ℝ, (p / 100) * 8 = 0.06 ∧ p = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_eight_l2275_227509


namespace NUMINAMATH_CALUDE_final_balance_is_214_12_l2275_227594

/-- Calculates the credit card balance after five months given the initial balance and monthly transactions. -/
def creditCardBalance (initialBalance : ℚ) 
  (month1Interest : ℚ)
  (month2Spent month2Payment month2Interest : ℚ)
  (month3Spent month3Payment month3Interest : ℚ)
  (month4Spent month4Payment month4Interest : ℚ)
  (month5Spent month5Payment month5Interest : ℚ) : ℚ :=
  let balance1 := initialBalance * (1 + month1Interest)
  let balance2 := (balance1 + month2Spent - month2Payment) * (1 + month2Interest)
  let balance3 := (balance2 + month3Spent - month3Payment) * (1 + month3Interest)
  let balance4 := (balance3 + month4Spent - month4Payment) * (1 + month4Interest)
  let balance5 := (balance4 + month5Spent - month5Payment) * (1 + month5Interest)
  balance5

/-- Theorem stating that the credit card balance after five months is $214.12 given the specific transactions. -/
theorem final_balance_is_214_12 : 
  creditCardBalance 50 0.2 20 15 0.18 30 5 0.22 25 20 0.15 40 10 0.2 = 214.12 := by
  sorry

end NUMINAMATH_CALUDE_final_balance_is_214_12_l2275_227594


namespace NUMINAMATH_CALUDE_alex_jogging_speed_l2275_227599

/-- Given the jogging speeds of Eugene, Brianna, Katie, and Alex, prove that Alex jogs at 2.4 miles per hour. -/
theorem alex_jogging_speed 
  (eugene_speed : ℝ) 
  (brianna_speed : ℝ) 
  (katie_speed : ℝ) 
  (alex_speed : ℝ) 
  (h1 : eugene_speed = 5)
  (h2 : brianna_speed = 4/5 * eugene_speed)
  (h3 : katie_speed = 6/5 * brianna_speed)
  (h4 : alex_speed = 1/2 * katie_speed) :
  alex_speed = 2.4 := by
  sorry

end NUMINAMATH_CALUDE_alex_jogging_speed_l2275_227599


namespace NUMINAMATH_CALUDE_problem_statement_l2275_227519

theorem problem_statement (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) :
  2 * (Real.cos (π / 6 + α / 2))^2 + 1 = 7 / 3 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2275_227519


namespace NUMINAMATH_CALUDE_cubic_factorization_l2275_227568

theorem cubic_factorization (a : ℝ) : a^3 - a = a * (a + 1) * (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l2275_227568


namespace NUMINAMATH_CALUDE_fruit_salad_oranges_l2275_227531

theorem fruit_salad_oranges :
  ∀ (s k a o : ℕ),
    s + k + a + o = 360 →
    s = k / 2 →
    a = 2 * o →
    o = 3 * s →
    o = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_fruit_salad_oranges_l2275_227531


namespace NUMINAMATH_CALUDE_A_inter_complement_B_l2275_227574

def U : Set Int := Set.univ

def A : Set Int := {-2, -1, 0, 1, 2}

def B : Set Int := {x | x^2 + 2*x = 0}

theorem A_inter_complement_B : A ∩ (U \ B) = {-1, 1, 2} := by sorry

end NUMINAMATH_CALUDE_A_inter_complement_B_l2275_227574


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2275_227593

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - 2*x - 3 < 0} = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2275_227593


namespace NUMINAMATH_CALUDE_floor_times_self_eq_54_l2275_227505

theorem floor_times_self_eq_54 (x : ℝ) :
  x > 0 ∧ (⌊x⌋ : ℝ) * x = 54 → x = 54 / 7 := by
  sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_54_l2275_227505


namespace NUMINAMATH_CALUDE_regression_correlation_zero_l2275_227506

/-- Regression coefficient -/
def regression_coefficient (X Y : List ℝ) : ℝ := sorry

/-- Correlation coefficient -/
def correlation_coefficient (X Y : List ℝ) : ℝ := sorry

theorem regression_correlation_zero (X Y : List ℝ) :
  regression_coefficient X Y = 0 → correlation_coefficient X Y = 0 := by
  sorry

end NUMINAMATH_CALUDE_regression_correlation_zero_l2275_227506


namespace NUMINAMATH_CALUDE_dans_purchases_cost_l2275_227521

/-- The total cost of Dan's purchases, given the cost of a snake toy, a cage, and finding a dollar bill. -/
theorem dans_purchases_cost (snake_toy_cost cage_cost found_money : ℚ) : 
  snake_toy_cost = 11.76 →
  cage_cost = 14.54 →
  found_money = 1 →
  snake_toy_cost + cage_cost - found_money = 25.30 := by
  sorry

end NUMINAMATH_CALUDE_dans_purchases_cost_l2275_227521


namespace NUMINAMATH_CALUDE_odd_expressions_l2275_227572

-- Define positive odd integers
def is_positive_odd (n : ℤ) : Prop := n > 0 ∧ ∃ k : ℤ, n = 2*k + 1

-- Theorem statement
theorem odd_expressions (p q : ℤ) 
  (hp : is_positive_odd p) (hq : is_positive_odd q) : 
  ∃ m n : ℤ, p * q + 2 = 2*m + 1 ∧ p^3 * q + q^2 = 2*n + 1 :=
sorry

end NUMINAMATH_CALUDE_odd_expressions_l2275_227572


namespace NUMINAMATH_CALUDE_triangle_cosine_sum_l2275_227530

/-- For a triangle ABC with angles A, B, C satisfying A/B = B/C = 1/3, 
    the sum of cosines of these angles is (1 + √13) / 4 -/
theorem triangle_cosine_sum (A B C : Real) : 
  A > 0 → B > 0 → C > 0 →
  A + B + C = Real.pi →
  A / B = 1 / 3 →
  B / C = 1 / 3 →
  Real.cos A + Real.cos B + Real.cos C = (1 + Real.sqrt 13) / 4 := by
  sorry

end NUMINAMATH_CALUDE_triangle_cosine_sum_l2275_227530


namespace NUMINAMATH_CALUDE_inscribed_parallelogram_sides_l2275_227510

/-- Triangle ABC with inscribed parallelogram BKLM -/
structure InscribedParallelogram where
  -- Side lengths of triangle ABC
  AB : ℝ
  BC : ℝ
  -- Sides of parallelogram BKLM
  BM : ℝ
  BK : ℝ
  -- Condition that BKLM is inscribed in ABC
  inscribed : BM ≤ BC ∧ BK ≤ AB

/-- The theorem stating the possible side lengths of the inscribed parallelogram -/
theorem inscribed_parallelogram_sides
  (T : InscribedParallelogram)
  (h_AB : T.AB = 18)
  (h_BC : T.BC = 12)
  (h_area : T.BM * T.BK = 48) :
  (T.BM = 8 ∧ T.BK = 6) ∨ (T.BM = 4 ∧ T.BK = 12) := by
  sorry

#check inscribed_parallelogram_sides

end NUMINAMATH_CALUDE_inscribed_parallelogram_sides_l2275_227510


namespace NUMINAMATH_CALUDE_cube_root_and_square_root_l2275_227523

theorem cube_root_and_square_root (a b : ℝ) : 
  (b - 4)^(1/3) = -2 → 
  b = -4 ∧ 
  Real.sqrt (5 * a - b) = 3 :=
by sorry

end NUMINAMATH_CALUDE_cube_root_and_square_root_l2275_227523


namespace NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l2275_227502

/-- A line in 2D space represented by the equation ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A triangle in 2D space represented by its three vertices -/
structure Triangle where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ

/-- Function to create a triangle from a line that intersects the coordinate axes -/
def triangleFromLine (l : Line) : Triangle := sorry

/-- Function to calculate the sum of altitudes of a triangle -/
def sumOfAltitudes (t : Triangle) : ℝ := sorry

/-- Theorem stating that for the given line, the sum of altitudes of the formed triangle
    is equal to 23 + 60/√409 -/
theorem sum_of_altitudes_for_specific_line :
  let l : Line := { a := 20, b := 3, c := 60 }
  let t : Triangle := triangleFromLine l
  sumOfAltitudes t = 23 + 60 / Real.sqrt 409 := by sorry

end NUMINAMATH_CALUDE_sum_of_altitudes_for_specific_line_l2275_227502


namespace NUMINAMATH_CALUDE_expression_simplification_l2275_227566

theorem expression_simplification :
  ((3 + 4 + 6 + 7) / 4) + ((2 * 6 + 10) / 4) = 10.5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2275_227566


namespace NUMINAMATH_CALUDE_pushup_comparison_l2275_227540

theorem pushup_comparison (zachary david emily : ℕ) 
  (h1 : zachary = 51)
  (h2 : david = 44)
  (h3 : emily = 37) :
  zachary = (david + emily) - 30 :=
by sorry

end NUMINAMATH_CALUDE_pushup_comparison_l2275_227540


namespace NUMINAMATH_CALUDE_coordinate_system_change_l2275_227524

/-- Represents a point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents a coordinate system in a 2D plane -/
structure CoordinateSystem where
  origin : Point2D

/-- Returns the coordinates of a point in a given coordinate system -/
def getCoordinates (p : Point2D) (cs : CoordinateSystem) : Point2D :=
  { x := p.x - cs.origin.x, y := p.y - cs.origin.y }

theorem coordinate_system_change 
  (A B : Point2D) 
  (csA csB : CoordinateSystem) 
  (h1 : csA.origin = A) 
  (h2 : csB.origin = B) 
  (h3 : getCoordinates B csA = { x := a, y := b }) :
  getCoordinates A csB = { x := -a, y := -b } := by
  sorry


end NUMINAMATH_CALUDE_coordinate_system_change_l2275_227524


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l2275_227544

def f (x : ℝ) := x^2 + 12*x - 15

theorem solution_exists_in_interval :
  ∃ x ∈ Set.Ioo 1.1 1.2, f x = 0 :=
by
  have h1 : f 1.1 < 0 := by sorry
  have h2 : f 1.2 > 0 := by sorry
  sorry

end NUMINAMATH_CALUDE_solution_exists_in_interval_l2275_227544


namespace NUMINAMATH_CALUDE_function_difference_l2275_227561

theorem function_difference (f : ℝ → ℝ) (h : ∀ x, f x = 9^x) :
  ∀ x, f (x + 1) - f x = 8 * f x := by
  sorry

end NUMINAMATH_CALUDE_function_difference_l2275_227561


namespace NUMINAMATH_CALUDE_simplify_expression_l2275_227547

theorem simplify_expression (x : ℝ) : (3*x)^5 + (5*x)*(x^4) - 7*x^5 = 241*x^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2275_227547


namespace NUMINAMATH_CALUDE_show_charge_day3_l2275_227548

/-- The charge per person on the first day in rupees -/
def charge_day1 : ℚ := 15

/-- The charge per person on the second day in rupees -/
def charge_day2 : ℚ := 15/2

/-- The ratio of attendance on the first day -/
def ratio_day1 : ℕ := 2

/-- The ratio of attendance on the second day -/
def ratio_day2 : ℕ := 5

/-- The ratio of attendance on the third day -/
def ratio_day3 : ℕ := 13

/-- The average charge per person for the whole show in rupees -/
def average_charge : ℚ := 5

/-- The charge per person on the third day in rupees -/
def charge_day3 : ℚ := 5/2

theorem show_charge_day3 :
  let total_ratio := ratio_day1 + ratio_day2 + ratio_day3
  let total_charge := ratio_day1 * charge_day1 + ratio_day2 * charge_day2 + ratio_day3 * charge_day3
  average_charge = total_charge / total_ratio := by
  sorry

end NUMINAMATH_CALUDE_show_charge_day3_l2275_227548


namespace NUMINAMATH_CALUDE_solve_for_e_l2275_227525

theorem solve_for_e (x e : ℝ) (h1 : (10 * x + 2) / 4 - (3 * x - e) / 18 = (2 * x + 4) / 3)
                     (h2 : x = 0.3) : e = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_e_l2275_227525


namespace NUMINAMATH_CALUDE_parking_problem_l2275_227555

/-- Calculates the number of vehicles that can still park in a lot -/
def vehiclesCanPark (totalSpaces : ℕ) (caravanSpaces : ℕ) (caravansParked : ℕ) : ℕ :=
  totalSpaces - (caravanSpaces * caravansParked)

/-- Theorem: Given the problem conditions, 24 vehicles can still park -/
theorem parking_problem :
  vehiclesCanPark 30 2 3 = 24 := by
  sorry

end NUMINAMATH_CALUDE_parking_problem_l2275_227555


namespace NUMINAMATH_CALUDE_sqrt_511100_approx_l2275_227584

-- Define the approximation relation
def approx (x y : ℝ) : Prop := ∃ ε > 0, |x - y| < ε

-- State the theorem
theorem sqrt_511100_approx :
  approx (Real.sqrt 51.11) 7.149 →
  approx (Real.sqrt 511100) 714.9 :=
by
  sorry

end NUMINAMATH_CALUDE_sqrt_511100_approx_l2275_227584


namespace NUMINAMATH_CALUDE_mississippi_arrangements_l2275_227500

def word : String := "MISSISSIPPI"

def letter_count (s : String) (c : Char) : Nat :=
  s.toList.filter (· = c) |>.length

theorem mississippi_arrangements :
  (Nat.factorial 11) / 
  (Nat.factorial (letter_count word 'S') * 
   Nat.factorial (letter_count word 'I') * 
   Nat.factorial (letter_count word 'P') * 
   Nat.factorial (letter_count word 'M')) = 34650 := by
  sorry

end NUMINAMATH_CALUDE_mississippi_arrangements_l2275_227500


namespace NUMINAMATH_CALUDE_range_of_f_l2275_227551

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 6*x - 9

-- Define the open interval (1, 4)
def open_interval : Set ℝ := {x | 1 < x ∧ x < 4}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ open_interval, f x = y} = {y | -18 ≤ y ∧ y < -14} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2275_227551


namespace NUMINAMATH_CALUDE_max_sum_of_digits_24hour_watch_l2275_227564

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours ≤ 23
  minutes_valid : minutes ≤ 59
  seconds_valid : seconds ≤ 59

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of all digits in a Time24 -/
def totalSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The theorem to be proved -/
theorem max_sum_of_digits_24hour_watch :
  ∃ (t : Time24), ∀ (t' : Time24), totalSumOfDigits t' ≤ totalSumOfDigits t ∧ totalSumOfDigits t = 38 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_digits_24hour_watch_l2275_227564


namespace NUMINAMATH_CALUDE_colored_regions_bound_l2275_227507

/-- A structure representing a plane with n lines and colored regions -/
structure ColoredPlane where
  n : ℕ
  n_ge_2 : n ≥ 2

/-- The number of colored regions in a ColoredPlane -/
def num_colored_regions (p : ColoredPlane) : ℕ := sorry

/-- Theorem stating that the number of colored regions is bounded -/
theorem colored_regions_bound (p : ColoredPlane) :
  num_colored_regions p ≤ (p.n^2 + p.n) / 3 := by sorry

end NUMINAMATH_CALUDE_colored_regions_bound_l2275_227507


namespace NUMINAMATH_CALUDE_parallelogram_dimensions_l2275_227592

/-- Proves the side lengths and perimeter of a parallelogram given its area, side ratio, and one angle -/
theorem parallelogram_dimensions (area : ℝ) (angle : ℝ) (h_area : area = 972) (h_angle : angle = 45 * π / 180) :
  ∃ (side1 side2 perimeter : ℝ),
    side1 / side2 = 4 / 3 ∧
    area = side1 * side2 * Real.sin angle ∧
    side1 = 36 * 2^(3/4) ∧
    side2 = 27 * 2^(3/4) ∧
    perimeter = 126 * 2^(3/4) := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_dimensions_l2275_227592


namespace NUMINAMATH_CALUDE_water_for_bathing_is_two_l2275_227579

/-- Calculates the water needed for bathing per horse per day -/
def water_for_bathing (initial_horses : ℕ) (added_horses : ℕ) (drinking_water_per_horse : ℕ) (total_days : ℕ) (total_water : ℕ) : ℚ :=
  let total_horses := initial_horses + added_horses
  let total_drinking_water := total_horses * drinking_water_per_horse * total_days
  let total_bathing_water := total_water - total_drinking_water
  (total_bathing_water : ℚ) / (total_horses * total_days : ℚ)

/-- Theorem: Given the conditions, each horse needs 2 liters of water for bathing per day -/
theorem water_for_bathing_is_two :
  water_for_bathing 3 5 5 28 1568 = 2 := by
  sorry

end NUMINAMATH_CALUDE_water_for_bathing_is_two_l2275_227579
