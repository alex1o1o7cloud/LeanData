import Mathlib

namespace NUMINAMATH_CALUDE_row_swap_matrix_l1744_174466

theorem row_swap_matrix : ∃ (N : Matrix (Fin 2) (Fin 2) ℝ),
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), N * A = ![![A 1 0, A 1 1], ![A 0 0, A 0 1]] :=
by
  sorry

end NUMINAMATH_CALUDE_row_swap_matrix_l1744_174466


namespace NUMINAMATH_CALUDE_maggots_eaten_first_correct_l1744_174429

/-- The number of maggots eaten by the beetle in the first feeding -/
def maggots_eaten_first : ℕ := 17

/-- The total number of maggots served -/
def total_maggots : ℕ := 20

/-- The number of maggots eaten in the second feeding -/
def maggots_eaten_second : ℕ := 3

/-- Theorem stating that the number of maggots eaten in the first feeding is correct -/
theorem maggots_eaten_first_correct : 
  maggots_eaten_first = total_maggots - maggots_eaten_second := by
  sorry

end NUMINAMATH_CALUDE_maggots_eaten_first_correct_l1744_174429


namespace NUMINAMATH_CALUDE_cosine_properties_l1744_174467

theorem cosine_properties (x : ℝ) : 
  (fun (x : ℝ) => Real.cos x) (Real.pi + x) = -(fun (x : ℝ) => Real.cos x) x ∧ 
  (fun (x : ℝ) => Real.cos x) (-x) = (fun (x : ℝ) => Real.cos x) x :=
by sorry

end NUMINAMATH_CALUDE_cosine_properties_l1744_174467


namespace NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1744_174438

theorem larger_solution_of_quadratic (x : ℝ) : 
  x^2 - 13*x + 42 = 0 ∧ x ≠ 6 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_larger_solution_of_quadratic_l1744_174438


namespace NUMINAMATH_CALUDE_subtraction_equivalence_l1744_174415

theorem subtraction_equivalence : 596 - 130 - 270 = 596 - (130 + 270) := by
  sorry

end NUMINAMATH_CALUDE_subtraction_equivalence_l1744_174415


namespace NUMINAMATH_CALUDE_integer_triple_divisibility_l1744_174453

theorem integer_triple_divisibility (a b c : ℕ+) : 
  (∃ k₁ k₂ k₃ : ℕ+, (a + 1 : ℕ) = k₁ * b ∧ (b + 1 : ℕ) = k₂ * c ∧ (c + 1 : ℕ) = k₃ * a) →
  ((a, b, c) = (1, 1, 1) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (1, 1, 2) ∨ (a, b, c) = (2, 1, 1)) :=
by sorry


end NUMINAMATH_CALUDE_integer_triple_divisibility_l1744_174453


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l1744_174455

theorem sin_2alpha_value (a α : ℝ) 
  (h : Real.sin (a + π/4) = Real.sqrt 2 * (Real.sin α + 2 * Real.cos α)) : 
  Real.sin (2 * α) = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l1744_174455


namespace NUMINAMATH_CALUDE_qr_equals_b_l1744_174489

def curve (c : ℝ) (x y : ℝ) : Prop := y / c = Real.cosh (x / c)

theorem qr_equals_b (a b c : ℝ) (h1 : curve c a b) (h2 : curve c 0 c) :
  let normal_slope := -1 / (Real.sinh (a / c) / c)
  let r_x := c * Real.sinh (a / c) / 2
  Real.sqrt ((r_x - 0)^2 + (0 - c)^2) = b := by sorry

end NUMINAMATH_CALUDE_qr_equals_b_l1744_174489


namespace NUMINAMATH_CALUDE_find_a_and_b_l1744_174480

-- Define the curve equation
def curve (x y a b : ℝ) : Prop := (x - a)^2 + (y - b)^2 = 36

-- Define the theorem
theorem find_a_and_b :
  ∀ a b : ℝ, curve 0 (-12) a b → curve 0 0 a b → a = 0 ∧ b = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_find_a_and_b_l1744_174480


namespace NUMINAMATH_CALUDE_solve_quadratic_sets_l1744_174414

-- Define the sets A and B
def A (p : ℝ) : Set ℝ := {x | x^2 + p*x - 8 = 0}
def B (q r : ℝ) : Set ℝ := {x | x^2 - q*x + r = 0}

-- State the theorem
theorem solve_quadratic_sets :
  ∃ (p q r : ℝ),
    A p ≠ B q r ∧
    A p ∪ B q r = {-2, 4} ∧
    A p ∩ B q r = {-2} ∧
    p = -2 ∧ q = -4 ∧ r = 4 :=
by sorry

end NUMINAMATH_CALUDE_solve_quadratic_sets_l1744_174414


namespace NUMINAMATH_CALUDE_chastity_final_money_is_16_49_l1744_174499

/-- Calculates the final amount of money Chastity has after buying candies and giving some to a friend --/
def chastity_final_money (
  lollipop_price : ℚ)
  (gummies_price : ℚ)
  (chips_price : ℚ)
  (chocolate_price : ℚ)
  (discount_rate : ℚ)
  (tax_rate : ℚ)
  (initial_money : ℚ) : ℚ :=
  let total_cost := 4 * lollipop_price + gummies_price + 3 * chips_price + chocolate_price
  let discounted_cost := total_cost * (1 - discount_rate)
  let taxed_cost := discounted_cost * (1 + tax_rate)
  let money_after_purchase := initial_money - taxed_cost
  let friend_payback := 2 * lollipop_price + chips_price
  money_after_purchase + friend_payback

/-- Theorem stating that Chastity's final amount of money is $16.49 --/
theorem chastity_final_money_is_16_49 :
  chastity_final_money 1.5 2 1.25 1.75 0.1 0.05 25 = 16.49 := by
  sorry

end NUMINAMATH_CALUDE_chastity_final_money_is_16_49_l1744_174499


namespace NUMINAMATH_CALUDE_bridge_weight_is_88_ounces_l1744_174450

/-- The weight of a toy bridge given the number of full soda cans, 
    the weight of soda in each can, the weight of an empty can, 
    and the number of additional empty cans. -/
def bridge_weight (full_cans : ℕ) (soda_weight : ℕ) (empty_can_weight : ℕ) (additional_empty_cans : ℕ) : ℕ :=
  (full_cans * (soda_weight + empty_can_weight)) + (additional_empty_cans * empty_can_weight)

/-- Theorem stating that the bridge must hold 88 ounces given the specified conditions. -/
theorem bridge_weight_is_88_ounces : 
  bridge_weight 6 12 2 2 = 88 := by
  sorry

end NUMINAMATH_CALUDE_bridge_weight_is_88_ounces_l1744_174450


namespace NUMINAMATH_CALUDE_soccer_team_wins_l1744_174411

/-- Given a soccer team that played 140 games and won 50 percent of them,
    prove that the number of games won is 70. -/
theorem soccer_team_wins (total_games : ℕ) (win_percentage : ℚ) (games_won : ℕ) :
  total_games = 140 →
  win_percentage = 1/2 →
  games_won = (total_games : ℚ) * win_percentage →
  games_won = 70 := by
  sorry

end NUMINAMATH_CALUDE_soccer_team_wins_l1744_174411


namespace NUMINAMATH_CALUDE_cube_root_simplification_l1744_174431

theorem cube_root_simplification :
  (80^3 + 100^3 + 120^3 : ℝ)^(1/3) = 20 * 405^(1/3) :=
by sorry

end NUMINAMATH_CALUDE_cube_root_simplification_l1744_174431


namespace NUMINAMATH_CALUDE_chess_tournament_proof_l1744_174440

theorem chess_tournament_proof (i g n : ℕ) (I G : ℚ) :
  g = 10 * i →
  n = i + g →
  G = (9/2) * I →
  (n * (n - 1)) / 2 = I + G →
  i = 1 ∧ g = 10 ∧ (n * (n - 1)) / 2 = 55 :=
by sorry

end NUMINAMATH_CALUDE_chess_tournament_proof_l1744_174440


namespace NUMINAMATH_CALUDE_b_completion_time_l1744_174459

/-- The number of days A needs to complete the work alone -/
def a_days : ℝ := 12

/-- The number of days A works before B joins -/
def a_solo_days : ℝ := 2

/-- The total number of days A and B work together to complete the job -/
def total_days : ℝ := 8

/-- The number of days B needs to complete the work alone -/
def b_days : ℝ := 18

/-- The theorem stating that given the conditions, B can complete the work alone in 18 days -/
theorem b_completion_time :
  (a_days = 12) →
  (a_solo_days = 2) →
  (total_days = 8) →
  (b_days = 18) →
  (1 / a_days * a_solo_days + (total_days - a_solo_days) * (1 / a_days + 1 / b_days) = 1) :=
by sorry

end NUMINAMATH_CALUDE_b_completion_time_l1744_174459


namespace NUMINAMATH_CALUDE_cookie_bags_l1744_174462

theorem cookie_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) (h1 : total_cookies = 33) (h2 : cookies_per_bag = 11) :
  total_cookies / cookies_per_bag = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_l1744_174462


namespace NUMINAMATH_CALUDE_miss_evans_class_contribution_l1744_174416

/-- Calculates the original contribution amount for a class given the number of students,
    individual contribution after using class funds, and the amount of class funds used. -/
def originalContribution (numStudents : ℕ) (individualContribution : ℕ) (classFunds : ℕ) : ℕ :=
  numStudents * individualContribution + classFunds

/-- Proves that for Miss Evans' class, the original contribution amount was $90. -/
theorem miss_evans_class_contribution :
  originalContribution 19 4 14 = 90 := by
  sorry

end NUMINAMATH_CALUDE_miss_evans_class_contribution_l1744_174416


namespace NUMINAMATH_CALUDE_dress_cost_calculation_l1744_174490

def dresses : ℕ := 5
def pants : ℕ := 3
def jackets : ℕ := 4
def pants_cost : ℕ := 12
def jackets_cost : ℕ := 30
def transportation_cost : ℕ := 5
def initial_money : ℕ := 400
def remaining_money : ℕ := 139

theorem dress_cost_calculation (dress_cost : ℕ) : 
  dress_cost * dresses + pants * pants_cost + jackets * jackets_cost + transportation_cost = initial_money - remaining_money → 
  dress_cost = 20 := by
sorry

end NUMINAMATH_CALUDE_dress_cost_calculation_l1744_174490


namespace NUMINAMATH_CALUDE_parabola_focus_line_intersection_l1744_174419

/-- Parabola struct representing x^2 = 2py -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (par : Parabola) where
  x : ℝ
  y : ℝ
  h : x^2 = 2 * par.p * y

/-- Line passing through the focus of a parabola with slope √3 -/
structure FocusLine (par : Parabola) where
  slope : ℝ
  hslope : slope = Real.sqrt 3
  pass_focus : ℝ → ℝ
  hpass : pass_focus 0 = par.p / 2

theorem parabola_focus_line_intersection (par : Parabola) 
  (l : FocusLine par) (M N : ParabolaPoint par) 
  (hM : M.x > 0) (hMN : M.x ≠ N.x) : 
  (par.p = 2 → M.x * N.x = -4) ∧ 
  (M.y * N.y = 1 → par.p = 2) ∧ 
  (par.p = 2 → Real.sqrt ((M.x - 0)^2 + (M.y - par.p/2)^2) = 8 + 4 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_parabola_focus_line_intersection_l1744_174419


namespace NUMINAMATH_CALUDE_division_result_l1744_174474

theorem division_result : (45 : ℝ) / 0.05 = 900 := by sorry

end NUMINAMATH_CALUDE_division_result_l1744_174474


namespace NUMINAMATH_CALUDE_valid_arrangements_l1744_174424

/- Define the number of students and schools -/
def total_students : ℕ := 4
def num_schools : ℕ := 2
def students_per_school : ℕ := 2

/- Define a function to calculate the number of arrangements -/
def num_arrangements (n : ℕ) (k : ℕ) (m : ℕ) : ℕ :=
  if n = total_students ∧ k = num_schools ∧ m = students_per_school ∧ n = k * m then
    2 * (Nat.factorial m) * (Nat.factorial m)
  else
    0

/- Theorem statement -/
theorem valid_arrangements :
  num_arrangements total_students num_schools students_per_school = 8 :=
by sorry

end NUMINAMATH_CALUDE_valid_arrangements_l1744_174424


namespace NUMINAMATH_CALUDE_original_number_l1744_174473

theorem original_number (x : ℝ) : x * 1.1 = 660 ↔ x = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l1744_174473


namespace NUMINAMATH_CALUDE_green_marble_probability_l1744_174464

/-- The probability of drawing a green marble from a box with 100 marbles -/
theorem green_marble_probability :
  ∀ (p_white p_red_or_blue p_green : ℝ),
  p_white = 1/4 →
  p_red_or_blue = 0.55 →
  p_white + p_red_or_blue + p_green = 1 →
  p_green = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_green_marble_probability_l1744_174464


namespace NUMINAMATH_CALUDE_trivia_team_points_per_member_l1744_174406

theorem trivia_team_points_per_member 
  (total_members : ℝ) 
  (absent_members : ℝ) 
  (total_points : ℝ) 
  (h1 : total_members = 5.0) 
  (h2 : absent_members = 2.0) 
  (h3 : total_points = 6.0) : 
  total_points / (total_members - absent_members) = 2.0 := by
sorry

end NUMINAMATH_CALUDE_trivia_team_points_per_member_l1744_174406


namespace NUMINAMATH_CALUDE_james_money_calculation_l1744_174402

theorem james_money_calculation (num_bills : ℕ) (bill_value : ℕ) (existing_amount : ℕ) : 
  num_bills = 3 → bill_value = 20 → existing_amount = 75 → 
  num_bills * bill_value + existing_amount = 135 := by
  sorry

end NUMINAMATH_CALUDE_james_money_calculation_l1744_174402


namespace NUMINAMATH_CALUDE_top_is_nine_l1744_174479

/-- Represents a valid labeling of the figure -/
structure Labeling where
  labels : Fin 9 → Fin 9
  bijective : Function.Bijective labels
  equal_sums : ∃ (s : ℕ), 
    (labels 0 + labels 1 + labels 3 + labels 4 = s) ∧
    (labels 1 + labels 2 + labels 4 + labels 5 = s) ∧
    (labels 0 + labels 3 + labels 6 = s) ∧
    (labels 1 + labels 4 + labels 7 = s) ∧
    (labels 2 + labels 5 + labels 8 = s) ∧
    (labels 3 + labels 4 + labels 5 = s)

/-- The theorem stating that the top number is always 9 in a valid labeling -/
theorem top_is_nine (l : Labeling) : l.labels 0 = 9 := by
  sorry

end NUMINAMATH_CALUDE_top_is_nine_l1744_174479


namespace NUMINAMATH_CALUDE_milk_replacement_percentage_l1744_174456

theorem milk_replacement_percentage (x : ℝ) : 
  (((100 - x) / 100) * ((100 - x) / 100) * ((100 - x) / 100)) * 100 = 51.20000000000001 → 
  x = 20 := by
sorry

end NUMINAMATH_CALUDE_milk_replacement_percentage_l1744_174456


namespace NUMINAMATH_CALUDE_line_circle_intersection_distance_l1744_174476

/-- The line y = kx + 1 intersects the circle (x - 2)² + y² = 9 at two points with distance 4 apart -/
theorem line_circle_intersection_distance (k : ℝ) : ∃ (A B : ℝ × ℝ),
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  (y₁ = k * x₁ + 1) ∧
  (y₂ = k * x₂ + 1) ∧
  ((x₁ - 2)^2 + y₁^2 = 9) ∧
  ((x₂ - 2)^2 + y₂^2 = 9) ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 = 16) := by
  sorry

#check line_circle_intersection_distance

end NUMINAMATH_CALUDE_line_circle_intersection_distance_l1744_174476


namespace NUMINAMATH_CALUDE_smallest_terminating_decimal_l1744_174471

/-- A positive integer n such that n/(n+51) is a terminating decimal -/
def is_terminating_decimal (n : ℕ+) : Prop :=
  ∃ (a b : ℕ), n.val / (n.val + 51) = (a : ℚ) / (10^b : ℚ)

/-- 74 is the smallest positive integer n such that n/(n+51) is a terminating decimal -/
theorem smallest_terminating_decimal :
  (∀ m : ℕ+, m.val < 74 → ¬is_terminating_decimal m) ∧ is_terminating_decimal 74 :=
sorry

end NUMINAMATH_CALUDE_smallest_terminating_decimal_l1744_174471


namespace NUMINAMATH_CALUDE_flour_for_one_loaf_l1744_174472

/-- Given that 5 cups of flour are needed for 2 loaves of bread,
    prove that 2.5 cups of flour are needed for 1 loaf of bread. -/
theorem flour_for_one_loaf (total_flour : ℝ) (total_loaves : ℝ) 
  (h1 : total_flour = 5)
  (h2 : total_loaves = 2) :
  total_flour / total_loaves = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_flour_for_one_loaf_l1744_174472


namespace NUMINAMATH_CALUDE_star_two_three_l1744_174442

-- Define the star operation
def star (a b : ℝ) : ℝ := a * b^3 - b + 2

-- Theorem statement
theorem star_two_three : star 2 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l1744_174442


namespace NUMINAMATH_CALUDE_min_buses_for_given_route_l1744_174486

/-- Represents the bus route configuration -/
structure BusRoute where
  one_way_time : ℕ
  stop_time : ℕ
  departure_interval : ℕ

/-- Calculates the minimum number of buses required for a given bus route -/
def min_buses_required (route : BusRoute) : ℕ :=
  let round_trip_time := 2 * (route.one_way_time + route.stop_time)
  (round_trip_time / route.departure_interval)

/-- Theorem stating that the minimum number of buses required for the given conditions is 20 -/
theorem min_buses_for_given_route :
  let route := BusRoute.mk 50 10 6
  min_buses_required route = 20 := by
  sorry

#eval min_buses_required (BusRoute.mk 50 10 6)

end NUMINAMATH_CALUDE_min_buses_for_given_route_l1744_174486


namespace NUMINAMATH_CALUDE_profit_share_ratio_l1744_174412

theorem profit_share_ratio (total_profit : ℚ) (difference : ℚ) 
  (h1 : total_profit = 800)
  (h2 : difference = 160) :
  ∃ (x y : ℚ), x + y = total_profit ∧ 
                |x - y| = difference ∧ 
                y / total_profit = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_profit_share_ratio_l1744_174412


namespace NUMINAMATH_CALUDE_alternating_integers_l1744_174448

theorem alternating_integers (n : ℕ) (a : ℕ → ℤ) : 
  (∀ i : ℕ, i < n → (a i + a ((i + 1) % n)) % 2 = 0) → 
  (∀ i j : ℕ, i < n → j < n → a i = a j) ∨ 
  (∃ x y : ℤ, ∀ i : ℕ, i < n → a i = if i % 2 = 0 then x else y) := by
sorry

end NUMINAMATH_CALUDE_alternating_integers_l1744_174448


namespace NUMINAMATH_CALUDE_function_composition_inverse_l1744_174435

-- Define the functions
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b
def g (x : ℝ) : ℝ := -5 * x + 3
def h (a b : ℝ) (x : ℝ) : ℝ := f a b (g x)

-- State the theorem
theorem function_composition_inverse (a b : ℝ) :
  (∀ x, h a b x = x - 9) → (a - b = 41 / 5) := by
  sorry

end NUMINAMATH_CALUDE_function_composition_inverse_l1744_174435


namespace NUMINAMATH_CALUDE_volleyball_league_female_fraction_l1744_174401

theorem volleyball_league_female_fraction 
  (last_year_male : ℕ)
  (total_increase : ℝ)
  (male_increase : ℝ)
  (female_increase : ℝ)
  (h1 : last_year_male = 30)
  (h2 : total_increase = 0.15)
  (h3 : male_increase = 0.10)
  (h4 : female_increase = 0.25) :
  let this_year_male : ℝ := last_year_male * (1 + male_increase)
  let last_year_female : ℝ := last_year_male * (1 + total_increase) / (2 + male_increase + female_increase) - last_year_male
  let this_year_female : ℝ := last_year_female * (1 + female_increase)
  let total_this_year : ℝ := this_year_male + this_year_female
  (this_year_female / total_this_year) = 25 / 47 := by
sorry

end NUMINAMATH_CALUDE_volleyball_league_female_fraction_l1744_174401


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1744_174443

theorem quadratic_inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ x^2 - 3*x - 2 - a > 0) → a < 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_range_l1744_174443


namespace NUMINAMATH_CALUDE_A_star_B_equals_zero_three_l1744_174461

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {1, 2, 3}

def star (X Y : Set ℕ) : Set ℕ :=
  {x | x ∈ X ∨ x ∈ Y ∧ x ∉ X ∩ Y}

theorem A_star_B_equals_zero_three : star A B = {0, 3} := by
  sorry

end NUMINAMATH_CALUDE_A_star_B_equals_zero_three_l1744_174461


namespace NUMINAMATH_CALUDE_factorial_square_root_equality_l1744_174468

def factorial (n : ℕ) : ℕ := 
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem factorial_square_root_equality : (((factorial 5) * (factorial 4) : ℝ).sqrt) ^ 2 = 2880 := by
  sorry

end NUMINAMATH_CALUDE_factorial_square_root_equality_l1744_174468


namespace NUMINAMATH_CALUDE_inverse_matrix_problem_l1744_174492

theorem inverse_matrix_problem (A : Matrix (Fin 2) (Fin 2) ℝ) :
  A⁻¹ = !![1, 0; 0, 2] → A = !![1, 0; 0, (1/2)] := by
  sorry

end NUMINAMATH_CALUDE_inverse_matrix_problem_l1744_174492


namespace NUMINAMATH_CALUDE_current_speed_l1744_174428

/-- The speed of the current given a woman's swimming times -/
theorem current_speed (v c : ℝ) 
  (h1 : v + c = 64 / 8)  -- Downstream speed
  (h2 : v - c = 24 / 8)  -- Upstream speed
  : c = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_current_speed_l1744_174428


namespace NUMINAMATH_CALUDE_circle_pencil_theorem_l1744_174452

/-- Definition of a circle in 2D plane -/
structure Circle where
  a : ℝ
  b : ℝ
  R : ℝ

/-- Left-hand side of circle equation -/
def K (C : Circle) (x y : ℝ) : ℝ :=
  (x - C.a)^2 + (y - C.b)^2 - C.R^2

/-- Type of circle pencil -/
inductive PencilType
  | Elliptic
  | Parabolic
  | Hyperbolic

/-- Theorem about circle pencils -/
theorem circle_pencil_theorem (C₁ C₂ : Circle) :
  ∃ (radical_axis : Set (ℝ × ℝ)) (pencil_type : PencilType),
    (∀ (t : ℝ), ∃ (C : Circle), ∀ (x y : ℝ),
      K C₁ x y + t * K C₂ x y = 0 ↔ K C x y = 0) ∧
    (∀ (C : Circle), (∀ (x y : ℝ), K C x y = 0 → (x, y) ∈ radical_axis) →
      ∃ (t : ℝ), ∀ (x y : ℝ), K C₁ x y + t * K C₂ x y = 0 ↔ K C x y = 0) ∧
    (pencil_type = PencilType.Elliptic →
      ∃ (p₁ p₂ : ℝ × ℝ), p₁ ≠ p₂ ∧ K C₁ p₁.1 p₁.2 = 0 ∧ K C₂ p₁.1 p₁.2 = 0 ∧
                          K C₁ p₂.1 p₂.2 = 0 ∧ K C₂ p₂.1 p₂.2 = 0) ∧
    (pencil_type = PencilType.Parabolic →
      ∃ (p : ℝ × ℝ), K C₁ p.1 p.2 = 0 ∧ K C₂ p.1 p.2 = 0 ∧
        ∀ (ε : ℝ), ε > 0 → ∃ (q : ℝ × ℝ), q ≠ p ∧ 
          abs (K C₁ q.1 q.2) < ε ∧ abs (K C₂ q.1 q.2) < ε) ∧
    (pencil_type = PencilType.Hyperbolic →
      ∀ (x y : ℝ), K C₁ x y = 0 → K C₂ x y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_pencil_theorem_l1744_174452


namespace NUMINAMATH_CALUDE_riku_sticker_count_l1744_174432

/-- The number of stickers Kristoff has -/
def kristoff_stickers : ℕ := 85

/-- The ratio of Riku's stickers to Kristoff's stickers -/
def riku_to_kristoff_ratio : ℕ := 25

/-- The number of stickers Riku has -/
def riku_stickers : ℕ := kristoff_stickers * riku_to_kristoff_ratio

theorem riku_sticker_count : riku_stickers = 2125 := by
  sorry

end NUMINAMATH_CALUDE_riku_sticker_count_l1744_174432


namespace NUMINAMATH_CALUDE_unique_n_for_prime_power_difference_l1744_174457

def is_power_of_three (x : ℕ) : Prop :=
  ∃ a : ℕ, x = 3^a ∧ a > 0

theorem unique_n_for_prime_power_difference :
  ∃! n : ℕ, n > 0 ∧ 
    (∃ p : ℕ, Nat.Prime p ∧ is_power_of_three (p^n - (p-1)^n)) :=
by
  sorry

end NUMINAMATH_CALUDE_unique_n_for_prime_power_difference_l1744_174457


namespace NUMINAMATH_CALUDE_min_time_to_target_l1744_174420

/-- Represents the number of steps to the right per minute -/
def right_steps : ℕ := 47

/-- Represents the number of steps to the left per minute -/
def left_steps : ℕ := 37

/-- Represents the target position (one step to the right) -/
def target : ℤ := 1

/-- Theorem stating the minimum time to reach the target position -/
theorem min_time_to_target :
  ∃ (x y : ℕ), 
    right_steps * x - left_steps * y = target ∧
    (∀ (a b : ℕ), right_steps * a - left_steps * b = target → x + y ≤ a + b) ∧
    x + y = 59 := by
  sorry

end NUMINAMATH_CALUDE_min_time_to_target_l1744_174420


namespace NUMINAMATH_CALUDE_melt_to_spend_ratio_is_80_l1744_174400

/-- The ratio of the value of melted quarters to spent quarters -/
def meltToSpendRatio : ℚ :=
  let quarterWeight : ℚ := 1 / 5
  let meltedValuePerOunce : ℚ := 100
  let spendingValuePerQuarter : ℚ := 1 / 4
  let quartersPerOunce : ℚ := 1 / quarterWeight
  let meltedValuePerQuarter : ℚ := meltedValuePerOunce / quartersPerOunce
  meltedValuePerQuarter / spendingValuePerQuarter

/-- The ratio of the value of melted quarters to spent quarters is 80 -/
theorem melt_to_spend_ratio_is_80 : meltToSpendRatio = 80 := by
  sorry

end NUMINAMATH_CALUDE_melt_to_spend_ratio_is_80_l1744_174400


namespace NUMINAMATH_CALUDE_speech_competition_score_l1744_174488

/-- Calculates the weighted average score for a speech competition --/
def weighted_average (content_score delivery_score effectiveness_score : ℚ) : ℚ :=
  (4 * content_score + 4 * delivery_score + 2 * effectiveness_score) / 10

/-- Theorem: The weighted average score for a student with scores 91, 94, and 90 is 92 --/
theorem speech_competition_score : weighted_average 91 94 90 = 92 := by
  sorry

end NUMINAMATH_CALUDE_speech_competition_score_l1744_174488


namespace NUMINAMATH_CALUDE_total_apples_l1744_174484

theorem total_apples (cecile_apples diane_apples : ℕ) : 
  cecile_apples = 15 → 
  diane_apples = cecile_apples + 20 → 
  cecile_apples + diane_apples = 50 := by
sorry

end NUMINAMATH_CALUDE_total_apples_l1744_174484


namespace NUMINAMATH_CALUDE_math_competition_average_score_l1744_174451

theorem math_competition_average_score 
  (total_people : ℕ) 
  (group_average : ℚ) 
  (xiaoming_score : ℚ) 
  (h1 : total_people = 10)
  (h2 : group_average = 84)
  (h3 : xiaoming_score = 93) :
  let remaining_people := total_people - 1
  let total_score := group_average * total_people
  let remaining_score := total_score - xiaoming_score
  remaining_score / remaining_people = 83 := by
sorry

end NUMINAMATH_CALUDE_math_competition_average_score_l1744_174451


namespace NUMINAMATH_CALUDE_principal_amount_proof_l1744_174403

/-- Proves that given the specified conditions, the principal amount is 7200 --/
theorem principal_amount_proof (rate : ℝ) (time : ℝ) (diff : ℝ) (P : ℝ) 
  (h1 : rate = 5 / 100)
  (h2 : time = 2)
  (h3 : diff = 18)
  (h4 : P * (1 + rate)^time - P - (P * rate * time) = diff) :
  P = 7200 := by
  sorry

#check principal_amount_proof

end NUMINAMATH_CALUDE_principal_amount_proof_l1744_174403


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1744_174495

theorem quadratic_two_distinct_roots (a : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + a*x₁ - 1 = 0 ∧ x₂^2 + a*x₂ - 1 = 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l1744_174495


namespace NUMINAMATH_CALUDE_quadrilateral_angle_ratio_concurrency_l1744_174469

-- Define the structure for a point in 2D space
structure Point :=
  (x : ℝ) (y : ℝ)

-- Define the quadrilateral
def Quadrilateral (A B C D : Point) : Prop :=
  -- Add conditions for convexity if needed
  true

-- Define a point inside the quadrilateral
def PointInside (P A B C D : Point) : Prop :=
  -- Add conditions for P being inside ABCD if needed
  true

-- Define the angle ratio condition
def AngleRatioCondition (P A B C D : Point) : Prop :=
  -- Represent the angle ratio condition
  true

-- Define a line
structure Line :=
  (a b c : ℝ)  -- ax + by + c = 0

-- Define the angle bisector
def AngleBisector (P Q R : Point) : Line :=
  sorry

-- Define the perpendicular bisector
def PerpendicularBisector (P Q : Point) : Line :=
  sorry

-- Define concurrency of lines
def Concurrent (l₁ l₂ l₃ : Line) : Prop :=
  -- Add conditions for concurrency
  true

theorem quadrilateral_angle_ratio_concurrency 
  (A B C D P : Point) 
  (h₁ : Quadrilateral A B C D) 
  (h₂ : PointInside P A B C D) 
  (h₃ : AngleRatioCondition P A B C D) :
  Concurrent 
    (AngleBisector A D P) 
    (AngleBisector P C B) 
    (PerpendicularBisector A B) :=
by
  sorry

end NUMINAMATH_CALUDE_quadrilateral_angle_ratio_concurrency_l1744_174469


namespace NUMINAMATH_CALUDE_price_quantity_change_l1744_174483

theorem price_quantity_change (original_price original_quantity : ℝ) :
  let price_increase_factor := 1.20
  let quantity_decrease_factor := 0.70
  let new_cost := original_price * price_increase_factor * original_quantity * quantity_decrease_factor
  let original_cost := original_price * original_quantity
  new_cost / original_cost = 0.84 :=
by sorry

end NUMINAMATH_CALUDE_price_quantity_change_l1744_174483


namespace NUMINAMATH_CALUDE_max_area_difference_rectangles_l1744_174485

/-- Represents a rectangle with integer dimensions -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- The perimeter of a rectangle -/
def perimeter (r : Rectangle) : ℕ := 2 * (r.length + r.width)

/-- The area of a rectangle -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Theorem: The maximum difference between areas of two rectangles with perimeter 144 is 1225 -/
theorem max_area_difference_rectangles :
  ∃ (r1 r2 : Rectangle),
    perimeter r1 = 144 ∧
    perimeter r2 = 144 ∧
    ∀ (r3 r4 : Rectangle),
      perimeter r3 = 144 →
      perimeter r4 = 144 →
      area r1 - area r2 ≥ area r3 - area r4 ∧
      area r1 - area r2 = 1225 :=
sorry

end NUMINAMATH_CALUDE_max_area_difference_rectangles_l1744_174485


namespace NUMINAMATH_CALUDE_heat_required_temperature_dependent_specific_heat_l1744_174427

/-- The amount of heat required to heat a body with temperature-dependent specific heat capacity. -/
theorem heat_required_temperature_dependent_specific_heat
  (m : ℝ) (c₀ : ℝ) (α : ℝ) (t₁ t₂ : ℝ)
  (hm : m = 2)
  (hc₀ : c₀ = 150)
  (hα : α = 0.05)
  (ht₁ : t₁ = 20)
  (ht₂ : t₂ = 100)
  : ∃ Q : ℝ, Q = 96000 ∧ Q = m * (c₀ * (1 + α * t₂) + c₀ * (1 + α * t₁)) / 2 * (t₂ - t₁) :=
by sorry

end NUMINAMATH_CALUDE_heat_required_temperature_dependent_specific_heat_l1744_174427


namespace NUMINAMATH_CALUDE_stream_speed_stream_speed_is_one_l1744_174493

/-- Given a man's swimming speed and the relative time to swim upstream vs downstream, 
    calculate the speed of the stream. -/
theorem stream_speed (mans_speed : ℝ) (upstream_time_ratio : ℝ) : ℝ :=
  let stream_speed := (mans_speed * (upstream_time_ratio - 1)) / (upstream_time_ratio + 1)
  stream_speed

/-- Prove that given the conditions, the stream speed is 1 km/h -/
theorem stream_speed_is_one :
  stream_speed 3 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_stream_speed_stream_speed_is_one_l1744_174493


namespace NUMINAMATH_CALUDE_circle_with_common_chord_as_diameter_l1744_174497

/-- C₁ is the first given circle -/
def C₁ (x y : ℝ) : Prop := x^2 + y^2 + 4*x + y + 1 = 0

/-- C₂ is the second given circle -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 2*y + 1 = 0

/-- C is the circle we need to prove -/
def C (x y : ℝ) : Prop := 5*x^2 + 5*y^2 + 6*x + 12*y + 5 = 0

/-- The common chord of C₁ and C₂ -/
def common_chord (x y : ℝ) : Prop := y = 2*x

theorem circle_with_common_chord_as_diameter :
  ∀ x y : ℝ, C x y ↔ 
    (∃ a b : ℝ, C₁ a b ∧ C₂ a b ∧ common_chord a b ∧
      (x - a)^2 + (y - b)^2 = ((x - a) - (b - y))^2 / 4) :=
sorry

end NUMINAMATH_CALUDE_circle_with_common_chord_as_diameter_l1744_174497


namespace NUMINAMATH_CALUDE_expression_simplification_l1744_174460

theorem expression_simplification (a b : ℚ) (h1 : a = 1) (h2 : b = -2) :
  ((a - 2*b)^2 - (a - 2*b)*(a + 2*b) + 4*b^2) / (-2*b) = 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1744_174460


namespace NUMINAMATH_CALUDE_units_digit_37_pow_37_l1744_174444

theorem units_digit_37_pow_37 : 37^37 ≡ 7 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_37_pow_37_l1744_174444


namespace NUMINAMATH_CALUDE_least_n_for_fraction_inequality_l1744_174423

theorem least_n_for_fraction_inequality : 
  (∃ n : ℕ, n > 0 ∧ (1 : ℚ) / n - (1 : ℚ) / (n + 1) < (1 : ℚ) / 15) ∧ 
  (∀ m : ℕ, m > 0 ∧ m < 4 → (1 : ℚ) / m - (1 : ℚ) / (m + 1) ≥ (1 : ℚ) / 15) ∧
  ((1 : ℚ) / 4 - (1 : ℚ) / 5 < (1 : ℚ) / 15) :=
by sorry

end NUMINAMATH_CALUDE_least_n_for_fraction_inequality_l1744_174423


namespace NUMINAMATH_CALUDE_find_point_c_l1744_174439

/-- Given two points A and B in a 2D plane, and a point C such that 
    vector BC is half of vector BA, find the coordinates of point C. -/
theorem find_point_c (A B : ℝ × ℝ) (C : ℝ × ℝ) : 
  A = (1, 1) → 
  B = (-1, 2) → 
  C - B = (1/2) • (A - B) → 
  C = (0, 3/2) := by
sorry

end NUMINAMATH_CALUDE_find_point_c_l1744_174439


namespace NUMINAMATH_CALUDE_distance_midpoint_endpoint_l1744_174470

theorem distance_midpoint_endpoint (t : ℝ) : 
  let A : ℝ × ℝ := (t - 4, -1)
  let B : ℝ × ℝ := (-2, t + 3)
  let midpoint : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  ((midpoint.1 - A.1)^2 + (midpoint.2 - A.2)^2 = t^2 / 2) →
  t = -5 := by
sorry

end NUMINAMATH_CALUDE_distance_midpoint_endpoint_l1744_174470


namespace NUMINAMATH_CALUDE_quiz_score_of_dropped_student_l1744_174436

theorem quiz_score_of_dropped_student 
  (initial_students : ℕ)
  (initial_average : ℚ)
  (curve_adjustment : ℕ)
  (remaining_students : ℕ)
  (final_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 61.5)
  (h3 : curve_adjustment = 5)
  (h4 : remaining_students = 15)
  (h5 : final_average = 64) :
  ∃ (dropped_score : ℕ), 
    (initial_students : ℚ) * initial_average - dropped_score + 
    (remaining_students : ℚ) * curve_adjustment = 
    (remaining_students : ℚ) * final_average ∧ 
    dropped_score = 99 := by
  sorry

end NUMINAMATH_CALUDE_quiz_score_of_dropped_student_l1744_174436


namespace NUMINAMATH_CALUDE_prob_monochromatic_triangle_l1744_174477

/-- A regular hexagon with colored edges -/
structure ColoredHexagon where
  /-- The probability of an edge being colored red -/
  p : ℝ
  /-- Assumption that p is between 0 and 1 -/
  h_p : 0 ≤ p ∧ p ≤ 1

/-- The number of edges (sides and diagonals) in a regular hexagon -/
def num_edges : ℕ := 15

/-- The number of triangles in a regular hexagon -/
def num_triangles : ℕ := 20

/-- The probability of a specific triangle not being monochromatic -/
def prob_not_monochromatic (h : ColoredHexagon) : ℝ :=
  3 * h.p^2 * (1 - h.p) + 3 * (1 - h.p)^2 * h.p

/-- The main theorem: probability of at least one monochromatic triangle -/
theorem prob_monochromatic_triangle (h : ColoredHexagon) :
  h.p = 1/2 → 1 - (prob_not_monochromatic h)^num_triangles = 1 - (3/4)^20 := by
  sorry

end NUMINAMATH_CALUDE_prob_monochromatic_triangle_l1744_174477


namespace NUMINAMATH_CALUDE_square_root_five_expansion_square_root_three_expansion_simplify_nested_square_root_l1744_174409

-- Part 1
theorem square_root_five_expansion (a b m n : ℤ) :
  a + b * Real.sqrt 5 = (m + n * Real.sqrt 5)^2 →
  a = m^2 + 5 * n^2 ∧ b = 2 * m * n :=
sorry

-- Part 2
theorem square_root_three_expansion :
  ∃ (x m n : ℕ+), x + 4 * Real.sqrt 3 = (m + n * Real.sqrt 3)^2 ∧
  ((m = 1 ∧ n = 2 ∧ x = 13) ∨ (m = 2 ∧ n = 1 ∧ x = 7)) :=
sorry

-- Part 3
theorem simplify_nested_square_root :
  Real.sqrt (5 + 2 * Real.sqrt 6) = Real.sqrt 2 + Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_square_root_five_expansion_square_root_three_expansion_simplify_nested_square_root_l1744_174409


namespace NUMINAMATH_CALUDE_missing_number_is_seven_l1744_174434

def known_numbers : List ℕ := [3, 11, 7, 9, 15, 13, 8, 19, 17, 21, 14]

theorem missing_number_is_seven (x : ℕ) :
  (known_numbers.sum + x) / 12 = 12 →
  x = 7 := by sorry

end NUMINAMATH_CALUDE_missing_number_is_seven_l1744_174434


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l1744_174482

theorem imaginary_part_of_z (z : ℂ) (h : (z - 2*Complex.I)*Complex.I = 1 + Complex.I) : 
  Complex.im z = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l1744_174482


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l1744_174430

/-- Given a quadratic equation ax^2 + bx + c = 0 with no real roots,
    if there exist two possible misinterpretations of the equation
    such that one yields roots 2 and 4, and the other yields roots -1 and 4,
    then (2b + 3c) / a = 12 -/
theorem quadratic_equation_roots (a b c : ℝ) : 
  (∀ x : ℝ, a * x^2 + b * x + c ≠ 0) →  -- No real roots
  (∃ a' : ℝ, a' * 4^2 + b * 4 + c = 0 ∧ a' * 2^2 + b * 2 + c = 0) →  -- Misinterpretation 1
  (∃ b' : ℝ, a * 4^2 + b' * 4 + c = 0 ∧ a * (-1)^2 + b' * (-1) + c = 0) →  -- Misinterpretation 2
  (2 * b + 3 * c) / a = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l1744_174430


namespace NUMINAMATH_CALUDE_steve_growth_l1744_174449

/-- Converts feet and inches to total inches -/
def feet_inches_to_inches (feet : ℕ) (inches : ℕ) : ℕ :=
  feet * 12 + inches

theorem steve_growth :
  let original_height := feet_inches_to_inches 5 6
  let new_height := 72
  new_height - original_height = 6 := by
  sorry

end NUMINAMATH_CALUDE_steve_growth_l1744_174449


namespace NUMINAMATH_CALUDE_bigger_part_is_38_l1744_174421

theorem bigger_part_is_38 (x y : ℕ) (h1 : x + y = 56) (h2 : 10 * x + 22 * y = 780) :
  max x y = 38 := by
sorry

end NUMINAMATH_CALUDE_bigger_part_is_38_l1744_174421


namespace NUMINAMATH_CALUDE_unique_solution_inequality_holds_max_value_l1744_174458

-- Define the functions f, g, and h
def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|
def h (a x : ℝ) : ℝ := |f x| + g a x

-- Theorem for part (1)
theorem unique_solution (a : ℝ) :
  (∃! x, |f x| = g a x) ↔ a < 0 :=
sorry

-- Theorem for part (2)
theorem inequality_holds (a : ℝ) :
  (∀ x, f x ≥ g a x) ↔ a ≤ -2 :=
sorry

-- Theorem for part (3)
theorem max_value (a : ℝ) :
  (∀ x ∈ Set.Icc (-2) 2, h a x ≤ 
    if a ≥ 0 then 3*a + 3
    else if a ≥ -3 then a + 3
    else 0) ∧
  (∃ x ∈ Set.Icc (-2) 2, h a x = 
    if a ≥ 0 then 3*a + 3
    else if a ≥ -3 then a + 3
    else 0) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_inequality_holds_max_value_l1744_174458


namespace NUMINAMATH_CALUDE_factorization_theorem_l1744_174433

theorem factorization_theorem (a b c : ℝ) :
  ((a^4 - b^4)^3 + (b^4 - c^4)^3 + (c^4 - a^4)^3) / ((a - b)^3 + (b - c)^3 + (c - a)^3) = (a + b) * (b + c) * (c + a) :=
by sorry

end NUMINAMATH_CALUDE_factorization_theorem_l1744_174433


namespace NUMINAMATH_CALUDE_subset_properties_l1744_174491

-- Define set A
def A : Set ℝ := {x | 0 < x ∧ x < 2}

-- Define the property that B is a subset of A
def is_subset_of_A (B : Set ℝ) : Prop := B ⊆ A

-- Theorem statement
theorem subset_properties (B : Set ℝ) (h : A ∩ B = B) :
  is_subset_of_A ∅ ∧
  is_subset_of_A {1} ∧
  is_subset_of_A A ∧
  ¬is_subset_of_A {x : ℝ | 0 ≤ x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_CALUDE_subset_properties_l1744_174491


namespace NUMINAMATH_CALUDE_rem_negative_five_ninths_seven_thirds_l1744_174410

def rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋

theorem rem_negative_five_ninths_seven_thirds :
  rem (-5/9 : ℚ) (7/3 : ℚ) = 16/9 := by
  sorry

end NUMINAMATH_CALUDE_rem_negative_five_ninths_seven_thirds_l1744_174410


namespace NUMINAMATH_CALUDE_twitch_income_per_subscriber_l1744_174425

/-- Calculates the income per subscriber for a Twitch streamer --/
theorem twitch_income_per_subscriber
  (initial_subscribers : ℕ)
  (gifted_subscribers : ℕ)
  (total_monthly_income : ℕ)
  (h1 : initial_subscribers = 150)
  (h2 : gifted_subscribers = 50)
  (h3 : total_monthly_income = 1800) :
  total_monthly_income / (initial_subscribers + gifted_subscribers) = 9 := by
sorry

end NUMINAMATH_CALUDE_twitch_income_per_subscriber_l1744_174425


namespace NUMINAMATH_CALUDE_bottles_not_in_crate_l1744_174478

/-- Given the number of bottles per crate, total bottles, and number of crates,
    calculate the number of bottles that will not be placed in a crate. -/
theorem bottles_not_in_crate
  (bottles_per_crate : ℕ)
  (total_bottles : ℕ)
  (num_crates : ℕ)
  (h1 : bottles_per_crate = 12)
  (h2 : total_bottles = 130)
  (h3 : num_crates = 10) :
  total_bottles - (bottles_per_crate * num_crates) = 10 :=
by sorry

end NUMINAMATH_CALUDE_bottles_not_in_crate_l1744_174478


namespace NUMINAMATH_CALUDE_clothing_sale_price_l1744_174407

theorem clothing_sale_price (a : ℝ) : 
  (∃ x y : ℝ, 
    x * 1.25 = a ∧ 
    y * 0.75 = a ∧ 
    x + y - 2*a = -8) → 
  a = 60 := by
sorry

end NUMINAMATH_CALUDE_clothing_sale_price_l1744_174407


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_half_open_interval_l1744_174447

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2*x}

-- Theorem statement
theorem A_intersect_B_eq_half_open_interval : A ∩ B = Set.Ioc 1 3 := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_half_open_interval_l1744_174447


namespace NUMINAMATH_CALUDE_remainder_n_plus_2023_l1744_174426

theorem remainder_n_plus_2023 (n : ℤ) (h : n % 7 = 3) : (n + 2023) % 7 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_n_plus_2023_l1744_174426


namespace NUMINAMATH_CALUDE_distance_las_vegas_to_los_angeles_l1744_174445

/-- Calculates the distance from Las Vegas to Los Angeles given the total drive time,
    average speed, and distance from Salt Lake City to Las Vegas. -/
theorem distance_las_vegas_to_los_angeles
  (total_time : ℝ)
  (average_speed : ℝ)
  (distance_salt_lake_to_vegas : ℝ)
  (h1 : total_time = 11)
  (h2 : average_speed = 63)
  (h3 : distance_salt_lake_to_vegas = 420) :
  total_time * average_speed - distance_salt_lake_to_vegas = 273 :=
by
  sorry

end NUMINAMATH_CALUDE_distance_las_vegas_to_los_angeles_l1744_174445


namespace NUMINAMATH_CALUDE_subset_sum_inequality_l1744_174481

theorem subset_sum_inequality (n m : ℕ) (A : Finset ℕ) (h_m : m > 0) (h_n : n > 0) 
  (h_subset : A ⊆ Finset.range n)
  (h_closure : ∀ (i j : ℕ), i ∈ A → j ∈ A → i + j ≤ n → i + j ∈ A) :
  (A.sum id) / m ≥ (n + 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_subset_sum_inequality_l1744_174481


namespace NUMINAMATH_CALUDE_average_age_decrease_l1744_174408

theorem average_age_decrease (original_strength : ℕ) (original_avg_age : ℝ) 
  (new_students : ℕ) (new_avg_age : ℝ) : 
  original_strength = 17 →
  original_avg_age = 40 →
  new_students = 17 →
  new_avg_age = 32 →
  let new_total_strength := original_strength + new_students
  let new_avg_age := (original_strength * original_avg_age + new_students * new_avg_age) / new_total_strength
  original_avg_age - new_avg_age = 4 := by
sorry

end NUMINAMATH_CALUDE_average_age_decrease_l1744_174408


namespace NUMINAMATH_CALUDE_opposite_numbers_fifth_power_sum_l1744_174475

theorem opposite_numbers_fifth_power_sum (a b : ℝ) : 
  a + b = 0 → a^5 + b^5 = 0 := by sorry

end NUMINAMATH_CALUDE_opposite_numbers_fifth_power_sum_l1744_174475


namespace NUMINAMATH_CALUDE_aunts_gift_amount_l1744_174417

def shirts_cost : ℕ := 5
def shirts_price : ℕ := 5
def pants_price : ℕ := 26
def remaining_money : ℕ := 20

theorem aunts_gift_amount : 
  shirts_cost * shirts_price + pants_price + remaining_money = 71 := by
  sorry

end NUMINAMATH_CALUDE_aunts_gift_amount_l1744_174417


namespace NUMINAMATH_CALUDE_gas_pressure_change_l1744_174437

/-- Given inverse proportionality of pressure and volume at constant temperature,
    prove that the pressure in a 6-liter container is 4 kPa, given initial conditions. -/
theorem gas_pressure_change (p₁ p₂ v₁ v₂ : ℝ) : 
  p₁ > 0 → v₁ > 0 → p₂ > 0 → v₂ > 0 →  -- Ensuring positive values
  (p₁ * v₁ = p₂ * v₂) →  -- Inverse proportionality
  p₁ = 8 → v₁ = 3 → v₂ = 6 →  -- Initial conditions and new volume
  p₂ = 4 := by sorry

end NUMINAMATH_CALUDE_gas_pressure_change_l1744_174437


namespace NUMINAMATH_CALUDE_puzzle_piece_increase_l1744_174454

/-- Represents the number of puzzles John buys -/
def num_puzzles : ℕ := 3

/-- Represents the number of pieces in the first puzzle -/
def first_puzzle_pieces : ℕ := 1000

/-- Represents the total number of pieces in all puzzles -/
def total_pieces : ℕ := 4000

/-- Represents the percentage increase in pieces for the second and third puzzles -/
def percentage_increase : ℚ := 50

theorem puzzle_piece_increase :
  ∃ (second_puzzle_pieces third_puzzle_pieces : ℕ),
    second_puzzle_pieces = third_puzzle_pieces ∧
    second_puzzle_pieces = first_puzzle_pieces + (percentage_increase / 100) * first_puzzle_pieces ∧
    first_puzzle_pieces + second_puzzle_pieces + third_puzzle_pieces = total_pieces :=
by sorry

#check puzzle_piece_increase

end NUMINAMATH_CALUDE_puzzle_piece_increase_l1744_174454


namespace NUMINAMATH_CALUDE_diver_B_depth_l1744_174422

/-- The depth of diver A in meters -/
def depth_A : ℝ := -55

/-- The vertical distance between diver B and diver A in meters -/
def distance_B_above_A : ℝ := 5

/-- The depth of diver B in meters -/
def depth_B : ℝ := depth_A + distance_B_above_A

theorem diver_B_depth : depth_B = -50 := by
  sorry

end NUMINAMATH_CALUDE_diver_B_depth_l1744_174422


namespace NUMINAMATH_CALUDE_intersection_M_N_l1744_174494

/-- Set M is defined as {x | 0 ≤ x ≤ 1} -/
def M : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

/-- Set N is defined as {x | |x| ≥ 1} -/
def N : Set ℝ := {x | abs x ≥ 1}

/-- The intersection of sets M and N is equal to the set containing only 1 -/
theorem intersection_M_N : M ∩ N = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1744_174494


namespace NUMINAMATH_CALUDE_expected_shots_value_l1744_174465

/-- The probability of hitting the target -/
def p : ℝ := 0.8

/-- The maximum number of bullets -/
def max_shots : ℕ := 3

/-- The expected number of shots -/
def expected_shots : ℝ := p + 2 * (1 - p) * p + 3 * (1 - p) * (1 - p)

/-- Theorem stating that the expected number of shots is 1.24 -/
theorem expected_shots_value : expected_shots = 1.24 := by
  sorry

end NUMINAMATH_CALUDE_expected_shots_value_l1744_174465


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l1744_174404

theorem solution_implies_a_value (x y a : ℝ) : 
  x = -2 → y = 1 → 2 * x + a * y = 3 → a = 7 := by sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l1744_174404


namespace NUMINAMATH_CALUDE_total_oranges_picked_l1744_174496

theorem total_oranges_picked (mary_oranges jason_oranges sarah_oranges : ℕ)
  (h1 : mary_oranges = 122)
  (h2 : jason_oranges = 105)
  (h3 : sarah_oranges = 137) :
  mary_oranges + jason_oranges + sarah_oranges = 364 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_picked_l1744_174496


namespace NUMINAMATH_CALUDE_max_ab_value_l1744_174463

theorem max_ab_value (a b : ℝ) : 
  (∃ x y : ℝ, (x - a)^2 + (y - b)^2 = 1 ∧ x + 2*y - 1 = 0) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ - a)^2 + (y₁ - b)^2 = 1 ∧ 
    (x₂ - a)^2 + (y₂ - b)^2 = 1 ∧ 
    x₁ + 2*y₁ - 1 = 0 ∧ 
    x₂ + 2*y₂ - 1 = 0 ∧ 
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = (4/5 * Real.sqrt 5)^2) →
  a * b ≤ 1/2 :=
sorry

end NUMINAMATH_CALUDE_max_ab_value_l1744_174463


namespace NUMINAMATH_CALUDE_binomial_inequality_l1744_174487

theorem binomial_inequality (x : ℝ) (n : ℕ) (h1 : x > -1) (h2 : x ≠ 0) (h3 : n ≥ 2) :
  (1 + x)^n > 1 + n * x := by
  sorry

end NUMINAMATH_CALUDE_binomial_inequality_l1744_174487


namespace NUMINAMATH_CALUDE_plane_centroid_sum_l1744_174498

-- Define the plane and points
def Plane := {plane : ℝ → ℝ → ℝ → Prop | ∃ (a b c : ℝ), ∀ x y z, plane x y z ↔ (x / a + y / b + z / c = 1)}

def distance_from_origin (plane : Plane) : ℝ := sorry

def intersect_x_axis (plane : Plane) : ℝ := sorry
def intersect_y_axis (plane : Plane) : ℝ := sorry
def intersect_z_axis (plane : Plane) : ℝ := sorry

def centroid (a b c : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := sorry

-- Theorem statement
theorem plane_centroid_sum (plane : Plane) :
  let a := (intersect_x_axis plane, 0, 0)
  let b := (0, intersect_y_axis plane, 0)
  let c := (0, 0, intersect_z_axis plane)
  let (p, q, r) := centroid a b c
  distance_from_origin plane = Real.sqrt 2 →
  a ≠ (0, 0, 0) ∧ b ≠ (0, 0, 0) ∧ c ≠ (0, 0, 0) →
  1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 2 := by
  sorry

end NUMINAMATH_CALUDE_plane_centroid_sum_l1744_174498


namespace NUMINAMATH_CALUDE_potato_difference_l1744_174418

/-- The number of potato wedges -/
def x : ℕ := 8 * 13

/-- The number of potatoes used for french fries or potato chips -/
def k : ℕ := (67 - 13) / 2

/-- The number of potato chips -/
def z : ℕ := 20 * k

/-- The difference between the number of potato chips and potato wedges -/
def d : ℤ := z - x

theorem potato_difference : d = 436 := by
  sorry

end NUMINAMATH_CALUDE_potato_difference_l1744_174418


namespace NUMINAMATH_CALUDE_multiply_polynomials_l1744_174405

theorem multiply_polynomials (x y : ℝ) :
  (3 * x^4 - 2 * y^3) * (9 * x^8 + 6 * x^4 * y^3 + 4 * y^6) = 27 * x^12 - 8 * y^9 := by
  sorry

end NUMINAMATH_CALUDE_multiply_polynomials_l1744_174405


namespace NUMINAMATH_CALUDE_bus_trip_time_calculation_l1744_174441

/-- Calculates the new trip time given the original time, original speed, distance increase factor, and new speed -/
def new_trip_time (original_time : ℚ) (original_speed : ℚ) (distance_increase : ℚ) (new_speed : ℚ) : ℚ :=
  (original_time * original_speed * (1 + distance_increase)) / new_speed

/-- Proves that the new trip time is 256/35 hours given the specified conditions -/
theorem bus_trip_time_calculation :
  let original_time : ℚ := 16 / 3  -- 5 1/3 hours
  let original_speed : ℚ := 80
  let distance_increase : ℚ := 1 / 5  -- 20% increase
  let new_speed : ℚ := 70
  new_trip_time original_time original_speed distance_increase new_speed = 256 / 35 := by
  sorry

#eval new_trip_time (16/3) 80 (1/5) 70

end NUMINAMATH_CALUDE_bus_trip_time_calculation_l1744_174441


namespace NUMINAMATH_CALUDE_selection_theorem_l1744_174446

/-- The number of students in the group -/
def total_students : Nat := 6

/-- The number of students to be selected -/
def selected_students : Nat := 4

/-- The number of subjects -/
def subjects : Nat := 4

/-- The number of students who cannot participate in a specific subject -/
def restricted_students : Nat := 2

/-- The number of different selection plans -/
def selection_plans : Nat := 240

theorem selection_theorem :
  (total_students.factorial / (total_students - selected_students).factorial) -
  (restricted_students * ((total_students - 1).factorial / (total_students - selected_students).factorial)) =
  selection_plans := by
  sorry

end NUMINAMATH_CALUDE_selection_theorem_l1744_174446


namespace NUMINAMATH_CALUDE_german_shepherd_vs_golden_retriever_pups_l1744_174413

/-- The number of pups each breed has -/
structure DogBreedPups where
  husky : Nat
  golden_retriever : Nat
  pitbull : Nat
  german_shepherd : Nat

/-- The number of dogs James has for each breed -/
structure DogCounts where
  huskies : Nat
  golden_retrievers : Nat
  pitbulls : Nat
  german_shepherds : Nat

/-- Calculate the difference in total pups between German shepherds and golden retrievers -/
def pup_difference (breed_pups : DogBreedPups) (counts : DogCounts) : Int :=
  (breed_pups.german_shepherd * counts.german_shepherds) - 
  (breed_pups.golden_retriever * counts.golden_retrievers)

theorem german_shepherd_vs_golden_retriever_pups : 
  ∀ (breed_pups : DogBreedPups) (counts : DogCounts),
  counts.huskies = 5 →
  counts.pitbulls = 2 →
  counts.golden_retrievers = 4 →
  counts.german_shepherds = 3 →
  breed_pups.husky = 4 →
  breed_pups.golden_retriever = breed_pups.husky + 2 →
  breed_pups.pitbull = 3 →
  breed_pups.german_shepherd = breed_pups.pitbull + 3 →
  pup_difference breed_pups counts = -6 :=
by
  sorry

end NUMINAMATH_CALUDE_german_shepherd_vs_golden_retriever_pups_l1744_174413
