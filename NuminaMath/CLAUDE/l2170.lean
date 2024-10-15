import Mathlib

namespace NUMINAMATH_CALUDE_no_xy_term_when_k_is_3_l2170_217023

/-- The polynomial that we're analyzing -/
def polynomial (x y k : ℝ) : ℝ := -x^2 - 3*k*x*y - 3*y^2 + 9*x*y - 8

/-- The coefficient of xy in the polynomial -/
def xy_coefficient (k : ℝ) : ℝ := -3*k + 9

theorem no_xy_term_when_k_is_3 :
  ∃ (k : ℝ), xy_coefficient k = 0 ∧ k = 3 :=
sorry

end NUMINAMATH_CALUDE_no_xy_term_when_k_is_3_l2170_217023


namespace NUMINAMATH_CALUDE_intersection_at_one_point_l2170_217029

theorem intersection_at_one_point (b : ℝ) : 
  (∃! x : ℝ, bx^2 + 2*x + 2 = -2*x - 2) ↔ b = 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_at_one_point_l2170_217029


namespace NUMINAMATH_CALUDE_units_digit_problem_l2170_217050

theorem units_digit_problem : (8 * 18 * 1988 - 8^3) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l2170_217050


namespace NUMINAMATH_CALUDE_candy_distribution_l2170_217079

/-- Proves that given the candy distribution conditions, the total number of children is 40 -/
theorem candy_distribution (total_candies : ℕ) (boys girls : ℕ) : 
  total_candies = 90 →
  total_candies / 3 = boys * 3 →
  2 * total_candies / 3 = girls * 2 →
  boys + girls = 40 := by
  sorry

#check candy_distribution

end NUMINAMATH_CALUDE_candy_distribution_l2170_217079


namespace NUMINAMATH_CALUDE_parabola_shift_down_2_l2170_217018

/-- The equation of a parabola after vertical shift -/
def shifted_parabola (a b : ℝ) : ℝ → ℝ := λ x => a * x^2 + b

/-- Theorem: Shifting y = x^2 down by 2 units results in y = x^2 - 2 -/
theorem parabola_shift_down_2 :
  shifted_parabola 1 (-2) = λ x => x^2 - 2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_shift_down_2_l2170_217018


namespace NUMINAMATH_CALUDE_constant_term_expansion_l2170_217017

theorem constant_term_expansion :
  let f := fun (x : ℝ) => (x - 1/x)^6
  ∃ (c : ℝ), c = -20 ∧ 
    ∀ (x : ℝ), x ≠ 0 → (∃ (g : ℝ → ℝ), f x = c + x * g x + (1/x) * g (1/x)) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_expansion_l2170_217017


namespace NUMINAMATH_CALUDE_pink_to_orange_ratio_l2170_217099

theorem pink_to_orange_ratio :
  -- Define the total number of balls
  let total_balls : ℕ := 50
  -- Define the number of red balls
  let red_balls : ℕ := 20
  -- Define the number of blue balls
  let blue_balls : ℕ := 10
  -- Define the number of orange balls
  let orange_balls : ℕ := 5
  -- Define the number of pink balls
  let pink_balls : ℕ := 15
  -- Ensure that the sum of all balls equals the total
  red_balls + blue_balls + orange_balls + pink_balls = total_balls →
  -- Prove that the ratio of pink to orange balls is 3:1
  (pink_balls : ℚ) / (orange_balls : ℚ) = 3 / 1 :=
by
  sorry

end NUMINAMATH_CALUDE_pink_to_orange_ratio_l2170_217099


namespace NUMINAMATH_CALUDE_max_third_side_length_l2170_217091

theorem max_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 10) :
  ∃ (x : ℕ), x ≤ 16 ∧
    (∀ (y : ℕ), (y : ℝ) + a > b ∧ (y : ℝ) + b > a ∧ a + b > (y : ℝ) → y ≤ x) ∧
    ((16 : ℝ) + a > b ∧ (16 : ℝ) + b > a ∧ a + b > 16) :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_length_l2170_217091


namespace NUMINAMATH_CALUDE_diana_video_game_time_l2170_217014

def video_game_time_per_hour_read : ℕ := 30
def raise_percentage : ℚ := 0.2
def chores_for_bonus_time : ℕ := 2
def bonus_time_per_chore_set : ℕ := 10
def max_bonus_time_from_chores : ℕ := 60
def hours_read : ℕ := 8
def chores_completed : ℕ := 10

theorem diana_video_game_time : 
  let base_time := hours_read * video_game_time_per_hour_read
  let raised_time := base_time + (base_time * raise_percentage).floor
  let chore_bonus_time := min (chores_completed / chores_for_bonus_time * bonus_time_per_chore_set) max_bonus_time_from_chores
  raised_time + chore_bonus_time = 338 := by
sorry

end NUMINAMATH_CALUDE_diana_video_game_time_l2170_217014


namespace NUMINAMATH_CALUDE_travelers_speed_l2170_217004

/-- Given two travelers A and B, where B travels 2 km/h faster than A,
    and they meet after 3 hours having traveled a total of 24 km,
    prove that A's speed is 3 km/h. -/
theorem travelers_speed (x : ℝ) : 3*x + 3*(x + 2) = 24 → x = 3 := by
  sorry

end NUMINAMATH_CALUDE_travelers_speed_l2170_217004


namespace NUMINAMATH_CALUDE_cube_root_sum_of_cubes_l2170_217051

theorem cube_root_sum_of_cubes : 
  (20^3 + 70^3 + 110^3 : ℝ)^(1/3) = 120 := by sorry

end NUMINAMATH_CALUDE_cube_root_sum_of_cubes_l2170_217051


namespace NUMINAMATH_CALUDE_player_current_average_l2170_217016

/-- Represents a cricket player's statistics -/
structure PlayerStats where
  matches_played : ℕ
  current_average : ℝ
  desired_increase : ℝ
  next_match_runs : ℕ

/-- Theorem stating the player's current average given the conditions -/
theorem player_current_average (player : PlayerStats)
  (h1 : player.matches_played = 10)
  (h2 : player.desired_increase = 4)
  (h3 : player.next_match_runs = 78) :
  player.current_average = 34 := by
  sorry

#check player_current_average

end NUMINAMATH_CALUDE_player_current_average_l2170_217016


namespace NUMINAMATH_CALUDE_general_form_equation_l2170_217087

theorem general_form_equation (x : ℝ) : 
  (x - 1) * (x - 2) = 4 ↔ x^2 - 3*x - 2 = 0 := by sorry

end NUMINAMATH_CALUDE_general_form_equation_l2170_217087


namespace NUMINAMATH_CALUDE_middle_number_is_five_l2170_217072

/-- Represents a triple of positive integers in increasing order -/
structure IncreasingTriple where
  a : ℕ+
  b : ℕ+
  c : ℕ+
  h1 : a < b
  h2 : b < c

/-- The set of all valid triples according to the problem conditions -/
def ValidTriples : Set IncreasingTriple :=
  { t : IncreasingTriple | t.a + t.b + t.c = 16 }

/-- A triple is ambiguous if there exists another valid triple with the same middle number -/
def IsAmbiguous (t : IncreasingTriple) : Prop :=
  ∃ t' : IncreasingTriple, t' ∈ ValidTriples ∧ t' ≠ t ∧ t'.b = t.b

theorem middle_number_is_five :
  ∀ t ∈ ValidTriples, IsAmbiguous t → t.b = 5 := by sorry

end NUMINAMATH_CALUDE_middle_number_is_five_l2170_217072


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2170_217007

theorem geometric_sequence_sum (a b c q : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a) = (a + b + c) * q ∧
  (c + a - b) = (a + b + c) * q^2 ∧
  (a + b - c) = (a + b + c) * q^3 →
  q^3 + q^2 + q = 1 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2170_217007


namespace NUMINAMATH_CALUDE_max_integer_value_of_fraction_l2170_217096

theorem max_integer_value_of_fraction (x : ℝ) : 
  (4*x^2 + 12*x + 23) / (4*x^2 + 12*x + 9) ≤ 8 ∧ 
  ∃ y : ℝ, (4*y^2 + 12*y + 23) / (4*y^2 + 12*y + 9) > 7 := by
  sorry

end NUMINAMATH_CALUDE_max_integer_value_of_fraction_l2170_217096


namespace NUMINAMATH_CALUDE_complex_modulus_one_l2170_217069

theorem complex_modulus_one (z : ℂ) (h : z * (1 + Complex.I) = 1 - Complex.I) : Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_one_l2170_217069


namespace NUMINAMATH_CALUDE_larry_jogging_days_l2170_217089

theorem larry_jogging_days (daily_jog_time : ℕ) (second_week_days : ℕ) (total_time : ℕ) : 
  daily_jog_time = 30 →
  second_week_days = 5 →
  total_time = 4 * 60 →
  (total_time - second_week_days * daily_jog_time) / daily_jog_time = 3 :=
by sorry

end NUMINAMATH_CALUDE_larry_jogging_days_l2170_217089


namespace NUMINAMATH_CALUDE_trapezoid_area_l2170_217009

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle -/
structure Rectangle where
  p : Point
  q : Point
  r : Point
  s : Point
  area : ℝ

/-- Represents a trapezoid -/
structure Trapezoid where
  t : Point
  u : Point
  v : Point
  s : Point

/-- Given a rectangle PQRS and points T, U, V forming a trapezoid TUVS, 
    prove that the area of TUVS is 10 square units -/
theorem trapezoid_area 
  (pqrs : Rectangle)
  (t : Point)
  (u : Point)
  (v : Point)
  (h1 : pqrs.area = 20)
  (h2 : t.x - pqrs.p.x = 2)
  (h3 : t.y = pqrs.p.y)
  (h4 : u.x - pqrs.q.x = 2)
  (h5 : u.y = pqrs.r.y)
  (h6 : v.x = pqrs.r.x)
  (h7 : v.y - t.y = pqrs.r.y - pqrs.p.y)
  : ∃ (tuvs : Trapezoid), tuvs.t = t ∧ tuvs.u = u ∧ tuvs.v = v ∧ tuvs.s = pqrs.s ∧ 
    (tuvs.v.x - tuvs.t.x + tuvs.s.x - tuvs.u.x) * (tuvs.u.y - tuvs.t.y) / 2 = 10 :=
by sorry

end NUMINAMATH_CALUDE_trapezoid_area_l2170_217009


namespace NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l2170_217055

/-- Given that α = 3, prove that the terminal side of α lies in the second quadrant. -/
theorem terminal_side_in_second_quadrant (α : ℝ) (h : α = 3) :
  (π / 2 : ℝ) < α ∧ α < π :=
sorry

end NUMINAMATH_CALUDE_terminal_side_in_second_quadrant_l2170_217055


namespace NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l2170_217027

theorem factorization_of_difference_of_squares (x : ℝ) :
  x^2 - 9 = (x + 3) * (x - 3) := by sorry

end NUMINAMATH_CALUDE_factorization_of_difference_of_squares_l2170_217027


namespace NUMINAMATH_CALUDE_function_monotonicity_l2170_217073

theorem function_monotonicity (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) :
  (∀ x, (x^2 - 3*x + 2) * (deriv (deriv f) x) ≤ 0) →
  (∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2) :=
by sorry

end NUMINAMATH_CALUDE_function_monotonicity_l2170_217073


namespace NUMINAMATH_CALUDE_expand_and_simplify_l2170_217078

theorem expand_and_simplify (x : ℝ) : -2 * (4 * x^3 - 5 * x^2 + 3 * x - 7) = -8 * x^3 + 10 * x^2 - 6 * x + 14 := by
  sorry

end NUMINAMATH_CALUDE_expand_and_simplify_l2170_217078


namespace NUMINAMATH_CALUDE_only_one_correct_statement_l2170_217036

/-- Represents the confidence level in the study conclusion -/
def confidence_level : ℝ := 0.99

/-- Represents the four statements about smoking and lung cancer -/
inductive Statement
  | all_smokers_have_cancer
  | high_probability_of_cancer
  | some_smokers_have_cancer
  | possibly_no_smokers_have_cancer

/-- Determines if a statement is correct given the confidence level -/
def is_correct (s : Statement) (conf : ℝ) : Prop :=
  match s with
  | Statement.possibly_no_smokers_have_cancer => conf < 1
  | _ => False

/-- The main theorem stating that only one statement is correct -/
theorem only_one_correct_statement : 
  (∃! s : Statement, is_correct s confidence_level) ∧ 
  (is_correct Statement.possibly_no_smokers_have_cancer confidence_level) :=
sorry

end NUMINAMATH_CALUDE_only_one_correct_statement_l2170_217036


namespace NUMINAMATH_CALUDE_inheritance_calculation_inheritance_value_l2170_217056

/-- The inheritance amount in dollars -/
def inheritance : ℝ := 49655

/-- The federal tax rate as a decimal -/
def federal_tax_rate : ℝ := 0.25

/-- The state tax rate as a decimal -/
def state_tax_rate : ℝ := 0.15

/-- The total tax paid in dollars -/
def total_tax_paid : ℝ := 18000

theorem inheritance_calculation :
  federal_tax_rate * inheritance + 
  state_tax_rate * (inheritance - federal_tax_rate * inheritance) = 
  total_tax_paid := by sorry

theorem inheritance_value :
  inheritance = 49655 := by sorry

end NUMINAMATH_CALUDE_inheritance_calculation_inheritance_value_l2170_217056


namespace NUMINAMATH_CALUDE_european_stamps_count_l2170_217064

/-- Represents the number of stamps from Asian countries -/
def asian_stamps : ℕ := sorry

/-- Represents the number of stamps from European countries -/
def european_stamps : ℕ := sorry

/-- The total number of stamps Jesse has -/
def total_stamps : ℕ := 444

/-- European stamps are three times the number of Asian stamps -/
axiom european_triple_asian : european_stamps = 3 * asian_stamps

/-- The sum of Asian and European stamps equals the total stamps -/
axiom sum_equals_total : asian_stamps + european_stamps = total_stamps

/-- Theorem stating that the number of European stamps is 333 -/
theorem european_stamps_count : european_stamps = 333 := by sorry

end NUMINAMATH_CALUDE_european_stamps_count_l2170_217064


namespace NUMINAMATH_CALUDE_least_n_for_jumpy_l2170_217076

/-- A permutation of 2021 elements -/
def Permutation := Fin 2021 → Fin 2021

/-- A function that reorders up to 1232 elements in a permutation -/
def reorder_1232 (p : Permutation) : Permutation :=
  sorry

/-- A function that reorders up to n elements in a permutation -/
def reorder_n (n : ℕ) (p : Permutation) : Permutation :=
  sorry

/-- The identity permutation -/
def id_perm : Permutation :=
  sorry

theorem least_n_for_jumpy :
  ∀ n : ℕ,
    (∀ p : Permutation,
      ∃ q : Permutation,
        reorder_n n (reorder_1232 p) = id_perm) ↔
    n ≥ 1234 :=
  sorry

end NUMINAMATH_CALUDE_least_n_for_jumpy_l2170_217076


namespace NUMINAMATH_CALUDE_wrong_mark_calculation_l2170_217039

theorem wrong_mark_calculation (n : ℕ) (initial_avg correct_avg correct_mark : ℝ) : 
  n = 10 ∧ 
  initial_avg = 100 ∧ 
  correct_avg = 96 ∧ 
  correct_mark = 10 → 
  ∃ wrong_mark : ℝ, 
    wrong_mark = 50 ∧ 
    n * initial_avg = (n - 1) * correct_avg + wrong_mark ∧
    n * correct_avg = (n - 1) * correct_avg + correct_mark :=
by sorry

end NUMINAMATH_CALUDE_wrong_mark_calculation_l2170_217039


namespace NUMINAMATH_CALUDE_bulb_arrangement_count_l2170_217057

/-- The number of ways to arrange bulbs in a garland with no consecutive white bulbs -/
def bulb_arrangements (blue red white : ℕ) : ℕ :=
  Nat.choose (blue + red) blue * Nat.choose (blue + red + 1) white

/-- Theorem stating the number of arrangements for the given bulb counts -/
theorem bulb_arrangement_count :
  bulb_arrangements 7 6 10 = 1717716 := by
  sorry

end NUMINAMATH_CALUDE_bulb_arrangement_count_l2170_217057


namespace NUMINAMATH_CALUDE_saras_weekly_savings_l2170_217013

/-- Sara's weekly savings to match Jim's savings after 820 weeks -/
theorem saras_weekly_savings (sara_initial : ℕ) (jim_weekly : ℕ) (weeks : ℕ) : 
  sara_initial = 4100 → jim_weekly = 15 → weeks = 820 →
  ∃ (sara_weekly : ℕ), sara_initial + weeks * sara_weekly = weeks * jim_weekly := by
  sorry

#check saras_weekly_savings

end NUMINAMATH_CALUDE_saras_weekly_savings_l2170_217013


namespace NUMINAMATH_CALUDE_distance_pole_to_line_rho_cos_theta_eq_two_l2170_217081

/-- The distance from the pole to a line in polar coordinates -/
def distance_pole_to_line (a : ℝ) : ℝ :=
  |a|

/-- Theorem: The distance from the pole to the line ρcosθ=2 is 2 -/
theorem distance_pole_to_line_rho_cos_theta_eq_two :
  distance_pole_to_line 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_pole_to_line_rho_cos_theta_eq_two_l2170_217081


namespace NUMINAMATH_CALUDE_g_of_3_l2170_217088

def g (x : ℝ) : ℝ := 5 * x^3 - 7 * x^2 + 3 * x - 2

theorem g_of_3 : g 3 = 79 := by
  sorry

end NUMINAMATH_CALUDE_g_of_3_l2170_217088


namespace NUMINAMATH_CALUDE_nested_fraction_equality_l2170_217067

theorem nested_fraction_equality : 
  (1 : ℚ) / (3 - 1 / (3 - 1 / (3 - 1 / 3))) = 8 / 21 := by sorry

end NUMINAMATH_CALUDE_nested_fraction_equality_l2170_217067


namespace NUMINAMATH_CALUDE_bears_captured_pieces_l2170_217074

theorem bears_captured_pieces (H B F : ℕ) : 
  (64 : ℕ) = H + B + F →
  H = B / 2 →
  H = F / 5 →
  (0 : ℕ) = 16 - B :=
by sorry

end NUMINAMATH_CALUDE_bears_captured_pieces_l2170_217074


namespace NUMINAMATH_CALUDE_four_digit_divisible_by_3_l2170_217006

/-- A function that returns true if a four-digit number of the form 258n is divisible by 3 -/
def isDivisibleBy3 (n : Nat) : Prop :=
  n ≥ 0 ∧ n ≤ 9 ∧ (2580 + n) % 3 = 0

/-- Theorem stating that a four-digit number 258n is divisible by 3 iff n is 0, 3, 6, or 9 -/
theorem four_digit_divisible_by_3 :
  ∀ n : Nat, isDivisibleBy3 n ↔ n = 0 ∨ n = 3 ∨ n = 6 ∨ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_divisible_by_3_l2170_217006


namespace NUMINAMATH_CALUDE_weaving_increase_proof_l2170_217049

/-- Represents the daily increase in weaving output -/
def daily_increase : ℚ := 16 / 29

/-- Represents the initial weaving output on the first day -/
def initial_output : ℚ := 5

/-- Represents the total number of days -/
def total_days : ℕ := 30

/-- Represents the total amount of fabric woven over the period -/
def total_output : ℚ := 390

theorem weaving_increase_proof :
  (initial_output + (total_days - 1) * daily_increase / 2) * total_days = total_output := by
  sorry

end NUMINAMATH_CALUDE_weaving_increase_proof_l2170_217049


namespace NUMINAMATH_CALUDE_sin_2005_equals_neg_sin_25_l2170_217063

theorem sin_2005_equals_neg_sin_25 :
  Real.sin (2005 * π / 180) = -Real.sin (25 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_sin_2005_equals_neg_sin_25_l2170_217063


namespace NUMINAMATH_CALUDE_existence_of_special_numbers_l2170_217028

theorem existence_of_special_numbers : ∃ (a b c : ℕ), 
  (a > 10^10 ∧ b > 10^10 ∧ c > 10^10) ∧
  (a * b * c) % (a + 2012) = 0 ∧
  (a * b * c) % (b + 2012) = 0 ∧
  (a * b * c) % (c + 2012) = 0 :=
by sorry

end NUMINAMATH_CALUDE_existence_of_special_numbers_l2170_217028


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2170_217047

/-- The cost of fruits at Zoe's Zesty Market -/
structure FruitCost where
  banana : ℕ
  apple : ℕ
  orange : ℕ

/-- The cost relationship between fruits -/
def cost_relationship (fc : FruitCost) : Prop :=
  5 * fc.banana = 4 * fc.apple ∧ 8 * fc.apple = 6 * fc.orange

/-- The theorem stating the equivalence of 40 bananas and 24 oranges in cost -/
theorem banana_orange_equivalence (fc : FruitCost) 
  (h : cost_relationship fc) : 40 * fc.banana = 24 * fc.orange := by
  sorry

#check banana_orange_equivalence

end NUMINAMATH_CALUDE_banana_orange_equivalence_l2170_217047


namespace NUMINAMATH_CALUDE_elberta_money_l2170_217048

theorem elberta_money (granny_smith : ℕ) (anjou elberta : ℝ) : 
  granny_smith = 72 →
  anjou = (1 / 4 : ℝ) * granny_smith →
  elberta = anjou + 3 →
  elberta = 21 := by sorry

end NUMINAMATH_CALUDE_elberta_money_l2170_217048


namespace NUMINAMATH_CALUDE_total_investment_total_investment_is_6647_l2170_217021

/-- The problem of calculating total investments --/
theorem total_investment (raghu_investment : ℕ) : ℕ :=
  let trishul_investment := raghu_investment - raghu_investment / 10
  let vishal_investment := trishul_investment + trishul_investment / 10
  raghu_investment + trishul_investment + vishal_investment

/-- The theorem stating that the total investment is 6647 when Raghu invests 2300 --/
theorem total_investment_is_6647 : total_investment 2300 = 6647 := by
  sorry

end NUMINAMATH_CALUDE_total_investment_total_investment_is_6647_l2170_217021


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_root_two_implies_k_value_l2170_217043

/-- The quadratic equation k^2*x^2 + 2*(k-1)*x + 1 = 0 -/
def quadratic_equation (k x : ℝ) : Prop :=
  k^2 * x^2 + 2*(k-1)*x + 1 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (k : ℝ) : ℝ :=
  4*(k-1)^2 - 4*k^2

theorem quadratic_two_distinct_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation k x ∧ quadratic_equation k y) ↔
  (k < 1/2 ∧ k ≠ 0) :=
sorry

theorem root_two_implies_k_value :
  ∀ k : ℝ, quadratic_equation k 2 → k = -3/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_root_two_implies_k_value_l2170_217043


namespace NUMINAMATH_CALUDE_disk_space_remaining_l2170_217030

/-- Calculates the remaining disk space given total space and used space -/
def remaining_space (total : ℕ) (used : ℕ) : ℕ :=
  total - used

/-- Theorem: Given 28 GB total space and 26 GB used space, the remaining space is 2 GB -/
theorem disk_space_remaining :
  remaining_space 28 26 = 2 := by
  sorry

end NUMINAMATH_CALUDE_disk_space_remaining_l2170_217030


namespace NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_1_range_f_geq_inequality_l2170_217070

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 1

-- Theorem for the first part of the problem
theorem solution_set_f_leq_x_plus_1 :
  {x : ℝ | f x ≤ x + 1} = Set.Icc 0 2 :=
sorry

-- Theorem for the second part of the problem
theorem range_f_geq_inequality (a : ℝ) (ha : a ≠ 0) :
  {x : ℝ | ∀ a : ℝ, a ≠ 0 → f x ≥ (|a + 1| - |2*a - 1|) / |a|} = 
    Set.Iic (-2) ∪ Set.Ici 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_leq_x_plus_1_range_f_geq_inequality_l2170_217070


namespace NUMINAMATH_CALUDE_factories_unchecked_l2170_217095

theorem factories_unchecked (total : ℕ) (group1 : ℕ) (group2 : ℕ) 
  (h1 : total = 169) 
  (h2 : group1 = 69) 
  (h3 : group2 = 52) : 
  total - (group1 + group2) = 48 := by
  sorry

end NUMINAMATH_CALUDE_factories_unchecked_l2170_217095


namespace NUMINAMATH_CALUDE_marys_friends_marys_friends_correct_l2170_217071

theorem marys_friends (total_stickers : ℕ) (stickers_per_friend : ℕ) (stickers_per_non_friend : ℕ) 
  (stickers_left : ℕ) (total_students : ℕ) : ℕ :=
  let num_friends := (total_stickers - stickers_left - 2 * (total_students - 1)) / 
    (stickers_per_friend - stickers_per_non_friend)
  num_friends

theorem marys_friends_correct : marys_friends 50 4 2 8 17 = 5 := by
  sorry

end NUMINAMATH_CALUDE_marys_friends_marys_friends_correct_l2170_217071


namespace NUMINAMATH_CALUDE_jose_maria_age_difference_jose_maria_age_difference_proof_l2170_217008

theorem jose_maria_age_difference : ℕ → ℕ → Prop :=
  fun jose_age maria_age =>
    (jose_age > maria_age) →
    (jose_age + maria_age = 40) →
    (maria_age = 14) →
    (jose_age - maria_age = 12)

-- The proof would go here, but we'll skip it as requested
theorem jose_maria_age_difference_proof : ∃ (j m : ℕ), jose_maria_age_difference j m :=
  sorry

end NUMINAMATH_CALUDE_jose_maria_age_difference_jose_maria_age_difference_proof_l2170_217008


namespace NUMINAMATH_CALUDE_projection_vector_l2170_217000

/-- Two parallel lines r and s in 2D space -/
structure ParallelLines where
  r : ℝ → ℝ × ℝ
  s : ℝ → ℝ × ℝ
  hr : ∀ t, r t = (2 + 5*t, 3 - 2*t)
  hs : ∀ u, s u = (1 + 5*u, -2 - 2*u)

/-- Points C, D, and Q in 2D space -/
structure Points (l : ParallelLines) where
  C : ℝ × ℝ
  D : ℝ × ℝ
  Q : ℝ × ℝ
  hC : ∃ t, l.r t = C
  hD : ∃ u, l.s u = D
  hQ : (Q.1 - C.1) * 5 + (Q.2 - C.2) * (-2) = 0 -- Q is on the perpendicular to s passing through C

/-- The theorem to be proved -/
theorem projection_vector (l : ParallelLines) (p : Points l) :
  ∃ k : ℝ, 
    (p.Q.1 - p.C.1, p.Q.2 - p.C.2) = k • (-2, -5) ∧
    (p.D.1 - p.C.1) * (-2) + (p.D.2 - p.C.2) * (-5) = 
      (p.Q.1 - p.C.1) * (-2) + (p.Q.2 - p.C.2) * (-5) ∧
    -2 - (-5) = 3 :=
  sorry

end NUMINAMATH_CALUDE_projection_vector_l2170_217000


namespace NUMINAMATH_CALUDE_b_fourth_zero_implies_b_squared_zero_l2170_217038

theorem b_fourth_zero_implies_b_squared_zero 
  (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : 
  B ^ 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_b_fourth_zero_implies_b_squared_zero_l2170_217038


namespace NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_l2170_217042

/-- Prove that one cubic meter is equal to 1,000,000 cubic centimeters -/
theorem cubic_meter_to_cubic_centimeters :
  (∀ m cm : ℕ, m = 100 * cm → m^3 = 1000000 * cm^3) :=
by sorry

end NUMINAMATH_CALUDE_cubic_meter_to_cubic_centimeters_l2170_217042


namespace NUMINAMATH_CALUDE_system_solution_l2170_217066

theorem system_solution : 
  ∃ (x y : ℚ), 
    (4 * x - 3 * y = -2) ∧ 
    (5 * x + 2 * y = 8) ∧ 
    (x = 20 / 23) ∧ 
    (y = 42 / 23) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l2170_217066


namespace NUMINAMATH_CALUDE_sequence_general_formula_l2170_217075

theorem sequence_general_formula (a : ℕ+ → ℝ) (S : ℕ+ → ℝ) :
  a 1 = 3 ∧
  (∀ n : ℕ+, S n = 2 * n * a (n + 1) - 3 * n^2 - 4 * n) →
  ∀ n : ℕ+, a n = 2 * n + 1 :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_formula_l2170_217075


namespace NUMINAMATH_CALUDE_equation_solution_range_l2170_217054

theorem equation_solution_range (k : ℝ) : 
  (∃! x : ℝ, x > 0 ∧ (x^2 + k*x + 3) / (x - 1) = 3*x + k) ↔ 
  (k = -33/8 ∨ k = -4 ∨ k ≥ -3) := by
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l2170_217054


namespace NUMINAMATH_CALUDE_triangle_area_l2170_217082

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) : 
  (1/2) * a * b = 54 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2170_217082


namespace NUMINAMATH_CALUDE_range_of_a_l2170_217058

-- Define propositions P and Q
def P (a : ℝ) : Prop := ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def Q (a : ℝ) : Prop := ∃ x y : ℝ, x^2 / a + y^2 / (a - 3) = 1 ∧ a * (a - 3) < 0

-- Define the theorem
theorem range_of_a (a : ℝ) : 
  ((P a ∨ Q a) ∧ ¬(P a ∧ Q a)) → (a = 0 ∨ (3 ≤ a ∧ a < 4)) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2170_217058


namespace NUMINAMATH_CALUDE_cacl₂_production_l2170_217084

/-- Represents the chemical reaction: CaCO₃ + 2HCl → CaCl₂ + CO₂ + H₂O -/
structure ChemicalReaction where
  cacO₃ : ℚ  -- moles of CaCO₃
  hcl : ℚ    -- moles of HCl
  cacl₂ : ℚ  -- moles of CaCl₂ produced

/-- The stoichiometric ratio of the reaction -/
def stoichiometricRatio : ℚ := 2

/-- Calculates the amount of CaCl₂ produced based on the limiting reactant -/
def calcCaCl₂Produced (reaction : ChemicalReaction) : ℚ :=
  min reaction.cacO₃ (reaction.hcl / stoichiometricRatio)

/-- Theorem stating that 2 moles of CaCl₂ are produced when 4 moles of HCl react with 2 moles of CaCO₃ -/
theorem cacl₂_production (reaction : ChemicalReaction) 
  (h1 : reaction.cacO₃ = 2)
  (h2 : reaction.hcl = 4) :
  calcCaCl₂Produced reaction = 2 := by
  sorry

end NUMINAMATH_CALUDE_cacl₂_production_l2170_217084


namespace NUMINAMATH_CALUDE_cone_base_radius_l2170_217061

theorem cone_base_radius 
  (unfolded_area : ℝ) 
  (generatrix : ℝ) 
  (h1 : unfolded_area = 15 * Real.pi) 
  (h2 : generatrix = 5) : 
  ∃ (base_radius : ℝ), base_radius = 3 ∧ unfolded_area = Real.pi * base_radius * generatrix :=
by sorry

end NUMINAMATH_CALUDE_cone_base_radius_l2170_217061


namespace NUMINAMATH_CALUDE_least_sum_exponents_1540_l2170_217068

/-- The function that computes the least sum of exponents for a given number -/
def leastSumOfExponents (n : ℕ) : ℕ := sorry

/-- The theorem stating that the least sum of exponents for 1540 is 21 -/
theorem least_sum_exponents_1540 : leastSumOfExponents 1540 = 21 := by sorry

end NUMINAMATH_CALUDE_least_sum_exponents_1540_l2170_217068


namespace NUMINAMATH_CALUDE_tangent_circle_existence_l2170_217024

-- Define the circle S
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point as a pair of real numbers
def Point := ℝ × ℝ

-- Define a line as a pair of points
structure Line where
  point1 : Point
  point2 : Point

-- Define tangency between two circles at a point
def CircleTangentToCircle (S S' : Circle) (A : Point) : Prop :=
  -- The centers of S and S' and point A are collinear
  sorry

-- Define tangency between a circle and a line at a point
def CircleTangentToLine (S : Circle) (l : Line) (B : Point) : Prop :=
  -- The radius of S at B is perpendicular to l
  sorry

-- The main theorem
theorem tangent_circle_existence 
  (S : Circle) (A : Point) (l : Line) : 
  ∃ (S' : Circle) (B : Point), 
    CircleTangentToCircle S S' A ∧ 
    CircleTangentToLine S' l B :=
  sorry

end NUMINAMATH_CALUDE_tangent_circle_existence_l2170_217024


namespace NUMINAMATH_CALUDE_initial_men_count_l2170_217034

/-- The number of men working initially -/
def initial_men : ℕ := 12

/-- The number of hours worked per day by the initial group -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked by the initial group -/
def initial_days : ℕ := 10

/-- The number of men in the new group -/
def new_men : ℕ := 6

/-- The number of hours worked per day by the new group -/
def new_hours_per_day : ℕ := 20

/-- The number of days worked by the new group -/
def new_days : ℕ := 8

/-- Theorem stating that the initial number of men is 12 -/
theorem initial_men_count : 
  initial_men * initial_hours_per_day * initial_days = 
  new_men * new_hours_per_day * new_days :=
by
  sorry

#check initial_men_count

end NUMINAMATH_CALUDE_initial_men_count_l2170_217034


namespace NUMINAMATH_CALUDE_total_subjects_is_41_l2170_217041

/-- The number of subjects taken by Monica -/
def monica_subjects : ℕ := 10

/-- The number of subjects taken by Marius -/
def marius_subjects : ℕ := monica_subjects + 4

/-- The number of subjects taken by Millie -/
def millie_subjects : ℕ := marius_subjects + 3

/-- The total number of subjects taken by all three students -/
def total_subjects : ℕ := monica_subjects + marius_subjects + millie_subjects

/-- Theorem stating that the total number of subjects is 41 -/
theorem total_subjects_is_41 : total_subjects = 41 := by
  sorry

end NUMINAMATH_CALUDE_total_subjects_is_41_l2170_217041


namespace NUMINAMATH_CALUDE_sea_horse_count_l2170_217037

theorem sea_horse_count : 
  ∀ (s p : ℕ), 
  (s : ℚ) / p = 5 / 11 → 
  p = s + 85 → 
  s = 70 := by
sorry

end NUMINAMATH_CALUDE_sea_horse_count_l2170_217037


namespace NUMINAMATH_CALUDE_green_apples_count_l2170_217020

theorem green_apples_count (total : ℕ) (red_to_green_ratio : ℕ) 
  (h1 : total = 496) 
  (h2 : red_to_green_ratio = 3) : 
  ∃ (green : ℕ), green = 124 ∧ total = green * (red_to_green_ratio + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_green_apples_count_l2170_217020


namespace NUMINAMATH_CALUDE_root_sum_squares_l2170_217005

theorem root_sum_squares (a b c : ℝ) : 
  (a^3 - 20*a^2 + 18*a - 7 = 0) →
  (b^3 - 20*b^2 + 18*b - 7 = 0) →
  (c^3 - 20*c^2 + 18*c - 7 = 0) →
  (a+b)^2 + (b+c)^2 + (c+a)^2 = 764 := by
sorry

end NUMINAMATH_CALUDE_root_sum_squares_l2170_217005


namespace NUMINAMATH_CALUDE_max_circle_sum_is_15_l2170_217032

/-- Represents a configuration of numbers in the circle diagram -/
def CircleConfiguration := Fin 7 → Fin 7

/-- The sum of numbers in a given circle of the configuration -/
def circle_sum (config : CircleConfiguration) (circle : Fin 3) : ℕ :=
  sorry

/-- Checks if a configuration is valid (uses all numbers 0 to 6 exactly once) -/
def is_valid_configuration (config : CircleConfiguration) : Prop :=
  sorry

/-- Checks if all circles in a configuration have the same sum -/
def all_circles_equal_sum (config : CircleConfiguration) : Prop :=
  sorry

/-- The maximum possible sum for each circle -/
def max_circle_sum : ℕ := 15

theorem max_circle_sum_is_15 :
  ∃ (config : CircleConfiguration),
    is_valid_configuration config ∧
    all_circles_equal_sum config ∧
    ∀ (c : Fin 3), circle_sum config c = max_circle_sum ∧
    ∀ (config' : CircleConfiguration),
      is_valid_configuration config' →
      all_circles_equal_sum config' →
      ∀ (c : Fin 3), circle_sum config' c ≤ max_circle_sum :=
sorry

end NUMINAMATH_CALUDE_max_circle_sum_is_15_l2170_217032


namespace NUMINAMATH_CALUDE_power_equation_solution_l2170_217010

theorem power_equation_solution : ∃ K : ℕ, (4 ^ 5) * (2 ^ 3) = 2 ^ K ∧ K = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l2170_217010


namespace NUMINAMATH_CALUDE_cafeteria_pies_l2170_217062

/-- Given a cafeteria with initial apples, apples handed out, and apples required per pie,
    calculates the number of pies that can be made with the remaining apples. -/
def calculate_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

/-- Proves that given 86 initial apples, after handing out 30 apples,
    and using 8 apples per pie, the number of pies that can be made is 7. -/
theorem cafeteria_pies :
  calculate_pies 86 30 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_cafeteria_pies_l2170_217062


namespace NUMINAMATH_CALUDE_intersection_when_a_2_B_subset_A_condition_l2170_217001

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | (x - 2) * (x - (3 * a + 1)) < 0}
def B (a : ℝ) : Set ℝ := {x | (x - 2 * a) / (x - (a^2 + 1)) < 0}

-- Part 1: Intersection when a = 2
theorem intersection_when_a_2 : 
  A 2 ∩ B 2 = {x : ℝ | 4 < x ∧ x < 5} := by sorry

-- Part 2: Condition for B to be a subset of A
theorem B_subset_A_condition (a : ℝ) :
  a ≠ 1 →
  (B a ⊆ A a ↔ (1 < a ∧ a ≤ 3) ∨ a = -1) := by sorry

end NUMINAMATH_CALUDE_intersection_when_a_2_B_subset_A_condition_l2170_217001


namespace NUMINAMATH_CALUDE_work_completion_time_l2170_217077

/-- Proves that A can complete the work in 15 days given the conditions -/
theorem work_completion_time (x : ℝ) : 
  (x > 0) →  -- A's completion time is positive
  (4 * (1 / x + 1 / 20) = 1 - 0.5333333333333333) →  -- Condition after 4 days of joint work
  x = 15 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l2170_217077


namespace NUMINAMATH_CALUDE_smallest_consecutive_triangle_perimeter_l2170_217035

/-- A triangle with consecutive integer side lengths. -/
structure ConsecutiveTriangle where
  a : ℕ
  valid : a > 0

/-- The three side lengths of a ConsecutiveTriangle. -/
def ConsecutiveTriangle.sides (t : ConsecutiveTriangle) : Fin 3 → ℕ
  | 0 => t.a
  | 1 => t.a + 1
  | 2 => t.a + 2

/-- The perimeter of a ConsecutiveTriangle. -/
def ConsecutiveTriangle.perimeter (t : ConsecutiveTriangle) : ℕ :=
  3 * t.a + 3

/-- Predicate for whether a ConsecutiveTriangle satisfies the Triangle Inequality. -/
def ConsecutiveTriangle.satisfiesTriangleInequality (t : ConsecutiveTriangle) : Prop :=
  t.sides 0 + t.sides 1 > t.sides 2 ∧
  t.sides 0 + t.sides 2 > t.sides 1 ∧
  t.sides 1 + t.sides 2 > t.sides 0

/-- The smallest ConsecutiveTriangle that satisfies the Triangle Inequality. -/
def smallestValidConsecutiveTriangle : ConsecutiveTriangle :=
  { a := 2
    valid := by simp }

/-- Theorem: The smallest possible perimeter of a triangle with consecutive integer side lengths is 9. -/
theorem smallest_consecutive_triangle_perimeter :
  (∀ t : ConsecutiveTriangle, t.satisfiesTriangleInequality → t.perimeter ≥ 9) ∧
  smallestValidConsecutiveTriangle.satisfiesTriangleInequality ∧
  smallestValidConsecutiveTriangle.perimeter = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_consecutive_triangle_perimeter_l2170_217035


namespace NUMINAMATH_CALUDE_squares_in_6x6_grid_l2170_217092

/-- Calculates the number of squares in a grid with n+1 lines in each direction -/
def count_squares (n : ℕ) : ℕ := 
  (n * (n + 1) * (2 * n + 1)) / 6

/-- Theorem: In a 6x6 grid, the total number of squares is 55 -/
theorem squares_in_6x6_grid : count_squares 5 = 55 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_6x6_grid_l2170_217092


namespace NUMINAMATH_CALUDE_jason_music_store_expenditure_l2170_217053

/-- The total cost of Jason's music store purchases --/
def total_cost : ℚ :=
  142.46 + 8.89 + 7.00 + 15.75 + 12.95 + 36.50 + 5.25

/-- Theorem stating that Jason's total music store expenditure is $229.80 --/
theorem jason_music_store_expenditure :
  total_cost = 229.80 := by sorry

end NUMINAMATH_CALUDE_jason_music_store_expenditure_l2170_217053


namespace NUMINAMATH_CALUDE_det_linear_combination_zero_l2170_217045

open Matrix

theorem det_linear_combination_zero
  (A B : Matrix (Fin 3) (Fin 3) ℝ)
  (h : A ^ 2 + B ^ 2 = 0) :
  ∀ (a b : ℝ), det (a • A + b • B) = 0 := by
sorry

end NUMINAMATH_CALUDE_det_linear_combination_zero_l2170_217045


namespace NUMINAMATH_CALUDE_relationship_abc_l2170_217098

theorem relationship_abc : 3^(1/10) > (1/2)^(1/10) ∧ (1/2)^(1/10) > (-1/2)^3 := by
  sorry

end NUMINAMATH_CALUDE_relationship_abc_l2170_217098


namespace NUMINAMATH_CALUDE_beth_peas_cans_l2170_217090

/-- The number of cans of corn Beth bought -/
def corn_cans : ℕ := 10

/-- The number of cans of peas Beth bought -/
def peas_cans : ℕ := 2 * corn_cans + 15

theorem beth_peas_cans : peas_cans = 35 := by
  sorry

end NUMINAMATH_CALUDE_beth_peas_cans_l2170_217090


namespace NUMINAMATH_CALUDE_abc_equation_solution_l2170_217022

theorem abc_equation_solution (a b c : ℕ+) (h1 : b ≤ c) 
  (h2 : (a * b - 1) * (a * c - 1) = 2023 * b * c) : 
  c = 82 ∨ c = 167 ∨ c = 1034 := by
sorry

end NUMINAMATH_CALUDE_abc_equation_solution_l2170_217022


namespace NUMINAMATH_CALUDE_segment_division_l2170_217080

/-- Given a segment of length a, prove that dividing it into n equal parts
    results in each part having a length of a/(n+1). -/
theorem segment_division (a : ℝ) (n : ℕ) (h : 0 < n) :
  ∃ (x : ℝ), x = a / (n + 1) ∧ n * x = a :=
by sorry

end NUMINAMATH_CALUDE_segment_division_l2170_217080


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l2170_217031

theorem quadratic_equation_root (b : ℝ) : 
  (2 * (4 : ℝ)^2 + b * 4 - 44 = 0) → b = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l2170_217031


namespace NUMINAMATH_CALUDE_monotone_quadratic_function_m_range_l2170_217093

/-- A function f is monotonically increasing on an interval (a, b) if for any x, y in (a, b) with x < y, we have f(x) < f(y) -/
def MonotonicallyIncreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x < f y

/-- The function f(x) = mx^2 + x - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 + x - 1

theorem monotone_quadratic_function_m_range :
  (∀ m : ℝ, MonotonicallyIncreasing (f m) (-1) Real.pi) ↔ 
  (∀ m : ℝ, 0 ≤ m ∧ m ≤ 1/2) :=
sorry

end NUMINAMATH_CALUDE_monotone_quadratic_function_m_range_l2170_217093


namespace NUMINAMATH_CALUDE_object3_length_is_15_l2170_217025

def longest_tape : ℕ := 5

def object1_length : ℕ := 225
def object2_length : ℕ := 780

def object3_length : ℕ := Nat.gcd object1_length object2_length

theorem object3_length_is_15 :
  longest_tape = 5 ∧
  object1_length = 225 ∧
  object2_length = 780 ∧
  object3_length = Nat.gcd object1_length object2_length →
  object3_length = 15 :=
by sorry

end NUMINAMATH_CALUDE_object3_length_is_15_l2170_217025


namespace NUMINAMATH_CALUDE_quadratic_intersection_points_l2170_217002

/-- A quadratic function with at least one y-intercept -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  has_y_intercept : ∃ x, a * x^2 + b * x + c = 0

/-- The number of intersection points between f(x) and -f(-x) -/
def intersection_points_f_u (f : QuadraticFunction) : ℕ := 1

/-- The number of intersection points between f(x) and f(x+1) -/
def intersection_points_f_v (f : QuadraticFunction) : ℕ := 0

/-- The main theorem -/
theorem quadratic_intersection_points (f : QuadraticFunction) :
  7 * (intersection_points_f_u f) + 3 * (intersection_points_f_v f) = 7 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_intersection_points_l2170_217002


namespace NUMINAMATH_CALUDE_vector_collinearity_l2170_217086

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b.1 = k * a.1 ∧ b.2 = k * a.2

theorem vector_collinearity :
  let a : ℝ × ℝ := (3, 6)
  let b : ℝ × ℝ := (x, 8)
  collinear a b → x = 4 :=
by sorry

end NUMINAMATH_CALUDE_vector_collinearity_l2170_217086


namespace NUMINAMATH_CALUDE_boat_journey_time_l2170_217059

/-- Calculates the total journey time for a boat traveling upstream and downstream in a river -/
theorem boat_journey_time 
  (river_speed : ℝ) 
  (boat_speed : ℝ) 
  (distance : ℝ) 
  (h1 : river_speed = 2)
  (h2 : boat_speed = 6)
  (h3 : distance = 64) : 
  (distance / (boat_speed - river_speed)) + (distance / (boat_speed + river_speed)) = 24 := by
  sorry

#check boat_journey_time

end NUMINAMATH_CALUDE_boat_journey_time_l2170_217059


namespace NUMINAMATH_CALUDE_existence_of_special_sequence_l2170_217046

theorem existence_of_special_sequence : ∃ (a b c : ℝ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧
  (a + b + c = 6) ∧
  (b - a = c - b) ∧
  ((a^2 = b * c) ∨ (b^2 = a * c) ∨ (c^2 = a * b)) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_special_sequence_l2170_217046


namespace NUMINAMATH_CALUDE_arrangements_starting_with_vowel_l2170_217085

def word : String := "basics"

def is_vowel (c : Char) : Bool :=
  c = 'a' || c = 'e' || c = 'i' || c = 'o' || c = 'u'

def count_vowels (s : String) : Nat :=
  s.toList.filter is_vowel |>.length

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def permutations_with_repetition (total : Nat) (repeated : List Nat) : Nat :=
  factorial total / (repeated.map factorial).prod

theorem arrangements_starting_with_vowel :
  let total_letters := word.length
  let vowels := count_vowels word
  let consonants := total_letters - vowels
  let arrangements := 
    vowels * permutations_with_repetition (total_letters - 1) [consonants, vowels - 1, 1]
  arrangements = 120 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_starting_with_vowel_l2170_217085


namespace NUMINAMATH_CALUDE_tower_configurations_count_l2170_217026

/-- The number of ways to build a tower of 10 cubes high using 3 red cubes, 4 blue cubes, and 5 yellow cubes, where two cubes are not used. -/
def towerConfigurations (red : Nat) (blue : Nat) (yellow : Nat) (towerHeight : Nat) : Nat :=
  sorry

/-- Theorem stating that the number of different tower configurations is 277,200 -/
theorem tower_configurations_count :
  towerConfigurations 3 4 5 10 = 277200 := by
  sorry

end NUMINAMATH_CALUDE_tower_configurations_count_l2170_217026


namespace NUMINAMATH_CALUDE_rex_cards_left_is_150_l2170_217011

/-- The number of Pokemon cards Rex has left after dividing his collection --/
def rexCardsLeft (nicolesCards : ℕ) : ℕ :=
  let cindysCards := nicolesCards * 2
  let totalCards := nicolesCards + cindysCards
  let rexCards := totalCards / 2
  rexCards / 4

/-- Theorem stating that Rex has 150 cards left --/
theorem rex_cards_left_is_150 : rexCardsLeft 400 = 150 := by
  sorry

end NUMINAMATH_CALUDE_rex_cards_left_is_150_l2170_217011


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2170_217003

def set_A : Set ℝ := {x | 2 * x + 1 > 0}
def set_B : Set ℝ := {x | |x - 1| < 2}

theorem intersection_of_A_and_B :
  set_A ∩ set_B = {x : ℝ | -1/2 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2170_217003


namespace NUMINAMATH_CALUDE_cuboid_volume_l2170_217015

/-- A cuboid with given height and base area has the specified volume. -/
theorem cuboid_volume (height : ℝ) (base_area : ℝ) :
  height = 13 → base_area = 14 → height * base_area = 182 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_volume_l2170_217015


namespace NUMINAMATH_CALUDE_gcd_of_840_and_1764_l2170_217033

theorem gcd_of_840_and_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_840_and_1764_l2170_217033


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2170_217012

theorem quadratic_roots_sum_product (k p : ℝ) : 
  (∃ x y : ℝ, 3 * x^2 - k * x + p = 0 ∧ 3 * y^2 - k * y + p = 0 ∧ x + y = -3 ∧ x * y = -6) →
  k + p = -27 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l2170_217012


namespace NUMINAMATH_CALUDE_bus_speed_problem_l2170_217094

theorem bus_speed_problem (distance : ℝ) (speed_ratio : ℝ) (time_difference : ℝ) :
  distance = 72 →
  speed_ratio = 1.2 →
  time_difference = 1/5 →
  ∀ (speed_large : ℝ),
    (distance / speed_large - distance / (speed_ratio * speed_large) = time_difference) →
    speed_large = 60 := by
  sorry

end NUMINAMATH_CALUDE_bus_speed_problem_l2170_217094


namespace NUMINAMATH_CALUDE_subtract_negative_l2170_217097

theorem subtract_negative (a b : ℝ) : a - (-b) = a + b := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l2170_217097


namespace NUMINAMATH_CALUDE_city_population_ratio_l2170_217019

theorem city_population_ratio (x y z : ℕ) 
  (h1 : y = 2 * z)
  (h2 : x = 12 * z) :
  x / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_city_population_ratio_l2170_217019


namespace NUMINAMATH_CALUDE_chips_on_line_after_moves_l2170_217060

/-- Represents a configuration of chips on a plane -/
structure ChipConfiguration where
  num_chips : ℕ
  num_lines : ℕ

/-- Represents a move that can be applied to a chip configuration -/
def apply_move (config : ChipConfiguration) : ChipConfiguration :=
  { num_chips := config.num_chips,
    num_lines := min config.num_lines (2 ^ (config.num_lines - 1)) }

/-- Represents the initial configuration of chips on a convex 2000-gon -/
def initial_config : ChipConfiguration :=
  { num_chips := 2000,
    num_lines := 2000 }

/-- Applies n moves to the initial configuration -/
def apply_n_moves (n : ℕ) : ChipConfiguration :=
  (List.range n).foldl (λ config _ => apply_move config) initial_config

theorem chips_on_line_after_moves :
  (∀ n : ℕ, n ≤ 9 → (apply_n_moves n).num_lines > 1) ∧
  ∃ m : ℕ, m = 10 ∧ (apply_n_moves m).num_lines = 1 :=
sorry

end NUMINAMATH_CALUDE_chips_on_line_after_moves_l2170_217060


namespace NUMINAMATH_CALUDE_exists_unique_subset_l2170_217052

theorem exists_unique_subset : ∃ (S : Set ℤ), 
  ∀ (n : ℤ), ∃! (pair : ℤ × ℤ), 
    pair.1 ∈ S ∧ pair.2 ∈ S ∧ n = 2 * pair.1 + pair.2 := by
  sorry

end NUMINAMATH_CALUDE_exists_unique_subset_l2170_217052


namespace NUMINAMATH_CALUDE_total_toys_l2170_217065

/-- Given that Annie has three times more toys than Mike, Annie has two less toys than Tom,
    and Mike has 6 toys, prove that the total number of toys Annie, Mike, and Tom have is 56. -/
theorem total_toys (mike_toys : ℕ) (annie_toys : ℕ) (tom_toys : ℕ)
  (h1 : annie_toys = 3 * mike_toys)
  (h2 : tom_toys = annie_toys + 2)
  (h3 : mike_toys = 6) :
  annie_toys + mike_toys + tom_toys = 56 :=
by sorry

end NUMINAMATH_CALUDE_total_toys_l2170_217065


namespace NUMINAMATH_CALUDE_doll_collection_problem_l2170_217044

theorem doll_collection_problem (original_count : ℕ) : 
  (original_count + 2 : ℚ) = original_count * (1 + 1/4) → 
  original_count + 2 = 10 := by
sorry

end NUMINAMATH_CALUDE_doll_collection_problem_l2170_217044


namespace NUMINAMATH_CALUDE_binomial_150_150_l2170_217040

theorem binomial_150_150 : (150 : ℕ).choose 150 = 1 := by sorry

end NUMINAMATH_CALUDE_binomial_150_150_l2170_217040


namespace NUMINAMATH_CALUDE_correct_average_calculation_l2170_217083

/-- Given a number of tables, women, and men, calculate the average number of customers per table. -/
def averageCustomersPerTable (tables : Float) (women : Float) (men : Float) : Float :=
  (women + men) / tables

/-- Theorem stating that for the given values, the average number of customers per table is correct. -/
theorem correct_average_calculation :
  averageCustomersPerTable 9.0 7.0 3.0 = (7.0 + 3.0) / 9.0 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_calculation_l2170_217083
