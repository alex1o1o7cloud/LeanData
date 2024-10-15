import Mathlib

namespace NUMINAMATH_CALUDE_car_efficiency_improvement_l907_90719

/-- Represents the additional miles a car can travel after improving fuel efficiency -/
def additional_miles (initial_efficiency : ℝ) (tank_capacity : ℝ) (efficiency_improvement : ℝ) : ℝ :=
  tank_capacity * (initial_efficiency * (1 + efficiency_improvement) - initial_efficiency)

/-- Theorem stating the additional miles a car can travel after modification -/
theorem car_efficiency_improvement :
  additional_miles 33 16 0.25 = 132 := by
  sorry

end NUMINAMATH_CALUDE_car_efficiency_improvement_l907_90719


namespace NUMINAMATH_CALUDE_dina_has_60_dolls_l907_90785

/-- The number of dolls Ivy has -/
def ivy_dolls : ℕ := 30

/-- The number of dolls Dina has -/
def dina_dolls : ℕ := 2 * ivy_dolls

/-- The number of collectors edition dolls Ivy has -/
def ivy_collectors : ℕ := 20

theorem dina_has_60_dolls :
  (2 * ivy_dolls = dina_dolls) →
  (2 * ivy_collectors = 3 * ivy_dolls) →
  (ivy_collectors = 20) →
  dina_dolls = 60 := by
  sorry

end NUMINAMATH_CALUDE_dina_has_60_dolls_l907_90785


namespace NUMINAMATH_CALUDE_christines_dog_weight_l907_90789

/-- The weight of Christine's dog given the weights of her two cats -/
def dogs_weight (cat1_weight cat2_weight : ℕ) : ℕ :=
  2 * (cat1_weight + cat2_weight)

/-- Theorem stating that Christine's dog weighs 34 pounds -/
theorem christines_dog_weight :
  dogs_weight 7 10 = 34 := by
  sorry

end NUMINAMATH_CALUDE_christines_dog_weight_l907_90789


namespace NUMINAMATH_CALUDE_consecutive_numbers_with_divisible_digit_sums_l907_90764

/-- Sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Theorem: There exist two consecutive natural numbers whose sum of digits are both divisible by 7 -/
theorem consecutive_numbers_with_divisible_digit_sums :
  ∃ n : ℕ, 7 ∣ sumOfDigits n ∧ 7 ∣ sumOfDigits (n + 1) := by sorry

end NUMINAMATH_CALUDE_consecutive_numbers_with_divisible_digit_sums_l907_90764


namespace NUMINAMATH_CALUDE_M_reflected_y_axis_l907_90754

/-- Reflects a point across the y-axis -/
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (-(p.1), p.2)

/-- The coordinates of point M -/
def M : ℝ × ℝ := (1, 2)

theorem M_reflected_y_axis :
  reflect_y_axis M = (-1, 2) := by sorry

end NUMINAMATH_CALUDE_M_reflected_y_axis_l907_90754


namespace NUMINAMATH_CALUDE_chess_tournament_games_l907_90780

theorem chess_tournament_games (n : ℕ) (total_games : ℕ) : 
  n = 24 → 
  total_games = 276 → 
  total_games = n * (n - 1) / 2 → 
  ∃ (games_per_participant : ℕ), 
    games_per_participant = n - 1 ∧ 
    games_per_participant = 23 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_games_l907_90780


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l907_90726

theorem quadratic_equation_from_means (a b : ℝ) 
  (h_arithmetic : (a + b) / 2 = 6)
  (h_geometric : Real.sqrt (a * b) = 5) :
  ∀ x, x^2 - 12*x + 25 = 0 ↔ (x = a ∨ x = b) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l907_90726


namespace NUMINAMATH_CALUDE_logarithm_order_comparison_l907_90749

theorem logarithm_order_comparison : 
  Real.log 4 / Real.log 3 > Real.log 3 / Real.log 4 ∧ 
  Real.log 3 / Real.log 4 > Real.log (3/4) / Real.log (4/3) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_order_comparison_l907_90749


namespace NUMINAMATH_CALUDE_robbie_afternoon_rice_l907_90788

/-- Represents the number of cups of rice Robbie eats at different times of the day and the fat content --/
structure RiceIntake where
  morning : ℕ
  evening : ℕ
  fat_per_cup : ℕ
  total_fat_per_week : ℕ

/-- Calculates the number of cups of rice Robbie eats in the afternoon --/
def afternoon_rice_cups (intake : RiceIntake) : ℕ :=
  (intake.total_fat_per_week - 7 * (intake.morning + intake.evening) * intake.fat_per_cup) / (7 * intake.fat_per_cup)

/-- Theorem stating that given the conditions, Robbie eats 14 cups of rice in the afternoon --/
theorem robbie_afternoon_rice 
  (intake : RiceIntake) 
  (h_morning : intake.morning = 3)
  (h_evening : intake.evening = 5)
  (h_fat_per_cup : intake.fat_per_cup = 10)
  (h_total_fat : intake.total_fat_per_week = 700) :
  afternoon_rice_cups intake = 14 := by
  sorry

end NUMINAMATH_CALUDE_robbie_afternoon_rice_l907_90788


namespace NUMINAMATH_CALUDE_arithmetic_sequence_first_term_range_l907_90768

theorem arithmetic_sequence_first_term_range (a : ℕ → ℝ) (d : ℝ) (h1 : d = π / 8) :
  (∀ n : ℕ, a (n + 1) = a n + d) →
  (a 10 ≤ 0) →
  (a 11 ≥ 0) →
  -5 * π / 4 ≤ a 1 ∧ a 1 ≤ -9 * π / 8 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_first_term_range_l907_90768


namespace NUMINAMATH_CALUDE_base4_equals_base2_l907_90772

-- Define a function to convert a number from base 4 to base 10
def base4ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

-- Define a function to convert a number from base 2 to base 10
def base2ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (2 ^ i)) 0

-- Theorem statement
theorem base4_equals_base2 :
  base4ToDecimal [0, 1, 0, 1] = base2ToDecimal [0, 0, 1, 0, 0, 0, 1] := by
  sorry

end NUMINAMATH_CALUDE_base4_equals_base2_l907_90772


namespace NUMINAMATH_CALUDE_cube_paint_theorem_l907_90794

/-- Given a cube of side length n, prove that if exactly one-third of the total number of faces
    of the n^3 unit cubes (obtained by cutting the original cube) are red, then n = 3. -/
theorem cube_paint_theorem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_paint_theorem_l907_90794


namespace NUMINAMATH_CALUDE_range_of_m_l907_90755

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x^2 + m*x + 2*m - 3 ≥ 0) → m ∈ Set.Icc 2 6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l907_90755


namespace NUMINAMATH_CALUDE_hcf_36_84_l907_90734

theorem hcf_36_84 : Nat.gcd 36 84 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hcf_36_84_l907_90734


namespace NUMINAMATH_CALUDE_shorts_cost_calculation_l907_90715

def jacket_cost : ℝ := 14.82
def shirt_cost : ℝ := 12.51
def total_cost : ℝ := 42.33

theorem shorts_cost_calculation : 
  total_cost - jacket_cost - shirt_cost = 15 :=
by sorry

end NUMINAMATH_CALUDE_shorts_cost_calculation_l907_90715


namespace NUMINAMATH_CALUDE_smallest_x_value_l907_90748

theorem smallest_x_value (x : ℝ) : x ≠ 1/4 →
  ((20 * x^2 - 49 * x + 20) / (4 * x - 1) + 7 * x = 3 * x + 2) →
  x ≥ 2/9 ∧ (∃ y : ℝ, y ≠ 1/4 ∧ ((20 * y^2 - 49 * y + 20) / (4 * y - 1) + 7 * y = 3 * y + 2) ∧ y = 2/9) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_value_l907_90748


namespace NUMINAMATH_CALUDE_a1_value_l907_90729

noncomputable section

def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = r * a n

theorem a1_value (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by sorry

end

end NUMINAMATH_CALUDE_a1_value_l907_90729


namespace NUMINAMATH_CALUDE_yellow_peaches_count_l907_90725

/-- The number of yellow peaches in a basket -/
def yellow_peaches (red green yellow total_green_yellow : ℕ) : Prop :=
  red = 5 ∧ green = 6 ∧ total_green_yellow = 20 → yellow = 14

theorem yellow_peaches_count : ∀ (red green yellow total_green_yellow : ℕ),
  yellow_peaches red green yellow total_green_yellow :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_peaches_count_l907_90725


namespace NUMINAMATH_CALUDE_game_strategy_sum_final_result_l907_90731

theorem game_strategy_sum (R S : ℕ) : R - S = 1010 :=
  by
  have h1 : R = (1010 : ℕ) * 2022 / 2 := by sorry
  have h2 : S = (1010 : ℕ) * 2020 / 2 := by sorry
  sorry

theorem final_result : (R - S) / 10 = 101 :=
  by
  have h : R - S = 1010 := game_strategy_sum R S
  sorry

end NUMINAMATH_CALUDE_game_strategy_sum_final_result_l907_90731


namespace NUMINAMATH_CALUDE_shyne_plants_l907_90774

/-- The number of plants Shyne can grow from her seed packets -/
def total_plants (eggplant_per_packet : ℕ) (sunflower_per_packet : ℕ) 
                 (eggplant_packets : ℕ) (sunflower_packets : ℕ) : ℕ :=
  eggplant_per_packet * eggplant_packets + sunflower_per_packet * sunflower_packets

/-- Proof that Shyne can grow 116 plants -/
theorem shyne_plants : 
  total_plants 14 10 4 6 = 116 := by
  sorry

end NUMINAMATH_CALUDE_shyne_plants_l907_90774


namespace NUMINAMATH_CALUDE_isosceles_triangles_bound_l907_90720

/-- The largest number of isosceles triangles whose vertices belong to some set of n points in the plane without three colinear points -/
noncomputable def f (n : ℕ) : ℕ := sorry

/-- Theorem stating the existence of positive real constants a and b bounding f(n) -/
theorem isosceles_triangles_bound :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ n : ℕ, n ≥ 3 → (a * n^2 : ℝ) < f n ∧ (f n : ℝ) < b * n^2 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangles_bound_l907_90720


namespace NUMINAMATH_CALUDE_min_posts_for_specific_plot_l907_90798

/-- Calculates the number of fence posts required for a given length -/
def posts_for_length (length : ℕ) : ℕ :=
  length / 10 + 1

/-- Represents a rectangular garden plot -/
structure GardenPlot where
  width : ℕ
  length : ℕ
  wall_length : ℕ

/-- Calculates the minimum number of fence posts required for a garden plot -/
def min_posts (plot : GardenPlot) : ℕ :=
  posts_for_length plot.length + 2 * (posts_for_length plot.width - 1)

/-- Theorem stating the minimum number of posts for the specific garden plot -/
theorem min_posts_for_specific_plot :
  ∃ (plot : GardenPlot), plot.width = 30 ∧ plot.length = 50 ∧ plot.wall_length = 80 ∧ min_posts plot = 12 := by
  sorry

end NUMINAMATH_CALUDE_min_posts_for_specific_plot_l907_90798


namespace NUMINAMATH_CALUDE_class_test_probabilities_l907_90751

theorem class_test_probabilities (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ)
  (h1 : p_first = 0.7)
  (h2 : p_second = 0.55)
  (h3 : p_neither = 0.2) :
  p_first + p_second - (1 - p_neither) = 0.45 := by
  sorry

end NUMINAMATH_CALUDE_class_test_probabilities_l907_90751


namespace NUMINAMATH_CALUDE_intersection_circle_equation_l907_90702

-- Define the curves C₁ and C₂
def C₁ (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 2 + t * Real.cos a ∧ p.2 = 1 + t * Real.sin a}

def C₂ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2/2 = 1}

-- Define the intersection points M and N
def intersection_points : Set (ℝ × ℝ) :=
  C₁ (Real.pi/4) ∩ C₂

-- State the theorem
theorem intersection_circle_equation :
  ∀ M N : ℝ × ℝ,
  M ∈ intersection_points → N ∈ intersection_points → M ≠ N →
  ∀ P : ℝ × ℝ,
  P ∈ {P : ℝ × ℝ | (P.1 - 1/3)^2 + (P.2 + 2/3)^2 = 8/9} ↔
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t • M + (1 - t) • N) :=
by sorry


end NUMINAMATH_CALUDE_intersection_circle_equation_l907_90702


namespace NUMINAMATH_CALUDE_rectangular_hall_dimensions_l907_90763

theorem rectangular_hall_dimensions (length width area : ℝ) : 
  width = (1/2) * length →
  area = length * width →
  area = 200 →
  length - width = 10 := by
sorry

end NUMINAMATH_CALUDE_rectangular_hall_dimensions_l907_90763


namespace NUMINAMATH_CALUDE_candy_bar_profit_l907_90769

/-- Calculates the profit from selling candy bars --/
def calculate_profit (
  total_bars : ℕ
  ) (buy_rate : ℚ × ℚ)
    (sell_rate : ℚ × ℚ)
    (discount_rate : ℕ × ℚ) : ℚ :=
  let cost_per_bar := buy_rate.2 / buy_rate.1
  let sell_per_bar := sell_rate.2 / sell_rate.1
  let total_cost := cost_per_bar * total_bars
  let total_revenue := sell_per_bar * total_bars
  let total_discounts := (total_bars / discount_rate.1) * discount_rate.2
  total_revenue - total_discounts - total_cost

theorem candy_bar_profit :
  calculate_profit 1200 (3, 1.5) (4, 3) (100, 2) = 276 := by
  sorry

end NUMINAMATH_CALUDE_candy_bar_profit_l907_90769


namespace NUMINAMATH_CALUDE_sqrt_2x_minus_1_meaningful_l907_90738

theorem sqrt_2x_minus_1_meaningful (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2*x - 1) ↔ x ≥ (1/2) :=
sorry

end NUMINAMATH_CALUDE_sqrt_2x_minus_1_meaningful_l907_90738


namespace NUMINAMATH_CALUDE_order_of_abc_l907_90757

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l907_90757


namespace NUMINAMATH_CALUDE_inequality_proof_l907_90747

theorem inequality_proof (a b c : ℝ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (2*a - b)^2 / (a - b)^2 + (2*b - c)^2 / (b - c)^2 + (2*c - a)^2 / (c - a)^2 ≥ 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l907_90747


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_tangent_line_condition_chord_length_condition_l907_90795

-- Define the line l: mx - y - 3m + 4 = 0
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x - y - 3 * m + 4 = 0

-- Define the circle O: x^2 + y^2 = 4
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the fixed point M(3,4) on line l
def point_M : ℝ × ℝ := (3, 4)

-- Theorem 1: Maximum distance from circle O to line l
theorem max_distance_circle_to_line :
  ∃ (m : ℝ), ∀ (x y : ℝ), circle_O x y →
    ∃ (max_dist : ℝ), max_dist = 7 ∧
      ∀ (x' y' : ℝ), line_l m x' y' →
        Real.sqrt ((x - x')^2 + (y - y')^2) ≤ max_dist :=
sorry

-- Theorem 2: Tangent line condition
theorem tangent_line_condition :
  ∃ (m : ℝ), m = (12 + 2 * Real.sqrt 21) / 5 ∨ m = (12 - 2 * Real.sqrt 21) / 5 →
    ∀ (x y : ℝ), line_l m x y →
      (∃! (x' y' : ℝ), circle_O x' y' ∧ x = x' ∧ y = y') :=
sorry

-- Theorem 3: Chord length condition
theorem chord_length_condition :
  ∃ (m : ℝ), m = (6 + Real.sqrt 6) / 4 ∨ m = (6 - Real.sqrt 6) / 4 →
    ∃ (x1 y1 x2 y2 : ℝ),
      line_l m x1 y1 ∧ line_l m x2 y2 ∧
      circle_O x1 y1 ∧ circle_O x2 y2 ∧
      Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_tangent_line_condition_chord_length_condition_l907_90795


namespace NUMINAMATH_CALUDE_function_derivative_problem_l907_90711

theorem function_derivative_problem (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = (2*x + a)^2)
  (h2 : deriv f 2 = 20) : 
  a = 1 := by sorry

end NUMINAMATH_CALUDE_function_derivative_problem_l907_90711


namespace NUMINAMATH_CALUDE_geometric_sequence_first_term_l907_90799

/-- A geometric sequence with common ratio 2 and fourth term 16 has first term equal to 2 -/
theorem geometric_sequence_first_term (a : ℕ → ℝ) : 
  (∀ n, a (n + 1) = 2 * a n) →  -- geometric sequence with common ratio 2
  a 4 = 16 →                    -- fourth term is 16
  a 1 = 2 :=                    -- prove that first term is 2
by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_first_term_l907_90799


namespace NUMINAMATH_CALUDE_candidate_x_win_percentage_l907_90736

theorem candidate_x_win_percentage :
  ∀ (total_voters : ℕ) (republican_ratio democrat_ratio : ℚ) 
    (republican_for_x democrat_for_x : ℚ),
  republican_ratio / democrat_ratio = 3 / 2 →
  republican_for_x = 70 / 100 →
  democrat_for_x = 25 / 100 →
  let republicans := (republican_ratio / (republican_ratio + democrat_ratio)) * total_voters
  let democrats := (democrat_ratio / (republican_ratio + democrat_ratio)) * total_voters
  let votes_for_x := republican_for_x * republicans + democrat_for_x * democrats
  let votes_for_y := total_voters - votes_for_x
  (votes_for_x - votes_for_y) / total_voters = 4 / 100 :=
by sorry

end NUMINAMATH_CALUDE_candidate_x_win_percentage_l907_90736


namespace NUMINAMATH_CALUDE_triangle_inradius_l907_90767

/-- Given a triangle with perimeter 48 and area 60, prove that its inradius is 2.5 -/
theorem triangle_inradius (P : ℝ) (A : ℝ) (r : ℝ) 
    (h1 : P = 48) 
    (h2 : A = 60) 
    (h3 : A = r * (P / 2)) : r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l907_90767


namespace NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_two_mod_seventeen_l907_90752

theorem smallest_five_digit_congruent_to_two_mod_seventeen : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ n ≡ 2 [ZMOD 17] → 
    n ≥ 10013 := by
  sorry

end NUMINAMATH_CALUDE_smallest_five_digit_congruent_to_two_mod_seventeen_l907_90752


namespace NUMINAMATH_CALUDE_ethanol_in_fuel_mix_l907_90770

/-- Calculates the total ethanol in a fuel tank -/
def total_ethanol (tank_capacity : ℝ) (fuel_a_volume : ℝ) (fuel_a_ethanol_percent : ℝ) (fuel_b_ethanol_percent : ℝ) : ℝ :=
  let fuel_b_volume := tank_capacity - fuel_a_volume
  let ethanol_a := fuel_a_volume * fuel_a_ethanol_percent
  let ethanol_b := fuel_b_volume * fuel_b_ethanol_percent
  ethanol_a + ethanol_b

/-- Theorem: The total ethanol in the specified fuel mix is 30 gallons -/
theorem ethanol_in_fuel_mix :
  total_ethanol 200 49.99999999999999 0.12 0.16 = 30 := by
  sorry

end NUMINAMATH_CALUDE_ethanol_in_fuel_mix_l907_90770


namespace NUMINAMATH_CALUDE_expected_groups_l907_90765

/-- The expected number of alternating groups in a random sequence of zeros and ones -/
theorem expected_groups (k m : ℕ) : 
  let total := k + m
  let prob_diff := (2 * k * m) / (total * (total - 1))
  1 + (total - 1) * prob_diff = 1 + (2 * k * m) / total := by
  sorry

end NUMINAMATH_CALUDE_expected_groups_l907_90765


namespace NUMINAMATH_CALUDE_science_club_neither_math_nor_physics_l907_90717

theorem science_club_neither_math_nor_physics 
  (total : ℕ) (math : ℕ) (physics : ℕ) (both : ℕ) 
  (h1 : total = 120)
  (h2 : math = 75)
  (h3 : physics = 50)
  (h4 : both = 15) :
  total - (math + physics - both) = 10 :=
by sorry

end NUMINAMATH_CALUDE_science_club_neither_math_nor_physics_l907_90717


namespace NUMINAMATH_CALUDE_no_solution_iff_k_equals_nine_l907_90709

theorem no_solution_iff_k_equals_nine :
  ∀ k : ℝ, (∀ x : ℝ, x ≠ 1 ∧ x ≠ 7 → (x - 3) / (x - 1) ≠ (x - k) / (x - 7)) ↔ k = 9 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_iff_k_equals_nine_l907_90709


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l907_90718

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l907_90718


namespace NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l907_90796

theorem contrapositive_square_sum_zero (a b : ℝ) :
  (¬(a = 0 ∧ b = 0) → a^2 + b^2 ≠ 0) ↔
  (a^2 + b^2 = 0 → a = 0 ∧ b = 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_square_sum_zero_l907_90796


namespace NUMINAMATH_CALUDE_quadratic_properties_l907_90782

/-- A quadratic function passing through specific points -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  f 0 = -7/2 ∧ f 1 = 1/2 ∧ f (3/2) = 1 ∧ f 2 = 1/2

theorem quadratic_properties (f : ℝ → ℝ) (hf : QuadraticFunction f) :
  (∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) ∧  -- f is quadratic
  f 0 = -7/2 ∧  -- y-axis intersection
  (∀ x, f (3/2 - x) = f (3/2 + x)) ∧  -- axis of symmetry
  (∀ x, f x ≤ f (3/2)) ∧  -- vertex
  (∀ x, f x = -2 * (x - 3/2)^2 + 1)  -- analytical expression
  := by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l907_90782


namespace NUMINAMATH_CALUDE_jordan_rectangle_width_l907_90724

/-- Given two rectangles with equal area, where one rectangle measures 8 inches by 15 inches
    and the other has a length of 4 inches, prove that the width of the second rectangle is 30 inches. -/
theorem jordan_rectangle_width (area carol_length carol_width jordan_length jordan_width : ℝ) :
  area = carol_length * carol_width →
  area = jordan_length * jordan_width →
  carol_length = 8 →
  carol_width = 15 →
  jordan_length = 4 →
  jordan_width = 30 := by
  sorry

end NUMINAMATH_CALUDE_jordan_rectangle_width_l907_90724


namespace NUMINAMATH_CALUDE_square_area_ratio_l907_90778

theorem square_area_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) (h_perimeter : 4 * a = 4 * (4 * b)) :
  a^2 = 16 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l907_90778


namespace NUMINAMATH_CALUDE_square_sum_from_means_l907_90713

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 100) : 
  x^2 + y^2 = 1400 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l907_90713


namespace NUMINAMATH_CALUDE_two_solutions_l907_90707

-- Define the matrix evaluation rule
def matrixEval (a b c d : ℝ) : ℝ := a * b - c * d + c

-- Define the equation
def equation (x : ℝ) : Prop := matrixEval (3 * x) x 2 (2 * x) = 2

-- Theorem statement
theorem two_solutions :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ ∧ equation x₂) ∧
  (∀ x : ℝ, equation x → x = 0 ∨ x = 4/3) :=
sorry

end NUMINAMATH_CALUDE_two_solutions_l907_90707


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l907_90743

-- Problem 1
theorem simplify_expression_1 (x : ℝ) :
  (2*x - 1) * (2*x - 3) - (1 - 2*x) * (2 - x) = 2*x^2 - 3*x + 1 := by sorry

-- Problem 2
theorem simplify_expression_2 (a : ℝ) (ha : a ≠ 0) (ha' : a ≠ 1) :
  (a^2 - 1) / a * (1 - (2*a + 1) / (a^2 + 2*a + 1)) / (a - 1) = a / (a + 1) := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_l907_90743


namespace NUMINAMATH_CALUDE_y_axis_reflection_l907_90737

/-- Given a point P(-2,3) in the Cartesian coordinate system, 
    its coordinates with respect to the y-axis are (2,3). -/
theorem y_axis_reflection :
  let P : ℝ × ℝ := (-2, 3)
  let reflected_P : ℝ × ℝ := (2, 3)
  reflected_P = (-(P.1), P.2) :=
by sorry

end NUMINAMATH_CALUDE_y_axis_reflection_l907_90737


namespace NUMINAMATH_CALUDE_complex_fourth_quadrant_range_l907_90797

theorem complex_fourth_quadrant_range (a : ℝ) : 
  let z₁ : ℂ := 3 - a * Complex.I
  let z₂ : ℂ := 1 + 2 * Complex.I
  (0 < (z₁ / z₂).re ∧ (z₁ / z₂).im < 0) → (-6 < a ∧ a < 3/2) :=
by sorry

end NUMINAMATH_CALUDE_complex_fourth_quadrant_range_l907_90797


namespace NUMINAMATH_CALUDE_max_ratio_system_l907_90727

theorem max_ratio_system (x y z u : ℕ+) (h1 : x + y = z + u) (h2 : 2 * x * y = z * u) (h3 : x ≥ y) :
  (x : ℝ) / y ≤ 3 + 2 * Real.sqrt 2 ∧ ∀ ε > 0, ∃ x' y' z' u' : ℕ+,
    x' + y' = z' + u' ∧
    2 * x' * y' = z' * u' ∧
    x' ≥ y' ∧
    (x' : ℝ) / y' > 3 + 2 * Real.sqrt 2 - ε :=
sorry

end NUMINAMATH_CALUDE_max_ratio_system_l907_90727


namespace NUMINAMATH_CALUDE_school_water_cases_l907_90791

theorem school_water_cases : 
  ∀ (bottles_per_case : ℕ) 
    (bottles_used_first_game : ℕ) 
    (bottles_used_second_game : ℕ) 
    (bottles_left : ℕ),
  bottles_per_case = 20 →
  bottles_used_first_game = 70 →
  bottles_used_second_game = 110 →
  bottles_left = 20 →
  (bottles_used_first_game + bottles_used_second_game + bottles_left) / bottles_per_case = 10 := by
sorry

end NUMINAMATH_CALUDE_school_water_cases_l907_90791


namespace NUMINAMATH_CALUDE_max_exterior_elements_sum_l907_90781

/-- A shape formed by adding a pyramid to a rectangular prism -/
structure PrismWithPyramid where
  prism_faces : ℕ
  prism_edges : ℕ
  prism_vertices : ℕ
  pyramid_base_edges : ℕ

/-- Calculate the total number of exterior elements after fusion -/
def total_exterior_elements (shape : PrismWithPyramid) : ℕ :=
  let new_faces := shape.prism_faces - 1 + shape.pyramid_base_edges
  let new_edges := shape.prism_edges + shape.pyramid_base_edges
  let new_vertices := shape.prism_vertices + 1
  new_faces + new_edges + new_vertices

/-- Theorem stating the maximum sum of exterior elements -/
theorem max_exterior_elements_sum :
  ∀ shape : PrismWithPyramid,
  shape.prism_faces = 6 →
  shape.prism_edges = 12 →
  shape.prism_vertices = 8 →
  shape.pyramid_base_edges = 4 →
  total_exterior_elements shape = 34 := by
  sorry


end NUMINAMATH_CALUDE_max_exterior_elements_sum_l907_90781


namespace NUMINAMATH_CALUDE_teacher_engineer_ratio_l907_90730

theorem teacher_engineer_ratio 
  (t e : ℕ) -- t is the number of teachers, e is the number of engineers
  (h_group : t + e > 0) -- ensures the group is not empty
  (h_avg : (40 * t + 55 * e) / (t + e) = 45) -- average age of the entire group is 45
  : t = 2 * e := by
sorry

end NUMINAMATH_CALUDE_teacher_engineer_ratio_l907_90730


namespace NUMINAMATH_CALUDE_sin_15_sin_105_equals_1_l907_90722

theorem sin_15_sin_105_equals_1 : 4 * Real.sin (15 * π / 180) * Real.sin (105 * π / 180) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sin_15_sin_105_equals_1_l907_90722


namespace NUMINAMATH_CALUDE_f_min_value_l907_90786

noncomputable def f (x : ℝ) := Real.exp x + 3 * x^2 - x + 2011

theorem f_min_value :
  ∃ (min : ℝ), min = 2012 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_f_min_value_l907_90786


namespace NUMINAMATH_CALUDE_adult_tickets_sold_l907_90750

/-- Represents the number of adult tickets sold -/
def adult_tickets : ℕ := sorry

/-- Represents the number of child tickets sold -/
def child_tickets : ℕ := sorry

/-- The cost of an adult ticket in dollars -/
def adult_cost : ℕ := 12

/-- The cost of a child ticket in dollars -/
def child_cost : ℕ := 4

/-- The total number of tickets sold -/
def total_tickets : ℕ := 130

/-- The total receipts in dollars -/
def total_receipts : ℕ := 840

theorem adult_tickets_sold : 
  adult_tickets = 40 ∧
  adult_tickets + child_tickets = total_tickets ∧
  adult_tickets * adult_cost + child_tickets * child_cost = total_receipts :=
by sorry

end NUMINAMATH_CALUDE_adult_tickets_sold_l907_90750


namespace NUMINAMATH_CALUDE_joy_reading_time_l907_90766

/-- Given that Joy can read 8 pages in 20 minutes, prove that it takes her 5 hours to read 120 pages. -/
theorem joy_reading_time : 
  -- Define Joy's reading speed
  let pages_per_20_min : ℚ := 8
  let total_pages : ℚ := 120
  -- Calculate the time in hours
  let time_in_hours : ℚ := (total_pages / pages_per_20_min) * (20 / 60)
  -- Prove that the time is 5 hours
  ∀ (pages_per_20_min total_pages time_in_hours : ℚ), 
    pages_per_20_min = 8 → 
    total_pages = 120 → 
    time_in_hours = (total_pages / pages_per_20_min) * (20 / 60) → 
    time_in_hours = 5 := by
  sorry


end NUMINAMATH_CALUDE_joy_reading_time_l907_90766


namespace NUMINAMATH_CALUDE_bacteria_growth_l907_90704

def b (t : ℝ) : ℝ := 105 + 104 * t - 1000 * t^2

theorem bacteria_growth (t : ℝ) :
  (deriv b 5 = 0) ∧
  (deriv b 10 = -10000) ∧
  (∀ t ∈ Set.Ioo 0 5, deriv b t > 0) ∧
  (∀ t ∈ Set.Ioi 5, deriv b t < 0) := by
  sorry

end NUMINAMATH_CALUDE_bacteria_growth_l907_90704


namespace NUMINAMATH_CALUDE_equation_solution_l907_90761

theorem equation_solution : ∃ x : ℚ, (3/4 : ℚ) + 1/x = 7/8 ∧ x = 8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l907_90761


namespace NUMINAMATH_CALUDE_handshake_problem_l907_90740

theorem handshake_problem (n : ℕ) : n * (n - 1) / 2 = 78 → n = 13 := by
  sorry

end NUMINAMATH_CALUDE_handshake_problem_l907_90740


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l907_90741

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 5 * a^2 + 7 * a + 2 = 1) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ (x : ℝ), 5 * x^2 + 7 * x + 2 = 1 → 3 * x + 2 ≥ m) ∧ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l907_90741


namespace NUMINAMATH_CALUDE_common_chord_intersection_l907_90703

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point where two circles intersect -/
def IntersectionPoint (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p | (p.1 - c1.center.1)^2 + (p.2 - c1.center.2)^2 = c1.radius^2 ∧
       (p.1 - c2.center.1)^2 + (p.2 - c2.center.2)^2 = c2.radius^2}

/-- The common chord of two intersecting circles -/
def CommonChord (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  IntersectionPoint c1 c2

/-- Theorem: For any three circles in a plane that intersect pairwise, 
    the common chords of these pairs of circles intersect at a single point -/
theorem common_chord_intersection (c1 c2 c3 : Circle) 
  (h12 : (CommonChord c1 c2).Nonempty)
  (h23 : (CommonChord c2 c3).Nonempty)
  (h31 : (CommonChord c3 c1).Nonempty) :
  ∃ p, p ∈ CommonChord c1 c2 ∧ p ∈ CommonChord c2 c3 ∧ p ∈ CommonChord c3 c1 :=
sorry

end NUMINAMATH_CALUDE_common_chord_intersection_l907_90703


namespace NUMINAMATH_CALUDE_double_dimensions_volume_l907_90787

/-- A cylindrical container with volume, height, and radius. -/
structure CylindricalContainer where
  volume : ℝ
  height : ℝ
  radius : ℝ
  volume_formula : volume = Real.pi * radius^2 * height

/-- Given a cylindrical container of 5 gallons, doubling its dimensions results in a 40-gallon container -/
theorem double_dimensions_volume (c : CylindricalContainer) 
  (h_volume : c.volume = 5) :
  let new_container : CylindricalContainer := {
    volume := Real.pi * (2 * c.radius)^2 * (2 * c.height),
    height := 2 * c.height,
    radius := 2 * c.radius,
    volume_formula := by sorry
  }
  new_container.volume = 40 := by
  sorry

end NUMINAMATH_CALUDE_double_dimensions_volume_l907_90787


namespace NUMINAMATH_CALUDE_remaining_money_l907_90742

def initial_amount : ℚ := 3
def purchase_amount : ℚ := 1

theorem remaining_money :
  initial_amount - purchase_amount = 2 := by sorry

end NUMINAMATH_CALUDE_remaining_money_l907_90742


namespace NUMINAMATH_CALUDE_max_t_for_tangent_slope_l907_90792

/-- Given t > 0 and f(x) = x²(x - t), prove that the maximum value of t for which
    the slope of the tangent line to f(x) is always greater than or equal to -1
    when x is in (0, 1] is 3/2. -/
theorem max_t_for_tangent_slope (t : ℝ) (h_t : t > 0) :
  (∀ x : ℝ, x ∈ (Set.Ioo 0 1) → (3 * x^2 - 2 * t * x) ≥ -1) ↔ t ≤ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_max_t_for_tangent_slope_l907_90792


namespace NUMINAMATH_CALUDE_parallel_lines_in_parallel_planes_not_always_parallel_l907_90746

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem parallel_lines_in_parallel_planes_not_always_parallel 
  (m n : Line) (α β : Plane) : 
  ¬(∀ m n α β, subset m α ∧ subset n β ∧ parallel_planes α β → parallel_lines m n) :=
sorry

end NUMINAMATH_CALUDE_parallel_lines_in_parallel_planes_not_always_parallel_l907_90746


namespace NUMINAMATH_CALUDE_integer_expression_l907_90706

theorem integer_expression (n : ℕ) : ∃ (k : ℤ), 
  (3^(2*n) : ℚ) / 112 - (4^(2*n) : ℚ) / 63 + (5^(2*n) : ℚ) / 144 = k := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_l907_90706


namespace NUMINAMATH_CALUDE_valid_numbers_l907_90793

def is_valid_number (n : ℕ) : Prop :=
  ∃ (A B C : ℕ),
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A ≥ 1 ∧ A ≤ 9 ∧ B ≥ 0 ∧ B ≤ 9 ∧ C ≥ 0 ∧ C ≤ 9 ∧
    n = 100001 * A + 10010 * B + 1100 * C ∧
    n % 7 = 0 ∧
    (100 * A + 10 * B + C) % 7 = 0

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {168861, 259952, 861168, 952259} :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_l907_90793


namespace NUMINAMATH_CALUDE_intersection_line_of_circles_l907_90732

/-- Given two circles in the xy-plane:
    Circle1: x^2 + y^2 - x + y - 2 = 0
    Circle2: x^2 + y^2 = 5
    This theorem states that the line x - y - 3 = 0 passes through their intersection points. -/
theorem intersection_line_of_circles (x y : ℝ) :
  (x^2 + y^2 - x + y - 2 = 0 ∧ x^2 + y^2 = 5) → (x - y - 3 = 0) := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_of_circles_l907_90732


namespace NUMINAMATH_CALUDE_remainder_three_power_45_plus_4_mod_5_l907_90773

theorem remainder_three_power_45_plus_4_mod_5 : (3^45 + 4) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_three_power_45_plus_4_mod_5_l907_90773


namespace NUMINAMATH_CALUDE_sum_of_edges_is_120_l907_90723

/-- A rectangular solid with specific properties -/
structure RectangularSolid where
  -- The three dimensions of the solid
  a : ℝ
  b : ℝ
  c : ℝ
  -- Volume is 1000 cm³
  volume_eq : a * b * c = 1000
  -- Surface area is 600 cm²
  surface_area_eq : 2 * (a * b + b * c + a * c) = 600
  -- Dimensions are in geometric progression
  geometric_progression : ∃ (r : ℝ), b = a * r ∧ c = b * r

/-- The sum of all edge lengths of a rectangular solid -/
def sum_of_edges (solid : RectangularSolid) : ℝ :=
  4 * (solid.a + solid.b + solid.c)

/-- Theorem stating that the sum of all edge lengths is 120 cm -/
theorem sum_of_edges_is_120 (solid : RectangularSolid) :
  sum_of_edges solid = 120 := by
  sorry

#check sum_of_edges_is_120

end NUMINAMATH_CALUDE_sum_of_edges_is_120_l907_90723


namespace NUMINAMATH_CALUDE_original_average_proof_l907_90760

theorem original_average_proof (n : ℕ) (original_avg new_avg : ℚ) : 
  n = 12 → 
  new_avg = 72 → 
  new_avg = 2 * original_avg → 
  original_avg = 36 := by
sorry

end NUMINAMATH_CALUDE_original_average_proof_l907_90760


namespace NUMINAMATH_CALUDE_sequence_relation_l907_90784

def x : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 4 * x (n + 1) - x n

def y : ℕ → ℤ
  | 0 => 1
  | 1 => 2
  | (n + 2) => 4 * y (n + 1) - y n

theorem sequence_relation (n : ℕ) : (y n)^2 = 3 * (x n)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_relation_l907_90784


namespace NUMINAMATH_CALUDE_opposite_number_pairs_l907_90776

theorem opposite_number_pairs : 
  (-(-(3 : ℤ)) = -(-|(-(3 : ℤ))|)) ∧ 
  ((-(2 : ℤ))^4 = -(2^4)) ∧ 
  ¬((-(2 : ℤ))^3 = -((-(3 : ℤ))^2)) ∧ 
  ¬((-(2 : ℤ))^3 = -(2^3)) := by
  sorry

end NUMINAMATH_CALUDE_opposite_number_pairs_l907_90776


namespace NUMINAMATH_CALUDE_min_value_of_expression_l907_90753

theorem min_value_of_expression (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : 3 * m + n = 1) :
  (1 / m + 3 / n) ≥ 12 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ 3 * m + n = 1 ∧ 1 / m + 3 / n = 12 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l907_90753


namespace NUMINAMATH_CALUDE_safe_locks_and_keys_l907_90721

/-- Represents the number of committee members -/
def n : ℕ := 11

/-- Represents the size of the smallest group that can open the safe -/
def k : ℕ := 6

/-- Calculates the number of locks needed -/
def num_locks : ℕ := Nat.choose n (k - 1)

/-- Calculates the total number of keys needed -/
def num_keys : ℕ := num_locks * k

/-- Theorem stating the minimum number of locks and keys needed -/
theorem safe_locks_and_keys : num_locks = 462 ∧ num_keys = 2772 := by
  sorry

#eval num_locks -- Should output 462
#eval num_keys  -- Should output 2772

end NUMINAMATH_CALUDE_safe_locks_and_keys_l907_90721


namespace NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l907_90712

theorem sin_cos_difference_equals_half : 
  Real.sin (65 * π / 180) * Real.cos (35 * π / 180) - 
  Real.sin (25 * π / 180) * Real.sin (35 * π / 180) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_difference_equals_half_l907_90712


namespace NUMINAMATH_CALUDE_dartboard_angle_l907_90714

theorem dartboard_angle (p : ℝ) (θ : ℝ) : 
  p = 1 / 8 → θ = p * 360 → θ = 45 :=
by sorry

end NUMINAMATH_CALUDE_dartboard_angle_l907_90714


namespace NUMINAMATH_CALUDE_jameson_medals_l907_90771

theorem jameson_medals (total_medals : ℕ) (badminton_medals : ℕ) 
  (h1 : total_medals = 20)
  (h2 : badminton_medals = 5) :
  ∃ track_medals : ℕ, 
    track_medals + 2 * track_medals + badminton_medals = total_medals ∧ 
    track_medals = 5 := by
  sorry

end NUMINAMATH_CALUDE_jameson_medals_l907_90771


namespace NUMINAMATH_CALUDE_jills_herd_sale_fraction_l907_90790

/-- Represents the number of llamas in Jill's herd -/
structure LlamaHerd where
  initial : ℕ
  single_births : ℕ
  twin_births : ℕ
  traded_calves : ℕ
  traded_adults : ℕ
  final : ℕ

/-- Calculates the fraction of the herd sold at the market -/
def fraction_sold (herd : LlamaHerd) : ℚ :=
  let total_calves := herd.single_births + 2 * herd.twin_births
  let before_trade := herd.initial + total_calves
  let after_trade := before_trade - herd.traded_calves + herd.traded_adults
  let sold := after_trade - herd.final
  sold / before_trade

/-- Theorem stating the fraction of the herd Jill sold at the market -/
theorem jills_herd_sale_fraction : 
  ∀ (herd : LlamaHerd), 
  herd.single_births = 9 → 
  herd.twin_births = 5 → 
  herd.traded_calves = 8 → 
  herd.traded_adults = 2 → 
  herd.final = 18 → 
  fraction_sold herd = 4 / 13 := by
  sorry


end NUMINAMATH_CALUDE_jills_herd_sale_fraction_l907_90790


namespace NUMINAMATH_CALUDE_wax_requirement_l907_90708

theorem wax_requirement (current_wax : ℕ) (additional_wax : ℕ) : 
  current_wax = 11 → additional_wax = 481 → current_wax + additional_wax = 492 := by
  sorry

end NUMINAMATH_CALUDE_wax_requirement_l907_90708


namespace NUMINAMATH_CALUDE_jiyoon_sum_l907_90739

theorem jiyoon_sum : 36 + 17 + 32 + 54 + 28 + 3 = 170 := by
  sorry

end NUMINAMATH_CALUDE_jiyoon_sum_l907_90739


namespace NUMINAMATH_CALUDE_smallest_seating_arrangement_l907_90775

/-- Represents a circular seating arrangement -/
structure CircularSeating :=
  (total_chairs : ℕ)
  (seated_people : ℕ)

/-- Checks if the seating arrangement satisfies the condition -/
def satisfies_condition (seating : CircularSeating) : Prop :=
  seating.seated_people > 0 ∧
  seating.seated_people ≤ seating.total_chairs ∧
  ∀ (new_seat : ℕ), new_seat < seating.total_chairs →
    ∃ (occupied_seat : ℕ), occupied_seat < seating.total_chairs ∧
      (new_seat = (occupied_seat + 1) % seating.total_chairs ∨
       new_seat = (occupied_seat + seating.total_chairs - 1) % seating.total_chairs)

/-- The main theorem to be proved -/
theorem smallest_seating_arrangement :
  ∃ (n : ℕ), n = 18 ∧
    satisfies_condition ⟨72, n⟩ ∧
    ∀ (m : ℕ), m < n → ¬satisfies_condition ⟨72, m⟩ :=
sorry

end NUMINAMATH_CALUDE_smallest_seating_arrangement_l907_90775


namespace NUMINAMATH_CALUDE_trigonometric_system_solution_l907_90756

theorem trigonometric_system_solution :
  let eq1 (x y : Real) := 
    (Real.sin x + Real.cos x) / (Real.sin y + Real.cos y) + 
    (Real.sin y - Real.cos y) / (Real.sin x + Real.cos x) = 
    1 / (Real.sin (x + y) + Real.cos (x - y))
  let eq2 (x y : Real) := 
    2 * (Real.sin x + Real.cos x)^2 - (2 * Real.cos y^2 + 1) = Real.sqrt 3 / 2
  let solutions : List (Real × Real) := 
    [(π/6, π/12), (π/6, 13*π/12), (π/3, 11*π/12), (π/3, 23*π/12)]
  ∀ (x y : Real), (x, y) ∈ solutions → eq1 x y ∧ eq2 x y :=
by
  sorry


end NUMINAMATH_CALUDE_trigonometric_system_solution_l907_90756


namespace NUMINAMATH_CALUDE_toddler_difference_l907_90759

/-- Represents the group of toddlers playing in the sandbox. -/
structure ToddlerGroup where
  total : ℕ
  forgot_bucket : ℕ
  forgot_shovel : ℕ
  bucket_implies_shovel : Bool

/-- The difference between toddlers with shovel but no bucket and toddlers with bucket -/
def shovel_no_bucket_minus_bucket (group : ToddlerGroup) : ℕ :=
  (group.total - group.forgot_shovel) - (group.total - group.forgot_bucket) - (group.total - group.forgot_bucket)

/-- The main theorem stating the difference is 4 -/
theorem toddler_difference (group : ToddlerGroup) 
  (h1 : group.total = 12)
  (h2 : group.forgot_bucket = 9)
  (h3 : group.forgot_shovel = 2)
  (h4 : group.bucket_implies_shovel = true) :
  shovel_no_bucket_minus_bucket group = 4 := by
  sorry

end NUMINAMATH_CALUDE_toddler_difference_l907_90759


namespace NUMINAMATH_CALUDE_cookie_bags_theorem_l907_90762

/-- Given a total number of cookies and cookies per bag, calculate the number of bags. -/
def number_of_bags (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

/-- Theorem: Given 33 cookies in total and 11 cookies per bag, the number of bags is 3. -/
theorem cookie_bags_theorem :
  number_of_bags 33 11 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cookie_bags_theorem_l907_90762


namespace NUMINAMATH_CALUDE_students_per_normal_class_l907_90745

theorem students_per_normal_class
  (total_students : ℕ)
  (moving_percentage : ℚ)
  (grade_levels : ℕ)
  (advanced_class_size : ℕ)
  (normal_classes_per_grade : ℕ)
  (h1 : total_students = 1590)
  (h2 : moving_percentage = 40 / 100)
  (h3 : grade_levels = 3)
  (h4 : advanced_class_size = 20)
  (h5 : normal_classes_per_grade = 6)
  : ℕ :=
by
  -- Proof goes here
  sorry

#check @students_per_normal_class

end NUMINAMATH_CALUDE_students_per_normal_class_l907_90745


namespace NUMINAMATH_CALUDE_infinitely_many_composite_numbers_l907_90701

theorem infinitely_many_composite_numbers :
  ∃ (N : Set ℕ), Set.Infinite N ∧
    ∀ n ∈ N, ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 50^n + (50*n + 1)^50 = a * b :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_composite_numbers_l907_90701


namespace NUMINAMATH_CALUDE_rectangular_box_diagonal_sum_l907_90710

theorem rectangular_box_diagonal_sum (a b c : ℝ) 
  (h_surface_area : 2 * (a * b + b * c + c * a) = 112)
  (h_edge_sum : 4 * (a + b + c) = 60) :
  4 * Real.sqrt (a^2 + b^2 + c^2) = 4 * Real.sqrt 113 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_box_diagonal_sum_l907_90710


namespace NUMINAMATH_CALUDE_function_value_ordering_l907_90735

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the properties of f
axiom even : ∀ x, f (-x) = f x
axiom periodic : ∀ x, f (x + 1) = f (x - 1)
axiom monotonic : ∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ 1 → f x < f y

-- State the theorem
theorem function_value_ordering : f (-3/2) < f (4/3) ∧ f (4/3) < f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_ordering_l907_90735


namespace NUMINAMATH_CALUDE_nadine_garage_sale_spend_l907_90783

/-- The amount Nadine spent at the garage sale -/
def garage_sale_total (table_price chair_price num_chairs : ℕ) : ℕ :=
  table_price + chair_price * num_chairs

/-- Theorem: Nadine spent $56 at the garage sale -/
theorem nadine_garage_sale_spend :
  garage_sale_total 34 11 2 = 56 := by
  sorry

end NUMINAMATH_CALUDE_nadine_garage_sale_spend_l907_90783


namespace NUMINAMATH_CALUDE_apple_pear_equivalence_l907_90744

/-- Given that 3/4 of 12 apples are worth as much as 6 pears,
    prove that 1/3 of 9 apples are worth as much as 2 pears. -/
theorem apple_pear_equivalence (apple pear : ℝ) 
    (h : (3/4 : ℝ) * 12 * apple = 6 * pear) : 
    (1/3 : ℝ) * 9 * apple = 2 * pear := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_equivalence_l907_90744


namespace NUMINAMATH_CALUDE_owls_on_fence_l907_90705

theorem owls_on_fence (initial_owls final_owls joined_owls : ℕ) : 
  final_owls = initial_owls + joined_owls →
  joined_owls = 2 →
  final_owls = 5 →
  initial_owls = 3 := by
  sorry

end NUMINAMATH_CALUDE_owls_on_fence_l907_90705


namespace NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_l907_90733

/-- The least number of digits in the repeating block of the decimal expansion of 7/13 -/
def repeating_block_length : ℕ := 6

/-- Theorem stating that the least number of digits in the repeating block 
    of the decimal expansion of 7/13 is equal to repeating_block_length -/
theorem seven_thirteenths_repeating_block : 
  (Nat.lcm 13 10 : ℕ).factorization 2 + (Nat.lcm 13 10 : ℕ).factorization 5 = repeating_block_length :=
sorry

end NUMINAMATH_CALUDE_seven_thirteenths_repeating_block_l907_90733


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l907_90716

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log x - 4 * (x - 1)

noncomputable def f' (x : ℝ) : ℝ := Real.log x + (x + 1) / x - 4

theorem tangent_line_at_one (x y : ℝ) :
  (f' 1 = -2) →
  (f 1 = 0) →
  (∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, |h| < δ → |f (1 + h) - (f 1 + f' 1 * h)| ≤ ε * |h|) →
  (2 * x + y - 2 = 0 ↔ y = f' 1 * (x - 1) + f 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l907_90716


namespace NUMINAMATH_CALUDE_product_first_three_eq_960_l907_90728

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  -- The seventh term is 20
  seventh_term : ℕ
  seventh_term_eq : seventh_term = 20
  -- The common difference is 2
  common_diff : ℕ
  common_diff_eq : common_diff = 2

/-- The product of the first three terms of the arithmetic sequence -/
def product_first_three (seq : ArithmeticSequence) : ℕ :=
  let a := seq.seventh_term - 6 * seq.common_diff -- First term
  let a2 := a + seq.common_diff -- Second term
  let a3 := a + 2 * seq.common_diff -- Third term
  a * a2 * a3

/-- Theorem stating that the product of the first three terms is 960 -/
theorem product_first_three_eq_960 (seq : ArithmeticSequence) :
  product_first_three seq = 960 := by
  sorry

end NUMINAMATH_CALUDE_product_first_three_eq_960_l907_90728


namespace NUMINAMATH_CALUDE_triangle_inequality_l907_90758

theorem triangle_inequality (a b c : ℝ) : 
  (a + b + c = 2) → 
  (a > 0) → (b > 0) → (c > 0) →
  (a + b ≥ c) → (b + c ≥ a) → (c + a ≥ b) →
  abc + 1/27 ≥ ab + bc + ca - 1 ∧ ab + bc + ca - 1 ≥ abc := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l907_90758


namespace NUMINAMATH_CALUDE_work_done_by_resultant_force_l907_90700

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Calculates the dot product of two 2D vectors -/
def dotProduct (v1 v2 : Vector2D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y

/-- Adds two 2D vectors -/
def addVectors (v1 v2 : Vector2D) : Vector2D :=
  ⟨v1.x + v2.x, v1.y + v2.y⟩

/-- Calculates the work done by a force over a displacement -/
def workDone (force displacement : Vector2D) : ℝ :=
  dotProduct force displacement

theorem work_done_by_resultant_force : 
  let f1 : Vector2D := ⟨3, -4⟩
  let f2 : Vector2D := ⟨2, -5⟩
  let f3 : Vector2D := ⟨3, 1⟩
  let a : Vector2D := ⟨1, 1⟩
  let b : Vector2D := ⟨0, 5⟩
  let resultantForce := addVectors (addVectors f1 f2) f3
  let displacement := ⟨b.x - a.x, b.y - a.y⟩
  workDone resultantForce displacement = -40 := by
  sorry


end NUMINAMATH_CALUDE_work_done_by_resultant_force_l907_90700


namespace NUMINAMATH_CALUDE_unique_solution_l907_90777

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem unique_solution :
  ∃! A : ℕ, ∃ B : ℕ,
    4 * A + (10 * B + 3) = 68 ∧
    is_two_digit (4 * A) ∧
    is_two_digit (10 * B + 3) ∧
    A ≤ 9 ∧ B ≤ 9 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_l907_90777


namespace NUMINAMATH_CALUDE_contractor_daily_wage_l907_90779

/-- Represents the contractor's payment scenario -/
structure ContractorPayment where
  totalDays : ℕ
  finePerAbsence : ℚ
  totalPayment : ℚ
  absentDays : ℕ

/-- Calculates the daily wage of the contractor -/
def dailyWage (c : ContractorPayment) : ℚ :=
  (c.totalPayment + c.finePerAbsence * c.absentDays) / (c.totalDays - c.absentDays)

/-- Theorem stating the contractor's daily wage is 25 -/
theorem contractor_daily_wage :
  let c := ContractorPayment.mk 30 (7.5) 425 10
  dailyWage c = 25 := by sorry

end NUMINAMATH_CALUDE_contractor_daily_wage_l907_90779
