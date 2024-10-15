import Mathlib

namespace NUMINAMATH_CALUDE_day_statistics_order_l2451_245145

/-- Represents the frequency distribution of days in a non-leap year -/
def day_frequency (n : ℕ) : ℕ :=
  if n ≤ 28 then 12
  else if n ≤ 30 then 11
  else if n = 31 then 6
  else 0

/-- The total number of days in a non-leap year -/
def total_days : ℕ := 365

/-- The median of modes for the day distribution -/
def median_of_modes : ℚ := 14.5

/-- The median of the day distribution -/
def median : ℕ := 13

/-- The mean of the day distribution -/
def mean : ℚ := 5707 / 365

theorem day_statistics_order :
  median_of_modes < median ∧ (median : ℚ) < mean :=
sorry

end NUMINAMATH_CALUDE_day_statistics_order_l2451_245145


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l2451_245123

theorem smallest_number_divisible (n : ℕ) : n = 6297 ↔ 
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 18 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 70 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 100 * k)) ∧
  (∀ m : ℕ, m < n → ¬(∃ k : ℕ, (m + 3) = 84 * k)) ∧
  (∃ k₁ k₂ k₃ k₄ : ℕ, (n + 3) = 18 * k₁ ∧ (n + 3) = 70 * k₂ ∧ (n + 3) = 100 * k₃ ∧ (n + 3) = 84 * k₄) :=
by sorry

#check smallest_number_divisible

end NUMINAMATH_CALUDE_smallest_number_divisible_l2451_245123


namespace NUMINAMATH_CALUDE_coordinates_wrt_origin_l2451_245107

def Point := ℝ × ℝ

theorem coordinates_wrt_origin (p : Point) : p = p := by sorry

end NUMINAMATH_CALUDE_coordinates_wrt_origin_l2451_245107


namespace NUMINAMATH_CALUDE_donut_selection_count_donut_selection_theorem_l2451_245101

theorem donut_selection_count : Nat → ℕ
  | n => 
    let total_donuts := 6
    let donut_types := 4
    let remaining_donuts := total_donuts - donut_types
    Nat.choose (remaining_donuts + donut_types - 1) (donut_types - 1)

theorem donut_selection_theorem : 
  donut_selection_count 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_donut_selection_count_donut_selection_theorem_l2451_245101


namespace NUMINAMATH_CALUDE_triangle_groups_count_l2451_245109

/-- The number of groups of 3 points from 12 points that can form a triangle -/
def triangle_groups : ℕ := 200

/-- The total number of points -/
def total_points : ℕ := 12

/-- Theorem stating that the number of groups of 3 points from 12 points 
    that can form a triangle is equal to 200 -/
theorem triangle_groups_count : 
  triangle_groups = 200 ∧ total_points = 12 := by sorry

end NUMINAMATH_CALUDE_triangle_groups_count_l2451_245109


namespace NUMINAMATH_CALUDE_abs_x_minus_5_lt_3_iff_2_lt_x_lt_8_l2451_245149

theorem abs_x_minus_5_lt_3_iff_2_lt_x_lt_8 :
  ∀ x : ℝ, |x - 5| < 3 ↔ 2 < x ∧ x < 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_5_lt_3_iff_2_lt_x_lt_8_l2451_245149


namespace NUMINAMATH_CALUDE_evaluate_expression_l2451_245166

theorem evaluate_expression : 6 - 5 * (10 - (2 + 3)^2) * 2 = 306 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2451_245166


namespace NUMINAMATH_CALUDE_rope_cutting_probability_l2451_245129

theorem rope_cutting_probability : 
  let rope_length : ℝ := 6
  let num_nodes : ℕ := 5
  let num_parts : ℕ := 6
  let min_segment_length : ℝ := 2

  let part_length : ℝ := rope_length / num_parts
  let favorable_cuts : ℕ := (num_nodes - 2)
  
  (favorable_cuts : ℝ) / num_nodes = 3 / 5 :=
by sorry

end NUMINAMATH_CALUDE_rope_cutting_probability_l2451_245129


namespace NUMINAMATH_CALUDE_original_car_cost_l2451_245130

/-- Proves that the original cost of a car is 42000 given the repair cost, selling price, and profit percentage. -/
theorem original_car_cost (repair_cost selling_price profit_percent : ℝ) : 
  repair_cost = 8000 →
  selling_price = 64900 →
  profit_percent = 29.8 →
  ∃ (original_cost : ℝ), 
    original_cost = 42000 ∧
    selling_price = (original_cost + repair_cost) * (1 + profit_percent / 100) :=
by
  sorry

end NUMINAMATH_CALUDE_original_car_cost_l2451_245130


namespace NUMINAMATH_CALUDE_color_tape_overlap_l2451_245179

theorem color_tape_overlap (total_length : ℝ) (tape_length : ℝ) (num_tapes : ℕ) 
  (h1 : total_length = 50.5)
  (h2 : tape_length = 18)
  (h3 : num_tapes = 3) :
  (num_tapes * tape_length - total_length) / 2 = 1.75 := by
  sorry

end NUMINAMATH_CALUDE_color_tape_overlap_l2451_245179


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l2451_245162

theorem quadratic_equation_roots (k : ℝ) (x₁ x₂ : ℝ) : 
  (∀ x, x^2 - 3*x + k = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁ * x₂ + 2*x₁ + 2*x₂ = 1) →
  k = -5 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l2451_245162


namespace NUMINAMATH_CALUDE_gcd_lcm_120_40_l2451_245194

theorem gcd_lcm_120_40 : 
  (Nat.gcd 120 40 = 40) ∧ (Nat.lcm 120 40 = 120) := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_120_40_l2451_245194


namespace NUMINAMATH_CALUDE_factorial_difference_l2451_245196

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end NUMINAMATH_CALUDE_factorial_difference_l2451_245196


namespace NUMINAMATH_CALUDE_root_implies_product_bound_l2451_245141

theorem root_implies_product_bound (a b : ℝ) 
  (h : (a + b + a) * (a + b + b) = 9) : a * b ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_product_bound_l2451_245141


namespace NUMINAMATH_CALUDE_cartoon_length_missy_cartoon_length_l2451_245170

/-- The length of a cartoon given specific TV watching conditions -/
theorem cartoon_length (reality_shows : ℕ) (reality_show_length : ℕ) (total_time : ℕ) : ℕ :=
  let cartoon_length := total_time - reality_shows * reality_show_length
  by
    sorry

/-- The length of Missy's cartoon is 10 minutes -/
theorem missy_cartoon_length : cartoon_length 5 28 150 = 10 := by
  sorry

end NUMINAMATH_CALUDE_cartoon_length_missy_cartoon_length_l2451_245170


namespace NUMINAMATH_CALUDE_hannah_apple_pie_apples_l2451_245185

/-- Calculates the number of pounds of apples needed for Hannah's apple pie. -/
def apple_pie_apples (
  servings : ℕ)
  (cost_per_serving : ℚ)
  (apple_cost_per_pound : ℚ)
  (pie_crust_cost : ℚ)
  (lemon_cost : ℚ)
  (butter_cost : ℚ) : ℚ :=
  let total_cost := servings * cost_per_serving
  let apple_cost := total_cost - pie_crust_cost - lemon_cost - butter_cost
  apple_cost / apple_cost_per_pound

/-- Theorem stating that Hannah needs 2 pounds of apples for her pie. -/
theorem hannah_apple_pie_apples :
  apple_pie_apples 8 1 2 2 (1/2) (3/2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_hannah_apple_pie_apples_l2451_245185


namespace NUMINAMATH_CALUDE_equation_solution_l2451_245167

theorem equation_solution : 
  ∃! x : ℝ, (16 : ℝ)^x * (16 : ℝ)^x * (16 : ℝ)^x * (4 : ℝ)^(3*x) = (64 : ℝ)^(4*x) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2451_245167


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2451_245180

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Theorem: In a geometric sequence, if a₃ * a₉ = 4 * a₄, then a₈ = 4 -/
theorem geometric_sequence_property 
  (a : ℕ → ℝ) 
  (h_geo : geometric_sequence a) 
  (h_cond : a 3 * a 9 = 4 * a 4) : 
  a 8 = 4 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2451_245180


namespace NUMINAMATH_CALUDE_four_point_theorem_l2451_245152

-- Define a type for points in a plane
variable (Point : Type)

-- Define a predicate for collinearity
variable (collinear : Point → Point → Point → Point → Prop)

-- Define a predicate for concyclicity
variable (concyclic : Point → Point → Point → Point → Prop)

-- Define a predicate for circle intersection
variable (circle_intersect : Point → Point → Point → Point → Prop)

-- Define the theorem
theorem four_point_theorem 
  (A B C D : Point) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) 
  (h_intersect : circle_intersect A B C D) : 
  collinear A B C D ∨ concyclic A B C D :=
sorry

end NUMINAMATH_CALUDE_four_point_theorem_l2451_245152


namespace NUMINAMATH_CALUDE_equation_solution_l2451_245153

theorem equation_solution :
  ∃ x : ℚ, (1 / 3 + 1 / x = 2 / 3) ∧ (x = 3) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2451_245153


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2451_245151

theorem inequality_solution_set (x : ℝ) :
  x ≠ 1 →
  ((x^2 + x - 6) / (x - 1) ≤ 0 ↔ x ∈ Set.Iic (-3) ∪ Set.Ioo 1 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2451_245151


namespace NUMINAMATH_CALUDE_incentive_savings_l2451_245160

/-- Calculates the amount saved given an initial amount and spending percentages -/
def calculate_savings (initial_amount : ℝ) (food_percent : ℝ) (clothes_percent : ℝ) 
  (household_percent : ℝ) (savings_percent : ℝ) : ℝ :=
  let remaining_after_food := initial_amount * (1 - food_percent)
  let remaining_after_clothes := remaining_after_food * (1 - clothes_percent)
  let remaining_after_household := remaining_after_clothes * (1 - household_percent)
  remaining_after_household * savings_percent

/-- Theorem stating that given the specified spending pattern, 
    the amount saved from a $600 incentive is $171.36 -/
theorem incentive_savings : 
  calculate_savings 600 0.3 0.2 0.15 0.6 = 171.36 := by
  sorry

end NUMINAMATH_CALUDE_incentive_savings_l2451_245160


namespace NUMINAMATH_CALUDE_hernandez_state_tax_l2451_245117

def calculate_state_tax (months_of_residency : ℕ) (taxable_income : ℝ) (tax_rate : ℝ) : ℝ :=
  let proportion_of_year := months_of_residency / 12
  let prorated_income := taxable_income * proportion_of_year
  prorated_income * tax_rate

theorem hernandez_state_tax :
  calculate_state_tax 9 42500 0.04 = 1275 := by
  sorry

end NUMINAMATH_CALUDE_hernandez_state_tax_l2451_245117


namespace NUMINAMATH_CALUDE_arrangements_six_people_one_restricted_l2451_245150

def number_of_arrangements (n : ℕ) : ℕ :=
  (n - 1) * (Nat.factorial (n - 1))

theorem arrangements_six_people_one_restricted :
  number_of_arrangements 6 = 600 := by
  sorry

end NUMINAMATH_CALUDE_arrangements_six_people_one_restricted_l2451_245150


namespace NUMINAMATH_CALUDE_pie_crust_flour_calculation_l2451_245138

/-- Given the initial conditions of pie crust baking and a new number of crusts,
    calculate the amount of flour required for each new crust. -/
theorem pie_crust_flour_calculation (initial_crusts : ℕ) (initial_flour : ℚ) (new_crusts : ℕ) :
  initial_crusts > 0 →
  initial_flour > 0 →
  new_crusts > 0 →
  (initial_flour / initial_crusts) * new_crusts = initial_flour →
  initial_flour / new_crusts = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_pie_crust_flour_calculation_l2451_245138


namespace NUMINAMATH_CALUDE_cymbal_triangle_tambourine_sync_l2451_245197

theorem cymbal_triangle_tambourine_sync (cymbal_beats : Nat) (triangle_beats : Nat) (tambourine_beats : Nat)
  (h1 : cymbal_beats = 13)
  (h2 : triangle_beats = 17)
  (h3 : tambourine_beats = 19) :
  Nat.lcm (Nat.lcm cymbal_beats triangle_beats) tambourine_beats = 4199 := by
  sorry

end NUMINAMATH_CALUDE_cymbal_triangle_tambourine_sync_l2451_245197


namespace NUMINAMATH_CALUDE_gift_cost_l2451_245186

theorem gift_cost (dave_money : ℕ) (kyle_initial : ℕ) (kyle_spent : ℕ) (kyle_remaining : ℕ) (lisa_money : ℕ) (gift_cost : ℕ) : 
  dave_money = 46 →
  kyle_initial = 3 * dave_money - 12 →
  kyle_spent = kyle_initial / 3 →
  kyle_remaining = kyle_initial - kyle_spent →
  lisa_money = kyle_remaining + 20 →
  gift_cost = (kyle_remaining + lisa_money) / 2 →
  gift_cost = 94 := by
sorry

end NUMINAMATH_CALUDE_gift_cost_l2451_245186


namespace NUMINAMATH_CALUDE_smallest_integers_difference_l2451_245121

theorem smallest_integers_difference : ∃ n₁ n₂ : ℕ,
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₁ % k = 1) ∧
  (∀ k : ℕ, 2 ≤ k → k ≤ 12 → n₂ % k = 1) ∧
  n₁ > 1 ∧ n₂ > 1 ∧ n₂ > n₁ ∧
  (∀ m : ℕ, m > 1 → (∀ k : ℕ, 2 ≤ k → k ≤ 12 → m % k = 1) → m ≥ n₁) ∧
  n₂ - n₁ = 27720 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integers_difference_l2451_245121


namespace NUMINAMATH_CALUDE_peggy_bought_three_folders_l2451_245137

/-- Represents the number of sheets in each folder -/
def sheets_per_folder : ℕ := 10

/-- Represents the number of stickers per sheet in the red folder -/
def red_stickers_per_sheet : ℕ := 3

/-- Represents the number of stickers per sheet in the green folder -/
def green_stickers_per_sheet : ℕ := 2

/-- Represents the number of stickers per sheet in the blue folder -/
def blue_stickers_per_sheet : ℕ := 1

/-- Represents the total number of stickers used -/
def total_stickers : ℕ := 60

/-- Theorem stating that Peggy bought 3 folders -/
theorem peggy_bought_three_folders :
  (sheets_per_folder * red_stickers_per_sheet) +
  (sheets_per_folder * green_stickers_per_sheet) +
  (sheets_per_folder * blue_stickers_per_sheet) = total_stickers :=
by sorry

end NUMINAMATH_CALUDE_peggy_bought_three_folders_l2451_245137


namespace NUMINAMATH_CALUDE_score_difference_is_five_l2451_245164

-- Define the score distribution
def score_distribution : List (ℝ × ℝ) := [
  (0.20, 60),
  (0.25, 75),
  (0.15, 85),
  (0.30, 90),
  (0.10, 95)
]

-- Define the median score
def median_score : ℝ := 85

-- Define the function to calculate the mean score
def mean_score (distribution : List (ℝ × ℝ)) : ℝ :=
  (distribution.map (λ (p, s) => p * s)).sum

-- Theorem statement
theorem score_difference_is_five :
  median_score - mean_score score_distribution = 5 := by
  sorry


end NUMINAMATH_CALUDE_score_difference_is_five_l2451_245164


namespace NUMINAMATH_CALUDE_class_average_mark_l2451_245158

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_average : ℝ) (remaining_average : ℝ) : 
  total_students = 35 →
  excluded_students = 5 →
  excluded_average = 20 →
  remaining_average = 90 →
  (total_students * (total_students * remaining_average - 
    excluded_students * excluded_average)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l2451_245158


namespace NUMINAMATH_CALUDE_per_minute_charge_plan_a_l2451_245159

/-- Represents the per-minute charge after the first 4 minutes under plan A -/
def x : ℝ := sorry

/-- The cost of an 18-minute call under plan A -/
def cost_plan_a : ℝ := 0.60 + 14 * x

/-- The cost of an 18-minute call under plan B -/
def cost_plan_b : ℝ := 0.08 * 18

/-- Theorem stating that the per-minute charge after the first 4 minutes under plan A is $0.06 -/
theorem per_minute_charge_plan_a : x = 0.06 := by
  have h1 : cost_plan_a = cost_plan_b := by sorry
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_per_minute_charge_plan_a_l2451_245159


namespace NUMINAMATH_CALUDE_boys_together_arrangements_l2451_245198

/-- The number of ways to arrange n distinct objects -/
def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

/-- The number of students -/
def total_students : ℕ := 5

/-- The number of boys -/
def num_boys : ℕ := 3

/-- The number of girls -/
def num_girls : ℕ := 2

/-- The number of arrangements where all boys stand together -/
def arrangements_with_boys_together : ℕ := factorial num_boys * factorial (total_students - num_boys + 1)

theorem boys_together_arrangements :
  arrangements_with_boys_together = 36 :=
sorry

end NUMINAMATH_CALUDE_boys_together_arrangements_l2451_245198


namespace NUMINAMATH_CALUDE_ratio_w_to_y_l2451_245105

theorem ratio_w_to_y (w x y z : ℚ) 
  (hw : w / x = 5 / 4)
  (hy : y / z = 4 / 3)
  (hz : z / x = 1 / 8) :
  w / y = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ratio_w_to_y_l2451_245105


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l2451_245143

theorem quadratic_roots_difference (P : ℝ) : 
  (∃ α β : ℝ, α^2 - 2*α - P = 0 ∧ β^2 - 2*β - P = 0 ∧ α - β = 12) → P = 35 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_l2451_245143


namespace NUMINAMATH_CALUDE_isabel_earnings_l2451_245199

/-- The number of bead necklaces sold -/
def bead_necklaces : ℕ := 3

/-- The number of gem stone necklaces sold -/
def gem_necklaces : ℕ := 3

/-- The cost of each necklace in dollars -/
def necklace_cost : ℕ := 6

/-- The total number of necklaces sold -/
def total_necklaces : ℕ := bead_necklaces + gem_necklaces

/-- The total money earned in dollars -/
def total_earned : ℕ := total_necklaces * necklace_cost

theorem isabel_earnings : total_earned = 36 := by
  sorry

end NUMINAMATH_CALUDE_isabel_earnings_l2451_245199


namespace NUMINAMATH_CALUDE_west_asian_percentage_approx_46_percent_l2451_245183

/-- Represents the Asian population (in millions) for each region of the U.S. -/
structure AsianPopulation where
  ne : ℕ
  mw : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the percentage of Asian population living in the West -/
def westAsianPercentage (pop : AsianPopulation) : ℚ :=
  (pop.west : ℚ) / (pop.ne + pop.mw + pop.south + pop.west)

/-- The given Asian population data for 1990 -/
def population1990 : AsianPopulation :=
  { ne := 2, mw := 3, south := 2, west := 6 }

theorem west_asian_percentage_approx_46_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1 : ℚ) / 100 ∧ 
  |westAsianPercentage population1990 - (46 : ℚ) / 100| < ε :=
sorry

end NUMINAMATH_CALUDE_west_asian_percentage_approx_46_percent_l2451_245183


namespace NUMINAMATH_CALUDE_midpoint_sum_scaled_triangle_l2451_245118

theorem midpoint_sum_scaled_triangle (a b c : ℝ) (h : a + b + c = 18) :
  let scaled_midpoint_sum := (2*a + 2*b) + (2*a + 2*c) + (2*b + 2*c)
  scaled_midpoint_sum = 36 := by
  sorry

end NUMINAMATH_CALUDE_midpoint_sum_scaled_triangle_l2451_245118


namespace NUMINAMATH_CALUDE_sum_of_squares_l2451_245177

theorem sum_of_squares (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 4)
  (h2 : a/x + b/y + c/z = 0) :
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 16 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2451_245177


namespace NUMINAMATH_CALUDE_solution_set_equality_l2451_245175

-- Define the polynomial function
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Define the set of x values satisfying both conditions
def S : Set ℝ := {x : ℝ | f x > 0 ∧ x < 3}

-- Theorem statement
theorem solution_set_equality : S = Set.Ioi (-1) ∪ Set.Ioo 1 3 :=
sorry

end NUMINAMATH_CALUDE_solution_set_equality_l2451_245175


namespace NUMINAMATH_CALUDE_solution_satisfies_system_l2451_245146

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := x + y = 3
def equation2 (x y : ℝ) : Prop := 2 * (x + y) - y = 5

-- Theorem stating that (2, 1) is the solution
theorem solution_satisfies_system :
  equation1 2 1 ∧ equation2 2 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_satisfies_system_l2451_245146


namespace NUMINAMATH_CALUDE_matrix_addition_a_l2451_245182

def A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 4; -1, 3]
def B : Matrix (Fin 2) (Fin 2) ℝ := !![-1, 3; 1, -4]

theorem matrix_addition_a : A + B = !![1, 7; 0, -1] := by sorry

end NUMINAMATH_CALUDE_matrix_addition_a_l2451_245182


namespace NUMINAMATH_CALUDE_baseball_theorem_l2451_245112

def baseball_problem (team_scores : List Nat) (lost_games : Nat) : Prop :=
  let total_games := team_scores.length
  let won_games := total_games - lost_games
  let opponent_scores := team_scores.map (λ score =>
    if score ∈ [2, 4, 6, 8] then score + 2 else score / 3)
  
  (total_games = 8) ∧
  (team_scores = [2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (lost_games = 4) ∧
  (opponent_scores.sum = 36)

theorem baseball_theorem :
  baseball_problem [2, 3, 4, 5, 6, 7, 8, 9] 4 := by
  sorry

end NUMINAMATH_CALUDE_baseball_theorem_l2451_245112


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2451_245139

theorem cubic_equation_solution : ∃ x : ℝ, (3 - x / 3) ^ (1/3 : ℝ) = -2 ∧ x = 33 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2451_245139


namespace NUMINAMATH_CALUDE_line_symmetrical_to_itself_l2451_245113

/-- A line in the 2D plane represented by its slope-intercept form y = mx + b -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  p.y = l.slope * p.x + l.intercept

/-- The line of symmetry -/
def lineOfSymmetry : Line :=
  { slope := 1, intercept := -2 }

/-- The original line -/
def originalLine : Line :=
  { slope := 3, intercept := 3 }

/-- Find the symmetric point of a given point with respect to the line of symmetry -/
def symmetricPoint (p : Point) : Point :=
  { x := p.x, y := p.y }

theorem line_symmetrical_to_itself :
  ∀ (p : Point), pointOnLine p originalLine →
  pointOnLine (symmetricPoint p) originalLine :=
sorry

end NUMINAMATH_CALUDE_line_symmetrical_to_itself_l2451_245113


namespace NUMINAMATH_CALUDE_derivative_exp_sin_derivative_frac_derivative_ln_derivative_product_derivative_cos_l2451_245144

variable (x : ℝ)

-- Function 1
theorem derivative_exp_sin (x : ℝ) : 
  deriv (fun x => Real.exp (Real.sin x)) x = Real.exp (Real.sin x) * Real.cos x := by sorry

-- Function 2
theorem derivative_frac (x : ℝ) : 
  deriv (fun x => (x + 3) / (x + 2)) x = - 1 / ((x + 2) ^ 2) := by sorry

-- Function 3
theorem derivative_ln (x : ℝ) : 
  deriv (fun x => Real.log (2 * x + 3)) x = 2 / (2 * x + 3) := by sorry

-- Function 4
theorem derivative_product (x : ℝ) : 
  deriv (fun x => (x^2 + 2) * (2*x - 1)) x = 6 * x^2 - 2 * x + 4 := by sorry

-- Function 5
theorem derivative_cos (x : ℝ) : 
  deriv (fun x => Real.cos (2*x + Real.pi/3)) x = -2 * Real.sin (2*x + Real.pi/3) := by sorry

end NUMINAMATH_CALUDE_derivative_exp_sin_derivative_frac_derivative_ln_derivative_product_derivative_cos_l2451_245144


namespace NUMINAMATH_CALUDE_melted_ice_cream_depth_l2451_245135

/-- Given a spherical scoop of ice cream with radius 3 inches that melts into a conical shape with radius 9 inches, prove that the height of the resulting cone is 4/3 inches, assuming constant density. -/
theorem melted_ice_cream_depth (sphere_radius : ℝ) (cone_radius : ℝ) (cone_height : ℝ) : 
  sphere_radius = 3 →
  cone_radius = 9 →
  (4 / 3) * Real.pi * sphere_radius^3 = (1 / 3) * Real.pi * cone_radius^2 * cone_height →
  cone_height = 4 / 3 := by
  sorry

#check melted_ice_cream_depth

end NUMINAMATH_CALUDE_melted_ice_cream_depth_l2451_245135


namespace NUMINAMATH_CALUDE_tan_3_degrees_decomposition_l2451_245188

theorem tan_3_degrees_decomposition :
  ∃ (p q r s : ℕ+),
    (Real.tan (3 * Real.pi / 180) = Real.sqrt p - Real.sqrt q + Real.sqrt r - s) ∧
    (p ≥ q) ∧ (q ≥ r) ∧ (r ≥ s) →
    p + q + r + s = 20 := by
  sorry

end NUMINAMATH_CALUDE_tan_3_degrees_decomposition_l2451_245188


namespace NUMINAMATH_CALUDE_book_cost_price_l2451_245165

/-- The cost price of a book given specific pricing conditions -/
theorem book_cost_price (marked_price selling_price cost_price : ℝ) :
  selling_price = 1.25 * cost_price →
  0.95 * marked_price = selling_price →
  selling_price = 62.5 →
  cost_price = 50 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l2451_245165


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l2451_245187

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 1) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 1 → x + y ≤ a + b :=
by sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l2451_245187


namespace NUMINAMATH_CALUDE_class_average_l2451_245134

/-- Proves that the average score of a class is 45.6 given the specified conditions -/
theorem class_average (total_students : ℕ) (top_scorers : ℕ) (zero_scorers : ℕ) 
  (top_score : ℕ) (rest_average : ℕ) :
  total_students = 25 →
  top_scorers = 3 →
  zero_scorers = 3 →
  top_score = 95 →
  rest_average = 45 →
  let rest_students := total_students - top_scorers - zero_scorers
  let total_score := top_scorers * top_score + zero_scorers * 0 + rest_students * rest_average
  (total_score : ℚ) / total_students = 45.6 := by
  sorry

end NUMINAMATH_CALUDE_class_average_l2451_245134


namespace NUMINAMATH_CALUDE_min_a_sqrt_sum_l2451_245115

theorem min_a_sqrt_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) →
  ∃ a_min : ℝ, a_min = Real.sqrt 2 ∧ ∀ a : ℝ, (∀ x y, x > 0 → y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) → a ≥ a_min :=
by sorry

end NUMINAMATH_CALUDE_min_a_sqrt_sum_l2451_245115


namespace NUMINAMATH_CALUDE_min_dividend_with_quotient_and_remainder_six_l2451_245190

theorem min_dividend_with_quotient_and_remainder_six (dividend : ℕ) (divisor : ℕ) : 
  dividend ≥ 48 → 
  dividend / divisor = 6 → 
  dividend % divisor = 6 → 
  dividend ≥ 48 :=
by
  sorry

end NUMINAMATH_CALUDE_min_dividend_with_quotient_and_remainder_six_l2451_245190


namespace NUMINAMATH_CALUDE_slope_of_line_intersecting_ellipse_l2451_245136

/-- Given an ellipse and a line that intersects it, this theorem proves
    that if (1,1) is the midpoint of the chord formed by the intersection,
    then the slope of the line is -1/4. -/
theorem slope_of_line_intersecting_ellipse 
  (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁^2/36 + y₁^2/9 = 1 →   -- Point (x₁, y₁) is on the ellipse
  x₂^2/36 + y₂^2/9 = 1 →   -- Point (x₂, y₂) is on the ellipse
  (x₁ + x₂)/2 = 1 →        -- x-coordinate of midpoint is 1
  (y₁ + y₂)/2 = 1 →        -- y-coordinate of midpoint is 1
  (y₂ - y₁)/(x₂ - x₁) = -1/4 :=  -- Slope of the line
by sorry

end NUMINAMATH_CALUDE_slope_of_line_intersecting_ellipse_l2451_245136


namespace NUMINAMATH_CALUDE_even_function_order_l2451_245148

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + 6 * m * x + 2

-- State the theorem
theorem even_function_order (m : ℝ) :
  (∀ x, f m x = f m (-x)) →
  f m (-2) < f m 1 ∧ f m 1 < f m 0 := by
sorry


end NUMINAMATH_CALUDE_even_function_order_l2451_245148


namespace NUMINAMATH_CALUDE_polar_bear_salmon_consumption_l2451_245168

def daily_fish_consumption : ℝ := 0.6
def daily_trout_consumption : ℝ := 0.2

theorem polar_bear_salmon_consumption :
  daily_fish_consumption - daily_trout_consumption = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_polar_bear_salmon_consumption_l2451_245168


namespace NUMINAMATH_CALUDE_cube_less_than_triple_l2451_245114

theorem cube_less_than_triple (x : ℤ) : x^3 < 3*x ↔ x = -3 ∨ x = -2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_less_than_triple_l2451_245114


namespace NUMINAMATH_CALUDE_order_of_f_l2451_245131

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_increasing_nonneg : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- Theorem statement
theorem order_of_f : f (-2) < f 3 ∧ f 3 < f (-π) :=
sorry

end NUMINAMATH_CALUDE_order_of_f_l2451_245131


namespace NUMINAMATH_CALUDE_coat_price_l2451_245133

/-- The original price of a coat given a specific price reduction and percentage decrease. -/
theorem coat_price (price_reduction : ℝ) (percent_decrease : ℝ) (original_price : ℝ) : 
  price_reduction = 300 ∧ 
  percent_decrease = 0.60 ∧ 
  price_reduction = percent_decrease * original_price → 
  original_price = 500 := by
sorry

end NUMINAMATH_CALUDE_coat_price_l2451_245133


namespace NUMINAMATH_CALUDE_highway_repair_time_l2451_245163

theorem highway_repair_time (x y : ℝ) : 
  (1 / x + 1 / y = 1 / 18) →  -- Combined work rate
  (2 * x / 3 + y / 3 = 40) →  -- Actual repair time
  (x = 45 ∧ y = 30) := by
  sorry

end NUMINAMATH_CALUDE_highway_repair_time_l2451_245163


namespace NUMINAMATH_CALUDE_greatest_consecutive_odd_integers_sum_400_l2451_245142

/-- The sum of the first n odd integers -/
def sum_odd_integers (n : ℕ) : ℕ := n^2

/-- The problem statement -/
theorem greatest_consecutive_odd_integers_sum_400 :
  (∃ (n : ℕ), sum_odd_integers n = 400) ∧
  (∀ (m : ℕ), sum_odd_integers m = 400 → m ≤ 20) :=
by sorry

end NUMINAMATH_CALUDE_greatest_consecutive_odd_integers_sum_400_l2451_245142


namespace NUMINAMATH_CALUDE_prime_composite_inequality_l2451_245176

theorem prime_composite_inequality (n : ℕ+) :
  (∀ (a : Fin n → ℕ+), (Function.Injective a) →
    ∃ (i j : Fin n), (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) ≥ 2 * n - 1) ∨
  (∃ (a : Fin n → ℕ+), (Function.Injective a) ∧
    ∀ (i j : Fin n), (a i + a j : ℚ) / (Nat.gcd (a i) (a j)) < 2 * n - 1) :=
by sorry

end NUMINAMATH_CALUDE_prime_composite_inequality_l2451_245176


namespace NUMINAMATH_CALUDE_gcd_372_684_l2451_245104

theorem gcd_372_684 : Nat.gcd 372 684 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_372_684_l2451_245104


namespace NUMINAMATH_CALUDE_unique_prime_303509_l2451_245126

theorem unique_prime_303509 :
  ∃! (B : ℕ), B < 10 ∧ Nat.Prime (303500 + B) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_303509_l2451_245126


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2451_245155

theorem fractional_equation_solution : 
  ∃ x : ℝ, (2 * x) / (x - 1) = 3 ∧ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2451_245155


namespace NUMINAMATH_CALUDE_center_digit_is_two_l2451_245110

/-- A function that checks if a number is a perfect square --/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A function that returns the tens digit of a three-digit number --/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

/-- The set of available digits --/
def digit_set : Finset ℕ := {2, 3, 4, 5, 6}

/-- A proposition stating that a number uses only digits from the digit set --/
def uses_digit_set (n : ℕ) : Prop :=
  (n / 100 ∈ digit_set) ∧ (tens_digit n ∈ digit_set) ∧ (n % 10 ∈ digit_set)

theorem center_digit_is_two :
  ∀ (a b : ℕ),
    a ≠ b
    → a ≥ 100 ∧ a < 1000
    → b ≥ 100 ∧ b < 1000
    → is_perfect_square a
    → is_perfect_square b
    → uses_digit_set a
    → uses_digit_set b
    → (Finset.card {a / 100, tens_digit a, a % 10, b / 100, tens_digit b, b % 10} = 5)
    → (tens_digit a = 2 ∨ tens_digit b = 2) :=
  sorry

end NUMINAMATH_CALUDE_center_digit_is_two_l2451_245110


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2451_245127

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 3 * k = 0) ↔ k = 6 :=
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2451_245127


namespace NUMINAMATH_CALUDE_laundry_detergent_price_l2451_245119

/-- Calculates the initial price of laundry detergent given grocery shopping conditions --/
theorem laundry_detergent_price
  (initial_amount : ℝ)
  (milk_price : ℝ)
  (bread_price : ℝ)
  (banana_price_per_pound : ℝ)
  (banana_quantity : ℝ)
  (detergent_coupon : ℝ)
  (amount_left : ℝ)
  (h1 : initial_amount = 20)
  (h2 : milk_price = 4)
  (h3 : bread_price = 3.5)
  (h4 : banana_price_per_pound = 0.75)
  (h5 : banana_quantity = 2)
  (h6 : detergent_coupon = 1.25)
  (h7 : amount_left = 4) :
  let discounted_milk_price := milk_price / 2
  let banana_total := banana_price_per_pound * banana_quantity
  let other_items_cost := discounted_milk_price + bread_price + banana_total
  let total_spent := initial_amount - amount_left
  let detergent_price_with_coupon := total_spent - other_items_cost
  let initial_detergent_price := detergent_price_with_coupon + detergent_coupon
  initial_detergent_price = 10.25 := by
sorry

end NUMINAMATH_CALUDE_laundry_detergent_price_l2451_245119


namespace NUMINAMATH_CALUDE_intersection_equiv_open_interval_l2451_245178

def set_A : Set ℝ := {x | x / (x - 1) ≤ 0}
def set_B : Set ℝ := {x | x^2 < 2*x}

theorem intersection_equiv_open_interval : 
  ∀ x : ℝ, x ∈ set_A ∩ set_B ↔ x ∈ Set.Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_intersection_equiv_open_interval_l2451_245178


namespace NUMINAMATH_CALUDE_benny_seashells_l2451_245191

/-- The number of seashells Benny has after giving some away -/
def remaining_seashells (initial : ℕ) (given_away : ℕ) : ℕ :=
  initial - given_away

/-- Theorem stating that Benny has 14 seashells after giving away 52 from his initial 66 -/
theorem benny_seashells : remaining_seashells 66 52 = 14 := by
  sorry

end NUMINAMATH_CALUDE_benny_seashells_l2451_245191


namespace NUMINAMATH_CALUDE_max_triangle_area_l2451_245157

/-- Given a triangle ABC where BC = 2 ∛3 and ∠BAC = π/3, the maximum possible area is 3 -/
theorem max_triangle_area (A B C : ℝ × ℝ) : 
  let BC := Real.sqrt (Real.sqrt 3) * 2
  let BAC := π / 3
  let area := Real.sqrt 3 * BC^2 / 4
  BC = Real.sqrt (Real.sqrt 3) * 2 →
  BAC = π / 3 →
  area ≤ 3 ∧ ∃ (A' B' C' : ℝ × ℝ), 
    let BC' := Real.sqrt (Real.sqrt 3) * 2
    let BAC' := π / 3
    let area' := Real.sqrt 3 * BC'^2 / 4
    BC' = Real.sqrt (Real.sqrt 3) * 2 ∧
    BAC' = π / 3 ∧
    area' = 3 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_area_l2451_245157


namespace NUMINAMATH_CALUDE_matrix_equality_implies_ratio_l2451_245103

theorem matrix_equality_implies_ratio (a b c d : ℝ) : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![2, 3; 4, 5]
  let B : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  A * B = B * A ∧ 4 * b ≠ c →
  (a - d) / (c - 4 * b) = 2 := by
  sorry

end NUMINAMATH_CALUDE_matrix_equality_implies_ratio_l2451_245103


namespace NUMINAMATH_CALUDE_line_intercepts_l2451_245174

/-- Given a line with equation x + 6y + 2 = 0, prove that its x-intercept is -2 and its y-intercept is -1/3 -/
theorem line_intercepts (x y : ℝ) :
  x + 6 * y + 2 = 0 →
  (x = -2 ∧ y = 0) ∨ (x = 0 ∧ y = -1/3) :=
by sorry

end NUMINAMATH_CALUDE_line_intercepts_l2451_245174


namespace NUMINAMATH_CALUDE_playstation_payment_l2451_245108

theorem playstation_payment (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_total : x₁ + x₂ + x₃ + x₄ + x₅ = 120)
  (h_x₁ : x₁ = (1/3) * (x₂ + x₃ + x₄ + x₅))
  (h_x₂ : x₂ = (1/4) * (x₁ + x₃ + x₄ + x₅))
  (h_x₃ : x₃ = (1/5) * (x₁ + x₂ + x₄ + x₅))
  (h_x₄ : x₄ = (1/6) * (x₁ + x₂ + x₃ + x₅)) :
  x₅ = 40 := by
sorry

end NUMINAMATH_CALUDE_playstation_payment_l2451_245108


namespace NUMINAMATH_CALUDE_amusement_park_problem_l2451_245181

/-- Proves that given the conditions of the amusement park problem, 
    the number of parents is 10 and the number of students is 5 -/
theorem amusement_park_problem 
  (total_people : ℕ)
  (adult_ticket_price : ℕ)
  (student_discount : ℚ)
  (total_spent : ℕ)
  (h1 : total_people = 15)
  (h2 : adult_ticket_price = 50)
  (h3 : student_discount = 0.6)
  (h4 : total_spent = 650) :
  ∃ (parents students : ℕ),
    parents + students = total_people ∧
    parents * adult_ticket_price + 
    students * (adult_ticket_price * (1 - student_discount)) = total_spent ∧
    parents = 10 ∧
    students = 5 :=
by sorry

end NUMINAMATH_CALUDE_amusement_park_problem_l2451_245181


namespace NUMINAMATH_CALUDE_fraction_sum_times_two_l2451_245192

theorem fraction_sum_times_two : 
  (3 / 20 + 5 / 200 + 7 / 2000) * 2 = 0.357 := by sorry

end NUMINAMATH_CALUDE_fraction_sum_times_two_l2451_245192


namespace NUMINAMATH_CALUDE_incorrect_log_values_l2451_245189

-- Define the logarithm function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the variables a, b, c
variable (a b c : ℝ)

-- Define the given correct logarithm values
axiom lg_2 : lg 2 = 1 - a - c
axiom lg_3 : lg 3 = 2*a - b
axiom lg_5 : lg 5 = a + c
axiom lg_9 : lg 9 = 4*a - 2*b

-- State the theorem
theorem incorrect_log_values :
  lg 1.5 ≠ 3*a - b + c ∧ lg 7 ≠ 2*(a + c) :=
sorry

end NUMINAMATH_CALUDE_incorrect_log_values_l2451_245189


namespace NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l2451_245106

theorem consecutive_page_numbers_sum (n : ℕ) : 
  n > 0 ∧ n * (n + 1) = 2156 → n + (n + 1) = 93 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_page_numbers_sum_l2451_245106


namespace NUMINAMATH_CALUDE_not_false_is_true_l2451_245195

theorem not_false_is_true (p q : Prop) (hp : p) (hq : ¬q) : ¬q := by
  sorry

end NUMINAMATH_CALUDE_not_false_is_true_l2451_245195


namespace NUMINAMATH_CALUDE_orchard_fruit_sales_l2451_245102

theorem orchard_fruit_sales (total_fruit : ℕ) (frozen_fruit : ℕ) (fresh_fruit : ℕ) :
  total_fruit = 9792 →
  frozen_fruit = 3513 →
  fresh_fruit = total_fruit - frozen_fruit →
  fresh_fruit = 6279 := by
sorry

end NUMINAMATH_CALUDE_orchard_fruit_sales_l2451_245102


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2451_245140

def is_isosceles_triangle (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)

def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem isosceles_triangle_side_lengths (x : ℝ) :
  (is_isosceles_triangle (x + 3) (2*x + 1) 11 ∧
   satisfies_triangle_inequality (x + 3) (2*x + 1) 11) →
  (x = 8 ∨ x = 5) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2451_245140


namespace NUMINAMATH_CALUDE_sum_of_integers_l2451_245128

theorem sum_of_integers (x y : ℕ+) (h1 : x - y = 8) (h2 : x * y = 180) : x + y = 28 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l2451_245128


namespace NUMINAMATH_CALUDE_simplify_expression_l2451_245100

theorem simplify_expression : (-5)^2 - Real.sqrt 3 = 5 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2451_245100


namespace NUMINAMATH_CALUDE_sum_congruence_l2451_245122

theorem sum_congruence : ∃ k : ℤ, (82 + 83 + 84 + 85 + 86 + 87 + 88 + 89) = 17 * k + 12 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_l2451_245122


namespace NUMINAMATH_CALUDE_expected_heads_after_turn_l2451_245132

/-- Represents the state of pennies on a table -/
structure PennyState where
  total : ℕ
  heads : ℕ
  tails : ℕ

/-- Represents the action of turning over pennies -/
def turn_pennies (state : PennyState) (num_turn : ℕ) : ℝ :=
  let p_heads := state.heads / state.total
  let p_tails := state.tails / state.total
  let expected_heads_turned := num_turn * p_heads
  let expected_tails_turned := num_turn * p_tails
  state.heads - expected_heads_turned + expected_tails_turned

/-- The main theorem to prove -/
theorem expected_heads_after_turn (initial_state : PennyState) 
  (h1 : initial_state.total = 100)
  (h2 : initial_state.heads = 30)
  (h3 : initial_state.tails = 70)
  (num_turn : ℕ)
  (h4 : num_turn = 40) :
  turn_pennies initial_state num_turn = 46 := by
  sorry


end NUMINAMATH_CALUDE_expected_heads_after_turn_l2451_245132


namespace NUMINAMATH_CALUDE_inequality_condition_l2451_245147

theorem inequality_condition (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  |A - B + C| ≤ 2 * Real.sqrt (A * C) :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l2451_245147


namespace NUMINAMATH_CALUDE_worker_pay_is_40_l2451_245173

/-- Represents the plant supplier's sales and expenses --/
structure PlantSupplier where
  orchids : ℕ
  orchidPrice : ℕ
  chinesePlants : ℕ
  chinesePlantPrice : ℕ
  potCost : ℕ
  leftover : ℕ
  workers : ℕ

/-- Calculates the amount paid to each worker --/
def workerPay (ps : PlantSupplier) : ℕ :=
  let totalEarnings := ps.orchids * ps.orchidPrice + ps.chinesePlants * ps.chinesePlantPrice
  let totalSpent := ps.potCost + ps.leftover
  (totalEarnings - totalSpent) / ps.workers

/-- Theorem stating that each worker is paid $40 --/
theorem worker_pay_is_40 (ps : PlantSupplier) 
  (h1 : ps.orchids = 20)
  (h2 : ps.orchidPrice = 50)
  (h3 : ps.chinesePlants = 15)
  (h4 : ps.chinesePlantPrice = 25)
  (h5 : ps.potCost = 150)
  (h6 : ps.leftover = 1145)
  (h7 : ps.workers = 2) :
  workerPay ps = 40 := by
  sorry

end NUMINAMATH_CALUDE_worker_pay_is_40_l2451_245173


namespace NUMINAMATH_CALUDE_crayons_left_l2451_245124

theorem crayons_left (initial : ℕ) (given_away : ℕ) (lost : ℕ) : 
  initial = 1453 → given_away = 563 → lost = 558 → 
  initial - given_away - lost = 332 := by
sorry

end NUMINAMATH_CALUDE_crayons_left_l2451_245124


namespace NUMINAMATH_CALUDE_equation_root_l2451_245171

theorem equation_root (a b c d : ℝ) (h1 : a + d = 2015) (h2 : b + c = 2015) (h3 : a ≠ c) :
  (∀ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d)) ↔ x = 1007.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_l2451_245171


namespace NUMINAMATH_CALUDE_david_orange_juice_purchase_l2451_245193

/-- Calculates the minimum cost to buy a given number of bottles -/
def min_cost (single_price : ℚ) (pack_price : ℚ) (total_bottles : ℕ) : ℚ :=
  let pack_count := total_bottles / 6
  let single_count := total_bottles % 6
  pack_count * pack_price + single_count * single_price

theorem david_orange_juice_purchase :
  min_cost (280/100) (1500/100) 22 = 5620/100 := by
  sorry

end NUMINAMATH_CALUDE_david_orange_juice_purchase_l2451_245193


namespace NUMINAMATH_CALUDE_return_journey_speed_l2451_245120

/-- Given a round trip with the following conditions:
    - The distance between home and the retreat is 300 miles each way
    - The average speed to the retreat was 50 miles per hour
    - The round trip took 10 hours
    - The same route was taken both ways
    Prove that the average speed on the return journey is 75 mph. -/
theorem return_journey_speed (distance : ℝ) (speed_to : ℝ) (total_time : ℝ) :
  distance = 300 →
  speed_to = 50 →
  total_time = 10 →
  let time_to : ℝ := distance / speed_to
  let time_from : ℝ := total_time - time_to
  let speed_from : ℝ := distance / time_from
  speed_from = 75 := by sorry

end NUMINAMATH_CALUDE_return_journey_speed_l2451_245120


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2451_245161

/-- An isosceles triangle with side lengths 5 and 8 has a perimeter of either 18 or 21. -/
theorem isosceles_triangle_perimeter : 
  ∀ a b c : ℝ, 
  (a = 5 ∧ b = 8) ∨ (a = 8 ∧ b = 5) → 
  (a = b ∨ a = c ∨ b = c) → 
  (a + b + c = 18 ∨ a + b + c = 21) := by
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l2451_245161


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2451_245125

theorem sum_of_numbers_in_ratio (x : ℝ) :
  x > 0 →
  x^2 + (2*x)^2 + (4*x)^2 = 1701 →
  x + 2*x + 4*x = 63 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2451_245125


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2451_245156

theorem sum_of_cubes (a b c : ℕ+) :
  (a + b + c)^3 - a^3 - b^3 - c^3 = 270 → a + b + c = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2451_245156


namespace NUMINAMATH_CALUDE_suv_wash_price_l2451_245169

/-- The price of a car wash in dollars -/
def car_price : ℕ := 5

/-- The price of a truck wash in dollars -/
def truck_price : ℕ := 6

/-- The total amount raised in dollars -/
def total_raised : ℕ := 100

/-- The number of SUVs washed -/
def num_suvs : ℕ := 5

/-- The number of trucks washed -/
def num_trucks : ℕ := 5

/-- The number of cars washed -/
def num_cars : ℕ := 7

/-- The price of an SUV wash in dollars -/
def suv_price : ℕ := 9

theorem suv_wash_price :
  car_price * num_cars + truck_price * num_trucks + suv_price * num_suvs = total_raised :=
by sorry

end NUMINAMATH_CALUDE_suv_wash_price_l2451_245169


namespace NUMINAMATH_CALUDE_chef_pies_l2451_245154

theorem chef_pies (apple_pies pecan_pies pumpkin_pies total_pies : ℕ) 
  (h1 : apple_pies = 2)
  (h2 : pecan_pies = 4)
  (h3 : total_pies = 13)
  (h4 : total_pies = apple_pies + pecan_pies + pumpkin_pies) :
  pumpkin_pies = 7 := by
  sorry

end NUMINAMATH_CALUDE_chef_pies_l2451_245154


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l2451_245184

theorem sum_and_reciprocal_sum (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h_sum : a + b = 6 * x) (h_reciprocal_sum : 1 / a + 1 / b = 6) : x = a * b :=
by sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l2451_245184


namespace NUMINAMATH_CALUDE_pm25_decrease_theorem_l2451_245172

/-- Calculates the PM2.5 concentration after two consecutive years of 10% decrease -/
def pm25_concentration (initial : ℝ) : ℝ :=
  initial * (1 - 0.1)^2

/-- Theorem stating that given an initial PM2.5 concentration of 50 micrograms per cubic meter
    two years ago, with a 10% decrease each year for two consecutive years,
    the resulting concentration is 40.5 micrograms per cubic meter -/
theorem pm25_decrease_theorem (initial : ℝ) (h : initial = 50) :
  pm25_concentration initial = 40.5 := by
  sorry

#eval pm25_concentration 50

end NUMINAMATH_CALUDE_pm25_decrease_theorem_l2451_245172


namespace NUMINAMATH_CALUDE_quadratic_root_sum_product_ratio_l2451_245116

theorem quadratic_root_sum_product_ratio : 
  ∀ x₁ x₂ : ℝ, x₁^2 - 2*x₁ - 8 = 0 → x₂^2 - 2*x₂ - 8 = 0 → 
  (x₁ + x₂) / (x₁ * x₂) = -1/4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_product_ratio_l2451_245116


namespace NUMINAMATH_CALUDE_min_x_coordinate_midpoint_l2451_245111

/-- Given a line segment AB of length m on the right branch of the hyperbola x²/a² - y²/b² = 1,
    where m > 2b²/a, the minimum x-coordinate of the midpoint M of AB is a(m + 2a) / (2√(a² + b²)). -/
theorem min_x_coordinate_midpoint (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hm : m > 2 * b^2 / a) :
  let min_x := a * (m + 2 * a) / (2 * Real.sqrt (a^2 + b^2))
  ∀ (x y z w : ℝ),
    x^2 / a^2 - y^2 / b^2 = 1 →
    z^2 / a^2 - w^2 / b^2 = 1 →
    (z - x)^2 + (w - y)^2 = m^2 →
    x > 0 →
    z > 0 →
    (x + z) / 2 ≥ min_x :=
by sorry

end NUMINAMATH_CALUDE_min_x_coordinate_midpoint_l2451_245111
