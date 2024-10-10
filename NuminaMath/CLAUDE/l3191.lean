import Mathlib

namespace chocolate_distribution_l3191_319122

/-- The number of ways to distribute n distinct objects among k recipients -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a distribution satisfies the given conditions -/
def validDistribution (d : List ℕ) : Prop :=
  d.length = 3 ∧ d.sum = 8 ∧ d.all (· > 0) ∧ d.Nodup

theorem chocolate_distribution :
  sumOfDigits (distribute 8 3) = 24 :=
sorry

end chocolate_distribution_l3191_319122


namespace remaining_episodes_l3191_319121

theorem remaining_episodes (seasons : Nat) (episodes_per_season : Nat) 
  (watched_fraction : Rat) (h1 : seasons = 12) (h2 : episodes_per_season = 20) 
  (h3 : watched_fraction = 1/3) : 
  seasons * episodes_per_season - (seasons * episodes_per_season * watched_fraction).floor = 160 := by
  sorry

end remaining_episodes_l3191_319121


namespace total_chairs_is_59_l3191_319130

/-- The number of chairs in the office canteen -/
def total_chairs : ℕ :=
  let round_tables : ℕ := 3
  let rectangular_tables : ℕ := 4
  let square_tables : ℕ := 2
  let chairs_per_round_table : ℕ := 6
  let chairs_per_rectangular_table : ℕ := 7
  let chairs_per_square_table : ℕ := 4
  let extra_chairs : ℕ := 5
  (round_tables * chairs_per_round_table) +
  (rectangular_tables * chairs_per_rectangular_table) +
  (square_tables * chairs_per_square_table) +
  extra_chairs

/-- Theorem stating that the total number of chairs in the office canteen is 59 -/
theorem total_chairs_is_59 : total_chairs = 59 := by
  sorry

end total_chairs_is_59_l3191_319130


namespace red_cars_count_l3191_319144

theorem red_cars_count (black_cars : ℕ) (ratio_red : ℕ) (ratio_black : ℕ) : 
  black_cars = 75 → ratio_red = 3 → ratio_black = 8 → 
  (ratio_red : ℚ) / (ratio_black : ℚ) * black_cars = 28 := by
  sorry

end red_cars_count_l3191_319144


namespace distance_product_range_l3191_319177

-- Define the curves C₁ and C₂
def C₁ (x y : ℝ) : Prop := y^2 = 4*x
def C₂ (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 8

-- Define a line with slope 45°
def line_slope_45 (x₁ y₁ x₂ y₂ : ℝ) : Prop := y₂ - y₁ = x₂ - x₁

-- Define the property of a point being on a curve
def point_on_curve (C : ℝ → ℝ → Prop) (x y : ℝ) : Prop := C x y

-- Define the property of a line intersecting a curve at two distinct points
def line_intersects_curve_at_two_points (C : ℝ → ℝ → Prop) (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  line_slope_45 x₁ y₁ x₂ y₂ ∧ line_slope_45 x₁ y₁ x₃ y₃ ∧
  point_on_curve C x₂ y₂ ∧ point_on_curve C x₃ y₃ ∧
  (x₂ ≠ x₃ ∨ y₂ ≠ y₃)

-- Define the distance between two points
def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ := ((x₂ - x₁)^2 + (y₂ - y₁)^2)^(1/2)

-- Define the product of distances
def distance_product (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : ℝ :=
  distance x₁ y₁ x₂ y₂ * distance x₁ y₁ x₃ y₃

-- Main theorem
theorem distance_product_range :
  ∀ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    point_on_curve C₁ x₁ y₁ →
    line_intersects_curve_at_two_points C₂ x₁ y₁ x₂ y₂ x₃ y₃ →
    4 ≤ distance_product x₁ y₁ x₂ y₂ x₃ y₃ ∧
    distance_product x₁ y₁ x₂ y₂ x₃ y₃ < 8 ∨
    8 < distance_product x₁ y₁ x₂ y₂ x₃ y₃ ∧
    distance_product x₁ y₁ x₂ y₂ x₃ y₃ ≤ 200 :=
sorry

end distance_product_range_l3191_319177


namespace g_one_equals_three_l3191_319186

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the properties of f and g
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x

-- Define the given equations
axiom eq1 : f (-1) + g 1 = 2
axiom eq2 : f 1 + g (-1) = 4

-- State the theorem to be proved
theorem g_one_equals_three : g 1 = 3 := by
  sorry

end g_one_equals_three_l3191_319186


namespace smallest_advantageous_discount_l3191_319136

theorem smallest_advantageous_discount : ∃ (n : ℕ), n = 29 ∧ 
  (∀ (x : ℝ), x > 0 → 
    (1 - n / 100) * x < (1 - 0.12) * (1 - 0.18) * x ∧
    (1 - n / 100) * x < (1 - 0.08) * (1 - 0.08) * (1 - 0.08) * x ∧
    (1 - n / 100) * x < (1 - 0.20) * (1 - 0.10) * x) ∧
  (∀ (m : ℕ), m < n → 
    ∃ (x : ℝ), x > 0 ∧
      ((1 - m / 100) * x ≥ (1 - 0.12) * (1 - 0.18) * x ∨
       (1 - m / 100) * x ≥ (1 - 0.08) * (1 - 0.08) * (1 - 0.08) * x ∨
       (1 - m / 100) * x ≥ (1 - 0.20) * (1 - 0.10) * x)) :=
by sorry

end smallest_advantageous_discount_l3191_319136


namespace min_value_expression_l3191_319100

theorem min_value_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : x + y = 2) :
  (4 / (x + 3 * y)) + (1 / (x - y)) ≥ 9 / 4 := by
  sorry

end min_value_expression_l3191_319100


namespace half_difference_donations_l3191_319173

theorem half_difference_donations (julie_donation margo_donation : ℕ) 
  (h1 : julie_donation = 4700)
  (h2 : margo_donation = 4300) :
  (julie_donation - margo_donation) / 2 = 200 := by
  sorry

end half_difference_donations_l3191_319173


namespace f_max_min_on_interval_l3191_319133

def f (x : ℝ) := x^4 - 8*x^2 + 3

theorem f_max_min_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ Set.Icc (-2) 2, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-2) 2, f x = max) ∧
    (∀ x ∈ Set.Icc (-2) 2, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-2) 2, f x = min) ∧
    max = 3 ∧ min = -13 := by sorry

end f_max_min_on_interval_l3191_319133


namespace calculate_expression_l3191_319107

theorem calculate_expression : 3000 * (3000 ^ 2999) * 2 ^ 3000 = 3000 ^ 3000 * 2 ^ 3000 := by
  sorry

end calculate_expression_l3191_319107


namespace undetermined_sum_l3191_319197

/-- Operation # defined for non-negative integers a and b, and positive integer c -/
def sharp (a b c : ℕ) : ℕ := 4 * a^3 + 4 * b^3 + 8 * a^2 * b + c

/-- Operation * defined for non-negative integers a and b, and positive integer d -/
def star (a b d : ℕ) : ℕ := 2 * a^2 - 3 * b^2 + d^3

/-- Theorem stating that the value of (a + b) + 6 cannot be determined -/
theorem undetermined_sum (a b x c d : ℕ) (hc : c > 0) (hd : d > 0) 
  (h1 : sharp a x c = 250) (h2 : star a b d + x = 50) : 
  ∃ (a' b' x' c' d' : ℕ), 
    c' > 0 ∧ d' > 0 ∧
    sharp a' x' c' = 250 ∧ 
    star a' b' d' + x' = 50 ∧
    a + b + 6 ≠ a' + b' + 6 :=
sorry

end undetermined_sum_l3191_319197


namespace expression_value_l3191_319126

theorem expression_value (a b c d e f : ℝ) 
  (h1 : a * b = 1)
  (h2 : c + d = 0)
  (h3 : |e| = Real.sqrt 2)
  (h4 : Real.sqrt f = 8) :
  (1/2) * a * b + (c + d) / 5 + e^2 + f^(1/3) = 13/2 := by
  sorry

end expression_value_l3191_319126


namespace tournament_rankings_l3191_319112

/-- Represents a team in the tournament -/
inductive Team : Type
| A | B | C | D | E | F

/-- Represents a match between two teams -/
structure Match :=
(team1 : Team)
(team2 : Team)

/-- Represents the tournament structure -/
structure Tournament :=
(saturday_matches : List Match)
(no_ties : Bool)

/-- Represents the final ranking of teams -/
structure Ranking :=
(first : Team)
(second : Team)
(third : Team)
(fourth : Team)
(fifth : Team)
(sixth : Team)

/-- Counts the number of possible ranking sequences for the given tournament -/
def countPossibleRankings (t : Tournament) : Nat :=
  sorry

/-- The main theorem stating the number of possible ranking sequences -/
theorem tournament_rankings (t : Tournament) 
  (h1 : t.saturday_matches = [Match.mk Team.A Team.B, Match.mk Team.C Team.D, Match.mk Team.E Team.F])
  (h2 : t.no_ties = true) : 
  countPossibleRankings t = 288 :=
sorry

end tournament_rankings_l3191_319112


namespace total_notebooks_bought_l3191_319111

/-- Represents the number of notebooks in a large pack -/
def large_pack_size : ℕ := 7

/-- Represents the number of large packs Wilson bought -/
def large_packs_bought : ℕ := 7

/-- Theorem stating that the total number of notebooks Wilson bought is 49 -/
theorem total_notebooks_bought : large_pack_size * large_packs_bought = 49 := by
  sorry

end total_notebooks_bought_l3191_319111


namespace spurs_basketball_distribution_l3191_319134

theorem spurs_basketball_distribution (num_players : ℕ) (total_basketballs : ℕ) 
  (h1 : num_players = 22) 
  (h2 : total_basketballs = 242) : 
  total_basketballs / num_players = 11 := by
  sorry

end spurs_basketball_distribution_l3191_319134


namespace sqrt_expression_simplification_fraction_simplification_l3191_319178

-- Problem 1
theorem sqrt_expression_simplification :
  3 * Real.sqrt 2 - (Real.sqrt 3 + 2 * Real.sqrt 2) * Real.sqrt 6 = -4 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem fraction_simplification (a : ℝ) (h1 : a^2 ≠ 4) (h2 : a ≠ 2) :
  a / (a^2 - 4) + 1 / (4 - 2*a) = 1 / (2*a + 4) := by
  sorry

end sqrt_expression_simplification_fraction_simplification_l3191_319178


namespace vector_norm_sum_l3191_319160

theorem vector_norm_sum (a b : ℝ × ℝ) :
  let m := (2 * a.1 + b.1, 2 * a.2 + b.2) / 2
  m = (-1, 5) →
  a.1 * b.1 + a.2 * b.2 = 10 →
  a.1^2 + a.2^2 + b.1^2 + b.2^2 = 16 := by
sorry

end vector_norm_sum_l3191_319160


namespace ellipse_major_axis_length_l3191_319114

-- Define the radius of the cylinder
def cylinder_radius : ℝ := 2

-- Define the relationship between major and minor axes
def major_axis_ratio : ℝ := 1.75

-- Theorem statement
theorem ellipse_major_axis_length :
  let minor_axis := 2 * cylinder_radius
  let major_axis := major_axis_ratio * minor_axis
  major_axis = 7 := by sorry

end ellipse_major_axis_length_l3191_319114


namespace dividend_calculation_l3191_319164

theorem dividend_calculation (dividend divisor : ℕ) : 
  (dividend / divisor = 4) → 
  (dividend % divisor = 3) → 
  (dividend + divisor + 4 + 3 = 100) → 
  dividend = 75 := by
sorry

end dividend_calculation_l3191_319164


namespace floor_of_e_eq_two_l3191_319113

/-- The floor of Euler's number is 2 -/
theorem floor_of_e_eq_two : ⌊Real.exp 1⌋ = 2 := by sorry

end floor_of_e_eq_two_l3191_319113


namespace sin_sum_to_product_l3191_319119

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end sin_sum_to_product_l3191_319119


namespace exponent_problem_l3191_319148

theorem exponent_problem (x m n : ℝ) (hm : x^m = 5) (hn : x^n = 1/4) :
  x^(2*m - n) = 100 := by
  sorry

end exponent_problem_l3191_319148


namespace triangle_angle_ratio_l3191_319158

theorem triangle_angle_ratio (A B C : ℝ) (x : ℝ) : 
  B = x * A →
  C = A + 12 →
  A = 24 →
  A + B + C = 180 →
  x = 5 := by sorry

end triangle_angle_ratio_l3191_319158


namespace inequality_properties_l3191_319109

theorem inequality_properties (m n : ℝ) : 
  (∀ a : ℝ, a > 0 → m * a^2 < n * a^2 → m < n) ∧
  (m < n → n < 0 → n / m < 1) :=
sorry

end inequality_properties_l3191_319109


namespace sum_f_negative_l3191_319137

/-- The function f(x) = 2x³ + 4x -/
def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x

/-- Theorem: Given f(x) = 2x³ + 4x and a + b < 0, b + c < 0, c + a < 0, then f(a) + f(b) + f(c) < 0 -/
theorem sum_f_negative (a b c : ℝ) (hab : a + b < 0) (hbc : b + c < 0) (hca : c + a < 0) :
  f a + f b + f c < 0 := by
  sorry

end sum_f_negative_l3191_319137


namespace smallest_marble_count_l3191_319159

theorem smallest_marble_count : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n ≤ 999) ∧ 
  (n + 7) % 10 = 0 ∧ 
  (n - 10) % 7 = 0 ∧
  n = 143 ∧
  ∀ (m : ℕ), (m ≥ 100 ∧ m ≤ 999) → (m + 7) % 10 = 0 → (m - 10) % 7 = 0 → m ≥ n :=
by sorry

end smallest_marble_count_l3191_319159


namespace weight_increase_percentage_shyam_weight_increase_percentage_l3191_319104

theorem weight_increase_percentage (ram_ratio : ℝ) (shyam_ratio : ℝ) 
  (ram_increase : ℝ) (total_weight : ℝ) (total_increase : ℝ) : ℝ :=
  let original_total := total_weight / (1 + total_increase / 100)
  let x := original_total / (ram_ratio + shyam_ratio)
  let ram_original := ram_ratio * x
  let shyam_original := shyam_ratio * x
  let ram_new := ram_original * (1 + ram_increase / 100)
  let shyam_new := total_weight - ram_new
  (shyam_new - shyam_original) / shyam_original * 100

/-- Given the weights of Ram and Shyam in a 7:5 ratio, Ram's weight increased by 10%,
    and the total weight after increase is 82.8 kg with a 15% total increase,
    prove that Shyam's weight increase percentage is 22%. -/
theorem shyam_weight_increase_percentage :
  weight_increase_percentage 7 5 10 82.8 15 = 22 := by
  sorry

end weight_increase_percentage_shyam_weight_increase_percentage_l3191_319104


namespace lowest_sale_price_percentage_l3191_319125

/-- Calculates the lowest possible sale price of a jersey as a percentage of its list price -/
theorem lowest_sale_price_percentage (list_price : ℝ) (max_regular_discount : ℝ) (summer_sale_discount : ℝ) :
  list_price = 80 ∧ 
  max_regular_discount = 0.5 ∧ 
  summer_sale_discount = 0.2 →
  (list_price * (1 - max_regular_discount) - list_price * summer_sale_discount) / list_price = 0.3 := by
sorry

end lowest_sale_price_percentage_l3191_319125


namespace ballpoint_pen_price_l3191_319168

-- Define the problem parameters
def total_pens : Nat := 30
def total_pencils : Nat := 75
def total_cost : ℝ := 690

def gel_pens : Nat := 20
def ballpoint_pens : Nat := 10
def standard_pencils : Nat := 50
def mechanical_pencils : Nat := 25

def avg_price_gel : ℝ := 1.5
def avg_price_mechanical : ℝ := 3
def avg_price_standard : ℝ := 2

-- Theorem to prove
theorem ballpoint_pen_price :
  ∃ (avg_price_ballpoint : ℝ),
    avg_price_ballpoint = 48.5 ∧
    total_cost = 
      gel_pens * avg_price_gel +
      mechanical_pencils * avg_price_mechanical +
      standard_pencils * avg_price_standard +
      ballpoint_pens * avg_price_ballpoint :=
by sorry

end ballpoint_pen_price_l3191_319168


namespace division_equations_l3191_319192

theorem division_equations (h : 40 * 60 = 2400) : 
  (2400 / 40 = 60) ∧ (2400 / 60 = 40) := by
  sorry

end division_equations_l3191_319192


namespace smallest_candy_count_l3191_319191

theorem smallest_candy_count : ∃ n : ℕ, 
  (100 ≤ n ∧ n ≤ 999) ∧ 
  (n + 7) % 9 = 0 ∧ 
  (n - 10) % 6 = 0 ∧
  (∀ m : ℕ, (100 ≤ m ∧ m < n ∧ (m + 7) % 9 = 0 ∧ (m - 10) % 6 = 0) → False) ∧
  n = 146 := by
sorry

end smallest_candy_count_l3191_319191


namespace book_selection_problem_l3191_319115

theorem book_selection_problem (n m k : ℕ) (h1 : n = 8) (h2 : m = 5) (h3 : k = 4) :
  (Nat.choose (n - 1) k) = (Nat.choose (n - 1) (m - 1)) :=
sorry

end book_selection_problem_l3191_319115


namespace article_cost_calculation_l3191_319110

/-- Proves that if an article is sold for $25 with a 25% gain, it was bought for $20. -/
theorem article_cost_calculation (selling_price : ℝ) (gain_percent : ℝ) : 
  selling_price = 25 → gain_percent = 25 → 
  ∃ (cost_price : ℝ), cost_price = 20 ∧ selling_price = cost_price * (1 + gain_percent / 100) :=
by sorry

end article_cost_calculation_l3191_319110


namespace only_translation_preserves_pattern_l3191_319102

/-- Represents the types of figures in the pattern -/
inductive Figure
| Triangle
| Square

/-- Represents a point on the line ℓ -/
structure PointOnLine where
  position : ℝ

/-- Represents the infinite pattern on line ℓ -/
def Pattern := ℕ → Figure

/-- Represents the possible rigid motion transformations -/
inductive RigidMotion
| Rotation (center : PointOnLine) (angle : ℝ)
| Translation (distance : ℝ)
| ReflectionAcrossL
| ReflectionPerpendicular (point : PointOnLine)

/-- Defines the alternating pattern of triangles and squares -/
def alternatingPattern : Pattern :=
  fun n => if n % 2 = 0 then Figure.Triangle else Figure.Square

/-- Checks if a rigid motion preserves the pattern -/
def preservesPattern (motion : RigidMotion) (pattern : Pattern) : Prop :=
  ∀ n, pattern n = pattern (n + 1)  -- This is a simplification; actual preservation would be more complex

/-- The main theorem stating that only translation preserves the pattern -/
theorem only_translation_preserves_pattern :
  ∀ motion : RigidMotion,
    preservesPattern motion alternatingPattern ↔ ∃ d, motion = RigidMotion.Translation d :=
sorry

end only_translation_preserves_pattern_l3191_319102


namespace bear_food_consumption_l3191_319162

/-- The weight of Victor in pounds -/
def victor_weight : ℝ := 126

/-- The number of "Victors" worth of food a bear eats in 3 weeks -/
def victors_in_three_weeks : ℝ := 15

/-- The number of weeks in the given condition -/
def given_weeks : ℝ := 3

/-- Theorem: For any number of weeks, the bear eats 5 times that many "Victors" worth of food -/
theorem bear_food_consumption (x : ℝ) : 
  (victors_in_three_weeks / given_weeks) * x = 5 * x := by
sorry

end bear_food_consumption_l3191_319162


namespace probability_no_consecutive_ones_l3191_319142

/-- Represents the number of valid sequences of length n -/
def validSequences : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | n+2 => validSequences (n+1) + validSequences n

/-- The length of the sequence -/
def sequenceLength : ℕ := 12

/-- The total number of possible sequences -/
def totalSequences : ℕ := 2^sequenceLength

theorem probability_no_consecutive_ones :
  (validSequences sequenceLength : ℚ) / totalSequences = 377 / 4096 := by
  sorry

#eval validSequences sequenceLength + totalSequences

end probability_no_consecutive_ones_l3191_319142


namespace mortdecai_charity_donation_l3191_319108

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the number of days Mortdecai collects eggs --/
def collection_days : ℕ := 2

/-- Represents the number of dozens of eggs Mortdecai collects per day --/
def collected_dozens_per_day : ℕ := 8

/-- Represents the number of dozens of eggs Mortdecai delivers to the market --/
def market_delivery : ℕ := 3

/-- Represents the number of dozens of eggs Mortdecai delivers to the mall --/
def mall_delivery : ℕ := 5

/-- Represents the number of dozens of eggs Mortdecai uses for pie --/
def pie_dozens : ℕ := 4

/-- Theorem stating that Mortdecai donates 48 eggs to charity --/
theorem mortdecai_charity_donation : 
  (collection_days * collected_dozens_per_day - (market_delivery + mall_delivery) - pie_dozens) * dozen = 48 := by
  sorry

end mortdecai_charity_donation_l3191_319108


namespace kelly_peanuts_weight_l3191_319131

/-- Given the total weight of snacks and the weight of raisins, 
    calculate the weight of peanuts Kelly bought. -/
theorem kelly_peanuts_weight 
  (total_snacks : ℝ) 
  (raisins_weight : ℝ) 
  (h1 : total_snacks = 0.5) 
  (h2 : raisins_weight = 0.4) : 
  total_snacks - raisins_weight = 0.1 := by
  sorry

#check kelly_peanuts_weight

end kelly_peanuts_weight_l3191_319131


namespace g_geq_neg_two_solution_set_f_minus_g_geq_m_plus_two_iff_l3191_319163

-- Define the functions f and g
def f (x : ℝ) : ℝ := |2*x - 1| + 2
def g (x : ℝ) : ℝ := -|x + 2| + 3

-- Theorem for the first part of the problem
theorem g_geq_neg_two_solution_set :
  {x : ℝ | g x ≥ -2} = {x : ℝ | -7 ≤ x ∧ x ≤ 3} :=
sorry

-- Theorem for the second part of the problem
theorem f_minus_g_geq_m_plus_two_iff (m : ℝ) :
  (∀ x : ℝ, f x - g x ≥ m + 2) ↔ m ≤ -1/2 :=
sorry

end g_geq_neg_two_solution_set_f_minus_g_geq_m_plus_two_iff_l3191_319163


namespace thursday_loaves_l3191_319138

def bakery_sequence : List ℕ := [5, 11, 10, 14, 19, 25]

def alternating_differences (seq : List ℕ) : List ℕ :=
  List.zipWith (λ a b => b - a) seq (seq.tail)

theorem thursday_loaves :
  let seq := bakery_sequence
  let diffs := alternating_differences seq
  (seq[1] = 11 ∧
   diffs[0] = diffs[2] + 1 ∧
   diffs[1] = diffs[3] - 1 ∧
   diffs[2] = diffs[4] + 1) →
  seq[1] = 11 := by sorry

end thursday_loaves_l3191_319138


namespace sum_of_a_and_b_l3191_319179

theorem sum_of_a_and_b (a b : ℝ) : 
  a^2*b^2 + a^2 + b^2 + 1 - 2*a*b = 2*a*b → a + b = 2 ∨ a + b = -2 := by
  sorry

end sum_of_a_and_b_l3191_319179


namespace tan_domain_shift_l3191_319193

theorem tan_domain_shift (x : ℝ) :
  (∃ k : ℤ, x = k * π / 2 + π / 12) ↔ (∃ k : ℤ, 2 * x + π / 3 = k * π + π / 2) :=
by sorry

end tan_domain_shift_l3191_319193


namespace jennifer_museum_trips_l3191_319166

/-- Calculates the total miles traveled for round trips to two museums -/
def total_miles_traveled (distance1 distance2 : ℕ) : ℕ :=
  2 * distance1 + 2 * distance2

/-- Theorem: Jennifer travels 40 miles in total to visit both museums -/
theorem jennifer_museum_trips : total_miles_traveled 5 15 = 40 := by
  sorry

end jennifer_museum_trips_l3191_319166


namespace tanning_salon_revenue_l3191_319116

/-- Calculate the revenue of a tanning salon for a month --/
theorem tanning_salon_revenue :
  let first_visit_charge : ℚ := 10
  let subsequent_visit_charge : ℚ := 8
  let discount_rate : ℚ := 0.1
  let premium_service_charge : ℚ := 15
  let premium_service_rate : ℚ := 0.2
  let total_customers : ℕ := 150
  let second_visit_customers : ℕ := 40
  let third_visit_customers : ℕ := 15
  let fourth_visit_customers : ℕ := 5

  let first_visit_revenue : ℚ := 
    (premium_service_rate * total_customers.cast) * premium_service_charge +
    ((1 - premium_service_rate) * total_customers.cast) * first_visit_charge
  let second_visit_revenue : ℚ := second_visit_customers.cast * subsequent_visit_charge
  let discounted_visit_charge : ℚ := subsequent_visit_charge * (1 - discount_rate)
  let third_visit_revenue : ℚ := third_visit_customers.cast * discounted_visit_charge
  let fourth_visit_revenue : ℚ := fourth_visit_customers.cast * discounted_visit_charge

  let total_revenue : ℚ := 
    first_visit_revenue + second_visit_revenue + third_visit_revenue + fourth_visit_revenue

  total_revenue = 2114 := by sorry

end tanning_salon_revenue_l3191_319116


namespace acid_solution_concentration_l3191_319185

/-- Proves that replacing half of a 50% acid solution with a solution of unknown concentration to obtain a 40% solution implies the unknown concentration is 30% -/
theorem acid_solution_concentration (original_concentration : ℝ) 
  (final_concentration : ℝ) (replaced_fraction : ℝ) (replacement_concentration : ℝ) :
  original_concentration = 50 →
  final_concentration = 40 →
  replaced_fraction = 0.5 →
  (1 - replaced_fraction) * original_concentration + replaced_fraction * replacement_concentration = 100 * final_concentration →
  replacement_concentration = 30 := by
sorry

end acid_solution_concentration_l3191_319185


namespace stratified_sampling_theorem_l3191_319135

/-- Represents the number of villages in each category -/
structure VillageCategories where
  total : ℕ
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the sample sizes for each category -/
structure SampleSizes where
  first : ℕ
  secondAndThird : ℕ

/-- Checks if the sampling is stratified -/
def isStratifiedSampling (vc : VillageCategories) (ss : SampleSizes) : Prop :=
  (ss.first : ℚ) / vc.first = (ss.first + ss.secondAndThird : ℚ) / vc.total

/-- The main theorem to prove -/
theorem stratified_sampling_theorem (vc : VillageCategories) (ss : SampleSizes) :
  vc.total = 300 →
  vc.first = 60 →
  vc.second = 100 →
  vc.third = vc.total - vc.first - vc.second →
  ss.first = 3 →
  isStratifiedSampling vc ss →
  ss.secondAndThird = 12 := by
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l3191_319135


namespace geometric_progression_special_ratio_l3191_319156

/-- A geometric progression with positive terms where each term is the average of the next two terms plus 2 has a common ratio of 1. -/
theorem geometric_progression_special_ratio :
  ∀ (a : ℝ) (r : ℝ),
  (a > 0) →  -- First term is positive
  (r > 0) →  -- Common ratio is positive
  (∀ n : ℕ, a * r^n = (a * r^(n+1) + a * r^(n+2)) / 2 + 2) →  -- Condition on terms
  r = 1 := by
sorry

end geometric_progression_special_ratio_l3191_319156


namespace range_of_f_l3191_319128

-- Define the function f
def f : ℝ → ℝ := λ x => x^2 - 10*x - 4

-- State the theorem
theorem range_of_f :
  ∀ t : ℝ, t ∈ Set.Ioo 0 8 → ∃ y : ℝ, y ∈ Set.Icc (-29) (-4) ∧ y = f t ∧
  ∀ z : ℝ, z ∈ Set.Icc (-29) (-4) → ∃ s : ℝ, s ∈ Set.Ioo 0 8 ∧ z = f s :=
by sorry

end range_of_f_l3191_319128


namespace solution_2015_squared_l3191_319118

theorem solution_2015_squared : 
  ∃ x : ℚ, (2015 + x)^2 = x^2 ∧ x = -2015/2 := by
sorry

end solution_2015_squared_l3191_319118


namespace angle_A_is_45_degrees_l3191_319129

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  -- The sum of angles in a triangle is 180°
  A + B + C = Real.pi ∧
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧
  b / Real.sin B = c / Real.sin C ∧
  a / Real.sin A = c / Real.sin C

-- State the theorem
theorem angle_A_is_45_degrees :
  ∀ (A B C : ℝ) (a b c : ℝ),
  triangle_ABC A B C a b c →
  a = Real.sqrt 2 →
  b = Real.sqrt 3 →
  B = Real.pi / 3 →
  A = Real.pi / 4 :=
by sorry

end angle_A_is_45_degrees_l3191_319129


namespace probability_heart_then_club_l3191_319161

theorem probability_heart_then_club (total_cards : Nat) (hearts : Nat) (clubs : Nat) :
  total_cards = 52 →
  hearts = 13 →
  clubs = 13 →
  (hearts : ℚ) / total_cards * clubs / (total_cards - 1) = 13 / 204 := by
  sorry

end probability_heart_then_club_l3191_319161


namespace committee_selection_count_l3191_319187

/-- Represents the total number of members in the class committee -/
def totalMembers : Nat := 5

/-- Represents the number of roles to be filled -/
def rolesToFill : Nat := 3

/-- Represents the number of members who cannot serve in a specific role -/
def restrictedMembers : Nat := 2

/-- Calculates the number of ways to select committee members under given constraints -/
def selectCommitteeMembers (total : Nat) (roles : Nat) (restricted : Nat) : Nat :=
  (total - restricted) * (total - 1) * (total - 2)

theorem committee_selection_count :
  selectCommitteeMembers totalMembers rolesToFill restrictedMembers = 36 := by
  sorry

#eval selectCommitteeMembers totalMembers rolesToFill restrictedMembers

end committee_selection_count_l3191_319187


namespace initial_number_proof_l3191_319165

theorem initial_number_proof (x : ℝ) : 
  x + 3889 - 47.80600000000004 = 3854.002 → x = 12.808000000000158 :=
by
  sorry

end initial_number_proof_l3191_319165


namespace quadratic_real_roots_range_l3191_319176

theorem quadratic_real_roots_range (k : ℝ) :
  (∃ x : ℝ, 2 * x^2 - 3 * x = k) ↔ k ≥ -9/8 := by sorry

end quadratic_real_roots_range_l3191_319176


namespace leahs_coin_value_l3191_319141

/-- Represents the number and value of coins --/
structure CoinCollection where
  pennies : ℕ
  quarters : ℕ

/-- Calculates the total value of coins in cents --/
def totalValue (coins : CoinCollection) : ℕ :=
  coins.pennies + 25 * coins.quarters

/-- Theorem stating the value of Leah's coin collection --/
theorem leahs_coin_value :
  ∀ (coins : CoinCollection),
    coins.pennies + coins.quarters = 15 →
    coins.pennies = 2 * (coins.quarters + 1) →
    totalValue coins = 110 := by
  sorry


end leahs_coin_value_l3191_319141


namespace sin_3phi_from_exponential_l3191_319103

theorem sin_3phi_from_exponential (φ : ℝ) :
  Complex.exp (Complex.I * φ) = (1 + Complex.I * Real.sqrt 8) / 3 →
  Real.sin (3 * φ) = -5 * Real.sqrt 8 / 9 := by
  sorry

end sin_3phi_from_exponential_l3191_319103


namespace tree_spacing_l3191_319120

theorem tree_spacing (yard_length : ℕ) (num_trees : ℕ) (distance : ℕ) : 
  yard_length = 273 ∧ num_trees = 14 → distance * (num_trees - 1) = yard_length → distance = 21 := by
  sorry

end tree_spacing_l3191_319120


namespace zoo_animal_count_l3191_319101

theorem zoo_animal_count (initial_count : ℕ) (gorillas_sent : ℕ) (hippo_adopted : ℕ) 
  (rhinos_taken : ℕ) (final_count : ℕ) : 
  initial_count = 68 →
  gorillas_sent = 6 →
  hippo_adopted = 1 →
  rhinos_taken = 3 →
  final_count = 90 →
  ∃ (lion_cubs : ℕ), 
    final_count = initial_count - gorillas_sent + hippo_adopted + rhinos_taken + lion_cubs + 2 * lion_cubs ∧
    lion_cubs = 8 := by
  sorry

end zoo_animal_count_l3191_319101


namespace speeding_ticket_percentage_l3191_319194

/-- The percentage of motorists who exceed the speed limit -/
def speeding_percentage : ℝ := 25

/-- The percentage of speeding motorists who do not receive tickets -/
def no_ticket_percentage : ℝ := 60

/-- The percentage of motorists who receive speeding tickets -/
def ticket_percentage : ℝ := 10

theorem speeding_ticket_percentage :
  ticket_percentage = speeding_percentage * (1 - no_ticket_percentage / 100) := by
  sorry

end speeding_ticket_percentage_l3191_319194


namespace towels_folded_in_one_hour_l3191_319154

/-- Represents the number of towels a person can fold in one hour --/
def towels_per_hour (
  jane_rate : ℕ → ℕ
) (
  kyla_rate : ℕ → ℕ
) (
  anthony_rate : ℕ → ℕ
) (
  david_rate : ℕ → ℕ
) : ℕ :=
  jane_rate 60 + kyla_rate 60 + anthony_rate 60 + david_rate 60

/-- Jane's folding rate: 5 towels in 5 minutes, 3-minute break after every 5 minutes --/
def jane_rate (minutes : ℕ) : ℕ :=
  (minutes / 8) * 5

/-- Kyla's folding rate: 12 towels in 10 minutes for first 30 minutes, then 6 towels in 10 minutes --/
def kyla_rate (minutes : ℕ) : ℕ :=
  min 36 (minutes / 10 * 12) + max 0 ((minutes - 30) / 10 * 6)

/-- Anthony's folding rate: 14 towels in 20 minutes, 10-minute break after 40 minutes --/
def anthony_rate (minutes : ℕ) : ℕ :=
  (min minutes 40) / 20 * 14

/-- David's folding rate: 4 towels in 15 minutes, speed increases by 1 towel per 15 minutes for every 3 sets --/
def david_rate (minutes : ℕ) : ℕ :=
  (minutes / 15) * 4 + (minutes / 45)

theorem towels_folded_in_one_hour :
  towels_per_hour jane_rate kyla_rate anthony_rate david_rate = 134 := by
  sorry

end towels_folded_in_one_hour_l3191_319154


namespace maya_shoe_probability_l3191_319149

/-- Represents the number of pairs for each shoe color --/
structure ShoePairs where
  black : Nat
  brown : Nat
  grey : Nat
  white : Nat

/-- Calculates the probability of picking two shoes of the same color,
    one left and one right, given a distribution of shoe pairs --/
def samePairColorProbability (pairs : ShoePairs) : Rat :=
  let totalShoes := 2 * (pairs.black + pairs.brown + pairs.grey + pairs.white)
  let numerator := pairs.black * pairs.black + pairs.brown * pairs.brown +
                   pairs.grey * pairs.grey + pairs.white * pairs.white
  numerator / (totalShoes * (totalShoes - 1))

/-- Maya's shoe collection --/
def mayasShoes : ShoePairs := ⟨8, 4, 3, 1⟩

theorem maya_shoe_probability :
  samePairColorProbability mayasShoes = 45 / 248 := by
  sorry

end maya_shoe_probability_l3191_319149


namespace geometric_sequence_problem_l3191_319155

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

def is_monotonically_increasing (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

theorem geometric_sequence_problem (a : ℕ → ℝ) 
  (h_geom : is_geometric_sequence a)
  (h_mono : is_monotonically_increasing a)
  (h_prod : a 1 * a 9 = 64)
  (h_sum : a 3 + a 7 = 20) :
  a 11 = 64 := by
  sorry

end geometric_sequence_problem_l3191_319155


namespace real_roots_imply_m_value_l3191_319153

theorem real_roots_imply_m_value (x m : ℝ) (i : ℂ) :
  (∃ x : ℝ, x^2 + (1 - 2*i)*x + 3*m - i = 0) → m = 1/12 := by
sorry

end real_roots_imply_m_value_l3191_319153


namespace building_has_at_least_43_floors_l3191_319106

/-- Represents a building with apartments -/
structure Building where
  apartments_per_floor : ℕ
  kolya_floor : ℕ
  kolya_apartment : ℕ
  vasya_floor : ℕ
  vasya_apartment : ℕ

/-- The specific building described in the problem -/
def problem_building : Building :=
  { apartments_per_floor := 4
  , kolya_floor := 5
  , kolya_apartment := 83
  , vasya_floor := 3
  , vasya_apartment := 169
  }

/-- Calculates the minimum number of floors in the building -/
def min_floors (b : Building) : ℕ :=
  ((b.vasya_apartment - 1) / b.apartments_per_floor) + 1

/-- Theorem stating that the building has at least 43 floors -/
theorem building_has_at_least_43_floors :
  min_floors problem_building ≥ 43 := by
  sorry


end building_has_at_least_43_floors_l3191_319106


namespace ellipse_points_equiv_target_set_l3191_319167

/-- Represents an ellipse passing through (2,1) with a > b > 0 -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0
  h_a_gt_b : a > b
  h_passes_through : 4 / a^2 + 1 / b^2 = 1

/-- The set of points (x, y) on the ellipse satisfying |y| > 1 -/
def ellipse_points (e : Ellipse) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / e.a^2 + p.2^2 / e.b^2 = 1 ∧ |p.2| > 1}

/-- The set of points (x, y) satisfying x^2 + y^2 < 5 and |y| > 1 -/
def target_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 < 5 ∧ |p.2| > 1}

/-- Theorem stating the equivalence of the two sets -/
theorem ellipse_points_equiv_target_set (e : Ellipse) :
  ellipse_points e = target_set := by sorry

end ellipse_points_equiv_target_set_l3191_319167


namespace team_a_games_l3191_319189

theorem team_a_games (a : ℕ) : 
  (2 : ℚ) / 3 * a + (1 : ℚ) / 3 * a = a → -- Team A's wins + losses = total games
  (3 : ℚ) / 5 * (a + 12) = (2 : ℚ) / 3 * a + 6 → -- Team B's wins = Team A's wins + 6
  (2 : ℚ) / 5 * (a + 12) = (1 : ℚ) / 3 * a + 6 → -- Team B's losses = Team A's losses + 6
  a = 18 := by
sorry

end team_a_games_l3191_319189


namespace train_time_theorem_l3191_319182

/-- The time in minutes for a train to travel between two platforms --/
def train_travel_time (X : ℝ) : Prop :=
  0 < X ∧ X < 60 ∧
  ∀ (start_hour start_minute end_hour end_minute : ℝ),
    -- Angle between hour and minute hands at start
    |30 * start_hour - 5.5 * start_minute| = X →
    -- Angle between hour and minute hands at end
    |30 * end_hour - 5.5 * end_minute| = X →
    -- Time difference between start and end
    (end_hour - start_hour) * 60 + (end_minute - start_minute) = X →
    X = 48

theorem train_time_theorem :
  ∀ X, train_travel_time X → X = 48 :=
by
  sorry

end train_time_theorem_l3191_319182


namespace two_digit_number_problem_l3191_319198

theorem two_digit_number_problem : 
  ∃! n : ℕ, 
    10 ≤ n ∧ n < 100 ∧  -- two-digit number
    (n / 10 + n % 10 = 11) ∧  -- sum of digits is 11
    (10 * (n % 10) + (n / 10) = n + 63) ∧  -- swapped number is 63 greater
    n = 29  -- the number is 29
  := by sorry

end two_digit_number_problem_l3191_319198


namespace toys_in_box_time_l3191_319180

/-- The time it takes to put all toys in the box -/
def time_to_put_toys_in_box (total_toys : ℕ) (toys_in_per_minute : ℕ) (toys_out_per_minute : ℕ) : ℕ :=
  sorry

/-- Theorem stating that it takes 25 minutes to put all toys in the box under given conditions -/
theorem toys_in_box_time : time_to_put_toys_in_box 50 5 3 = 25 := by
  sorry

end toys_in_box_time_l3191_319180


namespace mutually_exclusive_not_contradictory_l3191_319195

/-- Represents the color of a ball -/
inductive Color
| Red
| Black

/-- Represents the outcome of drawing two balls -/
structure Outcome :=
  (first second : Color)

/-- The set of all possible outcomes when drawing two balls from the bag -/
def allOutcomes : Finset Outcome := sorry

/-- The event "Exactly one black ball" -/
def exactlyOneBlack (outcome : Outcome) : Prop :=
  (outcome.first = Color.Black ∧ outcome.second = Color.Red) ∨
  (outcome.first = Color.Red ∧ outcome.second = Color.Black)

/-- The event "Exactly two black balls" -/
def exactlyTwoBlack (outcome : Outcome) : Prop :=
  outcome.first = Color.Black ∧ outcome.second = Color.Black

theorem mutually_exclusive_not_contradictory :
  (∀ o : Outcome, ¬(exactlyOneBlack o ∧ exactlyTwoBlack o)) ∧
  (∃ o : Outcome, ¬exactlyOneBlack o ∧ ¬exactlyTwoBlack o) :=
sorry

end mutually_exclusive_not_contradictory_l3191_319195


namespace train_length_l3191_319184

theorem train_length (t_platform : ℝ) (t_pole : ℝ) (l_platform : ℝ)
  (h1 : t_platform = 33)
  (h2 : t_pole = 18)
  (h3 : l_platform = 250) :
  ∃ l_train : ℝ, l_train = 300 ∧ (l_train + l_platform) / t_platform = l_train / t_pole :=
by
  sorry

end train_length_l3191_319184


namespace prayer_difference_l3191_319199

/-- Represents the number of prayers for a pastor in a week -/
structure WeeklyPrayers where
  weekday : ℕ  -- Number of prayers on a weekday
  sunday : ℕ   -- Number of prayers on Sunday

/-- Calculates the total number of prayers in a week -/
def totalPrayers (wp : WeeklyPrayers) : ℕ :=
  6 * wp.weekday + wp.sunday

/-- Pastor Paul's prayer schedule -/
def paulPrayers : WeeklyPrayers where
  weekday := 20
  sunday := 40

/-- Pastor Bruce's prayer schedule -/
def brucePrayers : WeeklyPrayers where
  weekday := paulPrayers.weekday / 2
  sunday := 2 * paulPrayers.sunday

theorem prayer_difference :
  totalPrayers paulPrayers - totalPrayers brucePrayers = 20 := by
  sorry

end prayer_difference_l3191_319199


namespace area_of_three_semicircle_intersection_l3191_319151

/-- The area of intersection of three semicircles forming a square -/
theorem area_of_three_semicircle_intersection (r : ℝ) (h : r = 2) : 
  let square_side := 2 * r
  let square_area := square_side ^ 2
  square_area = 16 := by sorry

end area_of_three_semicircle_intersection_l3191_319151


namespace max_value_of_f_l3191_319181

/-- The function f(x) = |x| - |x - 3| -/
def f (x : ℝ) : ℝ := |x| - |x - 3|

/-- The maximum value of f(x) is 3 -/
theorem max_value_of_f : 
  ∃ (M : ℝ), M = 3 ∧ ∀ x, f x ≤ M ∧ ∃ y, f y = M :=
sorry

end max_value_of_f_l3191_319181


namespace tangent_slope_angle_at_one_l3191_319175

noncomputable def f (x : ℝ) : ℝ := -Real.sqrt 3 / 3 * x^3 + 2

theorem tangent_slope_angle_at_one :
  let f' : ℝ → ℝ := λ x ↦ deriv f x
  let slope : ℝ := f' 1
  let slope_angle : ℝ := Real.pi - Real.arctan slope
  slope_angle = 2 * Real.pi / 3 := by sorry

end tangent_slope_angle_at_one_l3191_319175


namespace highway_problem_l3191_319147

-- Define the speeds and distances
def yi_initial_speed : ℝ := 60
def speed_reduction_jia : ℝ := 0.4
def speed_reduction_yi : ℝ := 0.25
def time_jia_to_bing : ℝ := 9
def extra_distance_yi : ℝ := 50

-- Define the theorem
theorem highway_problem :
  ∃ (jia_initial_speed : ℝ) (distance_AD : ℝ),
    jia_initial_speed = 125 ∧ distance_AD = 1880 := by
  sorry


end highway_problem_l3191_319147


namespace ammonia_formed_l3191_319171

-- Define the chemical species
structure ChemicalSpecies where
  name : String
  coefficient : ℕ

-- Define the chemical equation
structure ChemicalEquation where
  reactants : List ChemicalSpecies
  products : List ChemicalSpecies

-- Define the reaction conditions
structure ReactionConditions where
  li3n_amount : ℚ
  h2o_amount : ℚ
  lioh_amount : ℚ

-- Define the balanced equation
def balanced_equation : ChemicalEquation :=
  { reactants := [
      { name := "Li3N", coefficient := 1 },
      { name := "H2O", coefficient := 3 }
    ],
    products := [
      { name := "LiOH", coefficient := 3 },
      { name := "NH3", coefficient := 1 }
    ]
  }

-- Define the reaction conditions
def reaction_conditions : ReactionConditions :=
  { li3n_amount := 1,
    h2o_amount := 54,
    lioh_amount := 3 }

-- Theorem statement
theorem ammonia_formed (eq : ChemicalEquation) (conditions : ReactionConditions) :
  eq = balanced_equation ∧
  conditions = reaction_conditions →
  ∃ (nh3_amount : ℚ), nh3_amount = 1 :=
sorry

end ammonia_formed_l3191_319171


namespace right_triangle_hypotenuse_l3191_319174

theorem right_triangle_hypotenuse (a b c : ℝ) : 
  a = 90 → b = 120 → c^2 = a^2 + b^2 → c = 150 := by sorry

end right_triangle_hypotenuse_l3191_319174


namespace correct_categorization_l3191_319105

-- Define the teams
def IntegerTeam : Set ℝ := {0, -8}
def FractionTeam : Set ℝ := {1/7, 0.505}
def IrrationalTeam : Set ℝ := {Real.sqrt 13, Real.pi}

-- Define the properties for each team
def isInteger (x : ℝ) : Prop := ∃ n : ℤ, x = n
def isFraction (x : ℝ) : Prop := ∃ a b : ℤ, b ≠ 0 ∧ x = a / b
def isIrrational (x : ℝ) : Prop := ¬(isInteger x ∨ isFraction x)

-- Theorem to prove the correct categorization
theorem correct_categorization :
  (∀ x ∈ IntegerTeam, isInteger x) ∧
  (∀ x ∈ FractionTeam, isFraction x) ∧
  (∀ x ∈ IrrationalTeam, isIrrational x) :=
  sorry

end correct_categorization_l3191_319105


namespace expression_evaluation_l3191_319152

theorem expression_evaluation :
  let x : ℝ := 3
  let expr := (1 / (x^2 - 2*x) - 1 / (x^2 - 4*x + 4)) / (2 / (x^2 - 2*x))
  expr = -1 := by
  sorry

end expression_evaluation_l3191_319152


namespace annual_salary_is_20_l3191_319172

/-- Represents the total annual cash salary in rupees -/
def annual_salary : ℕ := sorry

/-- Represents the number of months the servant worked -/
def months_worked : ℕ := 9

/-- Represents the total amount received by the servant after 9 months in rupees -/
def amount_received : ℕ := 55

/-- Represents the price of the turban in rupees -/
def turban_price : ℕ := 50

/-- Theorem stating that the annual salary is 20 rupees -/
theorem annual_salary_is_20 :
  annual_salary = 20 :=
by sorry

end annual_salary_is_20_l3191_319172


namespace area_of_second_square_l3191_319146

/-- A right isosceles triangle with two inscribed squares -/
structure RightIsoscelesTriangleWithSquares where
  -- The side length of the triangle
  b : ℝ
  -- The side length of the first inscribed square (ADEF)
  a₁ : ℝ
  -- The side length of the second inscribed square (GHIJ)
  a : ℝ
  -- The first square is inscribed in the triangle
  h_a₁_inscribed : a₁ = b / 2
  -- The second square is inscribed in the triangle
  h_a_inscribed : a = (2 * b ^ 2) / (3 * b * Real.sqrt 2)

/-- The theorem statement -/
theorem area_of_second_square (t : RightIsoscelesTriangleWithSquares) 
    (h_area_first : t.a₁ ^ 2 = 2250) : 
    t.a ^ 2 = 2000 := by
  sorry

end area_of_second_square_l3191_319146


namespace find_genuine_coins_l3191_319117

/-- Represents the result of a weighing -/
inductive WeighResult
  | Equal : WeighResult
  | LeftHeavier : WeighResult
  | RightHeavier : WeighResult

/-- Represents a coin -/
inductive Coin
  | Genuine : Coin
  | Counterfeit : Coin

/-- Represents a set of coins -/
def CoinSet := Fin 7 → Coin

/-- A weighing function that compares two sets of coins -/
def weigh (coins : CoinSet) (left right : List (Fin 7)) : WeighResult :=
  sorry

/-- Checks if a given set of coins contains exactly 3 genuine coins -/
def isValidResult (coins : CoinSet) (result : List (Fin 7)) : Prop :=
  result.length = 3 ∧ ∀ i ∈ result.toFinset, coins i = Coin.Genuine

/-- The main theorem stating that it's possible to find 3 genuine coins in two weighings -/
theorem find_genuine_coins 
  (coins : CoinSet) 
  (h1 : ∃ (i j : Fin 7), i ≠ j ∧ coins i = Coin.Counterfeit ∧ coins j = Coin.Counterfeit)
  (h2 : ∀ (i : Fin 7), coins i ≠ Coin.Counterfeit → coins i = Coin.Genuine)
  : ∃ (w1 w2 : List (Fin 7) × List (Fin 7)) (result : List (Fin 7)),
    isValidResult coins result ∧ 
    (∀ (c1 c2 : CoinSet), 
      (∀ (i : Fin 7), coins i = Coin.Genuine ↔ c1 i = Coin.Genuine) →
      (∀ (i : Fin 7), coins i = Coin.Genuine ↔ c2 i = Coin.Genuine) →
      weigh c1 w1.1 w1.2 = weigh c2 w1.1 w1.2 →
      weigh c1 w2.1 w2.2 = weigh c2 w2.1 w2.2 →
      (∀ (i : Fin 7), i ∈ result → c1 i = Coin.Genuine)) :=
sorry

end find_genuine_coins_l3191_319117


namespace subset_removal_distinctness_l3191_319169

theorem subset_removal_distinctness (n : ℕ) :
  ∀ (S : Finset ℕ) (A : Fin n → Finset ℕ),
    S = Finset.range n →
    (∀ i j, i ≠ j → A i ≠ A j) →
    (∀ i, A i ⊆ S) →
    ∃ x ∈ S, ∀ i j, i ≠ j → A i \ {x} ≠ A j \ {x} :=
by sorry

end subset_removal_distinctness_l3191_319169


namespace john_needs_72_strings_l3191_319132

/-- Calculates the total number of strings needed for restringing instruments --/
def total_strings (num_basses : ℕ) (strings_per_bass : ℕ) (strings_per_guitar : ℕ) (strings_per_8string : ℕ) : ℕ :=
  let num_guitars := 2 * num_basses
  let num_8string := num_guitars - 3
  num_basses * strings_per_bass + num_guitars * strings_per_guitar + num_8string * strings_per_8string

theorem john_needs_72_strings :
  total_strings 3 4 6 8 = 72 := by
  sorry

end john_needs_72_strings_l3191_319132


namespace eighth_prime_is_19_l3191_319150

/-- Natural numbers are non-negative integers -/
def NaturalNumber (n : ℕ) : Prop := True

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself -/
def IsPrime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d > 0 → d < p → p % d ≠ 0

/-- The nth prime number -/
def NthPrime (n : ℕ) : ℕ :=
  sorry

theorem eighth_prime_is_19 : NthPrime 8 = 19 := by
  sorry

end eighth_prime_is_19_l3191_319150


namespace prob_two_boys_one_girl_l3191_319190

/-- A hobby group with boys and girls -/
structure HobbyGroup where
  boys : Nat
  girls : Nat

/-- The probability of selecting exactly one boy and one girl -/
def prob_one_boy_one_girl (group : HobbyGroup) : Rat :=
  if group.boys ≥ 1 ∧ group.girls ≥ 1 then
    (group.boys * group.girls : Rat) / (group.boys + group.girls).choose 2
  else
    0

/-- Theorem: The probability of selecting exactly one boy and one girl
    from a group of 2 boys and 1 girl is 2/3 -/
theorem prob_two_boys_one_girl :
  prob_one_boy_one_girl ⟨2, 1⟩ = 2/3 := by
  sorry


end prob_two_boys_one_girl_l3191_319190


namespace x_y_inequality_l3191_319145

theorem x_y_inequality (x y : ℝ) 
  (h1 : x < 1) 
  (h2 : 1 < y) 
  (h3 : 2 * Real.log x + Real.log (1 - x) ≥ 3 * Real.log y + Real.log (y - 1)) :
  x^3 + y^3 ≤ 2 := by
  sorry

end x_y_inequality_l3191_319145


namespace inequality_solution_range_l3191_319196

theorem inequality_solution_range (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (a * Real.cos x - 1) * (a * x^2 - x + 16 * a) < 0) ↔ 
  (a < -1 ∨ a > 0) :=
by sorry

end inequality_solution_range_l3191_319196


namespace function_characterization_l3191_319127

theorem function_characterization (f : ℕ → ℕ) 
  (h_increasing : ∀ x y : ℕ, x ≤ y → f x ≤ f y)
  (h_square1 : ∀ n : ℕ, ∃ k : ℕ, f n + n + 1 = k^2)
  (h_square2 : ∀ n : ℕ, ∃ k : ℕ, f (f n) - f n = k^2) :
  ∀ x : ℕ, f x = x^2 + x :=
sorry

end function_characterization_l3191_319127


namespace baby_nexus_monograms_l3191_319124

/-- The number of letters in the alphabet --/
def alphabet_size : ℕ := 26

/-- The number of letters to exclude (X and one other) --/
def excluded_letters : ℕ := 2

/-- The number of letters to choose for the monogram (first and middle initials) --/
def letters_to_choose : ℕ := 2

/-- Calculates the number of possible monograms for baby Nexus --/
def monogram_count : ℕ :=
  Nat.choose (alphabet_size - excluded_letters) letters_to_choose

theorem baby_nexus_monograms :
  monogram_count = 253 := by
  sorry

end baby_nexus_monograms_l3191_319124


namespace train_length_calculation_l3191_319140

theorem train_length_calculation (crossing_time : ℝ) (bridge_length : ℝ) (train_speed_kmph : ℝ) :
  crossing_time = 25.997920166386688 →
  bridge_length = 160 →
  train_speed_kmph = 36 →
  let train_speed_mps := train_speed_kmph * (5/18)
  let total_distance := train_speed_mps * crossing_time
  let train_length := total_distance - bridge_length
  train_length = 99.97920166386688 := by sorry

end train_length_calculation_l3191_319140


namespace card_sum_theorem_l3191_319139

theorem card_sum_theorem (a b c d e f g h : ℕ) :
  (a + b) * (c + d) * (e + f) * (g + h) = 330 →
  a + b + c + d + e + f + g + h = 21 := by
sorry

end card_sum_theorem_l3191_319139


namespace cyclist_round_time_l3191_319170

/-- Proves that a cyclist completes one round of a rectangular park in 8 minutes
    given the specified conditions. -/
theorem cyclist_round_time (length width : ℝ) (area perimeter : ℝ) (speed : ℝ) : 
  length / width = 4 →
  area = length * width →
  area = 102400 →
  perimeter = 2 * (length + width) →
  speed = 12 * 1000 / 3600 →
  (perimeter / speed) / 60 = 8 :=
by sorry

end cyclist_round_time_l3191_319170


namespace circumscribed_sphere_area_l3191_319123

/-- Given a rectangular solid with adjacent face areas √2, √3, and √6,
    the surface area of its circumscribed sphere is 6π. -/
theorem circumscribed_sphere_area (x y z : ℝ) 
  (h1 : x * y = Real.sqrt 6)
  (h2 : y * z = Real.sqrt 2)
  (h3 : z * x = Real.sqrt 3) :
  4 * Real.pi * ((Real.sqrt 6) / 2)^2 = 6 * Real.pi := by
  sorry


end circumscribed_sphere_area_l3191_319123


namespace license_plate_difference_l3191_319188

/-- The number of letters in the alphabet --/
def num_letters : ℕ := 26

/-- The number of digits available --/
def num_digits : ℕ := 10

/-- The number of possible Florida license plates --/
def florida_plates : ℕ := num_letters^4 * num_digits^2

/-- The number of possible North Dakota license plates --/
def north_dakota_plates : ℕ := num_letters^3 * num_digits^3

/-- The difference in the number of possible license plates between Florida and North Dakota --/
def plate_difference : ℕ := florida_plates - north_dakota_plates

theorem license_plate_difference : plate_difference = 28121600 := by
  sorry

end license_plate_difference_l3191_319188


namespace belle_weekly_treat_cost_l3191_319157

/-- The cost to feed Belle treats for a week -/
def weekly_treat_cost (dog_biscuits_per_day : ℕ) (rawhide_bones_per_day : ℕ) 
  (dog_biscuit_cost : ℚ) (rawhide_bone_cost : ℚ) (days_per_week : ℕ) : ℚ :=
  (dog_biscuits_per_day * dog_biscuit_cost + rawhide_bones_per_day * rawhide_bone_cost) * days_per_week

/-- Proof that Belle's weekly treat cost is $21.00 -/
theorem belle_weekly_treat_cost :
  weekly_treat_cost 4 2 0.25 1 7 = 21 := by
  sorry

#eval weekly_treat_cost 4 2 (1/4) 1 7

end belle_weekly_treat_cost_l3191_319157


namespace robot_number_difference_l3191_319183

def largest_three_digit (a b c : Nat) : Nat :=
  100 * max a (max b c) + 10 * max (min (max a b) (max b c)) (min a (min b c)) + min a (min b c)

def smallest_three_digit (a b c : Nat) : Nat :=
  if min a (min b c) = 0
  then 100 * min (max a b) (max b c) + 10 * max (min a (min b c)) (min (max a b) (max b c)) + 0
  else 100 * min a (min b c) + 10 * min (max a b) (max b c) + max a (max b c)

theorem robot_number_difference :
  largest_three_digit 2 3 5 - smallest_three_digit 4 0 6 = 126 := by
  sorry

end robot_number_difference_l3191_319183


namespace rectangular_prism_width_l3191_319143

theorem rectangular_prism_width (l h d w : ℝ) : 
  l = 5 → h = 7 → d = 14 → d^2 = l^2 + w^2 + h^2 → w^2 = 122 := by sorry

end rectangular_prism_width_l3191_319143
