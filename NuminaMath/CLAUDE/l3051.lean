import Mathlib

namespace function_property_l3051_305150

theorem function_property (f : ℕ+ → ℝ) 
  (h1 : f 1 = 2) 
  (h2 : ∀ (n1 n2 : ℕ+), f (n1 + n2) = f n1 * f n2) : 
  ∀ (n : ℕ+), f n = 2^(n:ℝ) := by
  sorry

end function_property_l3051_305150


namespace tax_base_amount_l3051_305171

/-- Proves that given a tax rate of 82% and a tax amount of $82, the base amount is $100. -/
theorem tax_base_amount (tax_rate : ℝ) (tax_amount : ℝ) (base_amount : ℝ) : 
  tax_rate = 82 ∧ tax_amount = 82 → base_amount = 100 := by
  sorry

end tax_base_amount_l3051_305171


namespace prob_three_heads_l3051_305191

/-- The probability of getting heads for a biased coin -/
def p : ℚ := sorry

/-- Condition: probability of 1 head equals probability of 2 heads in 4 flips -/
axiom condition : 4 * p * (1 - p)^3 = 6 * p^2 * (1 - p)^2

/-- Theorem: Probability of 3 heads in 4 flips is 96/625 -/
theorem prob_three_heads : 4 * p^3 * (1 - p) = 96/625 := by sorry

end prob_three_heads_l3051_305191


namespace complex_number_range_l3051_305125

theorem complex_number_range (z : ℂ) (h : Complex.abs (z - (3 - 4*I)) = 1) :
  4 ≤ Complex.abs z ∧ Complex.abs z ≤ 6 := by
  sorry

end complex_number_range_l3051_305125


namespace quadratic_root_implies_a_l3051_305102

theorem quadratic_root_implies_a (a : ℝ) :
  (∃ x : ℝ, x^2 + 3*x + a = 0) ∧ (2^2 + 3*2 + a = 0) → a = -10 := by
  sorry

end quadratic_root_implies_a_l3051_305102


namespace valid_arrangements_count_l3051_305193

def number_of_people : ℕ := 5

-- Define a function to calculate permutations
def permutations (n : ℕ) (k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / Nat.factorial (n - k)

theorem valid_arrangements_count :
  (permutations number_of_people number_of_people) -
  (permutations (number_of_people - 2) (number_of_people - 2)) -
  (permutations (number_of_people - 2) 1 * permutations (number_of_people - 2) (number_of_people - 2)) -
  (permutations (number_of_people - 1) 1 * permutations (number_of_people - 2) (number_of_people - 2)) = 72 := by
  sorry

end valid_arrangements_count_l3051_305193


namespace problem_solution_l3051_305114

-- Define the function f
def f (x : ℝ) := |2*x - 2| + |x + 2|

-- Define the theorem
theorem problem_solution :
  (∃ (S : Set ℝ), S = {x : ℝ | -3 ≤ x ∧ x ≤ 3/2} ∧ ∀ x, f x ≤ 6 - x ↔ x ∈ S) ∧
  (∃ (T : ℝ), T = 3 ∧ ∀ x, f x ≥ T) ∧
  (∀ a b c : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 3 → 1/a + 1/b + 4/c ≥ 16/3) :=
by sorry

end problem_solution_l3051_305114


namespace teacher_cai_running_stats_l3051_305110

/-- Represents the running distances for each day of the week -/
def running_distances : List Int := [460, 220, -250, -10, -330, 50, 560]

/-- The standard running distance per day in meters -/
def standard_distance : Nat := 3000

/-- The reward threshold in meters -/
def reward_threshold : Nat := 10000

theorem teacher_cai_running_stats :
  let max_distance := running_distances.maximum?
  let min_distance := running_distances.minimum?
  let total_distance := (running_distances.sum + standard_distance * running_distances.length)
  (∀ m n, m ∈ running_distances → n ∈ running_distances → m - n ≤ 890) ∧
  (total_distance = 21700) ∧
  (total_distance > reward_threshold) := by
  sorry


end teacher_cai_running_stats_l3051_305110


namespace tangent_line_equation_l3051_305165

-- Define the function f
def f (x : ℝ) : ℝ := x^4 - x

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  ∃ (m b : ℝ), (∀ (x y : ℝ), y = m * x + b ↔ m * x - y + b = 0) ∧
  (∀ (x : ℝ), (x - P.1) * (f x - P.2) ≤ m * (x - P.1)^2) ∧
  m * P.1 - P.2 + b = 0 ∧
  m = 3 ∧ b = -3 :=
sorry

end tangent_line_equation_l3051_305165


namespace trig_identities_l3051_305168

open Real

theorem trig_identities (α x : ℝ) (h : tan α = 2) :
  (2 * sin α - cos α) / (sin α + 2 * cos α) = 3/4 ∧
  2 * sin x ^ 2 - sin x * cos x + cos x ^ 2 = 2 - sin (2 * x) / 2 := by
  sorry

end trig_identities_l3051_305168


namespace share_multiple_l3051_305184

theorem share_multiple (total a b c k : ℕ) : 
  total = 585 →
  c = 260 →
  4 * a = k * b →
  4 * a = 3 * c →
  a + b + c = total →
  k = 6 :=
by sorry

end share_multiple_l3051_305184


namespace expand_product_l3051_305147

theorem expand_product (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := by
  sorry

end expand_product_l3051_305147


namespace ellipse_k_range_l3051_305179

/-- The ellipse equation -/
def ellipse (k x y : ℝ) : ℝ := k^2 * x^2 + y^2 - 4*k*x + 2*k*y + k^2 - 1

/-- Theorem stating the range of k for the given ellipse -/
theorem ellipse_k_range :
  ∀ k : ℝ, (∀ x y : ℝ, ellipse k x y = 0 → ellipse k 0 0 < 0) →
  (0 < |k| ∧ |k| < 1) :=
sorry

end ellipse_k_range_l3051_305179


namespace badger_walnuts_l3051_305105

theorem badger_walnuts (badger_walnuts_per_hole fox_walnuts_per_hole : ℕ)
  (h_badger_walnuts : badger_walnuts_per_hole = 5)
  (h_fox_walnuts : fox_walnuts_per_hole = 7)
  (h_hole_diff : ℕ)
  (h_hole_diff_value : h_hole_diff = 2)
  (badger_holes fox_holes : ℕ)
  (h_holes_relation : badger_holes = fox_holes + h_hole_diff)
  (total_walnuts : ℕ)
  (h_total_equality : badger_walnuts_per_hole * badger_holes = fox_walnuts_per_hole * fox_holes)
  (h_total_walnuts : total_walnuts = badger_walnuts_per_hole * badger_holes) :
  total_walnuts = 35 :=
by sorry

end badger_walnuts_l3051_305105


namespace snooker_ticket_difference_l3051_305145

/-- Proves the difference in ticket sales for a snooker tournament --/
theorem snooker_ticket_difference :
  ∀ (vip_tickets general_tickets : ℕ),
  vip_tickets + general_tickets = 320 →
  45 * vip_tickets + 20 * general_tickets = 7500 →
  general_tickets - vip_tickets = 232 :=
by
  sorry

end snooker_ticket_difference_l3051_305145


namespace xyz_sum_l3051_305178

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x^2 + x*y + y^2 = 147)
  (eq2 : y^2 + y*z + z^2 = 9)
  (eq3 : z^2 + x*z + x^2 = 156) :
  x*y + y*z + x*z = 42 := by
sorry

end xyz_sum_l3051_305178


namespace increasing_sequence_count_remainder_mod_1000_l3051_305149

def sequence_count (n : ℕ) (k : ℕ) (max : ℕ) : ℕ :=
  Nat.choose (n + k - 1) k

theorem increasing_sequence_count : 
  sequence_count 998 12 2008 = Nat.choose 1009 12 :=
sorry

theorem remainder_mod_1000 : 
  1009 % 1000 = 9 :=
sorry

end increasing_sequence_count_remainder_mod_1000_l3051_305149


namespace tree_planting_ratio_l3051_305101

theorem tree_planting_ratio (initial_mahogany : ℕ) (initial_narra : ℕ) 
  (total_fallen : ℕ) (final_trees : ℕ) :
  initial_mahogany = 50 →
  initial_narra = 30 →
  total_fallen = 5 →
  final_trees = 88 →
  ∃ (fallen_narra fallen_mahogany planted : ℕ),
    fallen_narra + fallen_mahogany = total_fallen ∧
    fallen_mahogany = fallen_narra + 1 ∧
    planted = final_trees - (initial_mahogany + initial_narra - total_fallen) ∧
    planted * 2 = fallen_narra * 13 :=
by sorry

end tree_planting_ratio_l3051_305101


namespace temperature_is_dependent_variable_l3051_305196

/-- Represents a variable in the solar water heating process -/
inductive Variable
  | Temperature
  | Duration
  | Intensity
  | Heater

/-- Represents the relationship between variables in the solar water heating process -/
structure SolarWaterHeating where
  temp : Variable
  duration : Variable
  changes_with : Variable → Variable → Prop

/-- Definition of a dependent variable -/
def is_dependent_variable (v : Variable) (swh : SolarWaterHeating) : Prop :=
  ∃ (other : Variable), swh.changes_with v other

/-- Theorem stating that the temperature is the dependent variable in the solar water heating process -/
theorem temperature_is_dependent_variable (swh : SolarWaterHeating) 
  (h1 : swh.temp = Variable.Temperature)
  (h2 : swh.duration = Variable.Duration)
  (h3 : swh.changes_with swh.temp swh.duration) :
  is_dependent_variable swh.temp swh :=
by sorry


end temperature_is_dependent_variable_l3051_305196


namespace max_mn_value_l3051_305173

theorem max_mn_value (m n : ℝ) : 
  m > 0 → n > 0 → m * 2 - 1 + n = 0 → m * n ≤ 1/8 := by
  sorry

end max_mn_value_l3051_305173


namespace product_sampling_is_srs_l3051_305142

/-- Represents a sampling method --/
structure SamplingMethod where
  total : Nat
  sample_size : Nat
  method : String

/-- Defines what constitutes a simple random sample --/
def is_simple_random_sample (sm : SamplingMethod) : Prop :=
  sm.sample_size ≤ sm.total ∧
  sm.method = "drawing lots" ∧
  ∀ (subset : Finset (Fin sm.total)),
    subset.card = sm.sample_size →
    ∃ (p : ℝ), p > 0 ∧ p = 1 / (Nat.choose sm.total sm.sample_size)

/-- The sampling method described in the problem --/
def product_sampling : SamplingMethod :=
  { total := 10
  , sample_size := 3
  , method := "drawing lots" }

/-- Theorem stating that the product sampling method is a simple random sample --/
theorem product_sampling_is_srs : is_simple_random_sample product_sampling := by
  sorry


end product_sampling_is_srs_l3051_305142


namespace min_value_of_expression_l3051_305118

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b - 4*a*b = 0) :
  ∀ x y, x > 0 → y > 0 → x + 2*y - 4*x*y = 0 → a + 8*b ≤ x + 8*y ∧ ∃ a b, a > 0 ∧ b > 0 ∧ a + 2*b - 4*a*b = 0 ∧ a + 8*b = 9/2 :=
by sorry

end min_value_of_expression_l3051_305118


namespace mask_production_growth_rate_equation_l3051_305136

/-- Represents the monthly average growth rate of mask production from January to March. -/
def monthly_growth_rate : ℝ → Prop :=
  λ x => (160000 : ℝ) * (1 + x)^2 = 250000

/-- The equation representing the monthly average growth rate of mask production
    from January to March is 16(1+x)^2 = 25, given initial production of 160,000 masks
    in January and 250,000 masks in March. -/
theorem mask_production_growth_rate_equation :
  ∃ x : ℝ, monthly_growth_rate x ∧ 16 * (1 + x)^2 = 25 :=
sorry

end mask_production_growth_rate_equation_l3051_305136


namespace sqrt_x_plus_2_meaningful_l3051_305175

theorem sqrt_x_plus_2_meaningful (x : ℝ) : Real.sqrt (x + 2) ≥ 0 ↔ x ≥ -2 := by sorry

end sqrt_x_plus_2_meaningful_l3051_305175


namespace valid_arrangement_exists_l3051_305115

/-- Represents a tile with a diagonal --/
inductive Tile
| TopLeftToBottomRight
| TopRightToBottomLeft

/-- Represents a position in the 6×6 grid --/
structure Position :=
  (x : Fin 6)
  (y : Fin 6)

/-- Represents an arrangement of tiles in the 6×6 grid --/
def Arrangement := Position → Option Tile

/-- Checks if two positions are adjacent --/
def adjacent (p1 p2 : Position) : Bool :=
  (p1.x = p2.x ∧ (p1.y.val + 1 = p2.y.val ∨ p2.y.val + 1 = p1.y.val)) ∨
  (p1.y = p2.y ∧ (p1.x.val + 1 = p2.x.val ∨ p2.x.val + 1 = p1.x.val))

/-- Checks if the arrangement is valid --/
def validArrangement (arr : Arrangement) : Prop :=
  (∀ p : Position, ∃ t : Tile, arr p = some t) ∧
  (∀ p1 p2 : Position, adjacent p1 p2 → arr p1 ≠ arr p2)

/-- The main theorem stating that a valid arrangement exists --/
theorem valid_arrangement_exists : ∃ arr : Arrangement, validArrangement arr :=
sorry


end valid_arrangement_exists_l3051_305115


namespace quadratic_inequality_solution_set_l3051_305187

theorem quadratic_inequality_solution_set (d : ℝ) : 
  (d > 0 ∧ ∃ x : ℝ, x^2 - 8*x + d < 0) ↔ 0 < d ∧ d < 16 :=
by sorry

end quadratic_inequality_solution_set_l3051_305187


namespace a_fourth_zero_implies_a_squared_zero_l3051_305144

theorem a_fourth_zero_implies_a_squared_zero (A : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : A ^ 4 = 0) : A ^ 2 = 0 := by
  sorry

end a_fourth_zero_implies_a_squared_zero_l3051_305144


namespace distance_sum_l3051_305161

/-- Represents a segment with a midpoint and a point Q -/
structure Segment where
  length : ℝ
  q_distance : ℝ

/-- The problem setup -/
def problem_setup (cd : Segment) (cd_prime : Segment) : Prop :=
  cd.length = 10 ∧
  cd_prime.length = 16 ∧
  cd.q_distance = 3 ∧
  cd.q_distance = 2 * (cd_prime.length / 2 - (cd_prime.length / 2 - cd_prime.q_distance))

/-- The theorem to prove -/
theorem distance_sum (cd : Segment) (cd_prime : Segment) 
  (h : problem_setup cd cd_prime) : 
  cd.q_distance + cd_prime.q_distance = 7 := by
  sorry


end distance_sum_l3051_305161


namespace valid_q_values_are_zero_two_neg_two_l3051_305121

/-- Given a set of 10 distinct real numbers, this function determines the values of q
    such that every number in the second line is also in the third line. -/
def valid_q_values (napkin : Finset ℝ) : Set ℝ :=
  { q : ℝ | ∀ (a b c d : ℝ), a ∈ napkin → b ∈ napkin → c ∈ napkin → d ∈ napkin →
    ∃ (w x y z : ℝ), w ∈ napkin ∧ x ∈ napkin ∧ y ∈ napkin ∧ z ∈ napkin →
    q * (a - b) * (c - d) = (w - x)^2 + (y - z)^2 - (x - y)^2 - (z - w)^2 }

/-- Theorem stating that for any set of 10 distinct real numbers, 
    the only valid q values are 0, 2, and -2. -/
theorem valid_q_values_are_zero_two_neg_two (napkin : Finset ℝ) 
  (h : napkin.card = 10) :
  valid_q_values napkin = {0, 2, -2} := by
  sorry


end valid_q_values_are_zero_two_neg_two_l3051_305121


namespace last_digit_sum_powers_l3051_305129

theorem last_digit_sum_powers : (3^1991 + 1991^3) % 10 = 8 := by sorry

end last_digit_sum_powers_l3051_305129


namespace factor_expression_l3051_305180

theorem factor_expression (x : ℝ) : 3*x*(x+1) + 7*(x+1) = (3*x+7)*(x+1) := by
  sorry

end factor_expression_l3051_305180


namespace index_cards_per_pack_l3051_305123

-- Define the given conditions
def cards_per_student : ℕ := 10
def periods_per_day : ℕ := 6
def students_per_class : ℕ := 30
def cost_per_pack : ℕ := 3
def total_spent : ℕ := 108

-- Define the theorem
theorem index_cards_per_pack :
  (cards_per_student * periods_per_day * students_per_class) / (total_spent / cost_per_pack) = 50 := by
  sorry

end index_cards_per_pack_l3051_305123


namespace inequality_proof_l3051_305155

theorem inequality_proof (x y : ℝ) (h : x * y < 0) :
  x^4 / y^4 + y^4 / x^4 - x^2 / y^2 - y^2 / x^2 + x / y + y / x ≥ 2 := by
  sorry

end inequality_proof_l3051_305155


namespace least_possible_b_value_l3051_305185

/-- The number of factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- The least possible value of b satisfying the given conditions -/
def least_b : ℕ := 42

theorem least_possible_b_value (a b : ℕ+) 
  (ha : num_factors a = 4)
  (hb : num_factors b = a.val)
  (hd : a.val + 1 ∣ b) :
  b ≥ least_b ∧ ∃ (b' : ℕ+), b' = least_b ∧ 
    num_factors b' = a.val ∧ 
    a.val + 1 ∣ b' := by sorry

end least_possible_b_value_l3051_305185


namespace promotion_savings_l3051_305190

/-- Represents a shoe promotion strategy -/
inductive Promotion
  | A
  | B

/-- Calculates the total cost for two pairs of shoes given a promotion -/
def calculateCost (originalPrice : ℕ) (promo : Promotion) : ℕ :=
  match promo with
  | Promotion.A => originalPrice + originalPrice / 2
  | Promotion.B => originalPrice + originalPrice - 15

/-- Calculates the savings from using one promotion over another -/
def calculateSavings (originalPrice : ℕ) (promo1 promo2 : Promotion) : ℕ :=
  calculateCost originalPrice promo2 - calculateCost originalPrice promo1

theorem promotion_savings :
  calculateSavings 50 Promotion.A Promotion.B = 10 := by
  sorry

#eval calculateSavings 50 Promotion.A Promotion.B

end promotion_savings_l3051_305190


namespace sqrt_two_minus_one_power_l3051_305135

theorem sqrt_two_minus_one_power (n : ℕ) (hn : n > 0) :
  ∃ (k : ℕ), k > 1 ∧ (Real.sqrt 2 - 1) ^ n = Real.sqrt k - Real.sqrt (k - 1) :=
by sorry

end sqrt_two_minus_one_power_l3051_305135


namespace factor_theorem_application_l3051_305119

-- Define the polynomial P(x)
def P (c : ℝ) (x : ℝ) : ℝ := x^3 + 2*x^2 + c*x + 10

-- Theorem statement
theorem factor_theorem_application (c : ℝ) :
  (∀ x, P c x = 0 ↔ x = 5) → c = -37 := by
  sorry

end factor_theorem_application_l3051_305119


namespace quadratic_max_value_l3051_305169

/-- The quadratic function y = -(x-m)^2 + m^2 + 1 has a maximum value of 4 when -2 ≤ x ≤ 1. -/
theorem quadratic_max_value (m : ℝ) : 
  (∀ x, -2 ≤ x ∧ x ≤ 1 → -(x-m)^2 + m^2 + 1 ≤ 4) ∧ 
  (∃ x, -2 ≤ x ∧ x ≤ 1 ∧ -(x-m)^2 + m^2 + 1 = 4) →
  m = 2 ∨ m = -Real.sqrt 3 :=
sorry

end quadratic_max_value_l3051_305169


namespace gcd_problems_l3051_305158

theorem gcd_problems :
  (Nat.gcd 840 1764 = 84) ∧ (Nat.gcd 440 556 = 4) := by sorry

end gcd_problems_l3051_305158


namespace straight_part_length_l3051_305103

/-- Represents a river with straight and crooked parts -/
structure River where
  total_length : ℝ
  straight_length : ℝ
  crooked_length : ℝ

/-- The condition that the straight part is three times shorter than the crooked part -/
def straight_three_times_shorter (r : River) : Prop :=
  r.straight_length = r.crooked_length / 3

/-- The theorem stating the length of the straight part given the conditions -/
theorem straight_part_length (r : River) 
  (h1 : r.total_length = 80)
  (h2 : r.total_length = r.straight_length + r.crooked_length)
  (h3 : straight_three_times_shorter r) : 
  r.straight_length = 20 := by
  sorry

end straight_part_length_l3051_305103


namespace line_l_properties_l3051_305148

-- Define the line l: ax + y + a = 0
def line_l (a : ℝ) (x y : ℝ) : Prop := a * x + y + a = 0

-- Theorem statement
theorem line_l_properties (a : ℝ) :
  -- 1. The line passes through the point (-1, 0)
  line_l a (-1) 0 ∧
  -- 2. When a = -1, the line is perpendicular to x + y - 2 = 0
  (a = -1 → ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ ≠ x₂ ∧
    line_l (-1) x₁ y₁ ∧
    line_l (-1) x₂ y₂ ∧
    x₁ + y₁ - 2 = 0 ∧
    x₂ + y₂ - 2 = 0 ∧
    (y₂ - y₁) * (x₂ - x₁) = -1) ∧
  -- 3. When a > 0, the line passes through the second, third, and fourth quadrants
  (a > 0 → ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    line_l a x₁ y₁ ∧ x₁ < 0 ∧ y₁ > 0 ∧  -- Second quadrant
    line_l a x₂ y₂ ∧ x₂ < 0 ∧ y₂ < 0 ∧  -- Third quadrant
    line_l a x₃ y₃ ∧ x₃ > 0 ∧ y₃ < 0)   -- Fourth quadrant
  := by sorry

end line_l_properties_l3051_305148


namespace line_parabola_intersection_l3051_305162

/-- The line x = k intersects the parabola x = -3y^2 - 4y + 7 at exactly one point if and only if k = 25/3 -/
theorem line_parabola_intersection (k : ℝ) : 
  (∃! y : ℝ, k = -3 * y^2 - 4 * y + 7) ↔ k = 25/3 := by
  sorry

end line_parabola_intersection_l3051_305162


namespace triangle_height_l3051_305188

theorem triangle_height (base : ℝ) (area : ℝ) (height : ℝ) : 
  base = 4.5 → area = 13.5 → area = (base * height) / 2 → height = 6 := by
sorry

end triangle_height_l3051_305188


namespace circle_and_line_problem_l3051_305124

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define point D
def point_D : ℝ × ℝ := (-2, 0)

-- Define the line l passing through D
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Define the isosceles right triangle condition
def is_isosceles_right_triangle (A B C : ℝ × ℝ) : Prop :=
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 2 * ((A.1 - C.1)^2 + (A.2 - C.2)^2)

-- Define the intersection points condition
def are_intersection_points (A B : ℝ × ℝ) (k : ℝ) : Prop :=
  circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
  line_l k A.1 A.2 ∧ line_l k B.1 B.2

-- Theorem statement
theorem circle_and_line_problem :
  ∃ (A B C : ℝ × ℝ) (k : ℝ),
    C = (0, 4) ∧
    are_intersection_points A B k ∧
    is_isosceles_right_triangle A B C →
    (∀ (x y : ℝ), (x + 1)^2 + (y - 2)^2 = 5 ↔ 
      ∃ (P Q : ℝ × ℝ), P = C ∧ Q = point_D ∧ 
      (x - (P.1 + Q.1)/2)^2 + (y - (P.2 + Q.2)/2)^2 = 
      ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) / 4) ∧
    (k = 1 ∨ k = 7) :=
sorry

end circle_and_line_problem_l3051_305124


namespace pyramid_with_14_edges_has_8_vertices_l3051_305151

/-- A pyramid is a polyhedron with a polygonal base and triangular faces meeting at a common point (apex) --/
structure Pyramid where
  num_edges : ℕ

/-- The number of vertices in a pyramid --/
def num_vertices (p : Pyramid) : ℕ :=
  (p.num_edges / 2) + 2

theorem pyramid_with_14_edges_has_8_vertices (p : Pyramid) (h : p.num_edges = 14) : 
  num_vertices p = 8 := by
  sorry

#check pyramid_with_14_edges_has_8_vertices

end pyramid_with_14_edges_has_8_vertices_l3051_305151


namespace quadratic_equations_solutions_l3051_305128

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 + 4*x - 5 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 - 3*x + 1 = 0
  let solutions1 : Set ℝ := {1, -5}
  let solutions2 : Set ℝ := {(3 + Real.sqrt 5) / 2, (3 - Real.sqrt 5) / 2}
  (∀ x ∈ solutions1, eq1 x) ∧ (∀ y, eq1 y → y ∈ solutions1) ∧
  (∀ x ∈ solutions2, eq2 x) ∧ (∀ y, eq2 y → y ∈ solutions2) :=
by sorry


end quadratic_equations_solutions_l3051_305128


namespace aladdin_theorem_l3051_305153

/-- A continuous function that takes all values in [0, 1) -/
def AllValuesContinuousFunction (φ : ℝ → ℝ) : Prop :=
  Continuous φ ∧ ∀ y ∈ Set.Iio 1, ∃ t, φ t = y

/-- The difference between max and min of an AllValuesContinuousFunction is at least 1 -/
theorem aladdin_theorem (φ : ℝ → ℝ) (h : AllValuesContinuousFunction φ) :
    ⨆ t, φ t - ⨅ t, φ t ≥ 1 := by
  sorry

end aladdin_theorem_l3051_305153


namespace tangent_parabola_difference_l3051_305138

/-- Given a parabola y = x^2 + ax + b and a tangent line y = kx + 1 at point (1, 3),
    prove that a - b = -2 -/
theorem tangent_parabola_difference (a b k : ℝ) : 
  (∀ x, x^2 + a*x + b = k*x + 1 → 2*x + a = k) →  -- Tangency condition
  1^2 + a*1 + b = 3 →                             -- Point (1, 3) is on the parabola
  k*1 + 1 = 3 →                                   -- Point (1, 3) is on the tangent line
  a - b = -2 := by
  sorry

end tangent_parabola_difference_l3051_305138


namespace complex_fraction_real_l3051_305143

theorem complex_fraction_real (a : ℝ) : (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (((a : ℂ) - Complex.I) / ((2 : ℂ) + Complex.I)).im = 0 → a = -2 := by
  sorry

end complex_fraction_real_l3051_305143


namespace temperature_range_l3051_305112

/-- Given the highest and lowest temperatures on a day, 
    prove that any temperature on that day lies within this range -/
theorem temperature_range (t : ℝ) (highest lowest : ℝ) 
  (h_highest : highest = 5)
  (h_lowest : lowest = -2)
  (h_t_le_highest : t ≤ highest)
  (h_t_ge_lowest : t ≥ lowest) : 
  -2 ≤ t ∧ t ≤ 5 := by
  sorry

end temperature_range_l3051_305112


namespace swimming_problem_l3051_305164

structure Triangle :=
  (A B C : ℝ × ℝ)

def isEquilateral (t : Triangle) : Prop :=
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = d t.B t.C ∧ d t.B t.C = d t.C t.A ∧ d t.C t.A = d t.A t.B

def isWestOf (p q : ℝ × ℝ) : Prop :=
  p.1 < q.1 ∧ p.2 = q.2

def swimmingPath (A B : ℝ × ℝ) (x y : ℕ) : Prop :=
  ∃ P : ℝ × ℝ, 
    (P.1 - A.1)^2 + (P.2 - A.2)^2 = x^2 ∧
    P.1 = B.1 + y ∧
    P.2 = B.2

theorem swimming_problem (t : Triangle) (x y : ℕ) :
  isEquilateral t →
  isWestOf t.B t.C →
  let d := λ p q : ℝ × ℝ => Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d t.A t.B = 86 →
  swimmingPath t.A t.B x y →
  x > 0 →
  y > 0 →
  y = 6 :=
by sorry

end swimming_problem_l3051_305164


namespace three_squares_side_length_l3051_305139

theorem three_squares_side_length (x : ℝ) :
  let middle := x + 17
  let right := middle - 6
  x + middle + right = 52 →
  x = 8 := by
sorry

end three_squares_side_length_l3051_305139


namespace x_plus_reciprocal_squared_l3051_305108

theorem x_plus_reciprocal_squared (x : ℝ) (h : x^2 + 1/x^2 = 7) : (x + 1/x)^2 = 9 := by
  sorry

end x_plus_reciprocal_squared_l3051_305108


namespace system2_solution_l3051_305106

variable (a b : ℝ)

-- Define the first system of equations and its solution
def system1_eq1 (x y : ℝ) : Prop := a * x - b * y = 3
def system1_eq2 (x y : ℝ) : Prop := a * x + b * y = 5
def system1_solution (x y : ℝ) : Prop := x = 2 ∧ y = 1

-- Define the second system of equations
def system2_eq1 (m n : ℝ) : Prop := a * (m + 2 * n) - 2 * b * n = 6
def system2_eq2 (m n : ℝ) : Prop := a * (m + 2 * n) + 2 * b * n = 10

-- State the theorem
theorem system2_solution :
  (∃ x y, system1_eq1 a b x y ∧ system1_eq2 a b x y ∧ system1_solution x y) →
  (∃ m n, system2_eq1 a b m n ∧ system2_eq2 a b m n ∧ m = 2 ∧ n = 1) :=
by sorry

end system2_solution_l3051_305106


namespace line_equation_final_line_equation_l3051_305137

/-- The ellipse with equation x²/9 + y²/4 = 1 -/
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The line passing through (1, 1) with slope k -/
def line (x y k : ℝ) : Prop := y = k * (x - 1) + 1

/-- The point (1, 1) is on the line -/
def point_on_line (k : ℝ) : Prop := line 1 1 k

/-- The point (1, 1) is the midpoint of the chord intercepted by the line from the ellipse -/
def is_midpoint (k : ℝ) : Prop :=
  ∃ x₁ x₂ y₁ y₂ : ℝ,
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    line x₁ y₁ k ∧ line x₂ y₂ k ∧
    (x₁ + x₂) / 2 = 1 ∧ (y₁ + y₂) / 2 = 1

theorem line_equation :
  ∀ k : ℝ, point_on_line k → is_midpoint k → k = -4/9 :=
sorry

theorem final_line_equation :
  ∀ x y : ℝ, line x y (-4/9) ↔ 4*x + 9*y = 13 :=
sorry

end line_equation_final_line_equation_l3051_305137


namespace six_balls_three_boxes_l3051_305130

/-- The number of ways to distribute n indistinguishable balls into k distinguishable boxes -/
def distribute_balls (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The problem statement -/
theorem six_balls_three_boxes : distribute_balls 6 3 = 28 := by
  sorry

end six_balls_three_boxes_l3051_305130


namespace section_b_students_l3051_305126

/-- Proves that given the conditions of the class sections, the number of students in section B is 20 -/
theorem section_b_students (students_a : ℕ) (avg_weight_a : ℚ) (avg_weight_b : ℚ) (avg_weight_total : ℚ) :
  students_a = 40 →
  avg_weight_a = 50 →
  avg_weight_b = 40 →
  avg_weight_total = 467/10 →
  ∃ (students_b : ℕ), students_b = 20 ∧
    (students_a * avg_weight_a + students_b * avg_weight_b) / (students_a + students_b) = avg_weight_total :=
by sorry


end section_b_students_l3051_305126


namespace bow_count_l3051_305192

theorem bow_count (red_fraction : ℚ) (blue_fraction : ℚ) (yellow_fraction : ℚ) (green_fraction : ℚ) 
  (white_count : ℕ) :
  red_fraction = 1/6 →
  blue_fraction = 1/3 →
  yellow_fraction = 1/12 →
  green_fraction = 1/8 →
  red_fraction + blue_fraction + yellow_fraction + green_fraction + (white_count : ℚ)/144 = 1 →
  white_count = 42 →
  144 = red_fraction * 144 + blue_fraction * 144 + yellow_fraction * 144 + green_fraction * 144 + white_count :=
by sorry

end bow_count_l3051_305192


namespace last_three_digits_of_8_to_1000_l3051_305183

theorem last_three_digits_of_8_to_1000 (h : 8^125 ≡ 2 [ZMOD 1250]) :
  8^1000 ≡ 256 [ZMOD 1000] := by
  sorry

end last_three_digits_of_8_to_1000_l3051_305183


namespace professor_newton_students_l3051_305154

theorem professor_newton_students (total : ℕ) (male : ℕ) (female : ℕ) : 
  total % 4 = 2 →
  total % 5 = 1 →
  female = 15 →
  female > male →
  total = male + female →
  male = 11 := by
sorry

end professor_newton_students_l3051_305154


namespace isoelectronic_pairs_l3051_305166

/-- Represents a molecule with its composition and valence electron count -/
structure Molecule where
  composition : List (Nat × Nat)  -- List of (atomic number, count) pairs
  valence_electrons : Nat

/-- Calculates the total number of valence electrons for a molecule -/
def calculate_valence_electrons (composition : List (Nat × Nat)) : Nat :=
  composition.foldl (fun acc (atomic_number, count) => 
    acc + count * match atomic_number with
      | 6 => 4  -- Carbon
      | 7 => 5  -- Nitrogen
      | 8 => 6  -- Oxygen
      | 16 => 6 -- Sulfur
      | _ => 0
    ) 0

/-- Determines if two molecules are isoelectronic -/
def are_isoelectronic (m1 m2 : Molecule) : Prop :=
  m1.valence_electrons = m2.valence_electrons

/-- N2 molecule -/
def N2 : Molecule := ⟨[(7, 2)], calculate_valence_electrons [(7, 2)]⟩

/-- CO molecule -/
def CO : Molecule := ⟨[(6, 1), (8, 1)], calculate_valence_electrons [(6, 1), (8, 1)]⟩

/-- N2O molecule -/
def N2O : Molecule := ⟨[(7, 2), (8, 1)], calculate_valence_electrons [(7, 2), (8, 1)]⟩

/-- CO2 molecule -/
def CO2 : Molecule := ⟨[(6, 1), (8, 2)], calculate_valence_electrons [(6, 1), (8, 2)]⟩

/-- NO2⁻ ion -/
def NO2_minus : Molecule := ⟨[(7, 1), (8, 2)], calculate_valence_electrons [(7, 1), (8, 2)] + 1⟩

/-- SO2 molecule -/
def SO2 : Molecule := ⟨[(16, 1), (8, 2)], calculate_valence_electrons [(16, 1), (8, 2)]⟩

/-- O3 molecule -/
def O3 : Molecule := ⟨[(8, 3)], calculate_valence_electrons [(8, 3)]⟩

theorem isoelectronic_pairs : 
  (are_isoelectronic N2 CO) ∧ 
  (are_isoelectronic N2O CO2) ∧ 
  (are_isoelectronic NO2_minus SO2) ∧ 
  (are_isoelectronic NO2_minus O3) := by
  sorry

end isoelectronic_pairs_l3051_305166


namespace race_finish_time_difference_l3051_305198

/-- Race problem statement -/
theorem race_finish_time_difference 
  (race_distance : ℝ) 
  (malcolm_speed : ℝ) 
  (joshua_speed : ℝ) 
  (h1 : race_distance = 12) 
  (h2 : malcolm_speed = 7) 
  (h3 : joshua_speed = 9) : 
  joshua_speed * race_distance - malcolm_speed * race_distance = 24 := by
  sorry

end race_finish_time_difference_l3051_305198


namespace arithmetic_sequence_common_difference_l3051_305116

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_a2 : a 2 = 3) 
  (h_a7 : a 7 = 13) : 
  ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) - a n = d :=
by sorry

end arithmetic_sequence_common_difference_l3051_305116


namespace calculator_result_is_very_large_l3051_305122

/-- The calculator function that replaces x with x^2 - 2 -/
def calc_function (x : ℝ) : ℝ := x^2 - 2

/-- Applies the calculator function n times to the initial value -/
def apply_n_times (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0 => x
  | n + 1 => calc_function (apply_n_times n x)

/-- A number is considered "very large" if it's greater than 10^100 -/
def is_very_large (x : ℝ) : Prop := x > 10^100

/-- Theorem stating that after 50 applications of the calculator function starting from 3, the result is very large -/
theorem calculator_result_is_very_large : 
  is_very_large (apply_n_times 50 3) := by sorry

end calculator_result_is_very_large_l3051_305122


namespace freshman_class_size_l3051_305167

theorem freshman_class_size :
  ∃! n : ℕ, n < 500 ∧ n % 19 = 17 ∧ n % 18 = 9 ∧ n = 207 := by
  sorry

end freshman_class_size_l3051_305167


namespace equation_solution_l3051_305100

theorem equation_solution : ∃ y : ℝ, (125 : ℝ)^(3*y) = 25^(4*y - 5) ∧ y = -10 := by
  sorry

end equation_solution_l3051_305100


namespace expression_equality_l3051_305120

theorem expression_equality : (3^1003 + 5^1003)^2 - (3^1003 - 5^1003)^2 = 4 * 15^1003 := by
  sorry

end expression_equality_l3051_305120


namespace negative_abs_not_equal_five_l3051_305127

theorem negative_abs_not_equal_five : -|(-5 : ℤ)| ≠ 5 := by
  sorry

end negative_abs_not_equal_five_l3051_305127


namespace random_events_charge_attraction_certain_no_impossible_events_l3051_305159

-- Define the type for events
inductive Event
| GlassCups
| CannonFiring
| PhoneNumber
| ChargeAttraction
| LotteryWin

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  match e with
  | Event.GlassCups => true
  | Event.CannonFiring => true
  | Event.PhoneNumber => true
  | Event.ChargeAttraction => false
  | Event.LotteryWin => true

-- Theorem stating which events are random
theorem random_events :
  (isRandomEvent Event.GlassCups) ∧
  (isRandomEvent Event.CannonFiring) ∧
  (isRandomEvent Event.PhoneNumber) ∧
  (¬isRandomEvent Event.ChargeAttraction) ∧
  (isRandomEvent Event.LotteryWin) :=
by sorry

-- Definition of a certain event
def isCertainEvent (e : Event) : Prop :=
  match e with
  | Event.ChargeAttraction => true
  | _ => false

-- Theorem stating that charge attraction is a certain event
theorem charge_attraction_certain :
  isCertainEvent Event.ChargeAttraction :=
by sorry

-- Definition of an impossible event
def isImpossibleEvent (e : Event) : Prop := false

-- Theorem stating that none of the given events are impossible
theorem no_impossible_events :
  ∀ e : Event, ¬(isImpossibleEvent e) :=
by sorry

end random_events_charge_attraction_certain_no_impossible_events_l3051_305159


namespace last_integer_in_sequence_l3051_305176

def sequence_term (n : ℕ) : ℚ :=
  (1024000 : ℚ) / (4 ^ n)

def is_integer (q : ℚ) : Prop :=
  ∃ (z : ℤ), q = z

theorem last_integer_in_sequence :
  ∃ (n : ℕ), (is_integer (sequence_term n) ∧ sequence_term n = 250) ∧
             ∀ (m : ℕ), m > n → ¬ is_integer (sequence_term m) :=
by sorry

end last_integer_in_sequence_l3051_305176


namespace degree_of_composed_product_l3051_305107

-- Define polynomials f and g with their respective degrees
def f : Polynomial ℝ := sorry
def g : Polynomial ℝ := sorry

-- State the theorem
theorem degree_of_composed_product :
  (Polynomial.degree f = 4) →
  (Polynomial.degree g = 5) →
  Polynomial.degree (f.comp (Polynomial.X ^ 2) * g.comp (Polynomial.X ^ 4)) = 28 := by
  sorry

end degree_of_composed_product_l3051_305107


namespace expression_simplification_l3051_305177

theorem expression_simplification (a : ℝ) (h1 : a ≠ 1) (h2 : a ≠ 1/2) :
  1 - (1 / (1 - a / (1 - a))) = -a / (1 - 2*a) :=
by sorry

end expression_simplification_l3051_305177


namespace common_root_condition_l3051_305113

theorem common_root_condition (m : ℝ) : 
  (∃ x : ℝ, m * x - 1000 = 1001 ∧ 1001 * x = m - 1000 * x) ↔ (m = 2001 ∨ m = -2001) := by
  sorry

end common_root_condition_l3051_305113


namespace inequality_solution_range_l3051_305132

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, |x + 1| - |x - 2| < a^2 - 4*a) → 
  (a < 1 ∨ a > 3) := by
  sorry

end inequality_solution_range_l3051_305132


namespace limit_rational_function_l3051_305199

/-- The limit of (x^3 - 3x - 2) / (x - 2) as x approaches 2 is 9 -/
theorem limit_rational_function : 
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 2| ∧ |x - 2| < δ → 
    |(x^3 - 3*x - 2) / (x - 2) - 9| < ε := by
  sorry

end limit_rational_function_l3051_305199


namespace quadratic_equation_solution_l3051_305111

theorem quadratic_equation_solution (a b : ℝ) : 
  ∃ x : ℝ, (a^2 - b^2) * x^2 + 2 * (a^3 - b^3) * x + (a^4 - b^4) = 0 := by
  sorry

end quadratic_equation_solution_l3051_305111


namespace hoseok_friends_left_l3051_305133

theorem hoseok_friends_left (total : ℕ) (right : ℕ) (h1 : total = 16) (h2 : right = 8) :
  total - (right + 1) = 7 := by
  sorry

end hoseok_friends_left_l3051_305133


namespace factor_tree_proof_l3051_305160

theorem factor_tree_proof (X Y Z F : ℕ) : 
  X = Y * Z → 
  Y = 7 * 11 → 
  Z = 7 * F → 
  F = 11 * 2 → 
  X = 11858 := by
  sorry

end factor_tree_proof_l3051_305160


namespace geometric_sequence_partial_sum_l3051_305197

/-- Given a geometric sequence {a_n} with S_2 = 7 and S_6 = 91, prove that S_4 = 35 -/
theorem geometric_sequence_partial_sum (a : ℕ → ℝ) (r : ℝ) :
  (∀ n, a (n + 1) = a n * r) →  -- Geometric sequence condition
  (a 1 + a 1 * r = 7) →         -- S_2 = 7
  (a 1 + a 1 * r + a 1 * r^2 + a 1 * r^3 + a 1 * r^4 + a 1 * r^5 = 91) →  -- S_6 = 91
  (a 1 + a 1 * r + a 1 * r^2 + a 1 * r^3 = 35) :=  -- S_4 = 35
by sorry

end geometric_sequence_partial_sum_l3051_305197


namespace saras_cake_price_l3051_305189

/-- Sara's cake selling problem -/
theorem saras_cake_price (cakes_per_day : ℕ) (working_days : ℕ) (weeks : ℕ) (total_revenue : ℕ) :
  cakes_per_day = 4 →
  working_days = 5 →
  weeks = 4 →
  total_revenue = 640 →
  total_revenue / (cakes_per_day * working_days * weeks) = 8 := by
  sorry

#check saras_cake_price

end saras_cake_price_l3051_305189


namespace wage_problem_solution_l3051_305195

/-- Represents daily wages -/
structure DailyWage where
  amount : ℝ
  amount_pos : amount > 0

/-- Represents a sum of money -/
def SumOfMoney : Type := ℝ

/-- Given conditions of the problem -/
structure WageProblem where
  S : SumOfMoney
  B : DailyWage
  C : DailyWage
  S_pays_C_24_days : S = 24 * C.amount
  S_pays_both_8_days : S = 8 * (B.amount + C.amount)

/-- The theorem to prove -/
theorem wage_problem_solution (p : WageProblem) : 
  p.S = 12 * p.B.amount := by sorry

end wage_problem_solution_l3051_305195


namespace alpha_and_function_range_l3051_305186

open Real

theorem alpha_and_function_range 
  (α : ℝ) 
  (h1 : 2 * sin α * tan α = 3) 
  (h2 : 0 < α) 
  (h3 : α < π) : 
  α = π / 3 ∧ 
  ∀ x ∈ Set.Icc 0 (π / 4), 
    -1 ≤ 4 * sin x * sin (x - α) ∧ 
    4 * sin x * sin (x - α) ≤ 0 := by
  sorry


end alpha_and_function_range_l3051_305186


namespace stating_intersection_points_count_l3051_305141

/-- 
Given a positive integer n ≥ 5 and n lines in a plane where:
- Exactly 3 lines are mutually parallel
- Any two lines that are not part of the 3 parallel lines are not parallel
- Any three lines do not intersect at a single point
This function calculates the number of intersection points.
-/
def intersectionPoints (n : ℕ) : ℕ :=
  (n^2 - n - 6) / 2

/-- 
Theorem stating that for n ≥ 5 lines in a plane satisfying the given conditions,
the number of intersection points is (n^2 - n - 6) / 2.
-/
theorem intersection_points_count (n : ℕ) (h : n ≥ 5) :
  let lines := n
  let parallel_lines := 3
  intersectionPoints n = (n^2 - n - 6) / 2 := by
  sorry

#eval intersectionPoints 5  -- Expected output: 7
#eval intersectionPoints 6  -- Expected output: 12
#eval intersectionPoints 10 -- Expected output: 42

end stating_intersection_points_count_l3051_305141


namespace mia_has_110_dollars_l3051_305117

def darwins_money : ℕ := 45

def mias_money : ℕ := 2 * darwins_money + 20

theorem mia_has_110_dollars : mias_money = 110 := by
  sorry

end mia_has_110_dollars_l3051_305117


namespace sum_of_sequences_l3051_305140

-- Define the arithmetic sequence
def arithmetic_sequence (a b : ℝ) : Prop :=
  ∃ d : ℝ, a = 6 + d ∧ b = 6 + 2*d ∧ 48 = 6 + 3*d

-- Define the geometric sequence
def geometric_sequence (c d : ℝ) : Prop :=
  ∃ r : ℝ, c = 6*r ∧ d = 6*r^2 ∧ 48 = 6*r^3

theorem sum_of_sequences (a b c d : ℝ) 
  (h1 : arithmetic_sequence a b) 
  (h2 : geometric_sequence c d) : 
  a + b + c + d = 90 := by
  sorry

end sum_of_sequences_l3051_305140


namespace initial_mean_calculation_l3051_305181

theorem initial_mean_calculation (n : ℕ) (wrong_value correct_value : ℝ) (corrected_mean : ℝ) :
  n = 40 ∧ wrong_value = 20 ∧ correct_value = 34 ∧ corrected_mean = 36.45 →
  ∃ initial_mean : ℝ, initial_mean = 36.1 ∧
    n * corrected_mean = n * initial_mean + (correct_value - wrong_value) :=
by sorry

end initial_mean_calculation_l3051_305181


namespace imaginary_part_of_z_l3051_305174

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + Complex.I) = 4) : 
  Complex.im z = -2 := by sorry

end imaginary_part_of_z_l3051_305174


namespace f_geq_one_iff_a_nonneg_l3051_305146

/-- The quadratic function f(x) = x^2 + 2ax + 2a + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 2*a + 1

/-- The theorem stating the range of a for which f(x) ≥ 1 for all x in [-1, 1] -/
theorem f_geq_one_iff_a_nonneg (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≥ 1) ↔ a ≥ 0 := by sorry

end f_geq_one_iff_a_nonneg_l3051_305146


namespace parabola_properties_l3051_305131

/-- Represents a quadratic function of the form f(x) = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Properties of a parabola -/
structure ParabolaProperties where
  opensDownward : Bool
  axisOfSymmetry : ℝ
  vertexX : ℝ
  vertexY : ℝ

/-- Compute the properties of a parabola given its quadratic function -/
def computeParabolaProperties (f : QuadraticFunction) : ParabolaProperties :=
  sorry

theorem parabola_properties (f : QuadraticFunction) 
  (h : f = QuadraticFunction.mk (-2) 4 8) :
  computeParabolaProperties f = 
    ParabolaProperties.mk true 1 1 10 := by sorry

end parabola_properties_l3051_305131


namespace neither_sufficient_nor_necessary_l3051_305163

theorem neither_sufficient_nor_necessary (p q : Prop) :
  ¬(((¬p ∧ ¬q) → (p ∨ q)) ∧ ((p ∨ q) → (¬p ∧ ¬q))) :=
by sorry

end neither_sufficient_nor_necessary_l3051_305163


namespace gcf_of_60_90_150_l3051_305156

theorem gcf_of_60_90_150 : Nat.gcd 60 (Nat.gcd 90 150) = 30 := by sorry

end gcf_of_60_90_150_l3051_305156


namespace simplify_expression_l3051_305104

theorem simplify_expression (n : ℕ) : 
  (3^(n+4) - 3*(3^n) - 3^(n+2)) / (3*(3^(n+3))) = 23/9 := by
  sorry

end simplify_expression_l3051_305104


namespace probability_of_snow_in_three_days_l3051_305152

theorem probability_of_snow_in_three_days 
  (p1 : ℚ) (p2 : ℚ) (p3 : ℚ)
  (h1 : p1 = 1/2) 
  (h2 : p2 = 2/3) 
  (h3 : p3 = 3/4) : 
  1 - (1 - p1) * (1 - p2) * (1 - p3) = 23/24 := by
  sorry

end probability_of_snow_in_three_days_l3051_305152


namespace parallel_line_equation_l3051_305170

/-- A line passing through point (2,1) and parallel to y = -3x + 2 has equation y = -3x + 7 -/
theorem parallel_line_equation :
  let point : ℝ × ℝ := (2, 1)
  let parallel_line : ℝ → ℝ := λ x => -3 * x + 2
  let line : ℝ → ℝ := λ x => -3 * x + 7
  (∀ x : ℝ, line x - parallel_line x = line 0 - parallel_line 0) ∧
  line point.1 = point.2 := by
sorry

end parallel_line_equation_l3051_305170


namespace bank_deposit_exceeds_400_l3051_305109

def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (r^n - 1) / (r - 1)

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Sunday"
  | 1 => "Monday"
  | 2 => "Tuesday"
  | 3 => "Wednesday"
  | 4 => "Thursday"
  | 5 => "Friday"
  | _ => "Saturday"

theorem bank_deposit_exceeds_400 :
  let a := 2  -- Initial deposit in cents
  let r := 3  -- Common ratio
  let target := 40000  -- Target amount in cents
  ∀ n : ℕ, n < 10 → geometric_sum a r n ≤ target ∧
  geometric_sum a r 10 > target ∧
  day_of_week 10 = "Tuesday" :=
by sorry

end bank_deposit_exceeds_400_l3051_305109


namespace fibonacci_sequence_ones_l3051_305172

-- Define Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 1
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Define x_n sequence
def x : ℕ → ℕ → ℕ → ℚ
| 0, k, m => fib k / fib m
| (n + 1), k, m => 
  let prev := x n k m
  if prev = 1 then 1 else (2 * prev - 1) / (1 - prev)

-- Theorem statement
theorem fibonacci_sequence_ones (k m : ℕ) (h : m > k) :
  (∃ n, x n k m = 1) ↔ (∃ i : ℕ, k = 2 * i ∧ m = 2 * i + 1) :=
sorry

end fibonacci_sequence_ones_l3051_305172


namespace calculate_expression_l3051_305182

theorem calculate_expression : -2⁻¹ * (-8) + (2022)^0 - Real.sqrt 9 - abs (-4) = -3 := by
  sorry

end calculate_expression_l3051_305182


namespace inscribed_circle_radius_when_area_twice_perimeter_l3051_305157

/-- Given a triangle where the area is twice the perimeter, 
    the radius of the inscribed circle is 4. -/
theorem inscribed_circle_radius_when_area_twice_perimeter 
  (T : Set ℝ × Set ℝ) -- T represents a triangle in 2D space
  (A : ℝ) -- A represents the area of the triangle
  (p : ℝ) -- p represents the perimeter of the triangle
  (r : ℝ) -- r represents the radius of the inscribed circle
  (h1 : A = 2 * p) -- condition that area is twice the perimeter
  : r = 4 := by
  sorry

end inscribed_circle_radius_when_area_twice_perimeter_l3051_305157


namespace at_op_properties_l3051_305134

def at_op (a b : ℝ) : ℝ := a * b - 1

theorem at_op_properties (x y z : ℝ) : 
  (¬ (at_op x (y + z) = at_op x y + at_op x z)) ∧ 
  (¬ (x + at_op y z = at_op (x + y) (x + z))) ∧ 
  (¬ (at_op x (at_op y z) = at_op (at_op x y) (at_op x z))) :=
sorry

end at_op_properties_l3051_305134


namespace x_equals_eight_l3051_305194

theorem x_equals_eight (y : ℝ) (some_number : ℝ) 
  (h1 : 2 * x - y = some_number) 
  (h2 : y = 2) 
  (h3 : some_number = 14) : x = 8 := by
  sorry

end x_equals_eight_l3051_305194
