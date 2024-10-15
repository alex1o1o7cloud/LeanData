import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_l1601_160171

theorem triangle_area (base height : ℝ) (h1 : base = 8) (h2 : height = 4) :
  (base * height) / 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l1601_160171


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1601_160101

theorem min_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  a^2 / b^2 + b^2 / c^2 + c^2 / a^2 ≥ 3 ∧
  (a^2 / b^2 + b^2 / c^2 + c^2 / a^2 = 3 ↔ a = b ∧ b = c) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1601_160101


namespace NUMINAMATH_CALUDE_total_shirts_bought_l1601_160157

theorem total_shirts_bought (cost_15 : ℕ) (price_15 : ℕ) (price_20 : ℕ) (total_cost : ℕ) :
  cost_15 = 3 →
  price_15 = 15 →
  price_20 = 20 →
  total_cost = 85 →
  ∃ (cost_20 : ℕ), cost_15 * price_15 + cost_20 * price_20 = total_cost ∧ cost_15 + cost_20 = 5 :=
by sorry

end NUMINAMATH_CALUDE_total_shirts_bought_l1601_160157


namespace NUMINAMATH_CALUDE_ratio_equation_solution_l1601_160137

theorem ratio_equation_solution (x y z a : ℤ) : 
  (∃ (k : ℤ), x = 3 * k ∧ y = 4 * k ∧ z = 7 * k) →
  z = 30 * a - 15 →
  (∀ b : ℤ, 0 < b ∧ b < a → ¬(∃ (k : ℤ), 3 * k = 30 * b - 15)) →
  (∃ (k : ℤ), 3 * k = 30 * a - 15) →
  a = 4 := by
sorry

end NUMINAMATH_CALUDE_ratio_equation_solution_l1601_160137


namespace NUMINAMATH_CALUDE_abc_inequality_l1601_160187

theorem abc_inequality (a b c : ℝ) (sum_zero : a + b + c = 0) (product_one : a * b * c = 1) :
  (a * b + b * c + c * a < 0) ∧ (max a (max b c) ≥ Real.rpow 4 (1/3)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1601_160187


namespace NUMINAMATH_CALUDE_lcm_24_36_40_l1601_160166

theorem lcm_24_36_40 : Nat.lcm (Nat.lcm 24 36) 40 = 360 := by
  sorry

end NUMINAMATH_CALUDE_lcm_24_36_40_l1601_160166


namespace NUMINAMATH_CALUDE_students_without_a_count_l1601_160189

/-- Represents the number of students in a school course with various grade distributions. -/
structure CourseData where
  total_students : ℕ
  history_as : ℕ
  math_as : ℕ
  both_as : ℕ
  math_only_a : ℕ
  history_only_attendees : ℕ

/-- Calculates the number of students who did not receive an A in either class. -/
def students_without_a (data : CourseData) : ℕ :=
  data.total_students - (data.history_as + data.math_as - data.both_as)

/-- Theorem stating the number of students who did not receive an A in either class. -/
theorem students_without_a_count (data : CourseData) 
  (h1 : data.total_students = 30)
  (h2 : data.history_only_attendees = 1)
  (h3 : data.history_as = 6)
  (h4 : data.math_as = 15)
  (h5 : data.both_as = 3)
  (h6 : data.math_only_a = 1) :
  students_without_a data = 12 := by
  sorry

#eval students_without_a {
  total_students := 30,
  history_as := 6,
  math_as := 15,
  both_as := 3,
  math_only_a := 1,
  history_only_attendees := 1
}

end NUMINAMATH_CALUDE_students_without_a_count_l1601_160189


namespace NUMINAMATH_CALUDE_max_value_of_inequality_l1601_160172

theorem max_value_of_inequality (x : ℝ) : 
  (∀ y : ℝ, (6 + 5*y + y^2) * Real.sqrt (2*y^2 - y^3 - y) ≤ 0 → y ≤ x) → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_inequality_l1601_160172


namespace NUMINAMATH_CALUDE_rocket_max_height_l1601_160145

/-- The height function of the rocket -/
def h (t : ℝ) : ℝ := -20 * t^2 + 80 * t + 50

/-- The maximum height reached by the rocket -/
theorem rocket_max_height :
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 130 :=
sorry

end NUMINAMATH_CALUDE_rocket_max_height_l1601_160145


namespace NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1601_160170

theorem necessary_not_sufficient_condition :
  (∀ x : ℝ, |x - 2| < 1 → 1 < x ∧ x < 4) ∧
  ¬(∀ x : ℝ, 1 < x ∧ x < 4 → |x - 2| < 1) := by
  sorry

end NUMINAMATH_CALUDE_necessary_not_sufficient_condition_l1601_160170


namespace NUMINAMATH_CALUDE_clearance_sale_gain_percentage_l1601_160140

-- Define the original selling price
def original_selling_price : ℝ := 30

-- Define the original gain percentage
def original_gain_percentage : ℝ := 20

-- Define the discount percentage during clearance sale
def clearance_discount_percentage : ℝ := 10

-- Theorem statement
theorem clearance_sale_gain_percentage :
  let cost_price := original_selling_price / (1 + original_gain_percentage / 100)
  let discounted_price := original_selling_price * (1 - clearance_discount_percentage / 100)
  let new_gain := discounted_price - cost_price
  let new_gain_percentage := (new_gain / cost_price) * 100
  new_gain_percentage = 8 := by sorry

end NUMINAMATH_CALUDE_clearance_sale_gain_percentage_l1601_160140


namespace NUMINAMATH_CALUDE_max_value_of_f_l1601_160120

-- Define the function f(x)
def f (x : ℝ) := x * (1 - x)

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 1/4 ∧ ∀ x, 0 < x → x < 1 → f x ≤ M :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_f_l1601_160120


namespace NUMINAMATH_CALUDE_consecutive_numbers_product_l1601_160173

theorem consecutive_numbers_product (A B : Nat) : 
  A ≠ B → 
  A < 10 → 
  B < 10 → 
  35 * 36 * 37 * 38 * 39 = 120 * (100000 * A + 10000 * B + 1000 * A + 100 * B + 10 * A + B) → 
  A = 5 ∧ B = 7 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_numbers_product_l1601_160173


namespace NUMINAMATH_CALUDE_bucket_filling_time_l1601_160129

theorem bucket_filling_time (total_time : ℝ) (total_fraction : ℝ) (partial_fraction : ℝ) : 
  total_time = 150 → total_fraction = 1 → partial_fraction = 2/3 →
  (partial_fraction * total_time) / total_fraction = 100 := by
sorry

end NUMINAMATH_CALUDE_bucket_filling_time_l1601_160129


namespace NUMINAMATH_CALUDE_eggs_cooked_per_year_l1601_160169

/-- The number of eggs Lisa cooks for her family for breakfast in a year -/
def eggs_per_year : ℕ :=
  let days_per_week : ℕ := 5
  let weeks_per_year : ℕ := 52
  let num_children : ℕ := 4
  let eggs_per_child : ℕ := 2
  let eggs_for_husband : ℕ := 3
  let eggs_for_self : ℕ := 2
  let eggs_per_day : ℕ := num_children * eggs_per_child + eggs_for_husband + eggs_for_self
  eggs_per_day * days_per_week * weeks_per_year

theorem eggs_cooked_per_year :
  eggs_per_year = 3380 := by
  sorry

end NUMINAMATH_CALUDE_eggs_cooked_per_year_l1601_160169


namespace NUMINAMATH_CALUDE_factor_polynomial_l1601_160191

theorem factor_polynomial (x : ℝ) : 54 * x^5 - 135 * x^9 = 27 * x^5 * (2 - 5 * x^4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l1601_160191


namespace NUMINAMATH_CALUDE_danai_decorations_l1601_160112

def total_decorations (skulls broomsticks spiderwebs cauldron additional_budget additional_left : ℕ) : ℕ :=
  skulls + broomsticks + spiderwebs + (2 * spiderwebs) + cauldron + additional_budget + additional_left

theorem danai_decorations :
  let skulls : ℕ := 12
  let broomsticks : ℕ := 4
  let spiderwebs : ℕ := 12
  let cauldron : ℕ := 1
  let additional_budget : ℕ := 20
  let additional_left : ℕ := 10
  total_decorations skulls broomsticks spiderwebs cauldron additional_budget additional_left = 83 :=
by
  sorry

end NUMINAMATH_CALUDE_danai_decorations_l1601_160112


namespace NUMINAMATH_CALUDE_bretschneiders_theorem_l1601_160110

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  m : ℝ
  n : ℝ
  A : ℝ
  C : ℝ
  a_positive : a > 0
  b_positive : b > 0
  c_positive : c > 0
  d_positive : d > 0
  m_positive : m > 0
  n_positive : n > 0
  A_range : 0 < A ∧ A < π
  C_range : 0 < C ∧ C < π

-- State Bretschneider's theorem
theorem bretschneiders_theorem (q : ConvexQuadrilateral) :
  q.m^2 * q.n^2 = q.a^2 * q.c^2 + q.b^2 * q.d^2 - 2 * q.a * q.b * q.c * q.d * Real.cos (q.A + q.C) :=
sorry

end NUMINAMATH_CALUDE_bretschneiders_theorem_l1601_160110


namespace NUMINAMATH_CALUDE_one_more_stork_than_birds_l1601_160143

/-- Given the initial number of storks and birds on a fence, and additional birds that join,
    prove that there is one more stork than the total number of birds. -/
theorem one_more_stork_than_birds 
  (initial_storks : ℕ) 
  (initial_birds : ℕ) 
  (new_birds : ℕ) 
  (h1 : initial_storks = 6) 
  (h2 : initial_birds = 2) 
  (h3 : new_birds = 3) :
  initial_storks - (initial_birds + new_birds) = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_more_stork_than_birds_l1601_160143


namespace NUMINAMATH_CALUDE_range_of_a_l1601_160180

def S (a : ℝ) := {x : ℝ | x^2 ≤ a}

theorem range_of_a (a : ℝ) : (∅ ⊂ S a) → a ∈ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1601_160180


namespace NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l1601_160138

theorem polynomial_root_implies_coefficients 
  (a b : ℝ) :
  (∃ x : ℂ, x^3 + a*x^2 + 6*x + b = 0 ∧ x = 1 - 3*I) →
  a = 0 ∧ b = 20 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_root_implies_coefficients_l1601_160138


namespace NUMINAMATH_CALUDE_parabola_equation_l1601_160192

/-- A parabola with axis of symmetry parallel to the y-axis -/
structure Parabola where
  a : ℝ
  eq : ℝ → ℝ
  eq_def : ∀ x, eq x = a * (x - 1) * (x - 4)

/-- The line y = 2x -/
def line (x : ℝ) : ℝ := 2 * x

theorem parabola_equation (p : Parabola) (h1 : p.eq 1 = 0) (h2 : p.eq 4 = 0)
  (h_tangent : ∃ x, p.eq x = line x ∧ ∀ y ≠ x, p.eq y ≠ line y) :
  p.a = -2/9 ∨ p.a = -2 :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l1601_160192


namespace NUMINAMATH_CALUDE_apple_price_per_kg_final_apple_price_is_correct_l1601_160152

/-- Calculates the final price per kilogram of apples after discounts -/
theorem apple_price_per_kg (weight : ℝ) (original_price : ℝ) 
  (discount_percent : ℝ) (volume_discount_percent : ℝ) 
  (volume_discount_threshold : ℝ) : ℝ :=
  let price_after_discount := original_price * (1 - discount_percent)
  let final_price := 
    if weight > volume_discount_threshold
    then price_after_discount * (1 - volume_discount_percent)
    else price_after_discount
  final_price / weight

/-- Proves that the final price per kilogram is $1.44 given the specific conditions -/
theorem final_apple_price_is_correct : 
  apple_price_per_kg 5 10 0.2 0.1 3 = 1.44 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_per_kg_final_apple_price_is_correct_l1601_160152


namespace NUMINAMATH_CALUDE_sequence_properties_l1601_160125

/-- Definition of the sequence a_n -/
def a : ℕ → ℝ
  | 0 => 4
  | n + 1 => 2 * a n - 2 * (n + 1) + 1

/-- Definition of the sequence b_n -/
def b (t : ℝ) (n : ℕ) : ℝ := t * n + 2

/-- Theorem statement -/
theorem sequence_properties :
  (∀ n : ℕ, a (n + 1) - 2 * (n + 1) - 1 = 2 * (a n - 2 * n - 1)) ∧
  (∀ t : ℝ, (∀ n : ℕ, b t (n + 1) < 2 * a (n + 1)) → t < 6) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l1601_160125


namespace NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l1601_160175

theorem divisibility_of_fifth_power_differences (x y z : ℤ) 
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x) : 
  ∃ k : ℤ, (x - y)^5 + (y - z)^5 + (z - x)^5 = k * (5 * (y - z) * (z - x) * (x - y)) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_fifth_power_differences_l1601_160175


namespace NUMINAMATH_CALUDE_hilt_friends_cant_go_l1601_160102

/-- Given a total number of friends and the number of friends that can go to the movies,
    calculate the number of friends who can't go to the movies. -/
def friends_cant_go (total_friends : ℕ) (friends_going : ℕ) : ℕ :=
  total_friends - friends_going

/-- Theorem stating that with 25 total friends and 6 friends going to the movies,
    19 friends can't go to the movies. -/
theorem hilt_friends_cant_go :
  friends_cant_go 25 6 = 19 := by
  sorry

end NUMINAMATH_CALUDE_hilt_friends_cant_go_l1601_160102


namespace NUMINAMATH_CALUDE_nikola_leaf_price_l1601_160188

/-- The price Nikola charges per leaf -/
def price_per_leaf : ℚ :=
  1 / 100

theorem nikola_leaf_price :
  let num_ants : ℕ := 400
  let food_per_ant : ℚ := 2
  let food_price : ℚ := 1 / 10
  let job_start_price : ℕ := 5
  let num_leaves : ℕ := 6000
  let num_jobs : ℕ := 4
  (↑num_jobs * job_start_price + ↑num_leaves * price_per_leaf : ℚ) =
    ↑num_ants * food_per_ant * food_price :=
by sorry

end NUMINAMATH_CALUDE_nikola_leaf_price_l1601_160188


namespace NUMINAMATH_CALUDE_simplify_fraction_l1601_160146

theorem simplify_fraction : (3^4 + 3^2) / (3^3 - 3) = 15 / 4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l1601_160146


namespace NUMINAMATH_CALUDE_marching_band_ratio_l1601_160118

theorem marching_band_ratio (total_students : ℕ) (alto_sax_players : ℕ)
  (h_total : total_students = 600)
  (h_alto : alto_sax_players = 4)
  (h_sax : ∃ sax_players : ℕ, 3 * alto_sax_players = sax_players)
  (h_brass : ∃ brass_players : ℕ, 5 * sax_players = brass_players)
  (h_band : ∃ band_students : ℕ, 2 * brass_players = band_students) :
  band_students / total_students = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_marching_band_ratio_l1601_160118


namespace NUMINAMATH_CALUDE_four_six_eight_triangle_l1601_160165

/-- A predicate that determines if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem stating that 4, 6, and 8 can form a triangle -/
theorem four_six_eight_triangle :
  canFormTriangle 4 6 8 := by sorry

end NUMINAMATH_CALUDE_four_six_eight_triangle_l1601_160165


namespace NUMINAMATH_CALUDE_prob_three_odd_in_eight_rolls_l1601_160183

/-- The probability of getting an odd number on a single roll of a fair six-sided die -/
def prob_odd : ℚ := 1/2

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of odd results we're interested in -/
def target_odd : ℕ := 3

/-- Binomial coefficient -/
def binom (n k : ℕ) : ℕ := Nat.choose n k

/-- The probability of exactly k successes in n trials with probability p of success on each trial -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (binom n k : ℚ) * p^k * (1 - p)^(n - k)

theorem prob_three_odd_in_eight_rolls :
  binomial_probability num_rolls target_odd prob_odd = 7/32 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_in_eight_rolls_l1601_160183


namespace NUMINAMATH_CALUDE_chair_sequence_l1601_160197

theorem chair_sequence (seq : ℕ → ℕ) 
  (h1 : seq 1 = 14)
  (h2 : seq 2 = 23)
  (h3 : seq 3 = 32)
  (h4 : seq 4 = 41)
  (h6 : seq 6 = 59)
  (h_arithmetic : ∀ n : ℕ, n ≥ 1 → seq (n + 1) - seq n = seq 2 - seq 1) :
  seq 5 = 50 := by
  sorry

end NUMINAMATH_CALUDE_chair_sequence_l1601_160197


namespace NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1601_160160

def U : Set Int := {-2, -1, 0, 1, 2}
def A : Set Int := {-1, 1}
def B : Set Int := {0, 1, 2}

theorem intersection_of_A_and_complement_of_B : A ∩ (U \ B) = {-1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_complement_of_B_l1601_160160


namespace NUMINAMATH_CALUDE_equations_represent_scenario_l1601_160168

/-- Represents the value of livestock in taels of silver -/
structure LivestockValue where
  cow : ℝ
  sheep : ℝ

/-- The system of equations representing the livestock values -/
def livestock_equations (v : LivestockValue) : Prop :=
  5 * v.cow + 2 * v.sheep = 19 ∧ 2 * v.cow + 3 * v.sheep = 12

/-- The given scenario of livestock values -/
def livestock_scenario (v : LivestockValue) : Prop :=
  5 * v.cow + 2 * v.sheep = 19 ∧ 2 * v.cow + 3 * v.sheep = 12

/-- Theorem stating that the system of equations correctly represents the scenario -/
theorem equations_represent_scenario :
  ∀ v : LivestockValue, livestock_equations v ↔ livestock_scenario v :=
by sorry

end NUMINAMATH_CALUDE_equations_represent_scenario_l1601_160168


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_sum_100_l1601_160181

theorem smallest_of_five_consecutive_sum_100 (n : ℕ) : 
  (∃ (a b c d e : ℕ), 
    a + b + c + d + e = 100 ∧ 
    b = a + 1 ∧ 
    c = a + 2 ∧ 
    d = a + 3 ∧ 
    e = a + 4) → 
  n = 18 := by sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_sum_100_l1601_160181


namespace NUMINAMATH_CALUDE_max_value_of_cyclic_sum_l1601_160128

theorem max_value_of_cyclic_sum (a b c d e f : ℝ) 
  (non_neg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d ∧ 0 ≤ e ∧ 0 ≤ f) 
  (sum_constraint : a + b + c + d + e + f = 6) : 
  a * b * c + b * c * d + c * d * e + d * e * f + e * f * a + f * a * b ≤ 8 := by
sorry

end NUMINAMATH_CALUDE_max_value_of_cyclic_sum_l1601_160128


namespace NUMINAMATH_CALUDE_trail_mix_weight_l1601_160186

/-- The weight of peanuts in pounds -/
def weight_peanuts : ℝ := 0.17

/-- The weight of chocolate chips in pounds -/
def weight_chocolate_chips : ℝ := 0.17

/-- The weight of raisins in pounds -/
def weight_raisins : ℝ := 0.08

/-- The total weight of trail mix in pounds -/
def total_weight : ℝ := weight_peanuts + weight_chocolate_chips + weight_raisins

theorem trail_mix_weight : total_weight = 0.42 := by
  sorry

end NUMINAMATH_CALUDE_trail_mix_weight_l1601_160186


namespace NUMINAMATH_CALUDE_chlorous_acid_weight_l1601_160136

/-- The atomic weight of Hydrogen in g/mol -/
def H_weight : ℝ := 1.01

/-- The atomic weight of Chlorine in g/mol -/
def Cl_weight : ℝ := 35.45

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The number of moles of Chlorous acid -/
def moles : ℝ := 6

/-- The molecular weight of Chlorous acid (HClO2) in g/mol -/
def HClO2_weight : ℝ := H_weight + Cl_weight + 2 * O_weight

/-- Theorem: The molecular weight of 6 moles of Chlorous acid (HClO2) is 410.76 grams -/
theorem chlorous_acid_weight : moles * HClO2_weight = 410.76 := by
  sorry

end NUMINAMATH_CALUDE_chlorous_acid_weight_l1601_160136


namespace NUMINAMATH_CALUDE_gerald_pie_purchase_l1601_160167

/-- The number of farthings Gerald has initially -/
def initial_farthings : ℕ := 54

/-- The cost of the meat pie in pfennigs -/
def pie_cost : ℕ := 2

/-- The number of pfennigs Gerald has left after buying the pie -/
def remaining_pfennigs : ℕ := 7

/-- The number of farthings in a pfennig -/
def farthings_per_pfennig : ℕ := 6

theorem gerald_pie_purchase :
  initial_farthings - pie_cost * farthings_per_pfennig = remaining_pfennigs * farthings_per_pfennig :=
sorry

end NUMINAMATH_CALUDE_gerald_pie_purchase_l1601_160167


namespace NUMINAMATH_CALUDE_double_in_fifty_years_l1601_160156

/-- The interest rate (in percentage) that doubles an initial sum in 50 years under simple interest -/
def double_interest_rate : ℝ := 2

theorem double_in_fifty_years (P : ℝ) (P_pos : P > 0) :
  P * (1 + double_interest_rate * 50 / 100) = 2 * P := by
  sorry

#check double_in_fifty_years

end NUMINAMATH_CALUDE_double_in_fifty_years_l1601_160156


namespace NUMINAMATH_CALUDE_gcd_of_256_180_600_l1601_160122

theorem gcd_of_256_180_600 : Nat.gcd 256 (Nat.gcd 180 600) = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_256_180_600_l1601_160122


namespace NUMINAMATH_CALUDE_two_books_cost_exceeds_min_preparation_l1601_160133

/-- The cost of one storybook in yuan -/
def storybook_cost : ℚ := 25.5

/-- The minimum amount Wang Hong needs to prepare in yuan -/
def min_preparation : ℚ := 50

/-- Theorem: The cost of two storybooks is greater than the minimum preparation amount -/
theorem two_books_cost_exceeds_min_preparation : 2 * storybook_cost > min_preparation := by
  sorry

end NUMINAMATH_CALUDE_two_books_cost_exceeds_min_preparation_l1601_160133


namespace NUMINAMATH_CALUDE_existence_of_m_l1601_160121

def x : ℕ → ℚ
  | 0 => 5
  | n + 1 => (x n ^ 2 + 5 * x n + 4) / (x n + 6)

theorem existence_of_m :
  ∃ m : ℕ, 19 ≤ m ∧ m ≤ 60 ∧ 
  x m ≤ 4 + 1 / 2^10 ∧
  ∀ k : ℕ, 0 < k ∧ k < m → x k > 4 + 1 / 2^10 :=
sorry

end NUMINAMATH_CALUDE_existence_of_m_l1601_160121


namespace NUMINAMATH_CALUDE_problem_solution_l1601_160113

theorem problem_solution (x : ℝ) (h_pos : x > 0) :
  x^(2 * x^6) = 3 → x = (3 : ℝ)^(1/6) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1601_160113


namespace NUMINAMATH_CALUDE_g_1001_value_l1601_160117

/-- A function satisfying the given functional equation -/
def FunctionalEquation (g : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, g (x * y) + x = x * g y + g x

theorem g_1001_value
  (g : ℝ → ℝ)
  (h1 : FunctionalEquation g)
  (h2 : g 1 = -3) :
  g 1001 = -2001 := by
  sorry

end NUMINAMATH_CALUDE_g_1001_value_l1601_160117


namespace NUMINAMATH_CALUDE_smallest_odd_integer_triangle_perimeter_l1601_160155

/-- A function that generates the nth odd integer -/
def nthOddInteger (n : ℕ) : ℕ := 2 * n + 1

/-- A predicate that checks if three numbers form a valid triangle -/
def isValidTriangle (a b c : ℕ) : Prop := a + b > c ∧ a + c > b ∧ b + c > a

/-- The theorem stating the smallest possible perimeter of a triangle with consecutive odd integer sides -/
theorem smallest_odd_integer_triangle_perimeter :
  ∃ (n : ℕ), 
    isValidTriangle (nthOddInteger n) (nthOddInteger (n + 1)) (nthOddInteger (n + 2)) ∧
    (∀ (m : ℕ), m < n → ¬isValidTriangle (nthOddInteger m) (nthOddInteger (m + 1)) (nthOddInteger (m + 2))) ∧
    nthOddInteger n + nthOddInteger (n + 1) + nthOddInteger (n + 2) = 15 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_integer_triangle_perimeter_l1601_160155


namespace NUMINAMATH_CALUDE_unique_u_exists_l1601_160135

-- Define the variables as natural numbers
variable (a b u k p t : ℕ)

-- Define the conditions
def condition1 : Prop := a + b = u
def condition2 : Prop := u + k = p
def condition3 : Prop := p + a = t
def condition4 : Prop := b + k + t = 20

-- Define the uniqueness condition
def unique_digits : Prop := 
  a ≠ 0 ∧ b ≠ 0 ∧ u ≠ 0 ∧ k ≠ 0 ∧ p ≠ 0 ∧ t ≠ 0 ∧
  a ≠ b ∧ a ≠ u ∧ a ≠ k ∧ a ≠ p ∧ a ≠ t ∧
  b ≠ u ∧ b ≠ k ∧ b ≠ p ∧ b ≠ t ∧
  u ≠ k ∧ u ≠ p ∧ u ≠ t ∧
  k ≠ p ∧ k ≠ t ∧
  p ≠ t

-- Theorem statement
theorem unique_u_exists :
  ∃! u : ℕ, ∃ a b k p t : ℕ,
    condition1 a b u ∧
    condition2 u k p ∧
    condition3 p a t ∧
    condition4 b k t ∧
    unique_digits a b u k p t :=
  sorry

end NUMINAMATH_CALUDE_unique_u_exists_l1601_160135


namespace NUMINAMATH_CALUDE_fish_swimming_north_l1601_160114

theorem fish_swimming_north (west east north caught_east caught_west left : ℕ) :
  west = 1800 →
  east = 3200 →
  caught_east = (2 * east) / 5 →
  caught_west = (3 * west) / 4 →
  left = 2870 →
  west + east + north = caught_east + caught_west + left →
  north = 500 := by
sorry

end NUMINAMATH_CALUDE_fish_swimming_north_l1601_160114


namespace NUMINAMATH_CALUDE_quadratic_function_negative_at_four_l1601_160123

def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_negative_at_four
  (a b c : ℝ)
  (h1 : f a b c (-1) = -3)
  (h2 : f a b c 0 = 1)
  (h3 : f a b c 1 = 3)
  (h4 : f a b c 3 = 1) :
  f a b c 4 < 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_negative_at_four_l1601_160123


namespace NUMINAMATH_CALUDE_calculation_proof_l1601_160144

theorem calculation_proof : 
  |Real.sqrt 3 - 1| - (-Real.sqrt 3)^2 - 12 * (-1/3) = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l1601_160144


namespace NUMINAMATH_CALUDE_proposition_A_sufficient_not_necessary_for_B_l1601_160182

theorem proposition_A_sufficient_not_necessary_for_B :
  (∀ x : ℝ, 0 < x ∧ x < 5 → |x - 2| < 3) ∧
  (∃ x : ℝ, |x - 2| < 3 ∧ ¬(0 < x ∧ x < 5)) :=
by sorry

end NUMINAMATH_CALUDE_proposition_A_sufficient_not_necessary_for_B_l1601_160182


namespace NUMINAMATH_CALUDE_student_rank_l1601_160132

theorem student_rank (total_students : ℕ) (rank_from_right : ℕ) (rank_from_left : ℕ) :
  total_students = 20 →
  rank_from_right = 13 →
  rank_from_left = total_students - rank_from_right + 1 →
  rank_from_left = 9 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_l1601_160132


namespace NUMINAMATH_CALUDE_max_rectangle_area_l1601_160195

/-- Given a wire of length 52 cm, the maximum area of a rectangle that can be formed is 169 cm². -/
theorem max_rectangle_area (wire_length : ℝ) (h : wire_length = 52) : 
  (∀ l w : ℝ, l > 0 → w > 0 → 2 * (l + w) ≤ wire_length → l * w ≤ 169) ∧ 
  (∃ l w : ℝ, l > 0 ∧ w > 0 ∧ 2 * (l + w) = wire_length ∧ l * w = 169) :=
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l1601_160195


namespace NUMINAMATH_CALUDE_truck_speed_on_dirt_road_l1601_160162

/-- A semi truck travels on two types of roads. This theorem proves the speed on the dirt road. -/
theorem truck_speed_on_dirt_road :
  ∀ (v : ℝ),
  (3 * v) + (2 * (v + 20)) = 200 →
  v = 32 := by
sorry

end NUMINAMATH_CALUDE_truck_speed_on_dirt_road_l1601_160162


namespace NUMINAMATH_CALUDE_samara_alligators_l1601_160190

theorem samara_alligators (group_size : ℕ) (friends_count : ℕ) (friends_average : ℕ) (total_alligators : ℕ) :
  group_size = friends_count + 1 →
  friends_count = 3 →
  friends_average = 10 →
  total_alligators = 50 →
  total_alligators = friends_count * friends_average + (total_alligators - friends_count * friends_average) →
  (total_alligators - friends_count * friends_average) = 20 := by
  sorry

end NUMINAMATH_CALUDE_samara_alligators_l1601_160190


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l1601_160115

/-- The profit function based on price increase -/
def profit (x : ℝ) : ℝ := (90 + x - 80) * (400 - 10 * x)

/-- The initial purchase price -/
def initial_purchase_price : ℝ := 80

/-- The initial selling price -/
def initial_selling_price : ℝ := 90

/-- The initial sales volume -/
def initial_sales_volume : ℝ := 400

/-- The rate of decrease in sales volume per unit price increase -/
def sales_decrease_rate : ℝ := 10

/-- Theorem stating that the profit-maximizing selling price is 105 yuan -/
theorem profit_maximizing_price :
  ∃ (x : ℝ), x = 15 ∧ 
  ∀ (y : ℝ), profit y ≤ profit x ∧
  initial_selling_price + x = 105 := by
  sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l1601_160115


namespace NUMINAMATH_CALUDE_estate_value_l1601_160184

/-- Represents the estate distribution problem --/
structure EstateDistribution where
  total : ℝ
  daughter_share : ℝ
  son_share : ℝ
  wife_share : ℝ
  brother_share : ℝ
  nanny_share : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem estate_value (e : EstateDistribution) : 
  e.daughter_share + e.son_share = (3/5) * e.total ∧ 
  e.daughter_share = (5/7) * (e.daughter_share + e.son_share) ∧
  e.son_share = (2/7) * (e.daughter_share + e.son_share) ∧
  e.wife_share = 3 * e.son_share ∧
  e.brother_share = e.daughter_share ∧
  e.nanny_share = 400 ∧
  e.total = e.daughter_share + e.son_share + e.wife_share + e.brother_share + e.nanny_share
  →
  e.total = 825 := by
  sorry

#eval 825 -- To display the result

end NUMINAMATH_CALUDE_estate_value_l1601_160184


namespace NUMINAMATH_CALUDE_set_membership_implies_m_values_l1601_160109

theorem set_membership_implies_m_values (m : ℝ) : 
  let A : Set ℝ := {1, m + 2, m^2 + 4}
  5 ∈ A → (m = 3 ∨ m = 1) :=
by sorry

end NUMINAMATH_CALUDE_set_membership_implies_m_values_l1601_160109


namespace NUMINAMATH_CALUDE_not_algebraic_expression_l1601_160116

-- Define what constitutes an algebraic expression
def is_algebraic_expression (e : Prop) : Prop :=
  ¬(∃ (x : ℝ), e ↔ x = 1)

-- Define the given expressions
def pi_expr : Prop := True
def x_equals_1 : Prop := ∃ (x : ℝ), x = 1
def one_over_x : Prop := True
def sqrt_3 : Prop := True

-- Theorem statement
theorem not_algebraic_expression :
  is_algebraic_expression pi_expr ∧
  is_algebraic_expression one_over_x ∧
  is_algebraic_expression sqrt_3 ∧
  ¬(is_algebraic_expression x_equals_1) :=
sorry

end NUMINAMATH_CALUDE_not_algebraic_expression_l1601_160116


namespace NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l1601_160194

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, (∃ x : ℝ, (a + 1 : ℝ) * x^2 - 2*x + 3 = 0) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l1601_160194


namespace NUMINAMATH_CALUDE_initial_number_proof_l1601_160130

theorem initial_number_proof (x : ℝ) : 
  x + 12.808 - 47.80600000000004 = 3854.002 ↔ x = 3889 := by
  sorry

end NUMINAMATH_CALUDE_initial_number_proof_l1601_160130


namespace NUMINAMATH_CALUDE_factor_theorem_application_l1601_160100

theorem factor_theorem_application (m k : ℝ) : 
  (∃ q : ℝ, m^2 - k*m - 24 = (m - 8) * q) → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_factor_theorem_application_l1601_160100


namespace NUMINAMATH_CALUDE_tripled_rectangle_area_l1601_160124

/-- Theorem: New area of a tripled rectangle --/
theorem tripled_rectangle_area (k m : ℝ) (hk : k > 0) (hm : m > 0) : 
  let original_area := (6 * k) * (4 * m)
  let new_area := 3 * original_area
  new_area = 72 * k * m := by
  sorry


end NUMINAMATH_CALUDE_tripled_rectangle_area_l1601_160124


namespace NUMINAMATH_CALUDE_jacqueline_apples_l1601_160148

/-- The number of plums Jacqueline had initially -/
def plums : ℕ := 16

/-- The number of guavas Jacqueline had initially -/
def guavas : ℕ := 18

/-- The number of fruits Jacqueline gave to Jane -/
def given_fruits : ℕ := 40

/-- The number of fruits Jacqueline had left after giving some to Jane -/
def left_fruits : ℕ := 15

/-- The number of apples Jacqueline had initially -/
def apples : ℕ := 21

theorem jacqueline_apples :
  plums + guavas + apples = given_fruits + left_fruits :=
sorry

end NUMINAMATH_CALUDE_jacqueline_apples_l1601_160148


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l1601_160196

/-- The area of a square given its diagonal length -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 10) : 
  (d ^ 2 / 2 : ℝ) = 50 := by sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l1601_160196


namespace NUMINAMATH_CALUDE_least_meeting_time_for_four_horses_l1601_160164

def horse_lap_time (k : Nat) : Nat := 2 * k

def is_meeting_time (t : Nat) (horses : List Nat) : Prop :=
  ∀ h ∈ horses, t % (horse_lap_time h) = 0

theorem least_meeting_time_for_four_horses :
  ∃ T : Nat,
    T > 0 ∧
    (∃ horses : List Nat, horses.length ≥ 4 ∧ horses.all (· ≤ 8) ∧ is_meeting_time T horses) ∧
    (∀ t : Nat, 0 < t ∧ t < T →
      ¬∃ horses : List Nat, horses.length ≥ 4 ∧ horses.all (· ≤ 8) ∧ is_meeting_time t horses) ∧
    T = 24 := by sorry

end NUMINAMATH_CALUDE_least_meeting_time_for_four_horses_l1601_160164


namespace NUMINAMATH_CALUDE_smallest_b_in_geometric_sequence_l1601_160154

/-- 
Given a geometric sequence of positive terms a, b, c with product 216,
this theorem states that the smallest possible value of b is 6.
-/
theorem smallest_b_in_geometric_sequence (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →  -- all terms are positive
  (∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r) →  -- geometric sequence
  a * b * c = 216 →  -- product is 216
  (∀ b' : ℝ, b' > 0 → 
    (∃ a' c' : ℝ, a' > 0 ∧ c' > 0 ∧ 
      (∃ r : ℝ, r > 0 ∧ b' = a' * r ∧ c' = b' * r) ∧ 
      a' * b' * c' = 216) → 
    b' ≥ 6) →  -- for any valid b', b' is at least 6
  b = 6  -- therefore, the smallest possible b is 6
:= by sorry

end NUMINAMATH_CALUDE_smallest_b_in_geometric_sequence_l1601_160154


namespace NUMINAMATH_CALUDE_walking_speed_problem_l1601_160163

/-- Two people A and B walk towards each other and meet. This theorem proves B's speed. -/
theorem walking_speed_problem (speed_A speed_B : ℝ) (initial_distance total_time : ℝ) : 
  speed_A = 5 →
  initial_distance = 24 →
  total_time = 2 →
  speed_A * total_time + speed_B * total_time = initial_distance →
  speed_B = 7 := by
  sorry

#check walking_speed_problem

end NUMINAMATH_CALUDE_walking_speed_problem_l1601_160163


namespace NUMINAMATH_CALUDE_line_passes_through_point_two_two_l1601_160185

/-- The line equation is of the form (1+4k)x-(2-3k)y+2-14k=0 where k is a real parameter -/
def line_equation (k x y : ℝ) : Prop :=
  (1 + 4*k)*x - (2 - 3*k)*y + 2 - 14*k = 0

/-- Theorem: The line passes through the point (2, 2) for all values of k -/
theorem line_passes_through_point_two_two :
  ∀ k : ℝ, line_equation k 2 2 := by sorry

end NUMINAMATH_CALUDE_line_passes_through_point_two_two_l1601_160185


namespace NUMINAMATH_CALUDE_equation_solutions_l1601_160141

theorem equation_solutions :
  (∃ x : ℝ, 2 * x^3 = 16 ∧ x = 2) ∧
  (∃ x₁ x₂ : ℝ, (x₁ - 1)^2 = 4 ∧ (x₂ - 1)^2 = 4 ∧ x₁ = 3 ∧ x₂ = -1) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1601_160141


namespace NUMINAMATH_CALUDE_tower_combinations_l1601_160153

/-- Represents the number of cubes of each color --/
structure CubeColors where
  red : Nat
  blue : Nat
  green : Nat
  yellow : Nat

/-- Calculates the number of different towers that can be built --/
def numTowers (colors : CubeColors) (towerHeight : Nat) : Nat :=
  if towerHeight ≠ colors.red + colors.blue + colors.green + colors.yellow - 1 then 0
  else if colors.yellow = 0 then 0
  else
    let n := towerHeight - 1
    Nat.factorial n / (Nat.factorial colors.red * Nat.factorial colors.blue * 
                       Nat.factorial colors.green * Nat.factorial (colors.yellow - 1))

/-- The main theorem to be proven --/
theorem tower_combinations : 
  let colors := CubeColors.mk 3 4 2 2
  numTowers colors 10 = 1260 := by
  sorry

end NUMINAMATH_CALUDE_tower_combinations_l1601_160153


namespace NUMINAMATH_CALUDE_total_snacks_weight_l1601_160106

-- Define the conversion rate from ounces to pounds
def ounces_to_pounds : ℚ → ℚ := (· / 16)

-- Define the weights of snacks
def peanuts_weight : ℚ := 0.1
def raisins_weight_oz : ℚ := 5
def almonds_weight : ℚ := 0.3

-- Theorem to prove
theorem total_snacks_weight :
  peanuts_weight + ounces_to_pounds raisins_weight_oz + almonds_weight = 0.7125 := by
  sorry

end NUMINAMATH_CALUDE_total_snacks_weight_l1601_160106


namespace NUMINAMATH_CALUDE_part_one_part_two_l1601_160178

-- Define the equation
def equation (x a : ℝ) : Prop := (x + a) / (x - 2) - 5 / x = 1

-- Part 1: When x = 5 is a root
theorem part_one (a : ℝ) : (5 + a) / 3 - 1 = 1 → a = 1 := by sorry

-- Part 2: When the equation has no solution
theorem part_two (a : ℝ) : (∀ x : ℝ, ¬ equation x a) ↔ a = 3 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l1601_160178


namespace NUMINAMATH_CALUDE_number_order_l1601_160104

theorem number_order (a : ℝ) (h1 : 0 < a) (h2 : a < 1) :
  (1 / a > Real.sqrt a) ∧ (Real.sqrt a > a) ∧ (a > a^2) := by
  sorry

end NUMINAMATH_CALUDE_number_order_l1601_160104


namespace NUMINAMATH_CALUDE_prob_two_same_school_correct_l1601_160161

/-- Represents the number of schools participating in the activity -/
def num_schools : ℕ := 5

/-- Represents the number of students each school sends -/
def students_per_school : ℕ := 2

/-- Represents the total number of students participating -/
def total_students : ℕ := num_schools * students_per_school

/-- Represents the number of students chosen to play the game -/
def chosen_students : ℕ := 4

/-- Calculates the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Represents the probability of exactly two students coming from the same school -/
def prob_two_same_school : ℚ := 5 / 14

theorem prob_two_same_school_correct :
  prob_two_same_school = 
    (choose num_schools 1 * choose students_per_school 2 * choose (total_students - students_per_school) 2) / 
    (choose total_students chosen_students) := by
  sorry

end NUMINAMATH_CALUDE_prob_two_same_school_correct_l1601_160161


namespace NUMINAMATH_CALUDE_alloy_cut_theorem_l1601_160103

/-- Represents an alloy piece with its mass and copper concentration -/
structure AlloyPiece where
  mass : ℝ
  copper_concentration : ℝ

/-- Represents the result of cutting and swapping parts of two alloy pieces -/
def cut_and_swap (piece1 piece2 : AlloyPiece) (cut_mass : ℝ) : Prop :=
  let new_piece1 := AlloyPiece.mk piece1.mass 
    ((cut_mass * piece2.copper_concentration + (piece1.mass - cut_mass) * piece1.copper_concentration) / piece1.mass)
  let new_piece2 := AlloyPiece.mk piece2.mass 
    ((cut_mass * piece1.copper_concentration + (piece2.mass - cut_mass) * piece2.copper_concentration) / piece2.mass)
  new_piece1.copper_concentration = new_piece2.copper_concentration

theorem alloy_cut_theorem (piece1 piece2 : AlloyPiece) (cut_mass : ℝ) :
  piece1.mass = piece2.mass →
  piece1.copper_concentration ≠ piece2.copper_concentration →
  cut_and_swap piece1 piece2 cut_mass →
  cut_mass = piece1.mass / 2 :=
sorry

end NUMINAMATH_CALUDE_alloy_cut_theorem_l1601_160103


namespace NUMINAMATH_CALUDE_consecutive_composites_l1601_160107

theorem consecutive_composites
  (a t d r : ℕ+)
  (ha : ¬ Nat.Prime a.val)
  (ht : ¬ Nat.Prime t.val)
  (hd : ¬ Nat.Prime d.val)
  (hr : ¬ Nat.Prime r.val) :
  ∃ k : ℕ, ∀ i : ℕ, i < r → ¬ Nat.Prime (a * t ^ (k + i) + d) :=
sorry

end NUMINAMATH_CALUDE_consecutive_composites_l1601_160107


namespace NUMINAMATH_CALUDE_simplify_sqrt_one_minus_sin_twenty_deg_l1601_160198

theorem simplify_sqrt_one_minus_sin_twenty_deg :
  Real.sqrt (1 - Real.sin (20 * π / 180)) = Real.cos (10 * π / 180) - Real.sin (10 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_one_minus_sin_twenty_deg_l1601_160198


namespace NUMINAMATH_CALUDE_peter_age_is_16_l1601_160139

/-- Peter's present age -/
def PeterAge : ℕ := sorry

/-- Jacob's present age -/
def JacobAge : ℕ := sorry

/-- Theorem stating the conditions and the result to prove -/
theorem peter_age_is_16 :
  (JacobAge = PeterAge + 12) ∧
  (PeterAge - 10 = (JacobAge - 10) / 3) →
  PeterAge = 16 := by sorry

end NUMINAMATH_CALUDE_peter_age_is_16_l1601_160139


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1601_160177

theorem sum_of_roots_quadratic (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ + 3 = 0) → 
  (x₂^2 - 4*x₂ + 3 = 0) → 
  (x₁ + x₂ = 4) := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1601_160177


namespace NUMINAMATH_CALUDE_solve_allowance_problem_l1601_160126

def allowance_problem (initial_amount spent_amount final_amount : ℕ) : Prop :=
  ∃ allowance : ℕ, 
    initial_amount - spent_amount + allowance = final_amount

theorem solve_allowance_problem :
  allowance_problem 5 2 8 → ∃ allowance : ℕ, allowance = 5 :=
by
  sorry

end NUMINAMATH_CALUDE_solve_allowance_problem_l1601_160126


namespace NUMINAMATH_CALUDE_alternating_draw_probability_l1601_160150

/-- The number of white balls in the box -/
def white_balls : Nat := 5

/-- The number of black balls in the box -/
def black_balls : Nat := 5

/-- The total number of balls in the box -/
def total_balls : Nat := white_balls + black_balls

/-- The number of ways to arrange white_balls white balls and black_balls black balls -/
def total_arrangements : Nat := Nat.choose total_balls white_balls

/-- The number of valid alternating color patterns -/
def valid_patterns : Nat := 2

/-- The probability of drawing all balls in an alternating color pattern -/
def alternating_probability : ℚ := valid_patterns / total_arrangements

theorem alternating_draw_probability : alternating_probability = 1 / 126 := by
  sorry

end NUMINAMATH_CALUDE_alternating_draw_probability_l1601_160150


namespace NUMINAMATH_CALUDE_initial_amounts_given_final_state_l1601_160147

/-- Represents the state of the game after each round -/
structure GameState where
  player1 : ℤ
  player2 : ℤ
  player3 : ℤ

/-- Simulates one round of the game where the specified player loses -/
def playRound (state : GameState) (loser : Fin 3) : GameState :=
  match loser with
  | 0 => ⟨state.player1 - (state.player2 + state.player3), 
          state.player2 + state.player1, 
          state.player3 + state.player1⟩
  | 1 => ⟨state.player1 + state.player2, 
          state.player2 - (state.player1 + state.player3), 
          state.player3 + state.player2⟩
  | 2 => ⟨state.player1 + state.player3, 
          state.player2 + state.player3, 
          state.player3 - (state.player1 + state.player2)⟩

/-- Theorem stating the initial amounts given the final state -/
theorem initial_amounts_given_final_state 
  (x y z : ℤ) 
  (h1 : playRound (playRound (playRound ⟨x, y, z⟩ 0) 1) 2 = ⟨104, 104, 104⟩) :
  x = 169 ∧ y = 91 ∧ z = 52 := by
  sorry


end NUMINAMATH_CALUDE_initial_amounts_given_final_state_l1601_160147


namespace NUMINAMATH_CALUDE_brick_factory_workers_l1601_160158

/-- The maximum number of workers that can be hired at a brick factory -/
def max_workers : ℕ := 8

theorem brick_factory_workers :
  ∀ n : ℕ,
  n ≤ max_workers ↔
  (10 * n - n * n ≥ 13) ∧
  ∀ m : ℕ, m > n → (10 * m - m * m < 13) :=
by sorry

end NUMINAMATH_CALUDE_brick_factory_workers_l1601_160158


namespace NUMINAMATH_CALUDE_line_not_parallel_intersects_plane_l1601_160151

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields for a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields for a plane

/-- Definition: A line is parallel to a plane -/
def is_parallel (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Definition: A line shares common points with a plane -/
def shares_common_points (l : Line3D) (α : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is not parallel to a plane, then it shares common points with the plane -/
theorem line_not_parallel_intersects_plane (l : Line3D) (α : Plane3D) :
  ¬(is_parallel l α) → shares_common_points l α :=
by
  sorry

end NUMINAMATH_CALUDE_line_not_parallel_intersects_plane_l1601_160151


namespace NUMINAMATH_CALUDE_basketball_game_points_l1601_160105

/-- Calculate total points for a player given their shot counts -/
def playerPoints (twoPoints threePoints freeThrows : ℕ) : ℕ :=
  2 * twoPoints + 3 * threePoints + freeThrows

/-- Calculate total points for a team given two players' shot counts -/
def teamPoints (p1TwoPoints p1ThreePoints p1FreeThrows
                p2TwoPoints p2ThreePoints p2FreeThrows : ℕ) : ℕ :=
  playerPoints p1TwoPoints p1ThreePoints p1FreeThrows +
  playerPoints p2TwoPoints p2ThreePoints p2FreeThrows

/-- Theorem: The combined points of both teams is 128 -/
theorem basketball_game_points : 
  teamPoints 7 5 4 4 6 7 + teamPoints 9 4 5 6 3 6 = 128 := by
  sorry

#eval teamPoints 7 5 4 4 6 7 + teamPoints 9 4 5 6 3 6

end NUMINAMATH_CALUDE_basketball_game_points_l1601_160105


namespace NUMINAMATH_CALUDE_extra_lambs_found_l1601_160174

def lambs_problem (initial_lambs : ℕ) (lambs_with_babies : ℕ) (babies_per_lamb : ℕ) 
                  (traded_lambs : ℕ) (final_lambs : ℕ) : ℕ :=
  let lambs_after_babies := initial_lambs + lambs_with_babies * babies_per_lamb
  let lambs_after_trade := lambs_after_babies - traded_lambs
  final_lambs - lambs_after_trade

theorem extra_lambs_found :
  lambs_problem 6 2 2 3 14 = 7 := by
  sorry

end NUMINAMATH_CALUDE_extra_lambs_found_l1601_160174


namespace NUMINAMATH_CALUDE_min_races_for_top_3_l1601_160179

/-- Represents a horse in the race. -/
structure Horse :=
  (id : Nat)

/-- Represents a race with up to 6 horses. -/
structure Race :=
  (horses : Finset Horse)
  (condition : Nat)  -- Represents different race conditions

/-- A function to determine the ranking of horses in a race. -/
def raceResult (r : Race) : List Horse := sorry

/-- The total number of horses. -/
def totalHorses : Nat := 30

/-- The maximum number of horses that can race together. -/
def maxHorsesPerRace : Nat := 6

/-- A function to determine if we have found the top 3 horses. -/
def hasTop3 (races : List Race) : Bool := sorry

/-- Theorem stating the minimum number of races needed. -/
theorem min_races_for_top_3 :
  ∃ (races : List Race),
    races.length = 7 ∧
    hasTop3 races ∧
    ∀ (other_races : List Race),
      hasTop3 other_races → other_races.length ≥ 7 := by sorry

end NUMINAMATH_CALUDE_min_races_for_top_3_l1601_160179


namespace NUMINAMATH_CALUDE_min_value_of_f_l1601_160176

noncomputable def f (x : ℝ) : ℝ := 9 / (8 * Real.cos (2 * x) + 16) - Real.sin x ^ 2

theorem min_value_of_f (x : ℝ) :
  f x ≥ 0 ∧ (f x = 0 ↔ Real.cos (2 * x) = -1/2) :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1601_160176


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1601_160199

theorem diophantine_equation_solution (x y z : ℕ+) :
  1 + 4^x.val + 4^y.val = z.val^2 ↔ 
  (∃ n : ℕ+, (x = n ∧ y = 2*n - 1 ∧ z = 1 + 2^(2*n.val - 1)) ∨ 
             (x = 2*n - 1 ∧ y = n ∧ z = 1 + 2^(2*n.val - 1))) :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1601_160199


namespace NUMINAMATH_CALUDE_necessary_condition_for_P_l1601_160131

-- Define the set A
def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

-- Define the proposition P(a)
def P (a : ℝ) : Prop := ∀ x ∈ A, x^2 - a < 0

-- Theorem statement
theorem necessary_condition_for_P :
  (∃ a : ℝ, P a) → (∀ a : ℝ, P a → a ≥ 1) ∧ ¬(∀ a : ℝ, a ≥ 1 → P a) := by
  sorry

end NUMINAMATH_CALUDE_necessary_condition_for_P_l1601_160131


namespace NUMINAMATH_CALUDE_correct_calculation_l1601_160149

theorem correct_calculation : ∃ x : ℕ, (x + 30 = 86) ∧ (x * 30 = 1680) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1601_160149


namespace NUMINAMATH_CALUDE_hyperbola_center_is_correct_l1601_160108

/-- The equation of a hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  9 * x^2 - 54 * x - 36 * y^2 + 360 * y - 864 = 0

/-- The center of a hyperbola -/
def hyperbola_center : ℝ × ℝ := (3, 5)

/-- Theorem: The center of the given hyperbola is (3, 5) -/
theorem hyperbola_center_is_correct :
  ∀ x y : ℝ, hyperbola_equation x y ↔ 
    ((x - hyperbola_center.1)^2 / 5 - (y - hyperbola_center.2)^2 / (5/4) = 1) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_center_is_correct_l1601_160108


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l1601_160159

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 8*x - 6 = 0) ∧
  (∃ x : ℝ, (x - 3)^2 + 2*x*(x - 3) = 0) ∧
  (∀ x : ℝ, x^2 - 8*x - 6 = 0 ↔ (x = 4 + Real.sqrt 22 ∨ x = 4 - Real.sqrt 22)) ∧
  (∀ x : ℝ, (x - 3)^2 + 2*x*(x - 3) = 0 ↔ (x = 3 ∨ x = 1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l1601_160159


namespace NUMINAMATH_CALUDE_circle_area_with_diameter_l1601_160119

theorem circle_area_with_diameter (d : ℝ) (A : ℝ) :
  d = 7.5 →
  A = π * (d / 2)^2 →
  A = 14.0625 * π :=
by sorry

end NUMINAMATH_CALUDE_circle_area_with_diameter_l1601_160119


namespace NUMINAMATH_CALUDE_consecutive_product_square_appendage_l1601_160127

theorem consecutive_product_square_appendage (n : ℕ) :
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ ∃ (k : ℕ), 100 * (n * (n + 1)) + 10 * a + b = k ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_square_appendage_l1601_160127


namespace NUMINAMATH_CALUDE_f_101_form_l1601_160134

def is_valid_f (f : ℕ → ℕ) : Prop :=
  ∀ m n : ℕ, m > 0 ∧ n > 0 → (m * n + 1) ∣ (f m * f n + 1)

theorem f_101_form (f : ℕ → ℕ) (h : is_valid_f f) :
  ∃ k : ℕ, k % 2 = 1 ∧ f 101 = 101^k :=
by sorry

end NUMINAMATH_CALUDE_f_101_form_l1601_160134


namespace NUMINAMATH_CALUDE_equation_solution_l1601_160193

/-- Converts a number from base b to base 10 -/
def to_base_10 (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- Checks if the equation 3△_4 = △2_11 is satisfied -/
def equation_satisfied (triangle : Nat) : Prop :=
  to_base_10 [3, triangle] 4 = to_base_10 [triangle, 2] 11

/-- Theorem stating that the equation is satisfied when triangle is 1 -/
theorem equation_solution :
  equation_satisfied 1 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l1601_160193


namespace NUMINAMATH_CALUDE_range_of_a_l1601_160111

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (Real.exp x - a)^2 + x^2 - 2*a*x + a^2 ≤ 1/2) → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l1601_160111


namespace NUMINAMATH_CALUDE_total_wheels_is_64_l1601_160142

/-- The number of wheels on a four-wheeler -/
def wheels_per_four_wheeler : ℕ := 4

/-- The number of four-wheelers parked in the school -/
def num_four_wheelers : ℕ := 16

/-- The total number of wheels for all four-wheelers parked in the school -/
def total_wheels_four_wheelers : ℕ := num_four_wheelers * wheels_per_four_wheeler

/-- Theorem: The total number of wheels for the four-wheelers parked in the school is 64 -/
theorem total_wheels_is_64 : total_wheels_four_wheelers = 64 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_is_64_l1601_160142
