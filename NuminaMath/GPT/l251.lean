import Mathlib

namespace relationships_with_correlation_l251_25165

-- Definitions for each of the relationships as conditions
def person_age_wealth := true -- placeholder definition 
def curve_points_coordinates := true -- placeholder definition
def apple_production_climate := true -- placeholder definition
def tree_diameter_height := true -- placeholder definition
def student_school := true -- placeholder definition

-- Statement to prove which relationships involve correlation
theorem relationships_with_correlation :
  person_age_wealth ∧ apple_production_climate ∧ tree_diameter_height :=
by
  sorry

end relationships_with_correlation_l251_25165


namespace two_digit_number_representation_l251_25172

theorem two_digit_number_representation (m n : ℕ) (hm : m < 10) (hn : n < 10) : 10 * n + m = m + 10 * n :=
by sorry

end two_digit_number_representation_l251_25172


namespace range_of_a_l251_25152

-- Definitions capturing the given conditions
variables (a b c : ℝ)

-- Conditions are stated as assumptions
def condition1 := a^2 - b * c - 8 * a + 7 = 0
def condition2 := b^2 + c^2 + b * c - 6 * a + 6 = 0

-- The mathematically equivalent proof problem
theorem range_of_a (h1 : condition1 a b c) (h2 : condition2 a b c) : 1 ≤ a ∧ a ≤ 9 := 
sorry

end range_of_a_l251_25152


namespace Ludwig_daily_salary_l251_25168

theorem Ludwig_daily_salary 
(D : ℝ)
(h_weekly_earnings : 4 * D + (3 / 2) * D = 55) :
D = 10 := 
by
  sorry

end Ludwig_daily_salary_l251_25168


namespace fraction_product_equals_64_l251_25170

theorem fraction_product_equals_64 : 
  (1 / 4) * (8 / 1) * (1 / 32) * (64 / 1) * (1 / 128) * (256 / 1) * (1 / 512) * (1024 / 1) * (1 / 2048) * (4096 / 1) * (1 / 8192) * (16384 / 1) = 64 :=
by
  sorry

end fraction_product_equals_64_l251_25170


namespace shape_is_cylinder_l251_25141

def positive_constant (c : ℝ) := c > 0

def is_cylinder (r θ z : ℝ) (c : ℝ) : Prop :=
  r = c

theorem shape_is_cylinder (c : ℝ) (r θ z : ℝ) 
  (h_pos : positive_constant c) (h_eq : r = c) :
  is_cylinder r θ z c := by
  sorry

end shape_is_cylinder_l251_25141


namespace smallest_integer_y_solution_l251_25132

theorem smallest_integer_y_solution :
  ∃ y : ℤ, (∀ z : ℤ, (z / 4 + 3 / 7 > 9 / 4) → (z ≥ y)) ∧ (y = 8) := 
by
  sorry

end smallest_integer_y_solution_l251_25132


namespace joan_carrots_grown_correct_l251_25143

variable (total_carrots : ℕ) (jessica_carrots : ℕ) (joan_carrots : ℕ)

theorem joan_carrots_grown_correct (h1 : total_carrots = 40) (h2 : jessica_carrots = 11) (h3 : total_carrots = joan_carrots + jessica_carrots) : joan_carrots = 29 :=
by
  sorry

end joan_carrots_grown_correct_l251_25143


namespace circle_area_ratio_l251_25134

/-- If the diameter of circle R is 60% of the diameter of circle S, 
the area of circle R is 36% of the area of circle S. -/
theorem circle_area_ratio (D_S D_R A_S A_R : ℝ) (h : D_R = 0.60 * D_S) 
  (hS : A_S = Real.pi * (D_S / 2) ^ 2) (hR : A_R = Real.pi * (D_R / 2) ^ 2): 
  A_R = 0.36 * A_S := 
sorry

end circle_area_ratio_l251_25134


namespace correct_product_l251_25123

theorem correct_product (a b c : ℕ) (ha : 10 * c + 1 = a) (hb : 10 * c + 7 = a) 
(hl : (10 * c + 1) * b = 255) (hw : (10 * c + 7 + 6) * b = 335) : 
  a * b = 285 := 
  sorry

end correct_product_l251_25123


namespace option_D_correct_l251_25160

theorem option_D_correct (a b : ℝ) (h : a > b) : 3 * a > 3 * b :=
sorry

end option_D_correct_l251_25160


namespace inequality_solution_l251_25180

theorem inequality_solution 
  (x : ℝ) 
  (h1 : (x + 3) / 2 ≤ x + 2) 
  (h2 : 2 * (x + 4) > 4 * x + 2) : 
  -1 ≤ x ∧ x < 3 := sorry

end inequality_solution_l251_25180


namespace rods_in_one_mile_l251_25150

-- Define the given conditions
def mile_to_chains : ℕ := 10
def chain_to_rods : ℕ := 4

-- Prove the number of rods in one mile
theorem rods_in_one_mile : (1 * mile_to_chains * chain_to_rods) = 40 := by
  sorry

end rods_in_one_mile_l251_25150


namespace rajans_position_l251_25159

theorem rajans_position
    (total_boys : ℕ)
    (vinay_position_from_right : ℕ)
    (boys_between_rajan_and_vinay : ℕ)
    (total_boys_eq : total_boys = 24)
    (vinay_position_from_right_eq : vinay_position_from_right = 10)
    (boys_between_eq : boys_between_rajan_and_vinay = 8) :
    ∃ R : ℕ, R = 6 :=
by
  sorry

end rajans_position_l251_25159


namespace find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l251_25191

noncomputable def f (a m x : ℝ) := Real.log (x + m) / Real.log a
noncomputable def g (a x : ℝ) := Real.log (1 - x) / Real.log a
noncomputable def F (a m x : ℝ) := f a m x - g a x

theorem find_m_and_domain (a : ℝ) (m : ℝ) (h : F a m 0 = 0) : m = 1 ∧ ∀ x, -1 < x ∧ x < 1 :=
sorry

theorem parity_of_F (a : ℝ) (m : ℝ) (h : m = 1) : ∀ x, F a m (-x) = -F a m x :=
sorry

theorem range_of_x_for_F_positive (a : ℝ) (m : ℝ) (h : m = 1) :
  (a > 1 → ∀ x, 0 < x ∧ x < 1 → F a m x > 0) ∧ (0 < a ∧ a < 1 → ∀ x, -1 < x ∧ x < 0 → F a m x > 0) :=
sorry

end find_m_and_domain_parity_of_F_range_of_x_for_F_positive_l251_25191


namespace longest_side_of_garden_l251_25188

theorem longest_side_of_garden (l w : ℝ) (h1 : 2 * l + 2 * w = 225) (h2 : l * w = 8 * 225) :
  l = 93.175 ∨ w = 93.175 :=
by
  sorry

end longest_side_of_garden_l251_25188


namespace real_estate_commission_l251_25174

theorem real_estate_commission (commission_rate commission selling_price : ℝ) 
  (h1 : commission_rate = 0.06) 
  (h2 : commission = 8880) : 
  selling_price = 148000 :=
by
  sorry

end real_estate_commission_l251_25174


namespace cos_pi_plus_2alpha_l251_25190

-- Define the main theorem using the given condition and the result to be proven
theorem cos_pi_plus_2alpha (α : ℝ) (h : Real.sin (π / 2 - α) = 1 / 3) : Real.cos (π + 2 * α) = 7 / 9 :=
sorry

end cos_pi_plus_2alpha_l251_25190


namespace katherine_has_5_bananas_l251_25164

/-- Katherine has 4 apples -/
def apples : ℕ := 4

/-- Katherine has 3 times as many pears as apples -/
def pears : ℕ := 3 * apples

/-- Katherine has a total of 21 pieces of fruit (apples + pears + bananas) -/
def total_fruit : ℕ := 21

/-- Define the number of bananas Katherine has -/
def bananas : ℕ := total_fruit - (apples + pears)

/-- Prove that Katherine has 5 bananas -/
theorem katherine_has_5_bananas : bananas = 5 := by
  sorry

end katherine_has_5_bananas_l251_25164


namespace trajectory_midpoint_l251_25171

theorem trajectory_midpoint (P M D : ℝ × ℝ) (hP : P.1 ^ 2 + P.2 ^ 2 = 16) (hD : D = (P.1, 0)) (hM : M = ((P.1 + D.1)/2, (P.2 + D.2)/2)) :
  (M.1 ^ 2) / 4 + (M.2 ^ 2) / 16 = 1 :=
by
  sorry

end trajectory_midpoint_l251_25171


namespace min_focal_length_l251_25111

theorem min_focal_length (a b c : ℝ) (h : a > 0 ∧ b > 0) 
    (hyperbola_eq : ∀ x y, ((x^2 / a^2) - (y^2 / b^2) = 1))
    (line_intersects_asymptotes_at : x = a)
    (area_of_triangle : 1/2 * a * (2 * b) = 8) :
    2 * c = 8 :=
by
  sorry

end min_focal_length_l251_25111


namespace minimum_value_of_f_range_of_t_l251_25199

noncomputable def f (x : ℝ) : ℝ := x + 9 / (x - 3)

theorem minimum_value_of_f :
  (∃ x > 3, f x = 9) :=
by
  sorry

theorem range_of_t (t : ℝ) :
  (∀ x > 3, f x ≥ t / (t + 1) + 7) ↔ (t ≤ -2 ∨ t > -1) :=
by
  sorry

end minimum_value_of_f_range_of_t_l251_25199


namespace modem_B_download_time_l251_25118

theorem modem_B_download_time
    (time_A : ℝ) (speed_ratio : ℝ) 
    (h1 : time_A = 25.5) 
    (h2 : speed_ratio = 0.17) : 
    ∃ t : ℝ, t = 110.5425 := 
by
  sorry

end modem_B_download_time_l251_25118


namespace olivia_spent_38_l251_25113

def initial_amount : ℕ := 128
def amount_left : ℕ := 90
def money_spent (initial amount_left : ℕ) : ℕ := initial - amount_left

theorem olivia_spent_38 :
  money_spent initial_amount amount_left = 38 :=
by 
  sorry

end olivia_spent_38_l251_25113


namespace eval_expression_l251_25196

theorem eval_expression :
  let x := 2
  let y := -3
  let z := 1
  x^2 + y^2 - z^2 + 2 * x * y + 3 * z = 0 := by
sorry

end eval_expression_l251_25196


namespace minimum_sum_of_dimensions_l251_25179

theorem minimum_sum_of_dimensions {a b c : ℕ} (h1 : a * b * c = 2310) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  a + b + c ≥ 42 := 
sorry

end minimum_sum_of_dimensions_l251_25179


namespace segment_area_l251_25197

theorem segment_area (d : ℝ) (θ : ℝ) (r := d / 2)
  (A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180))
  (A_sector := (θ / 360) * Real.pi * r^2) :
  θ = 60 →
  d = 10 →
  A_sector - A_triangle = (100 * Real.pi - 75 * Real.sqrt 3) / 24 :=
by
  sorry

end segment_area_l251_25197


namespace quadrilateral_area_is_correct_l251_25147

-- Let's define the situation
structure TriangleDivisions where
  T1_area : ℝ
  T2_area : ℝ
  T3_area : ℝ
  Q_area : ℝ

def triangleDivisionExample : TriangleDivisions :=
  { T1_area := 4,
    T2_area := 9,
    T3_area := 9,
    Q_area := 36 }

-- The statement to prove
theorem quadrilateral_area_is_correct (T : TriangleDivisions) (h1 : T.T1_area = 4) 
  (h2 : T.T2_area = 9) (h3 : T.T3_area = 9) : T.Q_area = 36 :=
by
  sorry

end quadrilateral_area_is_correct_l251_25147


namespace solve_for_n_l251_25166

variable (n : ℚ)

theorem solve_for_n (h : 22 + Real.sqrt (-4 + 18 * n) = 24) : n = 4 / 9 := by
  sorry

end solve_for_n_l251_25166


namespace polynomial_inequality_solution_l251_25169

theorem polynomial_inequality_solution (x : ℝ) :
  x^4 + x^3 - 10 * x^2 + 25 * x > 0 ↔ x > 0 :=
sorry

end polynomial_inequality_solution_l251_25169


namespace count_routes_from_A_to_B_l251_25151

-- Define cities as an inductive type
inductive City
| A
| B
| C
| D
| E

-- Define roads as a list of pairs of cities
def roads : List (City × City) := [
  (City.A, City.B),
  (City.A, City.D),
  (City.B, City.D),
  (City.C, City.D),
  (City.D, City.E),
  (City.B, City.E)
]

-- Define the problem statement
noncomputable def route_count : ℕ :=
  3  -- This should be proven

theorem count_routes_from_A_to_B : route_count = 3 :=
  by
    sorry  -- Proof goes here

end count_routes_from_A_to_B_l251_25151


namespace total_animals_l251_25182

-- Definitions of the initial conditions
def initial_beavers := 20
def initial_chipmunks := 40
def doubled_beavers := 2 * initial_beavers
def decreased_chipmunks := initial_chipmunks - 10

theorem total_animals (initial_beavers initial_chipmunks doubled_beavers decreased_chipmunks : ℕ)
    (h1 : doubled_beavers = 2 * initial_beavers)
    (h2 : decreased_chipmunks = initial_chipmunks - 10) :
    (initial_beavers + initial_chipmunks) + (doubled_beavers + decreased_chipmunks) = 130 :=
by 
  sorry

end total_animals_l251_25182


namespace projection_problem_l251_25101

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

theorem projection_problem :
  let v : ℝ × ℝ := (1, -1/2)
  let sum_v := (v.1 + 1, v.2 + 1)
  projection (3, 5) sum_v = (104/17, 26/17) :=
by
  sorry

end projection_problem_l251_25101


namespace parallel_lines_when_m_is_neg7_l251_25185

-- Given two lines l1 and l2 defined as:
def l1 (m : ℤ) (x y : ℤ) := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℤ) (x y : ℤ) := 2 * x + (5 + m) * y = 8

-- The proof problem to show that l1 is parallel to l2 when m = -7
theorem parallel_lines_when_m_is_neg7 :
  ∃ m : ℤ, (∀ x y : ℤ, l1 m x y → l2 m x y) → m = -7 := 
sorry

end parallel_lines_when_m_is_neg7_l251_25185


namespace geometric_sequence_sum_range_l251_25198

theorem geometric_sequence_sum_range {a : ℕ → ℝ}
  (h4_8: a 4 * a 8 = 9) :
  a 3 + a 9 ∈ Set.Iic (-6) ∪ Set.Ici 6 :=
sorry

end geometric_sequence_sum_range_l251_25198


namespace greatest_possible_sum_of_two_consecutive_even_integers_l251_25122

theorem greatest_possible_sum_of_two_consecutive_even_integers
  (n : ℤ) (h1 : Even n) (h2 : n * (n + 2) < 800) :
  n + (n + 2) = 54 := 
sorry

end greatest_possible_sum_of_two_consecutive_even_integers_l251_25122


namespace brittany_age_when_returning_l251_25173

def rebecca_age : ℕ := 25
def age_difference : ℕ := 3
def vacation_duration : ℕ := 4

theorem brittany_age_when_returning : (rebecca_age + age_difference + vacation_duration) = 32 := by
  sorry

end brittany_age_when_returning_l251_25173


namespace find_y_l251_25112

variable (A B C : Point)

def carla_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees clockwise about point B lands at point C
  sorry

def devon_rotate (θ : ℕ) (A B C : Point) : Prop := 
  -- Definition to indicate point A rotated by θ degrees counterclockwise about point B lands at point C
  sorry

theorem find_y
  (h1 : carla_rotate 690 A B C)
  (h2 : ∀ y, devon_rotate y A B C)
  (h3 : y < 360) :
  ∃ y, y = 30 :=
by
  sorry

end find_y_l251_25112


namespace number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l251_25135

-- Definitions based on the conditions
def peopleO : ℕ := 28
def peopleA : ℕ := 7
def peopleB : ℕ := 9
def peopleAB : ℕ := 3

-- Proof for Question 1
theorem number_of_ways_to_select_one_person : peopleO + peopleA + peopleB + peopleAB = 47 := by
  sorry

-- Proof for Question 2
theorem number_of_ways_to_select_one_person_each_type : peopleO * peopleA * peopleB * peopleAB = 5292 := by
  sorry

end number_of_ways_to_select_one_person_number_of_ways_to_select_one_person_each_type_l251_25135


namespace percentage_fullness_before_storms_l251_25105

def capacity : ℕ := 200 -- capacity in billion gallons
def water_added_by_storms : ℕ := 15 + 30 + 75 -- total water added by storms in billion gallons
def percentage_after : ℕ := 80 -- percentage of fullness after storms
def amount_of_water_after_storms : ℕ := capacity * percentage_after / 100

theorem percentage_fullness_before_storms :
  (amount_of_water_after_storms - water_added_by_storms) * 100 / capacity = 20 := by
  sorry

end percentage_fullness_before_storms_l251_25105


namespace find_denominator_l251_25120

theorem find_denominator (y x : ℝ) (hy : y > 0) (h : (1 * y) / x + (3 * y) / 10 = 0.35 * y) : x = 20 := by
  sorry

end find_denominator_l251_25120


namespace sqrt_25_eq_pm_5_l251_25127

theorem sqrt_25_eq_pm_5 : {x : ℝ | x^2 = 25} = {5, -5} :=
by
  sorry

end sqrt_25_eq_pm_5_l251_25127


namespace cost_buses_minimize_cost_buses_l251_25138

theorem cost_buses
  (x y : ℕ) 
  (h₁ : x + y = 500)
  (h₂ : 2 * x + 3 * y = 1300) :
  x = 200 ∧ y = 300 :=
by 
  sorry

theorem minimize_cost_buses
  (m : ℕ) 
  (h₃: 15 * m + 25 * (8 - m) ≥ 180) :
  m = 2 ∧ (200 * m + 300 * (8 - m) = 2200) :=
by 
  sorry

end cost_buses_minimize_cost_buses_l251_25138


namespace dot_product_eq_neg29_l251_25186

-- Given definitions and conditions
variables (a b : ℝ × ℝ)

-- Theorem to prove the dot product condition.
theorem dot_product_eq_neg29 (h1 : a + b = (2, -4)) (h2 : 3 • a - b = (-10, 16)) :
  a.1 * b.1 + a.2 * b.2 = -29 :=
sorry

end dot_product_eq_neg29_l251_25186


namespace second_box_capacity_l251_25175

-- Given conditions
def height1 := 4 -- height of the first box in cm
def width1 := 2 -- width of the first box in cm
def length1 := 6 -- length of the first box in cm
def clay_capacity1 := 48 -- weight capacity of the first box in grams

def height2 := 3 * height1 -- height of the second box in cm
def width2 := 2 * width1 -- width of the second box in cm
def length2 := length1 -- length of the second box in cm

-- Hypothesis: weight capacity increases quadratically with height
def quadratic_relationship (h1 h2 : ℕ) (capacity1 : ℕ) : ℕ :=
  (h2 / h1) * (h2 / h1) * capacity1

-- The proof problem
theorem second_box_capacity :
  quadratic_relationship height1 height2 clay_capacity1 = 432 :=
by
  -- proof omitted
  sorry

end second_box_capacity_l251_25175


namespace flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l251_25145

-- Problem (a)
theorem flea_reach_B_with_7_jumps (A B : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  B = A + 5 → jumps = 7 → distance = 5 → 
  ways = Nat.choose (7) (1) := 
sorry

-- Problem (b)
theorem flea_reach_C_with_9_jumps (A C : ℤ) (jumps : ℤ) (distance : ℤ) (ways : ℕ) :
  C = A + 5 → jumps = 9 → distance = 5 → 
  ways = Nat.choose (9) (2) :=
sorry

-- Problem (c)
theorem flea_cannot_reach_D_with_2028_jumps (A D : ℤ) (jumps : ℤ) (distance : ℤ) :
  D = A + 2013 → jumps = 2028 → distance = 2013 → 
  ∃ x y : ℤ, x + y = 2028 ∧ x - y = 2013 → false :=
sorry

end flea_reach_B_with_7_jumps_flea_reach_C_with_9_jumps_flea_cannot_reach_D_with_2028_jumps_l251_25145


namespace find_a_l251_25167

noncomputable def f (x a : ℝ) : ℝ := 4 * x ^ 2 - 4 * a * x + a ^ 2 - 2 * a + 2

theorem find_a (a : ℝ) : 
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧  ∀ y : ℝ, 0 ≤ y ∧ y ≤ 2 → f y a ≤ f x a) ∧ f 0 a = 3 ∧ f 2 a = 3 → 
  a = 5 - Real.sqrt 10 ∨ a = 1 + Real.sqrt 2 := 
sorry

end find_a_l251_25167


namespace max_distinct_sums_l251_25133

/-- Given 3 boys and 20 girls standing in a row, each child counts the number of girls to their 
left and the number of boys to their right and adds these two counts together. Prove that 
the maximum number of different sums that the children could have obtained is 20. -/
theorem max_distinct_sums (boys girls : ℕ) (total_children : ℕ) 
  (h_boys : boys = 3) (h_girls : girls = 20) (h_total : total_children = boys + girls) : 
  ∃ (max_sums : ℕ), max_sums = 20 := 
by 
  sorry

end max_distinct_sums_l251_25133


namespace find_younger_age_l251_25144

def younger_age (y e : ℕ) : Prop :=
  (e = y + 20) ∧ (e - 5 = 5 * (y - 5))

theorem find_younger_age (y e : ℕ) (h : younger_age y e) : y = 10 :=
by sorry

end find_younger_age_l251_25144


namespace largest_expression_is_A_l251_25146

noncomputable def A : ℝ := 3009 / 3008 + 3009 / 3010
noncomputable def B : ℝ := 3011 / 3010 + 3011 / 3012
noncomputable def C : ℝ := 3010 / 3009 + 3010 / 3011

theorem largest_expression_is_A : A > B ∧ A > C := by
  sorry

end largest_expression_is_A_l251_25146


namespace even_function_implies_a_is_2_l251_25116

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l251_25116


namespace average_side_lengths_of_squares_l251_25195

theorem average_side_lengths_of_squares:
  let a₁ := 25
  let a₂ := 36
  let a₃ := 64

  let s₁ := Real.sqrt a₁
  let s₂ := Real.sqrt a₂
  let s₃ := Real.sqrt a₃

  (s₁ + s₂ + s₃) / 3 = 19 / 3 :=
by 
  sorry

end average_side_lengths_of_squares_l251_25195


namespace solve_for_b_l251_25126

noncomputable def g (a b : ℝ) (x : ℝ) := 1 / (2 * a * x + 3 * b)

theorem solve_for_b (a b : ℝ) (h_a : a ≠ 0) (h_b : b ≠ 0) :
  (g a b (2) = 1 / (4 * a + 3 * b)) → (4 * a + 3 * b = 1 / 2) → b = (1 - 4 * a) / 3 :=
by
  sorry

end solve_for_b_l251_25126


namespace problem_equivalence_l251_25121

theorem problem_equivalence (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (⌊(a^2 : ℚ) / b⌋ + ⌊(b^2 : ℚ) / a⌋ = ⌊(a^2 + b^2 : ℚ) / (a * b)⌋ + a * b) ↔
  (∃ k : ℕ, k > 0 ∧ ((a = k ∧ b = k^2 + 1) ∨ (a = k^2 + 1 ∧ b = k))) :=
sorry

end problem_equivalence_l251_25121


namespace gen_sequence_term_l251_25102

theorem gen_sequence_term (a : ℕ → ℕ) (n : ℕ) (h1 : a 1 = 1) (h2 : ∀ k, a (k + 1) = 3 * a k + 1) :
  a n = (3^n - 1) / 2 := by
  sorry

end gen_sequence_term_l251_25102


namespace transformed_function_correct_l251_25187

-- Given function
def f (x : ℝ) : ℝ := 2 * x + 1

-- Main theorem to be proven
theorem transformed_function_correct (x : ℝ) (h : 2 ≤ x ∧ x ≤ 4) : 
  f (x - 1) = 2 * x - 1 :=
by {
  sorry
}

end transformed_function_correct_l251_25187


namespace number_of_days_b_worked_l251_25139

variables (d_a : ℕ) (d_c : ℕ) (total_earnings : ℝ)
variables (wage_ratio : ℝ) (wage_c : ℝ) (d_b : ℕ) (wages : ℝ)
variables (total_wage_a : ℝ) (total_wage_c : ℝ) (total_wage_b : ℝ)

-- Given conditions
def given_conditions :=
  d_a = 6 ∧
  d_c = 4 ∧
  wage_c = 95 ∧
  wage_ratio = wage_c / 5 ∧
  wages = 3 * wage_ratio ∧
  total_earnings = 1406 ∧
  total_wage_a = d_a * wages ∧
  total_wage_c = d_c * wage_c ∧
  total_wage_b = d_b * (4 * wage_ratio) ∧
  total_wage_a + total_wage_b + total_wage_c = total_earnings

-- Theorem to prove
theorem number_of_days_b_worked :
  given_conditions d_a d_c total_earnings wage_ratio wage_c d_b wages total_wage_a total_wage_c total_wage_b →
  d_b = 9 :=
by
  intro h
  sorry

end number_of_days_b_worked_l251_25139


namespace tangent_line_ellipse_l251_25117

variables {x y x0 y0 r a b : ℝ}

/-- Given the tangent line to the circle x^2 + y^2 = r^2 at the point (x0, y0) is x0 * x + y0 * y = r^2,
we prove the tangent line to the ellipse x^2 / a^2 + y^2 / b^2 = 1 at the point (x0, y0) is x0 * x / a^2 + y0 * y / b^2 = 1. -/
theorem tangent_line_ellipse :
  (x0 * x + y0 * y = r^2) →
  (x0^2 / a^2 + y0^2 / b^2 = 1) →
  (x0 * x / a^2 + y0 * y / b^2 = 1) :=
by
  intros hc he
  sorry

end tangent_line_ellipse_l251_25117


namespace boyden_family_tickets_l251_25140

theorem boyden_family_tickets (child_ticket_cost : ℕ) (adult_ticket_cost : ℕ) (total_cost : ℕ) (num_adults : ℕ) (num_children : ℕ) :
  adult_ticket_cost = child_ticket_cost + 6 →
  total_cost = 77 →
  adult_ticket_cost = 19 →
  num_adults = 2 →
  num_children = 3 →
  num_adults * adult_ticket_cost + num_children * child_ticket_cost = total_cost →
  num_adults + num_children = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end boyden_family_tickets_l251_25140


namespace yellow_more_than_green_by_l251_25130

-- Define the problem using the given conditions.
def weight_yellow_block : ℝ := 0.6
def weight_green_block  : ℝ := 0.4

-- State the theorem that the yellow block weighs 0.2 pounds more than the green block.
theorem yellow_more_than_green_by : weight_yellow_block - weight_green_block = 0.2 :=
by sorry

end yellow_more_than_green_by_l251_25130


namespace lisa_walks_distance_per_minute_l251_25184

-- Variables and conditions
variable (d : ℤ) -- distance that Lisa walks each minute (what we're solving for)
variable (daily_distance : ℤ) -- distance that Lisa walks each hour
variable (total_distance_in_two_days : ℤ := 1200) -- total distance in two days
variable (hours_per_day : ℤ := 1) -- one hour per day

-- Given conditions
axiom walks_for_an_hour_each_day : ∀ (d: ℤ), daily_distance = d * 60
axiom walks_1200_meters_in_two_days : ∀ (d: ℤ), total_distance_in_two_days = 2 * daily_distance

-- The theorem we want to prove
theorem lisa_walks_distance_per_minute : (d = 10) :=
by
  -- TODO: complete the proof
  sorry

end lisa_walks_distance_per_minute_l251_25184


namespace remainder_of_sum_l251_25108

theorem remainder_of_sum (x y z : ℕ) (h1 : x % 15 = 6) (h2 : y % 15 = 9) (h3 : z % 15 = 3) : 
  (x + y + z) % 15 = 3 := 
  sorry

end remainder_of_sum_l251_25108


namespace marbles_left_l251_25158

def initial_marbles : ℝ := 150
def lost_marbles : ℝ := 58.5
def given_away_marbles : ℝ := 37.2
def found_marbles : ℝ := 10.8

theorem marbles_left :
  initial_marbles - lost_marbles - given_away_marbles + found_marbles = 65.1 :=
by 
  sorry

end marbles_left_l251_25158


namespace total_shaded_area_l251_25109

theorem total_shaded_area (side_len : ℝ) (segment_len : ℝ) (h : ℝ) :
  side_len = 8 ∧ segment_len = 1 ∧ 0 ≤ h ∧ h ≤ 8 →
  (segment_len * h / 2 + segment_len * (side_len - h) / 2) = 4 := 
by
  intro h_cond
  rcases h_cond with ⟨h_side_len, h_segment_len, h_nonneg, h_le⟩
  -- Directly state the simplified computation
  sorry

end total_shaded_area_l251_25109


namespace edric_monthly_salary_l251_25156

theorem edric_monthly_salary 
  (hours_per_day : ℝ)
  (days_per_week : ℝ)
  (weeks_per_month : ℝ)
  (hourly_rate : ℝ) :
  hours_per_day = 8 ∧ days_per_week = 6 ∧ weeks_per_month = 4.33 ∧ hourly_rate = 3 →
  (hours_per_day * days_per_week * weeks_per_month * hourly_rate) = 623.52 :=
by
  intros h
  sorry

end edric_monthly_salary_l251_25156


namespace min_f_x_eq_one_implies_a_eq_zero_or_two_l251_25124

theorem min_f_x_eq_one_implies_a_eq_zero_or_two (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x + a| = 1) → (a = 0 ∨ a = 2) := by
  sorry

end min_f_x_eq_one_implies_a_eq_zero_or_two_l251_25124


namespace statement_1_statement_2_statement_3_all_statements_correct_l251_25183

-- Define the function f and the axioms/conditions given in the problem
def f : ℕ → ℕ → ℕ := sorry

-- Conditions
axiom f_initial : f 1 1 = 1
axiom f_nat : ∀ m n : ℕ, m > 0 → n > 0 → f m n > 0
axiom f_condition_1 : ∀ m n : ℕ, m > 0 → n > 0 → f m (n + 1) = f m n + 2
axiom f_condition_2 : ∀ m : ℕ, m > 0 → f (m + 1) 1 = 2 * f m 1

-- Statements to be proved
theorem statement_1 : f 1 5 = 9 := sorry
theorem statement_2 : f 5 1 = 16 := sorry
theorem statement_3 : f 5 6 = 26 := sorry

theorem all_statements_correct : (f 1 5 = 9) ∧ (f 5 1 = 16) ∧ (f 5 6 = 26) := by
  exact ⟨statement_1, statement_2, statement_3⟩

end statement_1_statement_2_statement_3_all_statements_correct_l251_25183


namespace abs_neg_six_l251_25178

theorem abs_neg_six : abs (-6) = 6 :=
sorry

end abs_neg_six_l251_25178


namespace numBoysInClassroom_l251_25161

-- Definitions based on the problem conditions
def numGirls : ℕ := 10
def girlsToBoysRatio : ℝ := 0.5

-- The statement to prove
theorem numBoysInClassroom : ∃ B : ℕ, girlsToBoysRatio * B = numGirls ∧ B = 20 :=
by
  -- Proof goes here
  sorry

end numBoysInClassroom_l251_25161


namespace evaluate_custom_operation_l251_25106

def custom_operation (A B : ℕ) : ℕ :=
  (A + 2 * B) * (A - B)

theorem evaluate_custom_operation : custom_operation 7 5 = 34 :=
by
  sorry

end evaluate_custom_operation_l251_25106


namespace average_salary_of_feb_mar_apr_may_l251_25107

theorem average_salary_of_feb_mar_apr_may
  (avg_salary_jan_feb_mar_apr : ℝ)
  (salary_jan : ℝ)
  (salary_may : ℝ)
  (total_salary_feb_mar_apr : ℝ)
  (total_salary_feb_mar_apr_may: ℝ)
  (n_months: ℝ): 
  avg_salary_jan_feb_mar_apr = 8000 ∧ 
  salary_jan = 6100 ∧ 
  salary_may = 6500 ∧ 
  total_salary_feb_mar_apr = (avg_salary_jan_feb_mar_apr * 4 - salary_jan) ∧
  total_salary_feb_mar_apr_may = (total_salary_feb_mar_apr + salary_may) ∧
  n_months = (total_salary_feb_mar_apr_may / 8100) →
  n_months = 4 :=
by
  intros 
  sorry

end average_salary_of_feb_mar_apr_may_l251_25107


namespace total_cost_of_products_l251_25114

-- Conditions
def smartphone_price := 300
def personal_computer_price := smartphone_price + 500
def advanced_tablet_price := smartphone_price + personal_computer_price

-- Theorem statement for the total cost of one of each product
theorem total_cost_of_products :
  smartphone_price + personal_computer_price + advanced_tablet_price = 2200 := by
  sorry

end total_cost_of_products_l251_25114


namespace find_sum_of_smallest_multiples_l251_25115

-- Define c as the smallest positive two-digit multiple of 5
def is_smallest_two_digit_multiple_of_5 (c : ℕ) : Prop :=
  c ≥ 10 ∧ c % 5 = 0 ∧ ∀ n, (n ≥ 10 ∧ n % 5 = 0) → n ≥ c

-- Define d as the smallest positive three-digit multiple of 7
def is_smallest_three_digit_multiple_of_7 (d : ℕ) : Prop :=
  d ≥ 100 ∧ d % 7 = 0 ∧ ∀ n, (n ≥ 100 ∧ n % 7 = 0) → n ≥ d

theorem find_sum_of_smallest_multiples :
  ∃ c d : ℕ, is_smallest_two_digit_multiple_of_5 c ∧ is_smallest_three_digit_multiple_of_7 d ∧ c + d = 115 :=
by
  sorry

end find_sum_of_smallest_multiples_l251_25115


namespace coins_in_bag_l251_25157

theorem coins_in_bag (x : ℕ) (h : x + x / 2 + x / 4 = 105) : x = 60 :=
by
  sorry

end coins_in_bag_l251_25157


namespace paint_containers_left_l251_25149

theorem paint_containers_left (initial_containers : ℕ)
  (tiled_wall_containers : ℕ)
  (ceiling_containers : ℕ)
  (gradient_walls : ℕ)
  (additional_gradient_containers_per_wall : ℕ)
  (remaining_containers : ℕ) :
  initial_containers = 16 →
  tiled_wall_containers = 1 →
  ceiling_containers = 1 →
  gradient_walls = 3 →
  additional_gradient_containers_per_wall = 1 →
  remaining_containers = initial_containers - tiled_wall_containers - (ceiling_containers + gradient_walls * additional_gradient_containers_per_wall) →
  remaining_containers = 11 :=
by
  intros h_initial h_tiled h_ceiling h_gradient_walls h_additional_gradient h_remaining_calc
  rw [h_initial, h_tiled, h_ceiling, h_gradient_walls, h_additional_gradient] at h_remaining_calc
  exact h_remaining_calc

end paint_containers_left_l251_25149


namespace coronavirus_transmission_l251_25148

theorem coronavirus_transmission (x : ℝ) 
  (H: (1 + x) ^ 2 = 225) : (1 + x) ^ 2 = 225 :=
  by
    sorry

end coronavirus_transmission_l251_25148


namespace new_apps_added_l251_25104

theorem new_apps_added (x : ℕ) (h1 : 15 + x - (x + 1) = 14) : x = 0 :=
by
  sorry

end new_apps_added_l251_25104


namespace max_value_y_l251_25131

theorem max_value_y (x : ℝ) (h : x < 5 / 4) : 
  (4 * x - 2 + 1 / (4 * x - 5)) ≤ 1 :=
sorry

end max_value_y_l251_25131


namespace second_number_deduction_l251_25162

theorem second_number_deduction
  (x : ℝ)
  (h1 : (10 * 16 = 10 * x + (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9)))
  (h2 : 2.5 + (x+1 - y) + 6.5 + 8.5 + 10.5 + 12.5 + 14.5 + 16.5 + 18.5 + 20.5 = 115)
  : y = 8 :=
by
  -- This is where the proof would go, but we'll leave it as 'sorry' for now.
  sorry

end second_number_deduction_l251_25162


namespace find_second_sum_l251_25193

theorem find_second_sum (S : ℤ) (x : ℤ) (h_S : S = 2678)
  (h_eq_interest : x * 3 * 8 = (S - x) * 5 * 3) : (S - x) = 1648 :=
by {
  sorry
}

end find_second_sum_l251_25193


namespace geese_flock_size_l251_25129

theorem geese_flock_size : 
  ∃ x : ℕ, x + x + (x / 2) + (x / 4) + 1 = 100 ∧ x = 36 := 
by
  sorry

end geese_flock_size_l251_25129


namespace domain_of_f_l251_25155

def denominator (x : ℝ) : ℝ := x^2 - 4 * x + 3

def is_defined (x : ℝ) : Prop := denominator x ≠ 0

theorem domain_of_f :
  {x : ℝ // is_defined x} = {x : ℝ | x < 1} ∪ {x : ℝ | 1 < x ∧ x < 3} ∪ {x : ℝ | x > 3} :=
by
  sorry

end domain_of_f_l251_25155


namespace weekly_allowance_l251_25176

theorem weekly_allowance (A : ℝ) (h1 : A / 2 + 6 = 11) : A = 10 := 
by 
  sorry

end weekly_allowance_l251_25176


namespace exterior_angle_of_octagon_is_45_degrees_l251_25142

noncomputable def exterior_angle_of_regular_octagon : ℝ :=
  let n : ℝ := 8
  let interior_angle_sum := 180 * (n - 2) -- This is the sum of interior angles of any n-gon
  let each_interior_angle := interior_angle_sum / n -- Each interior angle in a regular polygon
  let each_exterior_angle := 180 - each_interior_angle -- Exterior angle is supplement of interior angle
  each_exterior_angle

theorem exterior_angle_of_octagon_is_45_degrees :
  exterior_angle_of_regular_octagon = 45 := by
  sorry

end exterior_angle_of_octagon_is_45_degrees_l251_25142


namespace area_of_rectangular_garden_l251_25110

theorem area_of_rectangular_garden (length width : ℝ) (h_length : length = 2.5) (h_width : width = 0.48) :
  length * width = 1.2 :=
by
  sorry

end area_of_rectangular_garden_l251_25110


namespace car_clock_time_correct_l251_25128

noncomputable def car_clock (t : ℝ) : ℝ := t * (4 / 3)

theorem car_clock_time_correct :
  ∀ t_real t_car,
  (car_clock 0 = 0) ∧
  (car_clock 0.5 = 2 / 3) ∧
  (car_clock t_real = t_car) ∧
  (t_car = (8 : ℝ)) → (t_real = 6) → (t_real + 1 = 7) :=
by
  intro t_real t_car h
  sorry

end car_clock_time_correct_l251_25128


namespace product_of_three_numbers_is_correct_l251_25163

noncomputable def sum_three_numbers_product (x y z n : ℚ) : Prop :=
  x + y + z = 200 ∧
  8 * x = y - 12 ∧
  8 * x = z + 12 ∧
  (x * y * z = 502147200 / 4913)

theorem product_of_three_numbers_is_correct :
  ∃ (x y z n : ℚ), sum_three_numbers_product x y z n :=
by
  sorry

end product_of_three_numbers_is_correct_l251_25163


namespace fruit_bowl_l251_25100

variable {A P B : ℕ}

theorem fruit_bowl : (P = A + 2) → (B = P + 3) → (A + P + B = 19) → B = 9 :=
by
  intros h1 h2 h3
  sorry

end fruit_bowl_l251_25100


namespace single_elimination_games_l251_25192

theorem single_elimination_games (n : ℕ) (h : n = 512) : ∃ g : ℕ, g = 511 := by
  sorry

end single_elimination_games_l251_25192


namespace P_iff_Q_l251_25154

def P (x : ℝ) := x > 1 ∨ x < -1
def Q (x : ℝ) := |x + 1| + |x - 1| > 2

theorem P_iff_Q : ∀ x, P x ↔ Q x :=
by
  intros x
  sorry

end P_iff_Q_l251_25154


namespace max_black_balls_C_is_22_l251_25125

-- Define the given parameters
noncomputable def balls_A : ℕ := 100
noncomputable def black_balls_A : ℕ := 15
noncomputable def balls_B : ℕ := 50
noncomputable def balls_C : ℕ := 80
noncomputable def probability : ℚ := 101 / 600

-- Define the maximum number of black balls in box C given the conditions
theorem max_black_balls_C_is_22 (y : ℕ) (h : (1/3 * (black_balls_A / balls_A) + 1/3 * (y / balls_B) + 1/3 * (22 / balls_C)) = probability  ) :
  ∃ (x : ℕ), x ≤ 22 := sorry

end max_black_balls_C_is_22_l251_25125


namespace quadratic_range_l251_25189

theorem quadratic_range (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + 1 < 0) → (-1 ≤ a ∧ a ≤ 1) :=
by
  sorry

end quadratic_range_l251_25189


namespace pictures_remaining_l251_25194

-- Define the initial number of pictures taken at the zoo and museum
def zoo_pictures : Nat := 50
def museum_pictures : Nat := 8
-- Define the number of pictures deleted
def deleted_pictures : Nat := 38

-- Define the total number of pictures taken initially and remaining after deletion
def total_pictures : Nat := zoo_pictures + museum_pictures
def remaining_pictures : Nat := total_pictures - deleted_pictures

theorem pictures_remaining : remaining_pictures = 20 := 
by 
  -- This theorem states that, given the conditions, the remaining pictures count must be 20
  sorry

end pictures_remaining_l251_25194


namespace handshake_count_l251_25181

-- Define the conditions
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_people : ℕ := num_companies * reps_per_company
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

-- Define the theorem to prove
theorem handshake_count : 
  total_people * (total_people - 1 - (reps_per_company - 1)) / 2 = 160 := 
by
  -- Here should be the proof, but it is omitted
  sorry

end handshake_count_l251_25181


namespace equidistant_point_x_axis_l251_25119

theorem equidistant_point_x_axis (x : ℝ) (C D : ℝ × ℝ)
  (hC : C = (-3, 0))
  (hD : D = (0, 5))
  (heqdist : ∀ p : ℝ × ℝ, p.2 = 0 → 
    dist p C = dist p D) :
  x = 8 / 3 :=
by
  sorry

end equidistant_point_x_axis_l251_25119


namespace jerry_original_butterflies_l251_25153

/-- Define the number of butterflies Jerry originally had -/
def original_butterflies (let_go : ℕ) (now_has : ℕ) : ℕ := let_go + now_has

/-- Given conditions -/
def let_go : ℕ := 11
def now_has : ℕ := 82

/-- Theorem to prove the number of butterflies Jerry originally had -/
theorem jerry_original_butterflies : original_butterflies let_go now_has = 93 :=
by
  sorry

end jerry_original_butterflies_l251_25153


namespace sum_of_roots_is_zero_l251_25136

variable {R : Type*} [LinearOrderedField R]

-- Define the function f : R -> R and its properties
variable (f : R → R)
variable (even_f : ∀ x, f x = f (-x))
variable (roots_f : Finset R)
variable (roots_f_four : roots_f.card = 4)
variable (roots_f_set : ∀ x, x ∈ roots_f → f x = 0)

theorem sum_of_roots_is_zero : (roots_f.sum id) = 0 := 
sorry

end sum_of_roots_is_zero_l251_25136


namespace minimum_omega_l251_25137

theorem minimum_omega 
  (ω : ℝ)
  (hω : ω > 0)
  (h_shift : ∃ T > 0, T = 2 * π / ω ∧ T = 2 * π / 3) : 
  ω = 3 := 
sorry

end minimum_omega_l251_25137


namespace baseball_cards_remaining_l251_25103

-- Define the number of baseball cards Mike originally had
def original_cards : ℕ := 87

-- Define the number of baseball cards Sam bought from Mike
def cards_bought : ℕ := 13

-- Prove that the remaining number of baseball cards Mike has is 74
theorem baseball_cards_remaining : original_cards - cards_bought = 74 := by
  sorry

end baseball_cards_remaining_l251_25103


namespace remainder_six_n_mod_four_l251_25177

theorem remainder_six_n_mod_four (n : ℤ) (h : n % 4 = 3) : (6 * n) % 4 = 2 := 
by sorry

end remainder_six_n_mod_four_l251_25177
