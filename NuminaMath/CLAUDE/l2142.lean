import Mathlib

namespace expression_value_l2142_214271

theorem expression_value (x y : ℝ) (h : (x + 2)^2 + |y - 1/2| = 0) :
  (x - 2*y) * (x + 2*y) - (x - 2*y)^2 = -6 := by
  sorry

end expression_value_l2142_214271


namespace power_boat_travel_time_l2142_214252

/-- The time taken for a power boat to travel downstream from dock A to dock B,
    given the conditions of the river journey problem. -/
theorem power_boat_travel_time
  (r : ℝ) -- speed of the river current
  (p : ℝ) -- relative speed of the power boat with respect to the river
  (h1 : r > 0) -- river speed is positive
  (h2 : p > r) -- power boat speed is greater than river speed
  : ∃ t : ℝ,
    t > 0 ∧
    t = (12 * r) / (6 * p - r) ∧
    (p + r) * t + (p - r) * (12 - t) = 12 * r :=
by sorry

end power_boat_travel_time_l2142_214252


namespace layla_phone_probability_l2142_214270

def first_segment_choices : ℕ := 3
def last_segment_digits : ℕ := 4

theorem layla_phone_probability :
  (1 : ℚ) / (first_segment_choices * Nat.factorial last_segment_digits) = 1 / 72 :=
by sorry

end layla_phone_probability_l2142_214270


namespace price_reduction_profit_l2142_214221

/-- Represents the daily sales and profit scenario of a product in a shopping mall -/
structure MallSales where
  initialSales : ℕ  -- Initial daily sales in units
  initialProfit : ℕ  -- Initial profit per unit in yuan
  salesIncrease : ℕ  -- Increase in sales units per yuan of price reduction
  priceReduction : ℕ  -- Price reduction per unit in yuan

/-- Calculates the daily profit based on the given sales scenario -/
def dailyProfit (m : MallSales) : ℕ :=
  (m.initialSales + m.salesIncrease * m.priceReduction) * (m.initialProfit - m.priceReduction)

/-- Theorem stating that a price reduction of 20 yuan results in a daily profit of 2100 yuan -/
theorem price_reduction_profit (m : MallSales) 
  (h1 : m.initialSales = 30)
  (h2 : m.initialProfit = 50)
  (h3 : m.salesIncrease = 2)
  (h4 : m.priceReduction = 20) :
  dailyProfit m = 2100 := by
  sorry

#eval dailyProfit { initialSales := 30, initialProfit := 50, salesIncrease := 2, priceReduction := 20 }

end price_reduction_profit_l2142_214221


namespace rectangle_circle_intersection_area_l2142_214207

/-- The area of intersection between a rectangle and a circle with shared center -/
theorem rectangle_circle_intersection_area :
  ∀ (rectangle_length rectangle_width circle_radius : ℝ),
  rectangle_length = 10 →
  rectangle_width = 2 * Real.sqrt 3 →
  circle_radius = 3 →
  ∃ (intersection_area : ℝ),
  intersection_area = (9 * Real.pi) / 2 :=
by sorry

end rectangle_circle_intersection_area_l2142_214207


namespace sin_240_degrees_l2142_214230

theorem sin_240_degrees : Real.sin (240 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end sin_240_degrees_l2142_214230


namespace asaf_age_l2142_214222

theorem asaf_age :
  ∀ (asaf_age alexander_age asaf_pencils : ℕ),
    -- Sum of ages is 140
    asaf_age + alexander_age = 140 →
    -- Age difference is half of Asaf's pencils
    alexander_age - asaf_age = asaf_pencils / 2 →
    -- Total pencils is 220
    asaf_pencils + (asaf_pencils + 60) = 220 →
    -- Asaf's age is 90
    asaf_age = 90 := by
  sorry

end asaf_age_l2142_214222


namespace parallelogram_height_l2142_214283

/-- Given a parallelogram with area 576 cm² and base 32 cm, its height is 18 cm. -/
theorem parallelogram_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 576 ∧ base = 32 ∧ area = base * height → height = 18 :=
by sorry

end parallelogram_height_l2142_214283


namespace perimeter_semicircular_bounded_rectangle_l2142_214297

/-- The perimeter of a region bounded by semicircular arcs constructed on each side of a rectangle --/
theorem perimeter_semicircular_bounded_rectangle (l w : ℝ) (hl : l = 4 / π) (hw : w = 2 / π) :
  let semicircle_length := π * l / 2
  let semicircle_width := π * w / 2
  semicircle_length + semicircle_length + semicircle_width + semicircle_width = 6 := by
  sorry

end perimeter_semicircular_bounded_rectangle_l2142_214297


namespace jen_candy_profit_l2142_214234

/-- Calculates the profit from selling candy bars --/
def candy_profit (buy_price sell_price : ℕ) (bought sold : ℕ) : ℕ :=
  (sell_price - buy_price) * sold

/-- Proves that Jen's profit from selling candy bars is 960 cents --/
theorem jen_candy_profit : candy_profit 80 100 50 48 = 960 := by
  sorry

end jen_candy_profit_l2142_214234


namespace sum_of_squares_lower_bound_l2142_214231

theorem sum_of_squares_lower_bound (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (sum_eq_6 : a + b + c = 6) : 
  a^2 + b^2 + c^2 ≥ 12 := by
  sorry

end sum_of_squares_lower_bound_l2142_214231


namespace largest_power_is_396_l2142_214243

def pow (n : ℕ) : ℕ :=
  sorry

def largest_divisible_power (upper_bound : ℕ) : ℕ :=
  sorry

theorem largest_power_is_396 :
  largest_divisible_power 4000 = 396 :=
sorry

end largest_power_is_396_l2142_214243


namespace carreys_rental_cost_l2142_214262

/-- The cost per kilometer for Carrey's car rental -/
def carreys_cost_per_km : ℝ := 0.25

/-- The initial cost for Carrey's car rental -/
def carreys_initial_cost : ℝ := 20

/-- The initial cost for Samuel's car rental -/
def samuels_initial_cost : ℝ := 24

/-- The cost per kilometer for Samuel's car rental -/
def samuels_cost_per_km : ℝ := 0.16

/-- The distance driven by both Carrey and Samuel -/
def distance_driven : ℝ := 44.44444444444444

theorem carreys_rental_cost (x : ℝ) :
  carreys_initial_cost + x * distance_driven =
  samuels_initial_cost + samuels_cost_per_km * distance_driven →
  x = carreys_cost_per_km :=
by sorry

end carreys_rental_cost_l2142_214262


namespace average_speed_theorem_l2142_214279

theorem average_speed_theorem (total_distance : ℝ) (distance1 : ℝ) (speed1 : ℝ) (distance2 : ℝ) (speed2 : ℝ) 
  (h1 : total_distance = 80)
  (h2 : distance1 = 30)
  (h3 : speed1 = 30)
  (h4 : distance2 = 50)
  (h5 : speed2 = 50)
  (h6 : total_distance = distance1 + distance2) :
  (total_distance) / ((distance1 / speed1) + (distance2 / speed2)) = 40 := by
  sorry

end average_speed_theorem_l2142_214279


namespace symmetric_sine_cosine_l2142_214284

/-- A function f is symmetric about a line x = c if f(c + h) = f(c - h) for all h -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ h, f (c + h) = f (c - h)

/-- The main theorem -/
theorem symmetric_sine_cosine (a : ℝ) :
  SymmetricAbout (fun x ↦ Real.sin (2 * x) + a * Real.cos (2 * x)) (π / 8) → a = 1 := by
  sorry

end symmetric_sine_cosine_l2142_214284


namespace sum_relations_l2142_214249

theorem sum_relations (a b c d : ℝ) 
  (hab : a + b = 4)
  (hcd : c + d = 3)
  (had : a + d = 2) :
  b + c = 5 := by
  sorry

end sum_relations_l2142_214249


namespace nils_geese_count_l2142_214272

/-- Represents the number of geese on Nils' farm -/
def n : ℕ := sorry

/-- Represents the number of days the feed lasts initially -/
def k : ℕ := sorry

/-- The amount of feed consumed by one goose per day -/
def x : ℝ := sorry

/-- The total amount of feed available -/
def A : ℝ := sorry

/-- The feed lasts k days for n geese -/
axiom initial_feed : A = k * x * n

/-- The feed lasts (k + 20) days for (n - 75) geese -/
axiom sell_75_geese : A = (k + 20) * x * (n - 75)

/-- The feed lasts (k - 15) days for (n + 100) geese -/
axiom buy_100_geese : A = (k - 15) * x * (n + 100)

theorem nils_geese_count : n = 300 := by sorry

end nils_geese_count_l2142_214272


namespace hyperbola_perpendicular_product_l2142_214200

/-- The hyperbola equation -/
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 = 1

/-- The asymptotes of the hyperbola -/
def asymptote₁ (x y : ℝ) : Prop := x + Real.sqrt 3 * y = 0
def asymptote₂ (x y : ℝ) : Prop := x - Real.sqrt 3 * y = 0

/-- A point on the hyperbola -/
def P : ℝ × ℝ → Prop := λ p => hyperbola p.1 p.2

/-- Feet of perpendiculars from P to asymptotes -/
def A : (ℝ × ℝ) → (ℝ × ℝ) → Prop := λ p a => 
  asymptote₁ a.1 a.2 ∧ (p.1 - a.1) * (Real.sqrt 3) + (p.2 - a.2) = 0

def B : (ℝ × ℝ) → (ℝ × ℝ) → Prop := λ p b => 
  asymptote₂ b.1 b.2 ∧ (p.1 - b.1) * (Real.sqrt 3) - (p.2 - b.2) = 0

/-- The theorem to be proved -/
theorem hyperbola_perpendicular_product (p a b : ℝ × ℝ) :
  P p → A p a → B p b → 
  (p.1 - a.1) * (p.1 - b.1) + (p.2 - a.2) * (p.2 - b.2) = -3/8 := by
  sorry

end hyperbola_perpendicular_product_l2142_214200


namespace prob_second_white_given_first_white_l2142_214291

/-- Represents the total number of balls -/
def total_balls : ℕ := 9

/-- Represents the number of white balls -/
def white_balls : ℕ := 5

/-- Represents the number of black balls -/
def black_balls : ℕ := 4

/-- Represents the probability of drawing a white ball first -/
def prob_first_white : ℚ := white_balls / total_balls

/-- Represents the probability of drawing two white balls consecutively -/
def prob_both_white : ℚ := (white_balls * (white_balls - 1)) / (total_balls * (total_balls - 1))

/-- Theorem stating the probability of drawing a white ball second, given the first was white -/
theorem prob_second_white_given_first_white :
  prob_both_white / prob_first_white = 1 / 2 := by sorry

end prob_second_white_given_first_white_l2142_214291


namespace rose_group_size_l2142_214260

theorem rose_group_size (group_size : ℕ) : 
  (9 > 0) →
  (group_size > 0) →
  (Nat.lcm 9 group_size = 171) →
  (171 % group_size = 0) →
  (9 % group_size ≠ 0) →
  group_size = 19 := by
  sorry

end rose_group_size_l2142_214260


namespace sin_cos_difference_75_15_l2142_214237

theorem sin_cos_difference_75_15 :
  Real.sin (75 * π / 180) * Real.cos (15 * π / 180) -
  Real.cos (75 * π / 180) * Real.sin (15 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_difference_75_15_l2142_214237


namespace final_price_is_91_percent_l2142_214201

/-- Represents the price increase factor -/
def price_increase : ℝ := 1.4

/-- Represents the discount factor -/
def discount : ℝ := 0.65

/-- Theorem stating that the final price after increase and discount is 91% of the original price -/
theorem final_price_is_91_percent (original_price : ℝ) :
  discount * (price_increase * original_price) = 0.91 * original_price := by
  sorry

end final_price_is_91_percent_l2142_214201


namespace lg_6_equals_a_plus_b_l2142_214209

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Theorem statement
theorem lg_6_equals_a_plus_b (a b : ℝ) (h1 : lg 2 = a) (h2 : lg 3 = b) : lg 6 = a + b := by
  sorry

end lg_6_equals_a_plus_b_l2142_214209


namespace potato_harvest_problem_l2142_214211

theorem potato_harvest_problem :
  ∃! (x y : ℕ+), 
    x * y * 5 = 45715 ∧ 
    x ≤ 100 ∧  -- reasonable upper bound for number of students
    y ≤ 1000   -- reasonable upper bound for daily output per student
  := by sorry

end potato_harvest_problem_l2142_214211


namespace range_of_M_l2142_214288

theorem range_of_M (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a * b < 1) :
  let M := 1 / (1 + a) + 1 / (1 + b)
  1 < M ∧ M < 2 := by
sorry

end range_of_M_l2142_214288


namespace halfway_between_fractions_average_of_fractions_l2142_214282

theorem halfway_between_fractions : 
  (2 : ℚ) / 7 + (4 : ℚ) / 9 = (46 : ℚ) / 63 :=
by sorry

theorem average_of_fractions : 
  ((2 : ℚ) / 7 + (4 : ℚ) / 9) / 2 = (23 : ℚ) / 63 :=
by sorry

end halfway_between_fractions_average_of_fractions_l2142_214282


namespace ali_nada_difference_l2142_214208

def total_amount : ℕ := 67
def john_amount : ℕ := 48

theorem ali_nada_difference (ali_amount nada_amount : ℕ) 
  (h1 : ali_amount + nada_amount + john_amount = total_amount)
  (h2 : ali_amount < nada_amount)
  (h3 : john_amount = 4 * nada_amount) :
  nada_amount - ali_amount = 5 := by
sorry

end ali_nada_difference_l2142_214208


namespace ab_100_necessary_not_sufficient_for_log_sum_2_l2142_214202

theorem ab_100_necessary_not_sufficient_for_log_sum_2 :
  (∀ a b : ℝ, (Real.log a + Real.log b = 2) → (a * b = 100)) ∧
  (∃ a b : ℝ, a * b = 100 ∧ Real.log a + Real.log b ≠ 2) := by
  sorry

end ab_100_necessary_not_sufficient_for_log_sum_2_l2142_214202


namespace line_plane_perpendicularity_l2142_214203

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallelism and perpendicularity relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem line_plane_perpendicularity 
  (m : Line) (α β : Plane) (h_diff : α ≠ β) :
  parallel m α → perpendicular m β → plane_perpendicular α β :=
sorry

end line_plane_perpendicularity_l2142_214203


namespace inequality_equivalence_l2142_214217

-- Define the logarithm with base 0.5
noncomputable def log_half (x : ℝ) : ℝ := Real.log x / Real.log 0.5

-- State the theorem
theorem inequality_equivalence (x : ℝ) (h : x > 0) :
  2 * (log_half x)^2 + 9 * log_half x + 9 ≤ 0 ↔ 2 * Real.sqrt 2 ≤ x ∧ x ≤ 8 :=
by sorry

end inequality_equivalence_l2142_214217


namespace triangle_ratio_l2142_214246

/-- An ellipse with foci on the x-axis -/
structure Ellipse where
  p : ℝ
  q : ℝ
  equation : ∀ x y : ℝ, x^2 / p^2 + y^2 / q^2 = 1

/-- An equilateral triangle -/
structure EquilateralTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  is_equilateral : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
                   (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
                   (A.1 - B.1)^2 + (A.2 - B.2)^2 = 
                   (A.1 - C.1)^2 + (A.2 - C.2)^2

/-- The configuration described in the problem -/
structure Configuration where
  E : Ellipse
  T : EquilateralTriangle
  B_on_ellipse : T.B = (0, E.q)
  AC_parallel_x : T.A.2 = T.C.2
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  F₁_on_BC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             F₁ = (t * T.B.1 + (1 - t) * T.C.1, t * T.B.2 + (1 - t) * T.C.2)
  F₂_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             F₂ = (t * T.A.1 + (1 - t) * T.B.1, t * T.A.2 + (1 - t) * T.B.2)
  focal_distance : (F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2 = 4

theorem triangle_ratio (c : Configuration) : 
  let AB := ((c.T.A.1 - c.T.B.1)^2 + (c.T.A.2 - c.T.B.2)^2).sqrt
  let F₁F₂ := ((c.F₁.1 - c.F₂.1)^2 + (c.F₁.2 - c.F₂.2)^2).sqrt
  AB / F₁F₂ = 8 / 5 := by
  sorry

end triangle_ratio_l2142_214246


namespace probability_all_red_balls_l2142_214276

def total_balls : ℕ := 10
def red_balls : ℕ := 5
def blue_balls : ℕ := 5
def drawn_balls : ℕ := 5

theorem probability_all_red_balls :
  (Nat.choose red_balls drawn_balls) / (Nat.choose total_balls drawn_balls) = 1 / 252 := by
  sorry

end probability_all_red_balls_l2142_214276


namespace at_least_one_seedling_exactly_one_success_l2142_214289

-- Define the probabilities
def prob_A_seedling : ℝ := 0.6
def prob_B_seedling : ℝ := 0.5
def prob_A_survive : ℝ := 0.7
def prob_B_survive : ℝ := 0.9

-- Theorem 1: Probability that at least one type of fruit tree becomes a seedling
theorem at_least_one_seedling :
  1 - (1 - prob_A_seedling) * (1 - prob_B_seedling) = 0.8 := by sorry

-- Theorem 2: Probability that exactly one type of fruit tree is successfully cultivated and survives
theorem exactly_one_success :
  let prob_A_success := prob_A_seedling * prob_A_survive
  let prob_B_success := prob_B_seedling * prob_B_survive
  prob_A_success * (1 - prob_B_success) + (1 - prob_A_success) * prob_B_success = 0.492 := by sorry

end at_least_one_seedling_exactly_one_success_l2142_214289


namespace integer_roots_of_cubic_l2142_214206

def f (x : ℤ) : ℤ := x^3 - 3*x^2 - 13*x + 15

theorem integer_roots_of_cubic :
  {x : ℤ | f x = 0} = {-3, 1, 5} := by sorry

end integer_roots_of_cubic_l2142_214206


namespace problem_solution_l2142_214205

theorem problem_solution (x : ℕ) (h : x = 36) : 
  2 * ((((x + 10) * 2) / 2) - 2) = 88 := by
  sorry

end problem_solution_l2142_214205


namespace symmetric_point_coordinates_l2142_214219

/-- A point in a 2D plane represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines symmetry with respect to the x-axis -/
def symmetricXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The problem statement -/
theorem symmetric_point_coordinates :
  let M : Point := ⟨-2, 1⟩
  let N : Point := ⟨-2, -1⟩
  symmetricXAxis M N := by
  sorry


end symmetric_point_coordinates_l2142_214219


namespace blue_jellybean_probability_l2142_214214

/-- The probability of drawing 3 blue jellybeans in succession from a bag of 10 red and 10 blue jellybeans without replacement is 1/9.5. -/
theorem blue_jellybean_probability : 
  let total_jellybeans : ℕ := 20
  let blue_jellybeans : ℕ := 10
  let draws : ℕ := 3
  let prob_first : ℚ := blue_jellybeans / total_jellybeans
  let prob_second : ℚ := (blue_jellybeans - 1) / (total_jellybeans - 1)
  let prob_third : ℚ := (blue_jellybeans - 2) / (total_jellybeans - 2)
  prob_first * prob_second * prob_third = 1 / (19 / 2) :=
by sorry

end blue_jellybean_probability_l2142_214214


namespace apartment_rent_calculation_required_rent_is_correct_l2142_214257

/-- Calculate the required monthly rent for an apartment investment --/
theorem apartment_rent_calculation (investment : ℝ) (maintenance_rate : ℝ) 
  (annual_taxes : ℝ) (desired_return_rate : ℝ) : ℝ :=
  let annual_return := investment * desired_return_rate
  let total_annual_requirement := annual_return + annual_taxes
  let monthly_net_requirement := total_annual_requirement / 12
  let monthly_rent := monthly_net_requirement / (1 - maintenance_rate)
  monthly_rent

/-- The required monthly rent is approximately $153.70 --/
theorem required_rent_is_correct : 
  ∃ ε > 0, |apartment_rent_calculation 20000 0.1 460 0.06 - 153.70| < ε :=
sorry

end apartment_rent_calculation_required_rent_is_correct_l2142_214257


namespace johnson_family_has_four_children_l2142_214238

/-- Represents the Johnson family -/
structure JohnsonFamily where
  num_children : ℕ
  father_age : ℕ
  mother_age : ℕ
  children_ages : List ℕ

/-- The conditions of the Johnson family -/
def johnson_family_conditions (family : JohnsonFamily) : Prop :=
  family.father_age = 55 ∧
  family.num_children + 2 = 6 ∧
  (family.father_age + family.mother_age + family.children_ages.sum) / 6 = 25 ∧
  (family.mother_age + family.children_ages.sum) / (family.num_children + 1) = 18

/-- The theorem stating that the Johnson family has 4 children -/
theorem johnson_family_has_four_children (family : JohnsonFamily) 
  (h : johnson_family_conditions family) : family.num_children = 4 := by
  sorry

end johnson_family_has_four_children_l2142_214238


namespace transformation_result_l2142_214229

/-- Rotates a point (x, y) 180° clockwise around (h, k) -/
def rotate180 (x y h k : ℝ) : ℝ × ℝ :=
  (2 * h - x, 2 * k - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqualsX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem transformation_result (c d : ℝ) :
  let (x₁, y₁) := rotate180 c d 2 (-3)
  let (x₂, y₂) := reflectAboutYEqualsX x₁ y₁
  (x₂ = 5 ∧ y₂ = -4) → d - c = -19 := by
  sorry

end transformation_result_l2142_214229


namespace tan_graph_property_l2142_214285

theorem tan_graph_property (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∀ x, a * Real.tan (b * x) = a * Real.tan (b * (x + π))) → 
  a * Real.tan (b * (π / 4)) = 3 → 
  a * b = 3 := by
sorry

end tan_graph_property_l2142_214285


namespace number_of_schedules_l2142_214254

/-- Represents the number of periods in a day -/
def total_periods : ℕ := 6

/-- Represents the number of morning periods -/
def morning_periods : ℕ := 3

/-- Represents the number of afternoon periods -/
def afternoon_periods : ℕ := 3

/-- Represents the total number of classes -/
def total_classes : ℕ := 6

/-- Represents the constraint that Mathematics must be in the morning -/
def math_in_morning : Prop := true

/-- Represents the constraint that Art must be in the afternoon -/
def art_in_afternoon : Prop := true

/-- The main theorem stating the number of possible schedules -/
theorem number_of_schedules :
  math_in_morning →
  art_in_afternoon →
  (total_periods = morning_periods + afternoon_periods) →
  (∃ (n : ℕ), n = 216 ∧ n = number_of_possible_schedules) :=
sorry

end number_of_schedules_l2142_214254


namespace inequality_system_solution_l2142_214235

theorem inequality_system_solution (x : ℝ) : 
  ((3*x - 2) / (x - 6) ≤ 1 ∧ 2*x^2 - x - 1 > 0) ↔ 
  ((-2 ≤ x ∧ x < -1/2) ∨ (1 < x ∧ x < 6)) :=
sorry

end inequality_system_solution_l2142_214235


namespace mans_speed_in_still_water_l2142_214228

/-- The speed of a man in still water given his downstream and upstream swimming times and distances -/
theorem mans_speed_in_still_water
  (downstream_distance : ℝ) (upstream_distance : ℝ) (time : ℝ)
  (h_downstream : downstream_distance = 40)
  (h_upstream : upstream_distance = 56)
  (h_time : time = 8)
  : ∃ (v_m : ℝ), v_m = 6 ∧ 
    downstream_distance / time = v_m + (downstream_distance - upstream_distance) / (2 * time) ∧
    upstream_distance / time = v_m - (downstream_distance - upstream_distance) / (2 * time) :=
by sorry

end mans_speed_in_still_water_l2142_214228


namespace geometric_progression_property_l2142_214215

def geometric_progression (b : ℕ → ℝ) := 
  ∀ n : ℕ, b (n + 1) / b n = b 2 / b 1

theorem geometric_progression_property (b : ℕ → ℝ) 
  (h₁ : geometric_progression b) 
  (h₂ : ∀ n : ℕ, b n > 0) : 
  (b 1 * b 2 * b 3 * b 4 * b 5 * b 6) ^ (1/6) = (b 3 * b 4) ^ (1/2) := by
  sorry

end geometric_progression_property_l2142_214215


namespace diagonal_passes_through_840_cubes_l2142_214258

/-- The number of cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_through (x y z : ℕ) : ℕ :=
  x + y + z - (Nat.gcd x y + Nat.gcd y z + Nat.gcd z x) + Nat.gcd x (Nat.gcd y z)

/-- Theorem: An internal diagonal of a 200 × 360 × 450 rectangular solid passes through 840 cubes -/
theorem diagonal_passes_through_840_cubes :
  cubes_passed_through 200 360 450 = 840 := by
  sorry

end diagonal_passes_through_840_cubes_l2142_214258


namespace equation_solution_inequality_solution_l2142_214277

-- Equation problem
theorem equation_solution :
  ∀ x : ℝ, 6 * x - 2 * (x - 3) = 14 ↔ x = 2 := by sorry

-- Inequality problem
theorem inequality_solution :
  ∀ x : ℝ, 3 * (x + 3) < x + 7 ↔ x < -1 := by sorry

end equation_solution_inequality_solution_l2142_214277


namespace tate_total_years_l2142_214225

/-- Calculates the total years spent by Tate in education and experiences --/
def totalYears (typicalHighSchoolYears : ℕ) : ℕ :=
  let highSchoolYears := typicalHighSchoolYears - 1
  let travelYears := 2
  let bachelorsYears := 2 * highSchoolYears
  let workExperienceYears := 1
  let phdYears := 3 * (highSchoolYears + bachelorsYears)
  highSchoolYears + travelYears + bachelorsYears + workExperienceYears + phdYears

/-- Theorem stating that Tate's total years spent is 39 --/
theorem tate_total_years : totalYears 4 = 39 := by
  sorry

end tate_total_years_l2142_214225


namespace f_monotonicity_and_zeros_l2142_214210

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 2) * x - Real.log x

theorem f_monotonicity_and_zeros (a : ℝ) :
  (∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ → f a x₁ > f a x₂) ∨
  (∃ c, 0 < c ∧ 
    (∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ x₂ < c → f a x₁ > f a x₂) ∧
    (∀ x₁ x₂, c < x₁ ∧ x₁ < x₂ → f a x₁ < f a x₂)) ∧
  (∃ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0) ↔ 0 < a ∧ a < 1 :=
by sorry

end f_monotonicity_and_zeros_l2142_214210


namespace sufficient_condition_range_exclusive_or_range_l2142_214296

/-- Definition of proposition p -/
def p (x : ℝ) : Prop := (x + 2) * (x - 6) ≤ 0

/-- Definition of proposition q -/
def q (m x : ℝ) : Prop := 2 - m ≤ x ∧ x ≤ 2 + m

/-- Theorem for part (1) -/
theorem sufficient_condition_range (m : ℝ) :
  (∀ x : ℝ, p x → q m x) → m ∈ Set.Ici 4 :=
sorry

/-- Theorem for part (2) -/
theorem exclusive_or_range (x : ℝ) :
  (p x ∨ q 5 x) ∧ ¬(p x ∧ q 5 x) → x ∈ Set.Icc (-3) (-2) ∪ Set.Ioo 6 7 :=
sorry

end sufficient_condition_range_exclusive_or_range_l2142_214296


namespace selection_methods_equal_l2142_214266

def male_students : ℕ := 20
def female_students : ℕ := 30
def total_students : ℕ := male_students + female_students
def select_count : ℕ := 4

def selection_method_1 : ℕ := Nat.choose total_students select_count - 
                               Nat.choose male_students select_count - 
                               Nat.choose female_students select_count

def selection_method_2 : ℕ := Nat.choose male_students 1 * Nat.choose female_students 3 +
                               Nat.choose male_students 2 * Nat.choose female_students 2 +
                               Nat.choose male_students 3 * Nat.choose female_students 1

theorem selection_methods_equal : selection_method_1 = selection_method_2 := by
  sorry

end selection_methods_equal_l2142_214266


namespace tv_screen_diagonal_l2142_214224

theorem tv_screen_diagonal (s : ℝ) (h : s^2 = 256 + 34) :
  Real.sqrt (2 * s^2) = Real.sqrt 580 := by
  sorry

end tv_screen_diagonal_l2142_214224


namespace locus_of_P_l2142_214292

/-- Given two variable points A and B on the x-axis and y-axis respectively,
    such that AB is in the first quadrant and has fixed length 2d,
    and a point P such that P and the origin are on opposite sides of AB,
    and PC is perpendicular to AB with length d (where C is the midpoint of AB),
    prove that P lies on the line y = x and its distance from the origin
    is between d√2 and 2d inclusive. -/
theorem locus_of_P (d : ℝ) (A B P : ℝ × ℝ) (h_d : d > 0) :
  let C := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  (A.2 = 0) →
  (B.1 = 0) →
  (A.1 ≥ 0 ∧ B.2 ≥ 0) →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*d)^2 →
  ((P.1 - C.1) * (B.1 - A.1) + (P.2 - C.2) * (B.2 - A.2) = 0) →
  ((P.1 - C.1)^2 + (P.2 - C.2)^2 = d^2) →
  (P.1 * B.2 > P.2 * A.1) →
  (P.1 = P.2 ∧ d * Real.sqrt 2 ≤ Real.sqrt (P.1^2 + P.2^2) ∧ Real.sqrt (P.1^2 + P.2^2) ≤ 2*d) :=
by sorry

end locus_of_P_l2142_214292


namespace algebraic_simplification_l2142_214274

theorem algebraic_simplification (x y : ℝ) :
  ((-3 * x * y^2)^3 * (-6 * x^2 * y)) / (9 * x^4 * y^5) = 18 * x * y^2 := by
  sorry

end algebraic_simplification_l2142_214274


namespace total_age_problem_l2142_214269

theorem total_age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  b = 10 →
  a + b + c = 27 :=
by sorry

end total_age_problem_l2142_214269


namespace octagon_area_given_equal_perimeter_and_square_area_l2142_214218

/-- Given a square and a regular octagon with equal perimeters, 
    if the square's area is 16, then the area of the octagon is 8(1+√2) -/
theorem octagon_area_given_equal_perimeter_and_square_area (a b : ℝ) : 
  a > 0 → b > 0 →
  (4 * a = 8 * b) →  -- Equal perimeters
  (a ^ 2 = 16) →     -- Square's area is 16
  (2 * (1 + Real.sqrt 2) * b ^ 2 = 8 * (1 + Real.sqrt 2)) :=
by sorry

end octagon_area_given_equal_perimeter_and_square_area_l2142_214218


namespace magnitude_of_complex_fraction_l2142_214239

theorem magnitude_of_complex_fraction (z : ℂ) : 
  z = (3 + Complex.I) / (1 + Complex.I) → Complex.abs z = Real.sqrt 5 := by
  sorry

end magnitude_of_complex_fraction_l2142_214239


namespace product_of_roots_plus_two_l2142_214293

theorem product_of_roots_plus_two (u v w : ℝ) : 
  (u^3 - 18*u^2 + 20*u - 8 = 0) →
  (v^3 - 18*v^2 + 20*v - 8 = 0) →
  (w^3 - 18*w^2 + 20*w - 8 = 0) →
  (2+u)*(2+v)*(2+w) = 128 := by
sorry

end product_of_roots_plus_two_l2142_214293


namespace g_stable_point_fixed_points_subset_stable_points_l2142_214255

-- Define a function type
def RealFunction := ℝ → ℝ

-- Define fixed point
def is_fixed_point (f : RealFunction) (x : ℝ) : Prop := f x = x

-- Define stable point
def is_stable_point (f : RealFunction) (x : ℝ) : Prop := f (f x) = x

-- Define the set of fixed points
def fixed_points (f : RealFunction) : Set ℝ := {x | is_fixed_point f x}

-- Define the set of stable points
def stable_points (f : RealFunction) : Set ℝ := {x | is_stable_point f x}

-- Define the function g(x) = 3x - 8
def g : RealFunction := λ x ↦ 3 * x - 8

-- Theorem: The stable point of g(x) = 3x - 8 is x = 4
theorem g_stable_point : is_stable_point g 4 := by sorry

-- Theorem: For any function, the set of fixed points is a subset of the set of stable points
theorem fixed_points_subset_stable_points (f : RealFunction) : 
  fixed_points f ⊆ stable_points f := by sorry

end g_stable_point_fixed_points_subset_stable_points_l2142_214255


namespace only_1_and_4_perpendicular_l2142_214295

-- Define the slopes of the lines
def m1 : ℚ := 2/3
def m2 : ℚ := -2/3
def m3 : ℚ := -2/3
def m4 : ℚ := -3/2

-- Define a function to check if two lines are perpendicular
def are_perpendicular (m1 m2 : ℚ) : Prop := m1 * m2 = -1

-- Theorem statement
theorem only_1_and_4_perpendicular :
  (are_perpendicular m1 m4) ∧
  ¬(are_perpendicular m1 m2) ∧
  ¬(are_perpendicular m1 m3) ∧
  ¬(are_perpendicular m2 m3) ∧
  ¬(are_perpendicular m2 m4) ∧
  ¬(are_perpendicular m3 m4) :=
by sorry

end only_1_and_4_perpendicular_l2142_214295


namespace simplify_expression_l2142_214275

theorem simplify_expression (x : ℝ) : (3 * x)^3 + (2 * x) * (x^4) = 27 * x^3 + 2 * x^5 := by
  sorry

end simplify_expression_l2142_214275


namespace expression_evaluation_l2142_214299

theorem expression_evaluation :
  let x : ℝ := Real.sqrt 2 + 1
  ((x + 1) / (x - 1) + 1 / (x^2 - 2*x + 1)) / (x / (x - 1)) = 1 + Real.sqrt 2 / 2 := by
  sorry

end expression_evaluation_l2142_214299


namespace smallest_difference_vovochka_sum_l2142_214245

/-- Vovochka's sum method for three-digit numbers -/
def vovochkaSum (a b c d e f : ℕ) : ℕ := 
  (a + d) * 1000 + (b + e) * 100 + (c + f)

/-- Correct sum method for three-digit numbers -/
def correctSum (a b c d e f : ℕ) : ℕ := 
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f)

/-- Theorem: The smallest positive difference between Vovochka's sum and the correct sum is 1800 -/
theorem smallest_difference_vovochka_sum : 
  ∀ a b c d e f : ℕ, 
  a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 →
  vovochkaSum a b c d e f - correctSum a b c d e f ≥ 1800 ∧
  ∃ a b c d e f : ℕ, 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ f < 10 ∧
    vovochkaSum a b c d e f - correctSum a b c d e f = 1800 :=
by sorry

end smallest_difference_vovochka_sum_l2142_214245


namespace line_intersection_difference_l2142_214242

/-- Given a line y = 2x - 4 intersecting the x-axis at point A(m, 0) and the y-axis at point B(0, n), prove that m - n = 6 -/
theorem line_intersection_difference (m n : ℝ) : 
  (∀ x y, y = 2 * x - 4) →  -- Line equation
  0 = 2 * m - 4 →           -- A(m, 0) satisfies the line equation
  n = -4 →                  -- B(0, n) satisfies the line equation
  m - n = 6 := by
sorry

end line_intersection_difference_l2142_214242


namespace refrigerator_cost_l2142_214233

/-- Proves that the cost of the refrigerator is 25000 given the problem conditions -/
theorem refrigerator_cost
  (mobile_cost : ℕ)
  (refrigerator_loss_percent : ℚ)
  (mobile_profit_percent : ℚ)
  (total_profit : ℕ)
  (h1 : mobile_cost = 8000)
  (h2 : refrigerator_loss_percent = 4 / 100)
  (h3 : mobile_profit_percent = 10 / 100)
  (h4 : total_profit = 200) :
  ∃ (refrigerator_cost : ℕ),
    refrigerator_cost = 25000 ∧
    (refrigerator_cost : ℚ) * (1 - refrigerator_loss_percent) +
    (mobile_cost : ℚ) * (1 + mobile_profit_percent) =
    (refrigerator_cost + mobile_cost + total_profit : ℚ) :=
sorry

end refrigerator_cost_l2142_214233


namespace smallest_k_for_negative_three_in_range_l2142_214298

-- Define the function g
def g (k : ℝ) (x : ℝ) : ℝ := x^2 + 3*x + k

-- State the theorem
theorem smallest_k_for_negative_three_in_range :
  (∃ k₀ : ℝ, (∀ k : ℝ, (∃ x : ℝ, g k x = -3) → k ≥ k₀) ∧
             (∃ x : ℝ, g k₀ x = -3) ∧
             k₀ = -3/4) := by
  sorry

end smallest_k_for_negative_three_in_range_l2142_214298


namespace senate_committee_seating_arrangements_l2142_214256

/-- The number of ways to arrange n distinguishable people around a circular table,
    where rotations are considered the same arrangement -/
def circularArrangements (n : ℕ) : ℕ := (n - 1).factorial

theorem senate_committee_seating_arrangements :
  circularArrangements 10 = 362880 := by
  sorry

end senate_committee_seating_arrangements_l2142_214256


namespace least_three_digit_7_heavy_correct_l2142_214294

/-- A number is 7-heavy if its remainder when divided by 7 is greater than 4 -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The least three-digit 7-heavy number -/
def least_three_digit_7_heavy : ℕ := 103

theorem least_three_digit_7_heavy_correct :
  (least_three_digit_7_heavy ≥ 100) ∧
  (least_three_digit_7_heavy < 1000) ∧
  is_7_heavy least_three_digit_7_heavy ∧
  ∀ n : ℕ, (n ≥ 100) ∧ (n < 1000) ∧ is_7_heavy n → n ≥ least_three_digit_7_heavy :=
by sorry

end least_three_digit_7_heavy_correct_l2142_214294


namespace root_product_theorem_l2142_214212

theorem root_product_theorem (x₁ x₂ x₃ x₄ x₅ : ℂ) : 
  (x₁^5 - 3*x₁^3 + x₁ + 6 = 0) →
  (x₂^5 - 3*x₂^3 + x₂ + 6 = 0) →
  (x₃^5 - 3*x₃^3 + x₃ + 6 = 0) →
  (x₄^5 - 3*x₄^3 + x₄ + 6 = 0) →
  (x₅^5 - 3*x₅^3 + x₅ + 6 = 0) →
  ((x₁^2 - 2) * (x₂^2 - 2) * (x₃^2 - 2) * (x₄^2 - 2) * (x₅^2 - 2) = 10) := by
  sorry

end root_product_theorem_l2142_214212


namespace complex_number_problem_l2142_214286

/-- Given a complex number z = 3 + bi where b is a positive real number,
    and (z - 2)² is a pure imaginary number, prove that:
    1. z = 3 + i
    2. |z / (2 + i)| = √2 -/
theorem complex_number_problem (b : ℝ) (z : ℂ) 
    (h1 : b > 0)
    (h2 : z = 3 + b * I)
    (h3 : ∃ (y : ℝ), (z - 2)^2 = y * I) :
  z = 3 + I ∧ Complex.abs (z / (2 + I)) = Real.sqrt 2 := by
  sorry

end complex_number_problem_l2142_214286


namespace theater_revenue_l2142_214259

/-- Calculates the total revenue for a theater performance series -/
theorem theater_revenue (seats : ℕ) (capacity : ℚ) (ticket_price : ℕ) (days : ℕ) :
  seats = 400 →
  capacity = 4/5 →
  ticket_price = 30 →
  days = 3 →
  (seats : ℚ) * capacity * (ticket_price : ℚ) * (days : ℚ) = 28800 := by
sorry

end theater_revenue_l2142_214259


namespace at_least_one_not_greater_than_neg_four_l2142_214226

theorem at_least_one_not_greater_than_neg_four
  (a b c : ℝ)
  (ha : a < 0)
  (hb : b < 0)
  (hc : c < 0) :
  (a + 4 / b ≤ -4) ∨ (b + 4 / c ≤ -4) ∨ (c + 4 / a ≤ -4) := by
sorry

end at_least_one_not_greater_than_neg_four_l2142_214226


namespace expression_evaluation_l2142_214241

theorem expression_evaluation (a b : ℤ) (h1 : a = 4) (h2 : b = -3) :
  -2*a - b^2 + 2*a*b = -41 := by sorry

end expression_evaluation_l2142_214241


namespace max_intersection_area_l2142_214220

/-- A right prism with a square base centered at the origin --/
structure Prism :=
  (side_length : ℝ)
  (center : ℝ × ℝ × ℝ := (0, 0, 0))

/-- A plane in 3D space defined by its equation coefficients --/
structure Plane :=
  (a b c d : ℝ)

/-- The intersection of a prism and a plane --/
def intersection (p : Prism) (plane : Plane) : Set (ℝ × ℝ × ℝ) :=
  {pt : ℝ × ℝ × ℝ | 
    let (x, y, z) := pt
    plane.a * x + plane.b * y + plane.c * z = plane.d ∧
    |x| ≤ p.side_length / 2 ∧
    |y| ≤ p.side_length / 2}

/-- The area of a set in 3D space --/
noncomputable def area (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- The theorem stating that the maximum area of the intersection is equal to the area of the square base --/
theorem max_intersection_area (p : Prism) (plane : Plane) :
  p.side_length = 12 ∧
  plane = {a := 3, b := -6, c := 2, d := 24} →
  area (intersection p plane) ≤ p.side_length ^ 2 :=
sorry

end max_intersection_area_l2142_214220


namespace nested_sqrt_equality_l2142_214236

theorem nested_sqrt_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x))) = (x ^ (11 / 8)) :=
sorry

end nested_sqrt_equality_l2142_214236


namespace abc_product_magnitude_l2142_214287

theorem abc_product_magnitude (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
    (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
    (h_eq : a - 1/b = b - 1/c ∧ b - 1/c = c - 1/a) :
    |a * b * c| = 1 := by
  sorry

end abc_product_magnitude_l2142_214287


namespace four_digit_numbers_with_prime_factorization_property_l2142_214216

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def prime_factorization_sum_property (n : ℕ) : Prop :=
  ∃ (factors : List ℕ) (exponents : List ℕ),
    n = (factors.zip exponents).foldl (λ acc (p, e) => acc * p^e) 1 ∧
    factors.all Nat.Prime ∧
    factors.sum = exponents.sum

theorem four_digit_numbers_with_prime_factorization_property :
  {n : ℕ | is_four_digit n ∧ prime_factorization_sum_property n} =
  {1792, 2000, 3125, 3840, 5000, 5760, 6272, 8640, 9600} := by
  sorry

end four_digit_numbers_with_prime_factorization_property_l2142_214216


namespace john_phone_bill_cost_l2142_214265

/-- Calculates the total cost of a phone bill given the monthly fee, per-minute rate, and minutes used. -/
def phoneBillCost (monthlyFee : ℝ) (perMinuteRate : ℝ) (minutesUsed : ℝ) : ℝ :=
  monthlyFee + perMinuteRate * minutesUsed

theorem john_phone_bill_cost :
  phoneBillCost 5 0.25 28.08 = 12.02 := by
  sorry

end john_phone_bill_cost_l2142_214265


namespace train_average_speed_l2142_214244

/-- Proves that the average speed of a train including stoppages is 36 kmph,
    given its speed excluding stoppages and the duration of stoppages. -/
theorem train_average_speed
  (speed_without_stops : ℝ)
  (stop_duration : ℝ)
  (h1 : speed_without_stops = 60)
  (h2 : stop_duration = 24)
  : (speed_without_stops * (60 - stop_duration) / 60) = 36 := by
  sorry

end train_average_speed_l2142_214244


namespace reciprocal_inequality_l2142_214227

theorem reciprocal_inequality (a b : ℝ) : a < b → b < 0 → (1 : ℝ) / a > (1 : ℝ) / b := by
  sorry

end reciprocal_inequality_l2142_214227


namespace porch_width_calculation_l2142_214213

/-- Given a house and porch with specific dimensions, calculate the width of the porch. -/
theorem porch_width_calculation (house_length house_width porch_length total_shingle_area : ℝ)
  (h1 : house_length = 20.5)
  (h2 : house_width = 10)
  (h3 : porch_length = 6)
  (h4 : total_shingle_area = 232) :
  let house_area := house_length * house_width
  let porch_area := total_shingle_area - house_area
  porch_area / porch_length = 4.5 := by sorry

end porch_width_calculation_l2142_214213


namespace unique_number_property_l2142_214204

theorem unique_number_property : ∃! x : ℝ, x / 3 = x - 5 := by
  sorry

end unique_number_property_l2142_214204


namespace cos_alpha_value_l2142_214251

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (α + π/4) = 12/13)
  (h2 : π/4 < α) 
  (h3 : α < 3*π/4) : 
  Real.cos α = 7*Real.sqrt 2/26 := by
sorry

end cos_alpha_value_l2142_214251


namespace special_number_unique_l2142_214264

/-- The unique integer between 10000 and 99999 satisfying the given conditions -/
def special_number : ℕ := 11311

/-- Checks if a natural number is between 10000 and 99999 -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- Extracts the digits of a five-digit number -/
def digits (n : ℕ) : Fin 5 → ℕ
| 0 => n / 10000
| 1 => (n / 1000) % 10
| 2 => (n / 100) % 10
| 3 => (n / 10) % 10
| 4 => n % 10

theorem special_number_unique :
  ∀ n : ℕ, is_five_digit n →
    (digits n 0 = n % 2) →
    (digits n 1 = n % 3) →
    (digits n 2 = n % 4) →
    (digits n 3 = n % 5) →
    (digits n 4 = n % 6) →
    n = special_number := by sorry

end special_number_unique_l2142_214264


namespace numeric_methods_students_l2142_214232

/-- The total number of students in the faculty -/
def total_students : ℕ := 653

/-- The number of second-year students studying automatic control -/
def auto_control_students : ℕ := 423

/-- The number of second-year students studying both numeric methods and automatic control -/
def both_subjects_students : ℕ := 134

/-- The approximate percentage of second-year students in the faculty -/
def second_year_percentage : ℚ := 80/100

/-- The number of second-year students (rounded) -/
def second_year_students : ℕ := 522

/-- Theorem stating the number of second-year students studying numeric methods -/
theorem numeric_methods_students : 
  ∃ (n : ℕ), n = second_year_students - (auto_control_students - both_subjects_students) ∧ n = 233 :=
sorry

end numeric_methods_students_l2142_214232


namespace system_solution_l2142_214253

theorem system_solution (x y z : ℝ) : 
  x + y = 5 ∧ y + z = -1 ∧ x + z = -2 → x = 2 ∧ y = 3 ∧ z = -4 := by
  sorry

end system_solution_l2142_214253


namespace complex_modulus_l2142_214278

theorem complex_modulus (z : ℂ) (h : (1 + Complex.I) * z = 1 - 2 * Complex.I) : 
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end complex_modulus_l2142_214278


namespace chemistry_physics_difference_l2142_214261

/-- Given a student's scores in mathematics, physics, and chemistry, prove that the difference between chemistry and physics scores is 20 marks. -/
theorem chemistry_physics_difference
  (M P C : ℕ)  -- Marks in Mathematics, Physics, and Chemistry
  (h1 : M + P = 60)  -- Total marks in mathematics and physics is 60
  (h2 : ∃ X : ℕ, C = P + X)  -- Chemistry score is some marks more than physics
  (h3 : (M + C) / 2 = 40)  -- Average marks in mathematics and chemistry is 40
  : ∃ X : ℕ, C = P + X ∧ X = 20 := by
  sorry

#check chemistry_physics_difference

end chemistry_physics_difference_l2142_214261


namespace total_passengers_is_420_l2142_214240

/-- Represents the number of carriages in a train -/
def carriages_per_train : ℕ := 4

/-- Represents the original number of seats in each carriage -/
def original_seats_per_carriage : ℕ := 25

/-- Represents the additional number of passengers each carriage can accommodate -/
def additional_passengers_per_carriage : ℕ := 10

/-- Represents the number of trains -/
def number_of_trains : ℕ := 3

/-- Calculates the total number of passengers that can fill up the given number of trains -/
def total_passengers : ℕ :=
  number_of_trains * carriages_per_train * (original_seats_per_carriage + additional_passengers_per_carriage)

theorem total_passengers_is_420 : total_passengers = 420 := by
  sorry

end total_passengers_is_420_l2142_214240


namespace calculation_proof_l2142_214268

theorem calculation_proof :
  (((3 * Real.sqrt 48) - (2 * Real.sqrt 27)) / Real.sqrt 3 = 6) ∧
  ((Real.sqrt 3 + 1) * (Real.sqrt 3 - 1) - Real.sqrt ((-3)^2) + (1 / (2 - Real.sqrt 5)) = -3 - Real.sqrt 5) :=
by sorry

end calculation_proof_l2142_214268


namespace quadratic_y_values_order_l2142_214290

def quadratic_function (x : ℝ) : ℝ := 3 * (x + 2)^2

theorem quadratic_y_values_order :
  ∀ (y₁ y₂ y₃ : ℝ),
  quadratic_function 1 = y₁ →
  quadratic_function 2 = y₂ →
  quadratic_function (-3) = y₃ →
  y₃ < y₁ ∧ y₁ < y₂ :=
by sorry

end quadratic_y_values_order_l2142_214290


namespace f_properties_l2142_214223

noncomputable def f (a b c x : ℝ) : ℝ := -x^3 + a*x^2 + b*x + c

theorem f_properties (a b c : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ 0 → f a b c x₁ > f a b c x₂) →
  (∀ x₁ x₂ : ℝ, 0 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 1 → f a b c x₁ < f a b c x₂) →
  (∃ x₁ x₂ x₃ : ℝ, x₁ < x₂ ∧ x₂ < x₃ ∧ f a b c x₁ = 0 ∧ f a b c x₂ = 0 ∧ f a b c x₃ = 0) →
  f a b c 1 = 0 →
  b = 0 :=
by sorry

end f_properties_l2142_214223


namespace largest_digit_divisible_by_six_l2142_214267

theorem largest_digit_divisible_by_six : 
  ∀ N : ℕ, N ≤ 9 → (5678 * 10 + N) % 6 = 0 → N ≤ 4 := by
  sorry

end largest_digit_divisible_by_six_l2142_214267


namespace number_property_l2142_214281

theorem number_property (y : ℝ) : y = (1 / y) * (-y) + 3 → y = 2 := by
  sorry

end number_property_l2142_214281


namespace last_score_is_71_l2142_214280

def scores : List Nat := [71, 74, 79, 85, 88, 92]

def is_valid_last_score (last_score : Nat) : Prop :=
  last_score ∈ scores ∧
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 6 → 
    (scores.sum - last_score) % n = 0

theorem last_score_is_71 : 
  ∃! last_score, is_valid_last_score last_score ∧ last_score = 71 := by sorry

end last_score_is_71_l2142_214280


namespace smallest_four_digit_divisible_by_35_l2142_214247

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 35 = 0 → n ≥ 1015 := by
  sorry

end smallest_four_digit_divisible_by_35_l2142_214247


namespace unpainted_side_length_approx_l2142_214248

/-- A rectangular parking space with three painted sides -/
structure ParkingSpace where
  length : ℝ
  width : ℝ
  painted_sides_sum : length + 2 * width = 37
  area : length * width = 125

/-- The length of the unpainted side of the parking space -/
def unpainted_side_length (p : ParkingSpace) : ℝ := p.length

/-- The unpainted side length is approximately 8.90 feet -/
theorem unpainted_side_length_approx (p : ParkingSpace) :
  ∃ ε > 0, |unpainted_side_length p - 8.90| < ε :=
sorry

end unpainted_side_length_approx_l2142_214248


namespace conic_single_point_implies_d_eq_11_l2142_214250

/-- A conic section represented by the equation 2x^2 + y^2 + 4x - 6y + d = 0 -/
def conic (d : ℝ) (x y : ℝ) : Prop :=
  2 * x^2 + y^2 + 4 * x - 6 * y + d = 0

/-- The conic degenerates to a single point -/
def is_single_point (d : ℝ) : Prop :=
  ∃! p : ℝ × ℝ, conic d p.1 p.2

/-- If the conic degenerates to a single point, then d = 11 -/
theorem conic_single_point_implies_d_eq_11 :
  ∀ d : ℝ, is_single_point d → d = 11 := by sorry

end conic_single_point_implies_d_eq_11_l2142_214250


namespace expected_turns_to_second_ace_prove_expected_turns_l2142_214273

/-- A deck of cards -/
structure Deck :=
  (n : ℕ)  -- Total number of cards
  (h : n ≥ 3)  -- There are at least 3 cards (for the 3 aces)

/-- The expected number of cards turned up until the second ace appears -/
def expectedTurnsToSecondAce (d : Deck) : ℚ :=
  (d.n + 1) / 2

/-- Theorem stating that the expected number of cards turned up until the second ace appears is (n+1)/2 -/
theorem expected_turns_to_second_ace (d : Deck) :
  expectedTurnsToSecondAce d = (d.n + 1) / 2 := by
  sorry

/-- Main theorem proving the expected number of cards turned up -/
theorem prove_expected_turns (d : Deck) :
  ∃ (e : ℚ), e = expectedTurnsToSecondAce d ∧ e = (d.n + 1) / 2 := by
  sorry

end expected_turns_to_second_ace_prove_expected_turns_l2142_214273


namespace total_profit_is_8640_l2142_214263

/-- Represents the investment and profit distribution of a business partnership --/
structure BusinessPartnership where
  total_investment : ℕ
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  a_profit_share : ℕ

/-- Calculates the total profit based on the given business partnership --/
def calculate_total_profit (bp : BusinessPartnership) : ℕ :=
  let investment_ratio := bp.a_investment + bp.b_investment + bp.c_investment
  let profit_per_ratio := bp.a_profit_share * investment_ratio / bp.a_investment
  profit_per_ratio

/-- Theorem stating the total profit for the given business scenario --/
theorem total_profit_is_8640 (bp : BusinessPartnership) 
  (h1 : bp.total_investment = 90000)
  (h2 : bp.a_investment = bp.b_investment + 6000)
  (h3 : bp.c_investment = bp.b_investment + 3000)
  (h4 : bp.a_investment + bp.b_investment + bp.c_investment = bp.total_investment)
  (h5 : bp.a_profit_share = 3168) :
  calculate_total_profit bp = 8640 := by
  sorry

end total_profit_is_8640_l2142_214263
