import Mathlib

namespace final_red_probability_l3372_337246

-- Define the contents of each bag
def bagA : ℕ × ℕ := (5, 3)  -- (white, black)
def bagB : ℕ × ℕ := (4, 6)  -- (red, green)
def bagC : ℕ × ℕ := (3, 4)  -- (red, green)

-- Define the probability of drawing a specific marble from a bag
def probDraw (color : ℕ) (bag : ℕ × ℕ) : ℚ :=
  color / (bag.1 + bag.2)

-- Define the probability of the final marble being red
def probFinalRed : ℚ :=
  let probWhiteA := probDraw bagA.1 bagA
  let probBlackA := probDraw bagA.2 bagA
  let probGreenB := probDraw bagB.2 bagB
  let probRedB := probDraw bagB.1 bagB
  let probGreenC := probDraw bagC.2 bagC
  let probRedC := probDraw bagC.1 bagC
  probWhiteA * probGreenB * probRedB + probBlackA * probGreenC * probRedC

-- Theorem statement
theorem final_red_probability : probFinalRed = 79 / 980 := by
  sorry

end final_red_probability_l3372_337246


namespace max_profit_at_36_l3372_337255

/-- Represents the daily sales quantity of product A in kg -/
def y (x : ℝ) : ℝ := -2 * x + 100

/-- Represents the daily profit in yuan -/
def w (x : ℝ) : ℝ := -2 * x^2 + 160 * x - 2760

/-- The cost of product A in yuan per kg -/
def cost_A : ℝ := 20

/-- The maximum allowed price of product A (180% of cost) -/
def max_price_A : ℝ := cost_A * 1.8

theorem max_profit_at_36 :
  ∀ x : ℝ, cost_A ≤ x ∧ x ≤ max_price_A →
  w x ≤ w 36 ∧ w 36 = 408 := by
  sorry

#eval w 36

end max_profit_at_36_l3372_337255


namespace cost_of_18_pencils_13_notebooks_l3372_337284

/-- The cost of a pencil -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 9 pencils and 11 notebooks cost $6.05 -/
axiom condition1 : 9 * pencil_cost + 11 * notebook_cost = 6.05

/-- The second given condition: 6 pencils and 4 notebooks cost $2.68 -/
axiom condition2 : 6 * pencil_cost + 4 * notebook_cost = 2.68

/-- Theorem: The cost of 18 pencils and 13 notebooks is $8.45 -/
theorem cost_of_18_pencils_13_notebooks :
  18 * pencil_cost + 13 * notebook_cost = 8.45 := by sorry

end cost_of_18_pencils_13_notebooks_l3372_337284


namespace cyclic_quadrilateral_inequality_l3372_337299

/-- A cyclic quadrilateral is a quadrilateral whose vertices all lie on a single circle. -/
structure CyclicQuadrilateral (P : Type*) [MetricSpace P] :=
  (A B C D : P)
  (cyclic : ∃ (center : P) (radius : ℝ), dist center A = radius ∧ dist center B = radius ∧ dist center C = radius ∧ dist center D = radius)

/-- The inequality for cyclic quadrilaterals -/
theorem cyclic_quadrilateral_inequality {P : Type*} [MetricSpace P] (ABCD : CyclicQuadrilateral P) :
  |dist ABCD.A ABCD.B - dist ABCD.C ABCD.D| + |dist ABCD.A ABCD.D - dist ABCD.B ABCD.C| ≥ 2 * |dist ABCD.A ABCD.C - dist ABCD.B ABCD.D| :=
sorry

end cyclic_quadrilateral_inequality_l3372_337299


namespace domino_arrangements_equals_binomial_coefficient_l3372_337274

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a domino with width and height -/
structure Domino :=
  (width : ℕ)
  (height : ℕ)

/-- The number of distinct arrangements of dominoes on a grid -/
def distinct_arrangements (g : Grid) (d : Domino) (num_dominoes : ℕ) : ℕ :=
  sorry

/-- The binomial coefficient (n choose k) -/
def binomial_coefficient (n : ℕ) (k : ℕ) : ℕ :=
  sorry

theorem domino_arrangements_equals_binomial_coefficient :
  let g : Grid := { width := 5, height := 3 }
  let d : Domino := { width := 2, height := 1 }
  let num_dominoes : ℕ := 3
  distinct_arrangements g d num_dominoes = binomial_coefficient 6 2 :=
by sorry

end domino_arrangements_equals_binomial_coefficient_l3372_337274


namespace prime_power_sum_l3372_337295

theorem prime_power_sum (n : ℕ) : Prime (n^4 + 4^n) ↔ n = 1 :=
sorry

end prime_power_sum_l3372_337295


namespace arithmetic_sequence_sum_l3372_337271

theorem arithmetic_sequence_sum : 2016 - 2017 + 2018 - 2019 + 2020 = 2018 := by
  sorry

end arithmetic_sequence_sum_l3372_337271


namespace unanswered_questions_count_l3372_337241

/-- Represents the scoring system for AHSME competition --/
structure ScoringSystem where
  correct : Int
  incorrect : Int
  unanswered : Int

/-- Represents the AHSME competition --/
structure AHSMECompetition where
  new_scoring : ScoringSystem
  old_scoring : ScoringSystem
  total_questions : Nat
  new_score : Int
  old_score : Int

/-- Theorem stating that the number of unanswered questions is 9 --/
theorem unanswered_questions_count (comp : AHSMECompetition)
  (h_new_scoring : comp.new_scoring = { correct := 5, incorrect := 0, unanswered := 2 })
  (h_old_scoring : comp.old_scoring = { correct := 4, incorrect := -1, unanswered := 0 })
  (h_old_base : comp.old_score - 30 = 4 * (comp.new_score / 5) - (comp.total_questions - (comp.new_score / 5) - 9))
  (h_total : comp.total_questions = 30)
  (h_new_score : comp.new_score = 93)
  (h_old_score : comp.old_score = 84) :
  ∃ (correct incorrect : Nat), 
    correct + incorrect + 9 = comp.total_questions ∧
    5 * correct + 2 * 9 = comp.new_score ∧
    4 * correct - incorrect = comp.old_score - 30 :=
by sorry


end unanswered_questions_count_l3372_337241


namespace water_level_rise_l3372_337217

/-- Given a cube with edge length 15 cm and a rectangular vessel with base dimensions 20 cm × 15 cm,
    prove that the rise in water level when the cube is fully immersed is 11.25 cm. -/
theorem water_level_rise (cube_edge : ℝ) (vessel_length : ℝ) (vessel_width : ℝ) :
  cube_edge = 15 →
  vessel_length = 20 →
  vessel_width = 15 →
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 11.25 := by
  sorry

#check water_level_rise

end water_level_rise_l3372_337217


namespace system_solution_l3372_337216

theorem system_solution : 
  ∀ x y z : ℝ, 
    (y * z = 3 * y + 2 * z - 8) ∧ 
    (z * x = 4 * z + 3 * x - 8) ∧ 
    (x * y = 2 * x + y - 1) → 
    ((x = 2 ∧ y = 3 ∧ z = 1) ∨ (x = 3 ∧ y = 5/2 ∧ z = -1)) :=
by sorry

end system_solution_l3372_337216


namespace y_value_l3372_337224

theorem y_value (x y : ℝ) (h1 : x^2 = y - 7) (h2 : x = 7) : y = 56 := by
  sorry

end y_value_l3372_337224


namespace sum_of_fractions_l3372_337232

theorem sum_of_fractions : (1 / 1.01) + (1 / 1.1) + (1 / 1) + (1 / 11) + (1 / 101) = 3 := by
  sorry

end sum_of_fractions_l3372_337232


namespace quadratic_radicals_simplification_l3372_337208

theorem quadratic_radicals_simplification :
  (∀ a b m n : ℝ, a > 0 ∧ b > 0 ∧ m > 0 ∧ n > 0 →
    m^2 + n^2 = a ∧ m * n = Real.sqrt b →
    Real.sqrt (a + 2 * Real.sqrt b) = m + n) ∧
  Real.sqrt (6 + 2 * Real.sqrt 5) = Real.sqrt 5 + 1 ∧
  Real.sqrt (5 - 2 * Real.sqrt 6) = Real.sqrt 3 - Real.sqrt 2 ∧
  (∀ a : ℝ, Real.sqrt (a^2 + 4 * Real.sqrt 5) = 2 + Real.sqrt 5 →
    a = 3 ∨ a = -3) :=
by sorry

end quadratic_radicals_simplification_l3372_337208


namespace building_block_width_l3372_337223

/-- Given a box and building blocks with specified dimensions, prove that the width of the building block is 2 inches. -/
theorem building_block_width (box_height box_width box_length : ℕ)
  (block_height block_length : ℕ) (num_blocks : ℕ) :
  box_height = 8 →
  box_width = 10 →
  box_length = 12 →
  block_height = 3 →
  block_length = 4 →
  num_blocks = 40 →
  (box_height * box_width * box_length) / num_blocks = block_height * 2 * block_length :=
by sorry

end building_block_width_l3372_337223


namespace distance_to_focus_l3372_337231

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- Define a point on the parabola with x-coordinate 4
def point_on_parabola : {P : ℝ × ℝ // parabola P.1 P.2 ∧ P.1 = 4} :=
  sorry

-- Theorem statement
theorem distance_to_focus :
  let P := point_on_parabola.val
  (P.1 - 0)^2 = 4^2 →  -- Distance from P to y-axis is 4
  (P.1 - focus.1)^2 + (P.2 - focus.2)^2 = 6^2  -- Distance from P to focus is 6
:= by sorry

end distance_to_focus_l3372_337231


namespace second_class_average_l3372_337201

theorem second_class_average (students1 : ℕ) (students2 : ℕ) (avg1 : ℝ) (avg_combined : ℝ) :
  students1 = 25 →
  students2 = 30 →
  avg1 = 40 →
  avg_combined = 50.90909090909091 →
  ((students1 * avg1 + students2 * 60) / (students1 + students2) = avg_combined) := by
sorry

end second_class_average_l3372_337201


namespace solution_set_of_inequality_l3372_337262

theorem solution_set_of_inequality (x : ℝ) :
  x^2 < 2*x ↔ 0 < x ∧ x < 2 := by sorry

end solution_set_of_inequality_l3372_337262


namespace sum_of_fractions_l3372_337239

theorem sum_of_fractions : 
  (19 / ((2^3 - 1) * (3^3 - 1)) + 
   37 / ((3^3 - 1) * (4^3 - 1)) + 
   61 / ((4^3 - 1) * (5^3 - 1)) + 
   91 / ((5^3 - 1) * (6^3 - 1))) = 208 / 1505 := by
  sorry

end sum_of_fractions_l3372_337239


namespace a_2006_mod_7_l3372_337260

def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => a n + (a (n + 1))^2

theorem a_2006_mod_7 : a 2006 % 7 = 6 := by
  sorry

end a_2006_mod_7_l3372_337260


namespace system_solution_l3372_337275

theorem system_solution :
  ∀ x y : ℝ, x > 0 → y > 0 →
  (y - 2 * Real.sqrt (x * y) - Real.sqrt (y / x) + 2 = 0) →
  (3 * x^2 * y^2 + y^4 = 84) →
  ((x = 1/3 ∧ y = 3) ∨ (x = (21/76)^(1/4) ∧ y = 2 * (84/19)^(1/4))) :=
by sorry

end system_solution_l3372_337275


namespace triangle_angle_measure_l3372_337235

theorem triangle_angle_measure (A B C : ℝ) (h1 : 0 < A) (h2 : A < π) (h3 : 0 < B) (h4 : B < π) (h5 : 0 < C) (h6 : C < π) 
  (h7 : A + B + C = π) (h8 : Real.sqrt 3 / Real.sin A = 1 / Real.sin (π/6)) (h9 : B = π/6) : 
  A = π/3 ∨ A = 2*π/3 := by
sorry

end triangle_angle_measure_l3372_337235


namespace oldest_child_age_l3372_337247

theorem oldest_child_age (a b c : ℕ) (h1 : a = 6) (h2 : b = 8) 
  (h3 : (a + b + c) / 3 = 9) : c = 13 := by
  sorry

end oldest_child_age_l3372_337247


namespace num_arrangements_eq_162_l3372_337210

/-- The number of ways to select k items from n items --/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange k items from n items --/
def arrange (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements for dispatching volunteers --/
def num_arrangements : ℕ :=
  let total_volunteers := 5
  let dispatched_volunteers := 4
  let num_communities := 3
  let scenario1 := choose 3 2 * (choose 4 2 - 1) * arrange 3 3
  let scenario2 := choose 2 1 * choose 4 2 * arrange 3 3
  scenario1 + scenario2

theorem num_arrangements_eq_162 : num_arrangements = 162 := by sorry

end num_arrangements_eq_162_l3372_337210


namespace right_triangle_special_area_l3372_337281

theorem right_triangle_special_area (c : ℝ) (h : c > 0) : ∃ (S : ℝ),
  (∃ (x : ℝ), 0 < x ∧ x < c ∧ (c - x) / x = x / c) →
  S = (c^2 * Real.sqrt (Real.sqrt 5 - 2)) / 2 :=
by sorry

end right_triangle_special_area_l3372_337281


namespace square_sum_from_means_l3372_337270

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 24) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 168) : 
  x^2 + y^2 = 1968 := by
  sorry

end square_sum_from_means_l3372_337270


namespace f_range_is_1_2_5_l3372_337273

def f (x : Int) : Int := x^2 + 1

def domain : Set Int := {-1, 0, 1, 2}

theorem f_range_is_1_2_5 : 
  {y | ∃ x ∈ domain, f x = y} = {1, 2, 5} := by sorry

end f_range_is_1_2_5_l3372_337273


namespace total_fish_count_l3372_337202

/-- Represents a fish company with tuna and mackerel counts -/
structure FishCompany where
  tuna : ℕ
  mackerel : ℕ

/-- Calculates the total fish count for a company -/
def totalFish (company : FishCompany) : ℕ :=
  company.tuna + company.mackerel

/-- Theorem stating the total fish count for all three companies -/
theorem total_fish_count 
  (jerk_tuna : FishCompany)
  (tall_tuna : FishCompany)
  (swell_tuna : FishCompany)
  (h1 : jerk_tuna.tuna = 144)
  (h2 : jerk_tuna.mackerel = 80)
  (h3 : tall_tuna.tuna = 2 * jerk_tuna.tuna)
  (h4 : tall_tuna.mackerel = jerk_tuna.mackerel + (30 * jerk_tuna.mackerel) / 100)
  (h5 : swell_tuna.tuna = tall_tuna.tuna + (50 * tall_tuna.tuna) / 100)
  (h6 : swell_tuna.mackerel = jerk_tuna.mackerel + (25 * jerk_tuna.mackerel) / 100) :
  totalFish jerk_tuna + totalFish tall_tuna + totalFish swell_tuna = 1148 := by
  sorry

end total_fish_count_l3372_337202


namespace area_of_circle_portion_l3372_337251

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop := x^2 - 12*x + y^2 = 28

/-- The line equation -/
def line_equation (x y : ℝ) : Prop := y = x - 4

/-- The region of interest -/
def region_of_interest (x y : ℝ) : Prop :=
  circle_equation x y ∧ y ≥ 0 ∧ y ≥ x - 4

/-- The area of the region of interest -/
noncomputable def area_of_region : ℝ := sorry

theorem area_of_circle_portion : area_of_region = 48 * Real.pi :=
sorry

end area_of_circle_portion_l3372_337251


namespace movie_marathon_duration_l3372_337207

theorem movie_marathon_duration :
  let movie1 : ℝ := 2
  let movie2 : ℝ := movie1 * 1.5
  let movie3 : ℝ := movie1 + movie2 - 1
  movie1 + movie2 + movie3 = 9 := by sorry

end movie_marathon_duration_l3372_337207


namespace green_balloons_l3372_337229

theorem green_balloons (total : ℕ) (red : ℕ) (green : ℕ) : 
  total = 17 → red = 8 → green = total - red → green = 9 := by
  sorry

end green_balloons_l3372_337229


namespace quadratic_real_roots_l3372_337218

theorem quadratic_real_roots (n : ℕ+) :
  (∃ x : ℝ, x^2 - 4*x + n.val = 0) ↔ n.val = 1 ∨ n.val = 2 ∨ n.val = 3 ∨ n.val = 4 := by
  sorry

end quadratic_real_roots_l3372_337218


namespace distance_swum_back_l3372_337286

/-- The distance a person swims back against the current -/
def swim_distance (still_water_speed : ℝ) (water_speed : ℝ) (time : ℝ) : ℝ :=
  (still_water_speed - water_speed) * time

/-- Theorem: The distance swum back against the current is 8 km -/
theorem distance_swum_back (still_water_speed : ℝ) (water_speed : ℝ) (time : ℝ)
    (h1 : still_water_speed = 8)
    (h2 : water_speed = 4)
    (h3 : time = 2) :
    swim_distance still_water_speed water_speed time = 8 := by
  sorry

end distance_swum_back_l3372_337286


namespace chord_length_at_135_degrees_chord_equation_when_bisected_l3372_337209

-- Define the circle
def Circle (O : ℝ × ℝ) (r : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - O.1)^2 + (p.2 - O.2)^2 = r^2}

-- Define a point on the circle
def P : ℝ × ℝ := (-1, 2)

-- Define the chord AB
structure Chord where
  A : ℝ × ℝ
  B : ℝ × ℝ
  P : ℝ × ℝ
  passes_through_P : A.1 ≠ B.1 ∧ (P.2 - A.2) / (P.1 - A.1) = (B.2 - A.2) / (B.1 - A.1)

-- Theorem 1
theorem chord_length_at_135_degrees (O : ℝ × ℝ) (r : ℝ) (AB : Chord) :
  O = (0, 0) →
  r^2 = 8 →
  P ∈ Circle O r →
  AB.P = P →
  (AB.B.2 - AB.A.2) / (AB.B.1 - AB.A.1) = -1 →
  Real.sqrt ((AB.A.1 - AB.B.1)^2 + (AB.A.2 - AB.B.2)^2) = Real.sqrt 30 :=
sorry

-- Theorem 2
theorem chord_equation_when_bisected (O : ℝ × ℝ) (r : ℝ) (AB : Chord) :
  O = (0, 0) →
  r^2 = 8 →
  P ∈ Circle O r →
  AB.P = P →
  AB.A.1 - P.1 = P.1 - AB.B.1 →
  AB.A.2 - P.2 = P.2 - AB.B.2 →
  ∃ (a b c : ℝ), a * AB.A.1 + b * AB.A.2 + c = 0 ∧
                 a * AB.B.1 + b * AB.B.2 + c = 0 ∧
                 a = 1 ∧ b = -2 ∧ c = 5 :=
sorry

end chord_length_at_135_degrees_chord_equation_when_bisected_l3372_337209


namespace alan_pine_trees_l3372_337227

/-- The number of pine cones dropped by each tree -/
def pine_cones_per_tree : ℕ := 200

/-- The percentage of pine cones that fall on Alan's roof -/
def roof_percentage : ℚ := 30 / 100

/-- The weight of each pine cone in ounces -/
def pine_cone_weight : ℕ := 4

/-- The total weight of pine cones on Alan's roof in ounces -/
def total_roof_weight : ℕ := 1920

/-- The number of pine trees in Alan's backyard -/
def num_trees : ℕ := 8

theorem alan_pine_trees :
  num_trees * (pine_cones_per_tree * roof_percentage).floor * pine_cone_weight = total_roof_weight :=
sorry

end alan_pine_trees_l3372_337227


namespace arrow_sequence_for_multiples_of_four_l3372_337238

def arrow_direction (n : ℕ) : Bool × Bool :=
  if n % 4 = 0 then (false, true) else (true, false)

theorem arrow_sequence_for_multiples_of_four (n : ℕ) (h : n % 4 = 0) :
  arrow_direction n = (false, true) := by sorry

end arrow_sequence_for_multiples_of_four_l3372_337238


namespace sqrt_equation_solution_l3372_337215

theorem sqrt_equation_solution : ∃ (a b c : ℕ+), 
  (2 * Real.sqrt (Real.sqrt 4 - Real.sqrt 3) = Real.sqrt a.val - Real.sqrt b.val + Real.sqrt c.val) ∧
  (a.val + b.val + c.val = 22) := by
  sorry

end sqrt_equation_solution_l3372_337215


namespace remainder_3_305_mod_13_l3372_337240

theorem remainder_3_305_mod_13 : 3^305 % 13 = 9 := by
  sorry

end remainder_3_305_mod_13_l3372_337240


namespace right_triangle_existence_l3372_337297

theorem right_triangle_existence (a : ℤ) (h : a ≥ 5) :
  ∃ b c : ℤ, c ≥ b ∧ b ≥ a ∧ a^2 + b^2 = c^2 := by
sorry

end right_triangle_existence_l3372_337297


namespace parabola_intercepts_sum_l3372_337261

/-- Represents a parabola of the form x = 3y^2 - 9y + 5 -/
def Parabola := { p : ℝ × ℝ | p.1 = 3 * p.2^2 - 9 * p.2 + 5 }

/-- The x-coordinate of the x-intercept -/
def a : ℝ := 5

/-- The y-coordinates of the y-intercepts -/
def b : ℝ := sorry
def c : ℝ := sorry

theorem parabola_intercepts_sum : a + b + c = 8 := by
  sorry

end parabola_intercepts_sum_l3372_337261


namespace f_derivative_l3372_337276

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp (2 * x)

theorem f_derivative :
  deriv f = λ x => (2 * x + 2 * x^2) * Real.exp (2 * x) :=
sorry

end f_derivative_l3372_337276


namespace frog_eggs_difference_l3372_337234

/-- Represents the number of eggs laid by a frog over 4 days -/
def FrogEggs : Type :=
  { eggs : Fin 4 → ℕ // 
    eggs 0 = 50 ∧ 
    eggs 1 = 2 * eggs 0 ∧ 
    eggs 3 = 2 * (eggs 0 + eggs 1 + eggs 2) ∧
    eggs 0 + eggs 1 + eggs 2 + eggs 3 = 810 }

/-- The difference between eggs laid on the third day and second day is 20 -/
theorem frog_eggs_difference (e : FrogEggs) : e.val 2 - e.val 1 = 20 := by
  sorry

end frog_eggs_difference_l3372_337234


namespace inequality_proof_l3372_337265

theorem inequality_proof (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) :
  a * x^2 + b * y^2 ≥ (a * x + b * y)^2 := by
  sorry

end inequality_proof_l3372_337265


namespace tv_price_increase_l3372_337272

theorem tv_price_increase (x : ℝ) : 
  (1 + 0.3) * (1 + x) = 1 + 0.5600000000000001 ↔ x = 0.2 := by
  sorry

end tv_price_increase_l3372_337272


namespace fraction_equality_l3372_337264

theorem fraction_equality : (2523 - 2428)^2 / 121 = 75 := by
  sorry

end fraction_equality_l3372_337264


namespace factorization_x_squared_minus_3x_l3372_337257

theorem factorization_x_squared_minus_3x (x : ℝ) : x^2 - 3*x = x*(x - 3) := by
  sorry

end factorization_x_squared_minus_3x_l3372_337257


namespace square_of_binomial_constant_l3372_337228

theorem square_of_binomial_constant (b : ℚ) : 
  (∃ (c : ℚ), ∀ (x : ℚ), 9*x^2 + 27*x + b = (3*x + c)^2) → b = 81/4 := by
  sorry

end square_of_binomial_constant_l3372_337228


namespace percentage_of_green_caps_l3372_337259

def total_caps : ℕ := 125
def red_caps : ℕ := 50

theorem percentage_of_green_caps :
  (total_caps - red_caps : ℚ) / total_caps * 100 = 60 := by
  sorry

end percentage_of_green_caps_l3372_337259


namespace min_x_prime_factorization_sum_l3372_337221

theorem min_x_prime_factorization_sum (x y : ℕ+) (h : 3 * x ^ 12 = 5 * y ^ 17) :
  ∃ (a b c d : ℕ),
    (∀ (p : ℕ), p.Prime → p ∣ x → p = a ∨ p = b) ∧
    x = a ^ c * b ^ d ∧
    (∀ (x' : ℕ+), 3 * x' ^ 12 = 5 * y ^ 17 → x ≤ x') ∧
    a + b + c + d = 30 :=
by sorry

end min_x_prime_factorization_sum_l3372_337221


namespace range_of_m_l3372_337230

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 4/y = 2) :
  (∃ m : ℝ, x + y/4 < m^2 - m) ↔ ∃ m : ℝ, m < -1 ∨ m > 2 := by
sorry

end range_of_m_l3372_337230


namespace layla_and_alan_apples_l3372_337225

def maggie_apples : ℕ := 40
def kelsey_apples : ℕ := 28
def total_people : ℕ := 4
def average_apples : ℕ := 30

theorem layla_and_alan_apples :
  ∃ (layla_apples alan_apples : ℕ),
    maggie_apples + kelsey_apples + layla_apples + alan_apples = total_people * average_apples ∧
    layla_apples + alan_apples = 52 :=
by sorry

end layla_and_alan_apples_l3372_337225


namespace ellipse_major_axis_length_l3372_337277

/-- Given an ellipse with equation 4x^2 + y^2 = 16, its major axis has length 8 -/
theorem ellipse_major_axis_length :
  ∀ (x y : ℝ), 4 * x^2 + y^2 = 16 → ∃ (a b : ℝ), 
    a > b ∧ 
    x^2 / a^2 + y^2 / b^2 = 1 ∧ 
    2 * a = 8 := by
  sorry

end ellipse_major_axis_length_l3372_337277


namespace fraction_of_rotten_berries_l3372_337267

theorem fraction_of_rotten_berries 
  (total_berries : ℕ) 
  (berries_to_sell : ℕ) 
  (h1 : total_berries = 60) 
  (h2 : berries_to_sell = 20) 
  (h3 : berries_to_sell * 2 ≤ total_berries) :
  (total_berries - berries_to_sell * 2 : ℚ) / total_berries = 1 / 3 := by
sorry

end fraction_of_rotten_berries_l3372_337267


namespace distance_to_origin_l3372_337206

theorem distance_to_origin (x y n : ℝ) : 
  y = 15 → 
  x = 2 + Real.sqrt 105 → 
  x > 2 → 
  n = Real.sqrt (x^2 + y^2) →
  n = Real.sqrt (334 + 4 * Real.sqrt 105) := by
sorry

end distance_to_origin_l3372_337206


namespace oldest_child_age_l3372_337219

def average_age : ℝ := 7
def younger_child1_age : ℝ := 4
def younger_child2_age : ℝ := 7

theorem oldest_child_age :
  ∃ (oldest_age : ℝ),
    (younger_child1_age + younger_child2_age + oldest_age) / 3 = average_age ∧
    oldest_age = 10 := by
  sorry

end oldest_child_age_l3372_337219


namespace binomial_distribution_unique_parameters_l3372_337212

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (ξ : BinomialRV) : ℝ := ξ.n * ξ.p

/-- The variance of a binomial random variable -/
def variance (ξ : BinomialRV) : ℝ := ξ.n * ξ.p * (1 - ξ.p)

theorem binomial_distribution_unique_parameters (ξ : BinomialRV) 
  (h_exp : expectation ξ = 12) 
  (h_var : variance ξ = 2.4) : 
  ξ.n = 15 ∧ ξ.p = 4/5 := by
  sorry

end binomial_distribution_unique_parameters_l3372_337212


namespace roots_triangle_condition_l3372_337226

/-- A cubic equation with coefficients p, q, and r -/
structure CubicEquation where
  p : ℝ
  q : ℝ
  r : ℝ

/-- The roots of a cubic equation form a triangle -/
def roots_form_triangle (eq : CubicEquation) : Prop :=
  ∃ (u v w : ℝ), u > 0 ∧ v > 0 ∧ w > 0 ∧
    u^3 + eq.p * u^2 + eq.q * u + eq.r = 0 ∧
    v^3 + eq.p * v^2 + eq.q * v + eq.r = 0 ∧
    w^3 + eq.p * w^2 + eq.q * w + eq.r = 0 ∧
    u + v > w ∧ u + w > v ∧ v + w > u

/-- The theorem stating the condition for roots to form a triangle -/
theorem roots_triangle_condition (eq : CubicEquation) :
  roots_form_triangle eq ↔ eq.p^3 - 4 * eq.p * eq.q + 8 * eq.r > 0 :=
sorry

end roots_triangle_condition_l3372_337226


namespace endpoint_coordinate_sum_l3372_337288

/-- Given a line segment with one endpoint at (10, 4) and midpoint at (4, -8),
    the sum of the coordinates of the other endpoint is -22. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
    (4 = (x + 10) / 2) → 
    (-8 = (y + 4) / 2) → 
    x + y = -22 := by
  sorry

end endpoint_coordinate_sum_l3372_337288


namespace merry_saturday_boxes_l3372_337285

/-- The number of boxes Merry had on Sunday -/
def sunday_boxes : ℕ := 25

/-- The number of apples in each box -/
def apples_per_box : ℕ := 10

/-- The total number of apples sold on Saturday and Sunday -/
def total_apples_sold : ℕ := 720

/-- The number of boxes left after selling -/
def boxes_left : ℕ := 3

/-- The number of boxes Merry had on Saturday -/
def saturday_boxes : ℕ := 69

theorem merry_saturday_boxes :
  saturday_boxes = 69 :=
by sorry

end merry_saturday_boxes_l3372_337285


namespace butterfingers_count_l3372_337279

theorem butterfingers_count (total : ℕ) (snickers : ℕ) (mars : ℕ) (butterfingers : ℕ) : 
  total = 12 → snickers = 3 → mars = 2 → total = snickers + mars + butterfingers →
  butterfingers = 7 := by
sorry

end butterfingers_count_l3372_337279


namespace library_visitors_average_l3372_337253

theorem library_visitors_average (sunday_visitors : ℕ) (other_day_visitors : ℕ) 
  (month_days : ℕ) (sundays_in_month : ℕ) :
  sunday_visitors = 510 →
  other_day_visitors = 240 →
  month_days = 30 →
  sundays_in_month = 4 →
  (sundays_in_month * sunday_visitors + (month_days - sundays_in_month) * other_day_visitors) / month_days = 276 :=
by
  sorry

end library_visitors_average_l3372_337253


namespace fixed_point_theorem_l3372_337213

/-- A line with slope k passing through a fixed point (x₀, y₀) -/
def line_equation (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

/-- The fixed point theorem for a family of lines -/
theorem fixed_point_theorem :
  ∃! p : ℝ × ℝ, ∀ k : ℝ, line_equation k p.1 p.2 (-3) 4 :=
by sorry

end fixed_point_theorem_l3372_337213


namespace evaluate_expression_l3372_337269

theorem evaluate_expression (x y : ℝ) (hx : x = 3) (hy : y = 0) : y * (y - 3 * x) = 0 := by
  sorry

end evaluate_expression_l3372_337269


namespace alcohol_percentage_in_solution_x_l3372_337204

/-- The percentage of alcohol in a solution that, when mixed with another solution,
    results in a specific alcohol concentration. -/
theorem alcohol_percentage_in_solution_x 
  (volume_x : ℝ) 
  (volume_y : ℝ) 
  (percent_y : ℝ) 
  (percent_final : ℝ) 
  (h1 : volume_x = 300)
  (h2 : volume_y = 900)
  (h3 : percent_y = 0.30)
  (h4 : percent_final = 0.25)
  : ∃ (percent_x : ℝ), 
    percent_x = 0.10 ∧ 
    volume_x * percent_x + volume_y * percent_y = (volume_x + volume_y) * percent_final :=
sorry

end alcohol_percentage_in_solution_x_l3372_337204


namespace systematic_sample_theorem_l3372_337249

/-- Represents a systematic sample from a population -/
structure SystematicSample where
  population_size : ℕ
  sample_size : ℕ
  interval : ℕ
  first_element : ℕ

/-- Generates the nth element of a systematic sample -/
def nth_element (s : SystematicSample) (n : ℕ) : ℕ :=
  s.first_element + (n - 1) * s.interval

theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population_size = 52)
  (h2 : s.sample_size = 4)
  (h3 : s.first_element = 5)
  (h4 : nth_element s 3 = 31)
  (h5 : nth_element s 4 = 44) :
  nth_element s 2 = 18 := by
  sorry

end systematic_sample_theorem_l3372_337249


namespace train_length_l3372_337211

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 54 → time_s = 9 → speed_kmh * (1000 / 3600) * time_s = 135 := by
  sorry

#check train_length

end train_length_l3372_337211


namespace y_equivalent_condition_l3372_337283

theorem y_equivalent_condition (x y : ℝ) :
  y = 2 * x + 4 →
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 8) ↔ 
  (y ∈ Set.Icc (-6) 6 ∪ Set.Icc 14 26) :=
by sorry

end y_equivalent_condition_l3372_337283


namespace square_intersection_perimeter_l3372_337290

/-- Given a square with side length 2a and an intersecting line y = x + a/2,
    the perimeter of one part divided by a equals (√17 + 8) / 2 -/
theorem square_intersection_perimeter (a : ℝ) (a_pos : a > 0) :
  let square_vertices := [(-a, -a), (a, -a), (-a, a), (a, a)]
  let intersecting_line (x : ℝ) := x + a / 2
  let intersection_points := [(-a, -a/2), (a, -a), (a/2, a), (-a, a)]
  let perimeter := Real.sqrt (17 * a^2) / 2 + 4 * a
  perimeter / a = (Real.sqrt 17 + 8) / 2 :=
by sorry

end square_intersection_perimeter_l3372_337290


namespace xy_cube_plus_cube_xy_l3372_337291

theorem xy_cube_plus_cube_xy (x y : ℝ) (h1 : x + y = 3) (h2 : x * y = -4) :
  x * y^3 + x^3 * y = -68 := by
  sorry

end xy_cube_plus_cube_xy_l3372_337291


namespace chess_tournament_games_l3372_337287

/-- The number of games in a chess tournament -/
def num_games (n : ℕ) (games_per_pair : ℕ) : ℕ :=
  (n * (n - 1) / 2) * games_per_pair

/-- Theorem: In a chess tournament with 30 players, where each player plays 
    5 times against every other player, the total number of games is 2175 -/
theorem chess_tournament_games : num_games 30 5 = 2175 := by
  sorry


end chess_tournament_games_l3372_337287


namespace y_value_proof_l3372_337203

theorem y_value_proof (y : ℝ) :
  (y / 5) / 3 = 15 / (y / 3) → y = 15 * Real.sqrt 3 ∨ y = -15 * Real.sqrt 3 := by
  sorry

end y_value_proof_l3372_337203


namespace original_number_exists_and_unique_l3372_337205

theorem original_number_exists_and_unique : 
  ∃! x : ℚ, 4 * (3 * x + 29) = 212 := by sorry

end original_number_exists_and_unique_l3372_337205


namespace largest_three_digit_multiple_of_9_with_digit_sum_27_l3372_337214

def digit_sum (n : Nat) : Nat :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem largest_three_digit_multiple_of_9_with_digit_sum_27 :
  ∀ n : Nat, 100 ≤ n ∧ n < 1000 ∧ n % 9 = 0 ∧ digit_sum n = 27 → n ≤ 999 :=
by
  sorry

end largest_three_digit_multiple_of_9_with_digit_sum_27_l3372_337214


namespace correct_proposition_l3372_337243

theorem correct_proposition :
  let p := ∀ x : ℝ, 2 * x < 3 * x
  let q := ∃ x : ℝ, x^3 = 1 - x^2
  ¬p ∧ q := by sorry

end correct_proposition_l3372_337243


namespace first_group_size_is_three_l3372_337252

/-- The number of people in the first group -/
def first_group_size : ℕ := 3

/-- The amount of work completed by the first group in 3 days -/
def first_group_work : ℕ := 3

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 3

/-- The number of people in the second group -/
def second_group_size : ℕ := 5

/-- The amount of work completed by the second group in 3 days -/
def second_group_work : ℕ := 5

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 3

theorem first_group_size_is_three :
  first_group_size * first_group_work * second_group_days =
  second_group_size * second_group_work * first_group_days :=
by sorry

end first_group_size_is_three_l3372_337252


namespace roots_sum_minus_product_l3372_337292

theorem roots_sum_minus_product (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ + 3 = 0) → 
  (x₂^2 - 4*x₂ + 3 = 0) → 
  x₁ + x₂ - x₁*x₂ = 1 := by
sorry

end roots_sum_minus_product_l3372_337292


namespace f_max_at_zero_l3372_337233

-- Define the function f and its derivative
noncomputable def f : ℝ → ℝ := λ x => x^4 - 2*x^2 - 5
def f' : ℝ → ℝ := λ x => 4*x^3 - 4*x

-- State the theorem
theorem f_max_at_zero :
  (∀ x : ℝ, (f' x) = 4*x^3 - 4*x) →
  f 0 = -5 →
  (∀ x : ℝ, f x ≤ -5) ∧ f 0 = -5 :=
by sorry

end f_max_at_zero_l3372_337233


namespace paco_salty_cookies_l3372_337222

/-- Prove that Paco initially had 56 salty cookies -/
theorem paco_salty_cookies 
  (initial_sweet : ℕ) 
  (eaten_sweet : ℕ) 
  (eaten_salty : ℕ) 
  (remaining_sweet : ℕ) 
  (h1 : initial_sweet = 34)
  (h2 : eaten_sweet = 15)
  (h3 : eaten_salty = 56)
  (h4 : remaining_sweet = 19)
  (h5 : initial_sweet = eaten_sweet + remaining_sweet) :
  eaten_salty = 56 := by
  sorry

end paco_salty_cookies_l3372_337222


namespace simplify_algebraic_expression_l3372_337250

theorem simplify_algebraic_expression (x : ℝ) 
  (h1 : x ≠ 1) (h2 : x ≠ -1) (h3 : x ≠ 0) : 
  (x - x / (x + 1)) / (1 + 1 / (x^2 - 1)) = x - 1 := by
  sorry

end simplify_algebraic_expression_l3372_337250


namespace average_income_l3372_337200

/-- The average monthly income problem -/
theorem average_income (A B C : ℕ) : 
  (B + C) / 2 = 5250 →
  (A + C) / 2 = 4200 →
  A = 3000 →
  (A + B) / 2 = 4050 := by
sorry

end average_income_l3372_337200


namespace spinach_amount_l3372_337289

/-- The initial amount of raw spinach in ounces -/
def initial_spinach : ℝ := 40

/-- The percentage of initial volume after cooking -/
def cooking_ratio : ℝ := 0.20

/-- The amount of cream cheese in ounces -/
def cream_cheese : ℝ := 6

/-- The amount of eggs in ounces -/
def eggs : ℝ := 4

/-- The total volume of the quiche in ounces -/
def total_volume : ℝ := 18

theorem spinach_amount :
  initial_spinach * cooking_ratio + cream_cheese + eggs = total_volume :=
by sorry

end spinach_amount_l3372_337289


namespace next_two_terms_l3372_337268

def arithmetic_sequence (a₀ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₀ + n * d

def is_arithmetic_sequence (seq : ℕ → ℕ) (a₀ d : ℕ) : Prop :=
  ∀ n, seq n = arithmetic_sequence a₀ d n

theorem next_two_terms
  (seq : ℕ → ℕ)
  (h : is_arithmetic_sequence seq 3 4)
  (h0 : seq 0 = 3)
  (h1 : seq 1 = 7)
  (h2 : seq 2 = 11)
  (h3 : seq 3 = 15)
  (h4 : seq 4 = 19)
  (h5 : seq 5 = 23) :
  seq 6 = 27 ∧ seq 7 = 31 := by
sorry

end next_two_terms_l3372_337268


namespace tan_negative_585_deg_l3372_337256

-- Define the tangent function for degrees
noncomputable def tan_deg (x : ℝ) : ℝ := Real.tan (x * Real.pi / 180)

-- State the theorem
theorem tan_negative_585_deg : tan_deg (-585) = -1 := by
  sorry

end tan_negative_585_deg_l3372_337256


namespace smallest_solution_of_equation_l3372_337293

theorem smallest_solution_of_equation :
  ∃ x : ℝ, x = 1 - Real.sqrt 10 ∧
  (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 12 ∧
  ∀ y : ℝ, (3 * y) / (y - 3) + (3 * y^2 - 27) / y = 12 → y ≥ x :=
by sorry

end smallest_solution_of_equation_l3372_337293


namespace weight_of_B_l3372_337236

theorem weight_of_B (A B C : ℝ) 
  (h1 : (A + B + C) / 3 = 43)
  (h2 : (A + B) / 2 = 48)
  (h3 : (B + C) / 2 = 42) :
  B = 51 := by sorry

end weight_of_B_l3372_337236


namespace lcm_gcd_problem_l3372_337258

theorem lcm_gcd_problem (a b : ℕ+) : 
  Nat.lcm a b = 2400 →
  Nat.gcd a b = 30 →
  a = 150 →
  b = 480 := by
sorry

end lcm_gcd_problem_l3372_337258


namespace triangle_inequality_l3372_337237

theorem triangle_inequality (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := by
  sorry

end triangle_inequality_l3372_337237


namespace infinitely_many_superabundant_l3372_337294

/-- Sum of divisors function -/
def sigma (n : ℕ+) : ℕ := sorry

/-- Superabundant number -/
def is_superabundant (m : ℕ+) : Prop :=
  ∀ k : ℕ+, k < m → (sigma m : ℚ) / m > (sigma k : ℚ) / k

/-- There are infinitely many superabundant numbers -/
theorem infinitely_many_superabundant :
  ∀ N : ℕ, ∃ m : ℕ+, m > N ∧ is_superabundant m :=
sorry

end infinitely_many_superabundant_l3372_337294


namespace choose_four_from_seven_l3372_337220

-- Define the number of available paints
def n : ℕ := 7

-- Define the number of paints to be chosen
def k : ℕ := 4

-- Theorem stating that choosing 4 paints from 7 different ones results in 35 ways
theorem choose_four_from_seven :
  Nat.choose n k = 35 := by
  sorry

end choose_four_from_seven_l3372_337220


namespace minimum_distance_triangle_warehouse_l3372_337282

theorem minimum_distance_triangle_warehouse (a b c : ℝ) (h1 : a = 2) (h2 : b = Real.sqrt 7) (h3 : c = 3) :
  ∃ (p : ℝ × ℝ),
    p.1 > 0 ∧ p.1 < c ∧ p.2 > 0 ∧
    p.2 < Real.sqrt (a^2 - p.1^2) ∧
    (∀ (q : ℝ × ℝ),
      q.1 > 0 ∧ q.1 < c ∧ q.2 > 0 ∧ q.2 < Real.sqrt (a^2 - q.1^2) →
      6 * (Real.sqrt ((0 - p.1)^2 + p.2^2) +
           Real.sqrt ((c - p.1)^2 + p.2^2) +
           Real.sqrt (p.1^2 + (a - p.2)^2))
      ≤ 6 * (Real.sqrt ((0 - q.1)^2 + q.2^2) +
             Real.sqrt ((c - q.1)^2 + q.2^2) +
             Real.sqrt (q.1^2 + (a - q.2)^2))) ∧
    6 * (Real.sqrt ((0 - p.1)^2 + p.2^2) +
         Real.sqrt ((c - p.1)^2 + p.2^2) +
         Real.sqrt (p.1^2 + (a - p.2)^2)) = 6 * Real.sqrt 19 :=
by sorry


end minimum_distance_triangle_warehouse_l3372_337282


namespace sqrt_equality_l3372_337244

theorem sqrt_equality (x : ℝ) (hx : x > 0) : -x * Real.sqrt (2 / x) = -Real.sqrt (2 * x) := by
  sorry

end sqrt_equality_l3372_337244


namespace linda_max_servings_l3372_337266

/-- Represents the recipe and available ingredients for making smoothies -/
structure SmoothieIngredients where
  recipe_bananas : ℕ        -- Bananas needed for 4 servings
  recipe_yogurt : ℕ         -- Cups of yogurt needed for 4 servings
  recipe_honey : ℕ          -- Tablespoons of honey needed for 4 servings
  available_bananas : ℕ     -- Bananas Linda has
  available_yogurt : ℕ      -- Cups of yogurt Linda has
  available_honey : ℕ       -- Tablespoons of honey Linda has

/-- Calculates the maximum number of servings that can be made -/
def max_servings (ingredients : SmoothieIngredients) : ℕ :=
  min
    (ingredients.available_bananas * 4 / ingredients.recipe_bananas)
    (min
      (ingredients.available_yogurt * 4 / ingredients.recipe_yogurt)
      (ingredients.available_honey * 4 / ingredients.recipe_honey))

/-- Theorem stating the maximum number of servings Linda can make -/
theorem linda_max_servings :
  let ingredients := SmoothieIngredients.mk 3 2 1 10 9 4
  max_servings ingredients = 13 := by
  sorry


end linda_max_servings_l3372_337266


namespace fruit_bags_weight_l3372_337263

theorem fruit_bags_weight (x y z : ℝ) 
  (h1 : x + y = 90) 
  (h2 : y + z = 100) 
  (h3 : z + x = 110) 
  (pos_x : x > 0) 
  (pos_y : y > 0) 
  (pos_z : z > 0) : 
  x + y + z = 150 := by
sorry

end fruit_bags_weight_l3372_337263


namespace probability_of_losing_l3372_337280

theorem probability_of_losing (p_win p_draw : ℚ) (h1 : p_win = 1/3) (h2 : p_draw = 1/2) 
  (h3 : p_win + p_draw + p_lose = 1) : p_lose = 1/6 := by
  sorry

end probability_of_losing_l3372_337280


namespace symmetric_point_theorem_l3372_337298

/-- Given a point (r, θ) in polar coordinates and a line θ = α, 
    the symmetric point with respect to this line has coordinates (r, 2α - θ) -/
def symmetric_point (r : ℝ) (θ : ℝ) (α : ℝ) : ℝ × ℝ := (r, 2*α - θ)

/-- The point symmetric to (3, π/2) with respect to the line θ = π/6 
    has polar coordinates (3, -π/6) -/
theorem symmetric_point_theorem : 
  symmetric_point 3 (π/2) (π/6) = (3, -π/6) := by sorry

end symmetric_point_theorem_l3372_337298


namespace complete_square_l3372_337248

theorem complete_square (x : ℝ) : 
  (x^2 + 6*x + 5 = 0) ↔ ((x + 3)^2 = 4) :=
by sorry

end complete_square_l3372_337248


namespace vector_collinearity_l3372_337254

theorem vector_collinearity (x : ℝ) : 
  let a : Fin 2 → ℝ := ![2, 1]
  let b : Fin 2 → ℝ := ![-1, x]
  let sum : Fin 2 → ℝ := ![a 0 + b 0, a 1 + b 1]
  let diff : Fin 2 → ℝ := ![a 0 - b 0, a 1 - b 1]
  (sum 0 * diff 0 + sum 1 * diff 1 = 0) → (x = 2 ∨ x = -2) := by
  sorry

end vector_collinearity_l3372_337254


namespace all_f_zero_l3372_337242

-- Define the type for infinite sequences of integers
def T := ℕ → ℤ

-- Define the sum of two sequences
def seqSum (x y : T) : T := λ n => x n + y n

-- Define the property of having exactly one 1 and all others 0
def hasOneOne (x : T) : Prop :=
  ∃ i, x i = 1 ∧ ∀ j, j ≠ i → x j = 0

-- Define the function f with its properties
def isValidF (f : T → ℤ) : Prop :=
  (∀ x, hasOneOne x → f x = 0) ∧
  (∀ x y, f (seqSum x y) = f x + f y)

-- The theorem to prove
theorem all_f_zero (f : T → ℤ) (hf : isValidF f) :
  ∀ x : T, f x = 0 := by
  sorry

end all_f_zero_l3372_337242


namespace polynomial_factorization_l3372_337296

theorem polynomial_factorization (x : ℝ) : 12 * x^2 + 8 * x = 4 * x * (3 * x + 2) := by
  sorry

end polynomial_factorization_l3372_337296


namespace min_value_z_l3372_337278

theorem min_value_z (x y : ℝ) (h1 : x - y + 5 ≥ 0) (h2 : x + y ≥ 0) (h3 : x ≤ 3) :
  ∀ z : ℝ, z = (x + y + 2) / (x + 3) → z ≥ 1/3 := by
sorry

end min_value_z_l3372_337278


namespace parabola_symmetric_axis_l3372_337245

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop :=
  y = (1/2) * x^2 - 6*x + 21

/-- The symmetric axis of the parabola -/
def symmetric_axis (x : ℝ) : Prop :=
  x = 6

/-- Theorem: The symmetric axis of the given parabola is x = 6 -/
theorem parabola_symmetric_axis :
  ∀ x y : ℝ, parabola x y → symmetric_axis x :=
by sorry

end parabola_symmetric_axis_l3372_337245
