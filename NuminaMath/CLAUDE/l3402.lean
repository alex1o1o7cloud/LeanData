import Mathlib

namespace geometry_theorem_l3402_340264

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_lines : Line → Line → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- Theorem statement
theorem geometry_theorem 
  (a b : Line) (α β : Plane) 
  (h : perpendicular b α) :
  (parallel_line_plane a α → perpendicular_lines a b) ∧
  (perpendicular b β → parallel_planes α β) := by
  sorry

end geometry_theorem_l3402_340264


namespace cosine_B_in_special_triangle_l3402_340275

theorem cosine_B_in_special_triangle (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π/2 →  -- acute angle A
  0 < B ∧ B < π/2 →  -- acute angle B
  0 < C ∧ C < π/2 →  -- acute angle C
  a = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin A →  -- side-angle relation
  b = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin B →  -- side-angle relation
  c = (2 * Real.sin (A/2) * Real.sin (B/2) * Real.sin (C/2)) / Real.sin C →  -- side-angle relation
  a = Real.sqrt 7 →
  b = 3 →
  Real.sqrt 7 * Real.sin B + Real.sin A = 2 * Real.sqrt 3 →
  Real.cos B = Real.sqrt 7 / 14 := by
sorry

end cosine_B_in_special_triangle_l3402_340275


namespace rent_increase_problem_l3402_340280

/-- Proves that given the conditions of the rent increase scenario, 
    the original rent of the friend whose rent was increased was $1250 -/
theorem rent_increase_problem (num_friends : ℕ) (initial_avg : ℝ) 
  (increase_percent : ℝ) (new_avg : ℝ) : 
  num_friends = 4 →
  initial_avg = 800 →
  increase_percent = 0.16 →
  new_avg = 850 →
  ∃ (original_rent : ℝ), 
    original_rent * (1 + increase_percent) + 
    (num_friends - 1 : ℝ) * initial_avg = 
    num_friends * new_avg ∧ 
    original_rent = 1250 := by
  sorry

end rent_increase_problem_l3402_340280


namespace travel_time_ngapara_to_zipra_l3402_340266

/-- Proves that the time taken to travel from Ngapara to Zipra is 60 hours -/
theorem travel_time_ngapara_to_zipra (time_ngapara_zipra : ℝ) 
  (h1 : 0.8 * time_ngapara_zipra + time_ngapara_zipra = 108) : 
  time_ngapara_zipra = 60 := by
  sorry

end travel_time_ngapara_to_zipra_l3402_340266


namespace total_chocolates_in_month_l3402_340291

/-- Represents the number of chocolates Kantana buys for herself each Saturday -/
def self_chocolates_per_saturday : ℕ := 2

/-- Represents the number of chocolates Kantana buys for her sister each Saturday -/
def sister_chocolates_per_saturday : ℕ := 1

/-- Represents the number of Saturdays in a month -/
def saturdays_in_month : ℕ := 4

/-- Represents the number of chocolates Kantana bought for her friend Charlie -/
def charlie_chocolates : ℕ := 10

/-- Theorem stating the total number of chocolates Kantana bought in a month -/
theorem total_chocolates_in_month : 
  self_chocolates_per_saturday * saturdays_in_month + 
  sister_chocolates_per_saturday * saturdays_in_month + 
  charlie_chocolates = 22 := by
  sorry

end total_chocolates_in_month_l3402_340291


namespace smallest_odd_angle_in_right_triangle_l3402_340205

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

theorem smallest_odd_angle_in_right_triangle :
  ∀ y : ℕ, 
    (is_odd y) →
    (∃ x : ℕ, 
      (is_even x) ∧ 
      (x + y = 90) ∧ 
      (x > y)) →
    y ≥ 31 :=
by sorry

end smallest_odd_angle_in_right_triangle_l3402_340205


namespace total_letters_is_68_l3402_340253

/-- The total number of letters in all siblings' names -/
def total_letters : ℕ :=
  let jonathan_first := 8
  let jonathan_last := 10
  let younger_sister_first := 5
  let younger_sister_last := 10
  let older_brother_first := 6
  let older_brother_last := 10
  let youngest_sibling_first := 4
  let youngest_sibling_last := 15
  (jonathan_first + jonathan_last) +
  (younger_sister_first + younger_sister_last) +
  (older_brother_first + older_brother_last) +
  (youngest_sibling_first + youngest_sibling_last)

/-- Theorem stating that the total number of letters in all siblings' names is 68 -/
theorem total_letters_is_68 : total_letters = 68 := by
  sorry

end total_letters_is_68_l3402_340253


namespace hexagonal_solid_volume_l3402_340219

/-- The volume of a solid with a hexagonal base and scaled, rotated upper face -/
theorem hexagonal_solid_volume : 
  let s : ℝ := 4  -- side length of base
  let h : ℝ := 9  -- height of solid
  let base_area : ℝ := (3 * Real.sqrt 3 / 2) * s^2
  let upper_area : ℝ := (3 * Real.sqrt 3 / 2) * (1.5 * s)^2
  let avg_area : ℝ := (base_area + upper_area) / 2
  let volume : ℝ := avg_area * h
  volume = 351 * Real.sqrt 3 := by sorry

end hexagonal_solid_volume_l3402_340219


namespace diesel_cost_approximation_l3402_340276

/-- Calculates the approximate average cost of diesel per litre over three years -/
def average_diesel_cost (price1 price2 price3 yearly_spend : ℚ) : ℚ :=
  let litres1 := yearly_spend / price1
  let litres2 := yearly_spend / price2
  let litres3 := yearly_spend / price3
  let total_litres := litres1 + litres2 + litres3
  let total_spent := 3 * yearly_spend
  total_spent / total_litres

/-- Theorem stating that the average diesel cost is approximately 8.98 given the specified conditions -/
theorem diesel_cost_approximation :
  let price1 : ℚ := 8.5
  let price2 : ℚ := 9
  let price3 : ℚ := 9.5
  let yearly_spend : ℚ := 5000
  let result := average_diesel_cost price1 price2 price3 yearly_spend
  ∃ (ε : ℚ), ε > 0 ∧ ε < 0.01 ∧ |result - 8.98| < ε :=
sorry

end diesel_cost_approximation_l3402_340276


namespace tom_seashells_l3402_340257

/-- The number of seashells Tom found yesterday -/
def seashells_yesterday : ℕ := 7

/-- The number of seashells Tom found today -/
def seashells_today : ℕ := 4

/-- The total number of seashells Tom found -/
def total_seashells : ℕ := seashells_yesterday + seashells_today

/-- Proof that the total number of seashells Tom found is 11 -/
theorem tom_seashells : total_seashells = 11 := by
  sorry

end tom_seashells_l3402_340257


namespace problem_solution_l3402_340216

theorem problem_solution : (2013^2 - 2013 - 1) / 2013 = 2012 - 1/2013 := by sorry

end problem_solution_l3402_340216


namespace total_marbles_l3402_340281

def jungkook_marbles : ℕ := 3
def marble_difference : ℕ := 4

def jimin_marbles : ℕ := jungkook_marbles + marble_difference

theorem total_marbles :
  jungkook_marbles + jimin_marbles = 10 := by
  sorry

end total_marbles_l3402_340281


namespace eulerian_path_figures_l3402_340225

-- Define a structure for our figures
structure Figure where
  has_eulerian_path : Bool
  all_vertices_even_degree : Bool
  num_odd_degree_vertices : Nat

-- Define our theorem
theorem eulerian_path_figures :
  let figureA : Figure := { has_eulerian_path := true, all_vertices_even_degree := true, num_odd_degree_vertices := 0 }
  let figureB : Figure := { has_eulerian_path := true, all_vertices_even_degree := false, num_odd_degree_vertices := 0 }
  let figureC : Figure := { has_eulerian_path := false, all_vertices_even_degree := false, num_odd_degree_vertices := 3 }
  let figureD : Figure := { has_eulerian_path := true, all_vertices_even_degree := true, num_odd_degree_vertices := 0 }
  ∀ (f : Figure),
    (f.all_vertices_even_degree ∨ f.num_odd_degree_vertices = 2) ↔ f.has_eulerian_path :=
by
  sorry


end eulerian_path_figures_l3402_340225


namespace power_multiplication_division_equality_l3402_340271

theorem power_multiplication_division_equality : (12 : ℕ)^1 * 6^4 / 432 = 36 := by
  sorry

end power_multiplication_division_equality_l3402_340271


namespace polynomial_roots_product_l3402_340222

theorem polynomial_roots_product (b c : ℤ) : 
  (∀ r : ℝ, r^2 - r - 2 = 0 → r^5 - b*r - c = 0) → b*c = 110 := by
  sorry

end polynomial_roots_product_l3402_340222


namespace chrysanthemum_pots_count_l3402_340254

/-- The total number of chrysanthemum pots -/
def total_pots : ℕ := 360

/-- The number of rows after transportation -/
def remaining_rows : ℕ := 9

/-- The number of pots in each row -/
def pots_per_row : ℕ := 20

/-- Theorem stating that the total number of chrysanthemum pots is 360 -/
theorem chrysanthemum_pots_count :
  total_pots = 2 * remaining_rows * pots_per_row :=
by sorry

end chrysanthemum_pots_count_l3402_340254


namespace marble_bag_problem_l3402_340202

/-- Given a bag of black and white marbles, if removing one black marble
    results in 1/8 of the remaining marbles being black, and removing three
    white marbles results in 1/6 of the remaining marbles being black,
    then the initial number of marbles in the bag is 9. -/
theorem marble_bag_problem (x y : ℕ) : 
  x > 0 → y > 0 →
  (x - 1 : ℚ) / (x + y - 1 : ℚ) = 1 / 8 →
  x / (x + y - 3 : ℚ) = 1 / 6 →
  x + y = 9 :=
by sorry

end marble_bag_problem_l3402_340202


namespace probability_not_pulling_prize_l3402_340283

/-- Given odds of 5:6 for pulling a prize, prove that the probability of not pulling the prize is 6/11 -/
theorem probability_not_pulling_prize (favorable_outcomes unfavorable_outcomes : ℕ) 
  (h : favorable_outcomes = 5 ∧ unfavorable_outcomes = 6) : 
  (unfavorable_outcomes : ℚ) / (favorable_outcomes + unfavorable_outcomes) = 6 / 11 := by
  sorry

end probability_not_pulling_prize_l3402_340283


namespace excluded_students_average_mark_l3402_340236

theorem excluded_students_average_mark 
  (N : ℕ) 
  (A : ℚ) 
  (E : ℕ) 
  (A_remaining : ℚ) 
  (h1 : N = 25)
  (h2 : A = 80)
  (h3 : E = 5)
  (h4 : A_remaining = 95) :
  let A_excluded := ((N : ℚ) * A - (N - E : ℚ) * A_remaining) / E
  A_excluded = 20 := by
sorry

end excluded_students_average_mark_l3402_340236


namespace count_valid_n_l3402_340217

theorem count_valid_n : ∃! (s : Finset ℕ), 
  (∀ n ∈ s, 0 < n ∧ n < 35 ∧ ∃ k : ℕ, k > 0 ∧ n = k * (35 - n)) ∧ 
  s.card = 2 := by
  sorry

end count_valid_n_l3402_340217


namespace total_rhino_weight_l3402_340268

/-- The weight of a white rhino in pounds -/
def white_rhino_weight : ℕ := 5100

/-- The weight of a black rhino in pounds -/
def black_rhino_weight : ℕ := 2000

/-- The number of white rhinos -/
def num_white_rhinos : ℕ := 7

/-- The number of black rhinos -/
def num_black_rhinos : ℕ := 8

/-- Theorem: The total weight of 7 white rhinos and 8 black rhinos is 51,700 pounds -/
theorem total_rhino_weight :
  num_white_rhinos * white_rhino_weight + num_black_rhinos * black_rhino_weight = 51700 := by
  sorry

end total_rhino_weight_l3402_340268


namespace tom_spent_seven_tickets_on_hat_l3402_340209

/-- The number of tickets Tom spent on the hat -/
def tickets_spent_on_hat (whack_a_mole_tickets : ℕ) (skee_ball_tickets : ℕ) (remaining_tickets : ℕ) : ℕ :=
  whack_a_mole_tickets + skee_ball_tickets - remaining_tickets

/-- Proof that Tom spent 7 tickets on the hat -/
theorem tom_spent_seven_tickets_on_hat :
  tickets_spent_on_hat 32 25 50 = 7 := by
  sorry

end tom_spent_seven_tickets_on_hat_l3402_340209


namespace greatest_n_with_222_digits_l3402_340203

def a (n : ℕ) : ℚ := (2 * 10^(n+1) - 20 - 18*n) / 81

def number_of_digits (q : ℚ) : ℕ := sorry

theorem greatest_n_with_222_digits : 
  ∃ (n : ℕ), (∀ m : ℕ, number_of_digits (a m) = 222 → m ≤ n) ∧ 
  number_of_digits (a n) = 222 ∧ n = 222 := by sorry

end greatest_n_with_222_digits_l3402_340203


namespace parabola_line_intersection_l3402_340232

/-- Given a parabola y = ax^2 - a (a ≠ 0) intersecting a line y = kx at points 
    with x-coordinates summing to less than 0, prove that the line y = ax + k 
    passes through the first and fourth quadrants. -/
theorem parabola_line_intersection (a k : ℝ) (h_a : a ≠ 0) :
  (∃ x₁ x₂ : ℝ, a * x₁^2 - a = k * x₁ ∧ 
               a * x₂^2 - a = k * x₂ ∧ 
               x₁ + x₂ < 0) →
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = a * x + k) ∧
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ y = a * x + k) :=
by sorry

end parabola_line_intersection_l3402_340232


namespace exponent_sum_l3402_340285

theorem exponent_sum (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 8) : a^(m+n) = 16 := by
  sorry

end exponent_sum_l3402_340285


namespace integer_pairs_satisfying_equation_l3402_340290

theorem integer_pairs_satisfying_equation :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (p : ℤ × ℤ), p ∈ s ↔ p.1 + p.2 = p.1 * p.2 - 2) ∧ 
    s.card = 6 := by
  sorry

end integer_pairs_satisfying_equation_l3402_340290


namespace lucy_cake_packs_l3402_340247

/-- Represents the number of packs of cookies Lucy bought -/
def cookie_packs : ℕ := 23

/-- Represents the total number of grocery packs Lucy bought -/
def total_packs : ℕ := 27

/-- Represents the number of cake packs Lucy bought -/
def cake_packs : ℕ := total_packs - cookie_packs

/-- Proves that the number of cake packs Lucy bought is equal to 4 -/
theorem lucy_cake_packs : cake_packs = 4 := by
  sorry

end lucy_cake_packs_l3402_340247


namespace distribution_theorem_l3402_340224

-- Define the number of books and students
def num_books : ℕ := 5
def num_students : ℕ := 3

-- Define a function to calculate the number of distribution methods
def distribution_methods (n_books : ℕ) (n_students : ℕ) : ℕ :=
  -- Implementation details are not provided as per the instructions
  sorry

-- Theorem statement
theorem distribution_theorem :
  distribution_methods num_books num_students = 150 := by
  sorry

end distribution_theorem_l3402_340224


namespace geometric_sequence_ratio_l3402_340272

/-- Given a geometric sequence {a_n} with sum of first n terms S_n, prove S_5/a_5 = 31 -/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∃ q : ℝ, ∀ n, a (n + 1) = q * a n) →  -- a_n is a geometric sequence
  (a 1 + a 3 = 5/2) →                    -- First condition
  (a 2 + a 4 = 5/4) →                    -- Second condition
  (∀ n, S n = (a 1) * (1 - (a 2 / a 1)^n) / (1 - (a 2 / a 1))) →  -- Definition of S_n
  (S 5 / a 5 = 31) :=                    -- Conclusion to prove
by sorry

end geometric_sequence_ratio_l3402_340272


namespace cherry_pitting_time_l3402_340269

/-- Time required to pit cherries for a pie --/
theorem cherry_pitting_time
  (pounds_needed : ℕ)
  (cherries_per_pound : ℕ)
  (pitting_time : ℕ)
  (cherries_per_batch : ℕ)
  (h1 : pounds_needed = 3)
  (h2 : cherries_per_pound = 80)
  (h3 : pitting_time = 10)
  (h4 : cherries_per_batch = 20) :
  (pounds_needed * cherries_per_pound * pitting_time) / (cherries_per_batch * 60) = 2 :=
by sorry

end cherry_pitting_time_l3402_340269


namespace efficiency_increase_sakshi_to_tanya_l3402_340244

/-- The percentage increase in efficiency between two work rates -/
def efficiency_increase (rate1 rate2 : ℚ) : ℚ :=
  (rate2 - rate1) / rate1 * 100

/-- Sakshi's work rate in parts per day -/
def sakshi_rate : ℚ := 1 / 25

/-- Tanya's work rate in parts per day -/
def tanya_rate : ℚ := 1 / 20

theorem efficiency_increase_sakshi_to_tanya :
  efficiency_increase sakshi_rate tanya_rate = 25 := by
  sorry

end efficiency_increase_sakshi_to_tanya_l3402_340244


namespace employee_payment_l3402_340233

theorem employee_payment (total_payment x y z : ℝ) : 
  total_payment = 1000 →
  x = 1.2 * y →
  z = 0.8 * y →
  x + z = 600 →
  y = 300 := by sorry

end employee_payment_l3402_340233


namespace similar_polygons_perimeter_ratio_l3402_340292

/-- If the ratio of the areas of two similar polygons is 4:9, then the ratio of their perimeters is 2:3 -/
theorem similar_polygons_perimeter_ratio (A B : ℝ) (P Q : ℝ) 
  (h_area : A / B = 4 / 9) (h_positive : A > 0 ∧ B > 0 ∧ P > 0 ∧ Q > 0)
  (h_area_perimeter : A / B = (P / Q)^2) : P / Q = 2 / 3 := by
  sorry

end similar_polygons_perimeter_ratio_l3402_340292


namespace discount_calculation_l3402_340249

/-- Given a bill amount and a discount for double the time, calculate the discount for the original time. -/
theorem discount_calculation (bill_amount : ℝ) (double_time_discount : ℝ) 
  (h1 : bill_amount = 110) 
  (h2 : double_time_discount = 18.33) : 
  ∃ (original_discount : ℝ), original_discount = 9.165 ∧ 
  original_discount = double_time_discount / 2 := by
  sorry

#check discount_calculation

end discount_calculation_l3402_340249


namespace pen_price_calculation_l3402_340227

theorem pen_price_calculation (total_cost : ℝ) (num_pens : ℕ) (num_pencils : ℕ) (pencil_price : ℝ) :
  total_cost = 450 ∧ num_pens = 30 ∧ num_pencils = 75 ∧ pencil_price = 2 →
  (total_cost - num_pencils * pencil_price) / num_pens = 10 := by
sorry

end pen_price_calculation_l3402_340227


namespace sqrt_equation_solution_l3402_340218

theorem sqrt_equation_solution :
  ∀ x : ℝ, (Real.sqrt ((1 + Real.sqrt 2) ^ x) + Real.sqrt ((1 - Real.sqrt 2) ^ x) = 2) ↔ (x = 0) :=
by sorry

end sqrt_equation_solution_l3402_340218


namespace log_x3y2_equals_2_l3402_340298

theorem log_x3y2_equals_2 
  (x y : ℝ) 
  (h1 : Real.log (x^2 * y^5) = 2) 
  (h2 : Real.log (x^3 * y^2) = 2) : 
  Real.log (x^3 * y^2) = 2 := by
sorry

end log_x3y2_equals_2_l3402_340298


namespace birds_on_fence_l3402_340206

theorem birds_on_fence (initial_birds joining_birds : ℕ) : 
  initial_birds = 2 → joining_birds = 4 → initial_birds + joining_birds = 6 :=
by sorry

end birds_on_fence_l3402_340206


namespace brick_weighs_32_kg_l3402_340284

-- Define the weight of one brick
def brick_weight : ℝ := sorry

-- Define the weight of one statue
def statue_weight : ℝ := sorry

-- Theorem stating the weight of one brick is 32 kg
theorem brick_weighs_32_kg : brick_weight = 32 :=
  by
  -- Condition 1: 5 bricks weigh the same as 4 statues
  have h1 : 5 * brick_weight = 4 * statue_weight := sorry
  -- Condition 2: 2 statues weigh 80 kg
  have h2 : 2 * statue_weight = 80 := sorry
  sorry -- Proof goes here


end brick_weighs_32_kg_l3402_340284


namespace gcd_1515_600_l3402_340243

theorem gcd_1515_600 : Nat.gcd 1515 600 = 15 := by
  sorry

end gcd_1515_600_l3402_340243


namespace kerosene_cost_l3402_340289

/-- The cost of kerosene in a market with given price relationships -/
theorem kerosene_cost (rice_pound_cost : ℝ) (h1 : rice_pound_cost = 0.24) :
  let dozen_eggs_cost := rice_pound_cost
  let half_liter_kerosene_cost := dozen_eggs_cost / 2
  let liter_kerosene_cost := 2 * half_liter_kerosene_cost
  let cents_per_dollar := 100
  ⌊liter_kerosene_cost * cents_per_dollar⌋ = 24 := by sorry

end kerosene_cost_l3402_340289


namespace floor_square_minus_square_floor_l3402_340279

theorem floor_square_minus_square_floor (x : ℝ) : x = 13.7 → ⌊x^2⌋ - ⌊x⌋ * ⌊x⌋ = 18 := by
  sorry

end floor_square_minus_square_floor_l3402_340279


namespace sum_of_valid_m_is_three_l3402_340242

-- Define the linear function
def linear_function (m : ℤ) (x : ℝ) : ℝ := (4 - m) * x - 3

-- Define the fractional equation
def fractional_equation (m : ℤ) (z : ℤ) : Prop :=
  m / (z - 1 : ℝ) - 2 = 3 / (1 - z : ℝ)

-- Main theorem
theorem sum_of_valid_m_is_three :
  ∃ (S : Finset ℤ),
    (∀ m ∈ S,
      (∀ x y : ℝ, x < y → linear_function m x < linear_function m y) ∧
      (∃ z : ℤ, z > 1 ∧ fractional_equation m z)) ∧
    (∀ m : ℤ,
      (∀ x y : ℝ, x < y → linear_function m x < linear_function m y) ∧
      (∃ z : ℤ, z > 1 ∧ fractional_equation m z) →
      m ∈ S) ∧
    (S.sum id = 3) :=
by sorry

end sum_of_valid_m_is_three_l3402_340242


namespace expression_value_at_three_l3402_340210

theorem expression_value_at_three : 
  let f (x : ℝ) := (x^2 - 3*x - 10) / (x - 5)
  f 3 = 5 := by
  sorry

end expression_value_at_three_l3402_340210


namespace tree_planting_solution_l3402_340270

/-- Represents the tree planting problem during Arbor Day -/
structure TreePlanting where
  students : ℕ
  typeA : ℕ
  typeB : ℕ

/-- The conditions of the tree planting problem -/
def valid_tree_planting (tp : TreePlanting) : Prop :=
  3 * tp.students + 20 = tp.typeA + tp.typeB ∧
  4 * tp.students = tp.typeA + tp.typeB + 25 ∧
  30 * tp.typeA + 40 * tp.typeB ≤ 5400

/-- The theorem stating the solution to the tree planting problem -/
theorem tree_planting_solution :
  ∃ (tp : TreePlanting), valid_tree_planting tp ∧ tp.students = 45 ∧ tp.typeA ≥ 80 :=
sorry

end tree_planting_solution_l3402_340270


namespace domino_arrangement_theorem_l3402_340274

/-- Represents a domino piece -/
structure Domino :=
  (first : Nat)
  (second : Nat)
  (h1 : first ≤ 6)
  (h2 : second ≤ 6)

/-- Represents a set of dominoes -/
def DominoSet := List Domino

/-- Represents a square frame made of dominoes -/
structure Frame :=
  (dominoes : List Domino)
  (side_sum : Nat)

/-- The total number of points in a standard set of dominoes minus doubles 3, 4, 5, and 6 -/
def total_points : Nat := 132

/-- The number of frames to be formed -/
def num_frames : Nat := 3

/-- Theorem: It's possible to arrange 24 dominoes into 3 square frames with equal side sums -/
theorem domino_arrangement_theorem (dominoes : DominoSet) 
  (h1 : dominoes.length = 24)
  (h2 : (dominoes.map (λ d => d.first + d.second)).sum = total_points) :
  ∃ (frames : List Frame), 
    frames.length = num_frames ∧ 
    (∀ f ∈ frames, f.dominoes.length = 8 ∧ 
      (f.dominoes.map (λ d => d.first + d.second)).sum = total_points / num_frames) ∧
    (∀ f ∈ frames, ∀ side : List Domino, side.length = 3 → 
      (side.map (λ d => d.first + d.second)).sum = f.side_sum) := by
  sorry


end domino_arrangement_theorem_l3402_340274


namespace valid_triangle_divisions_l3402_340246

/-- Represents a division of an equilateral triangle into smaller triangles -/
structure TriangleDivision where
  n : ℕ  -- number of smaller triangles
  k : ℕ  -- number of identical polygons

/-- Predicate to check if a division is valid -/
def is_valid_division (d : TriangleDivision) : Prop :=
  d.n = 36 ∧ d.k ∣ d.n ∧ 
  (d.k = 1 ∨ d.k = 3 ∨ d.k = 4 ∨ d.k = 9 ∨ d.k = 12 ∨ d.k = 36)

/-- Theorem stating the valid divisions of the triangle -/
theorem valid_triangle_divisions :
  ∀ d : TriangleDivision, is_valid_division d ↔ 
    (d.k = 1 ∨ d.k = 3 ∨ d.k = 4 ∨ d.k = 9 ∨ d.k = 12 ∨ d.k = 36) :=
by sorry

end valid_triangle_divisions_l3402_340246


namespace parabola_tangent_intercept_l3402_340226

/-- Parabola structure -/
structure Parabola where
  equation : ℝ → ℝ → Prop

/-- Point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line on a 2D plane -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Definition of the specific parabola x² = 4y -/
def parabola_C : Parabola :=
  { equation := fun x y => x^2 = 4*y }

/-- Focus of the parabola -/
def F : Point :=
  { x := 0, y := 1 }

/-- Point E on the y-axis -/
def E : Point :=
  { x := 0, y := 3 }

/-- Origin -/
def O : Point :=
  { x := 0, y := 0 }

/-- Theorem statement -/
theorem parabola_tangent_intercept :
  ∀ (M : Point),
    parabola_C.equation M.x M.y →
    M.x ≠ 0 →
    (∃ (l : Line),
      -- l is tangent to the parabola at M
      (∀ (P : Point), P.y = l.slope * P.x + l.intercept → parabola_C.equation P.x P.y) →
      -- l passes through M
      M.y = l.slope * M.x + l.intercept →
      -- l is perpendicular to ME
      l.slope * ((M.y - E.y) / (M.x - E.x)) = -1 →
      -- y-intercept of l is -1
      l.intercept = -1) :=
sorry

end parabola_tangent_intercept_l3402_340226


namespace math_books_count_l3402_340234

theorem math_books_count (total_books : ℕ) (math_book_price history_book_price : ℚ) (total_price : ℚ) :
  total_books = 80 →
  math_book_price = 4 →
  history_book_price = 5 →
  total_price = 368 →
  ∃ (math_books : ℕ), 
    math_books ≤ total_books ∧
    math_book_price * math_books + history_book_price * (total_books - math_books) = total_price ∧
    math_books = 32 :=
by
  sorry


end math_books_count_l3402_340234


namespace negative_exponent_equality_l3402_340212

theorem negative_exponent_equality : -5^3 = -(5^3) := by
  sorry

end negative_exponent_equality_l3402_340212


namespace jump_rope_cost_l3402_340238

/-- The cost of Dalton's desired items --/
structure ItemCosts where
  board_game : ℕ
  playground_ball : ℕ
  jump_rope : ℕ

/-- Dalton's available money and additional need --/
structure DaltonMoney where
  allowance : ℕ
  uncle_gift : ℕ
  additional_need : ℕ

/-- Theorem: Given the costs of items and Dalton's available money, 
    prove that the jump rope costs $7 --/
theorem jump_rope_cost 
  (costs : ItemCosts) 
  (money : DaltonMoney) 
  (h1 : costs.board_game = 12)
  (h2 : costs.playground_ball = 4)
  (h3 : money.allowance = 6)
  (h4 : money.uncle_gift = 13)
  (h5 : money.additional_need = 4)
  (h6 : costs.board_game + costs.playground_ball + costs.jump_rope = 
        money.allowance + money.uncle_gift + money.additional_need) :
  costs.jump_rope = 7 := by
  sorry


end jump_rope_cost_l3402_340238


namespace complex_imaginary_condition_l3402_340258

theorem complex_imaginary_condition (m : ℝ) :
  (∃ z : ℂ, z = Complex.mk (3*m - 2) (m - 1) ∧ z.re = 0) → m ≠ 1 :=
by sorry

end complex_imaginary_condition_l3402_340258


namespace x_positive_iff_sum_geq_two_l3402_340260

theorem x_positive_iff_sum_geq_two (x : ℝ) : x > 0 ↔ x + 1/x ≥ 2 := by sorry

end x_positive_iff_sum_geq_two_l3402_340260


namespace smallest_dual_palindrome_is_17_l3402_340262

/-- Checks if a number is a palindrome in the given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop :=
  let digits := Nat.digits base n
  digits = digits.reverse

/-- The smallest positive integer greater than 10 that is a palindrome in both base 2 and base 4 -/
def smallestDualPalindrome : ℕ := 17

theorem smallest_dual_palindrome_is_17 :
  (smallestDualPalindrome > 10) ∧
  (isPalindrome smallestDualPalindrome 2) ∧
  (isPalindrome smallestDualPalindrome 4) ∧
  (∀ n : ℕ, n > 10 ∧ n < smallestDualPalindrome →
    ¬(isPalindrome n 2 ∧ isPalindrome n 4)) :=
by sorry

#eval smallestDualPalindrome

end smallest_dual_palindrome_is_17_l3402_340262


namespace billy_weight_l3402_340248

/-- Given the weights of Billy, Brad, and Carl, prove that Billy weighs 159 pounds. -/
theorem billy_weight (billy brad carl : ℕ) 
  (h1 : billy = brad + 9)
  (h2 : brad = carl + 5)
  (h3 : carl = 145) : 
  billy = 159 := by
  sorry

end billy_weight_l3402_340248


namespace camdens_dogs_legs_count_l3402_340278

theorem camdens_dogs_legs_count :
  ∀ (justin_dogs : ℕ) (rico_dogs : ℕ) (camden_dogs : ℕ),
    justin_dogs = 14 →
    rico_dogs = justin_dogs + 10 →
    camden_dogs = (3 * rico_dogs) / 4 →
    camden_dogs * 4 = 72 := by
  sorry

end camdens_dogs_legs_count_l3402_340278


namespace age_equation_solution_l3402_340241

/-- Given a person's age and a number of years, this function represents the equation in the problem. -/
def ageEquation (A : ℕ) (x : ℕ) : Prop :=
  3 * (A + x) - 3 * (A - x) = A

/-- The theorem states that for an age of 30, the equation is satisfied when x is 5. -/
theorem age_equation_solution :
  ageEquation 30 5 := by
  sorry

end age_equation_solution_l3402_340241


namespace ceiling_neg_sqrt_64_over_9_l3402_340235

theorem ceiling_neg_sqrt_64_over_9 : ⌈-Real.sqrt (64/9)⌉ = -2 := by
  sorry

end ceiling_neg_sqrt_64_over_9_l3402_340235


namespace equation_solutions_l3402_340200

theorem equation_solutions :
  (∀ x : ℝ, 5 * x + 2 = 3 * x - 4 ↔ x = -3) ∧
  (∀ x : ℝ, 1.2 * (x + 4) = 3.6 * (x - 14) ↔ x = 23) := by
  sorry

end equation_solutions_l3402_340200


namespace special_functions_properties_l3402_340273

/-- Two functions satisfying a specific functional equation -/
class SpecialFunctions (f g : ℝ → ℝ) : Prop where
  eq : ∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y
  f_zero : f 0 = 0
  f_nonzero : ∃ x : ℝ, f x ≠ 0

/-- f is an odd function -/
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

/-- g is an even function -/
def IsEvenFunction (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (-x) = g x

/-- Main theorem: f is odd and g is even -/
theorem special_functions_properties {f g : ℝ → ℝ} [SpecialFunctions f g] :
    IsOddFunction f ∧ IsEvenFunction g := by
  sorry

end special_functions_properties_l3402_340273


namespace simplify_expression_l3402_340294

theorem simplify_expression (x : ℝ) : (5 - 4*x) - (7 + 5*x - x^2) = x^2 - 9*x - 2 := by
  sorry

end simplify_expression_l3402_340294


namespace plan_A_first_9_minutes_charge_l3402_340211

/- Define the charge per minute after the first 9 minutes for plan A -/
def plan_A_rate : ℚ := 6 / 100

/- Define the charge per minute for plan B -/
def plan_B_rate : ℚ := 8 / 100

/- Define the duration at which both plans charge the same amount -/
def equal_duration : ℚ := 3

/- Theorem statement -/
theorem plan_A_first_9_minutes_charge : 
  ∃ (charge : ℚ), 
    charge = plan_B_rate * equal_duration ∧ 
    charge = 24 / 100 := by
  sorry

end plan_A_first_9_minutes_charge_l3402_340211


namespace sum_of_digits_of_greatest_prime_divisor_8191_l3402_340214

def greatest_prime_divisor (n : Nat) : Nat :=
  sorry

def sum_of_digits (n : Nat) : Nat :=
  sorry

theorem sum_of_digits_of_greatest_prime_divisor_8191 :
  sum_of_digits (greatest_prime_divisor 8191) = 10 := by
  sorry

end sum_of_digits_of_greatest_prime_divisor_8191_l3402_340214


namespace factorial_difference_l3402_340215

theorem factorial_difference : Nat.factorial 10 - Nat.factorial 9 = 3265920 := by
  sorry

end factorial_difference_l3402_340215


namespace least_questions_for_probability_l3402_340251

theorem least_questions_for_probability (n : ℕ) : n ≥ 4 ↔ (1/2 : ℝ)^n < 1/10 := by sorry

end least_questions_for_probability_l3402_340251


namespace xyz_value_l3402_340213

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 12) :
  4 * x * y * z = 48 := by
  sorry

end xyz_value_l3402_340213


namespace max_sum_of_squares_l3402_340255

/-- Given a system of equations, prove that the maximum value of a^2 + b^2 + c^2 + d^2 is 770 -/
theorem max_sum_of_squares (a b c d : ℝ) 
  (eq1 : a + b = 17)
  (eq2 : a * b + c + d = 98)
  (eq3 : a * d + b * c = 176)
  (eq4 : c * d = 105) :
  a^2 + b^2 + c^2 + d^2 ≤ 770 ∧ ∃ a b c d, a^2 + b^2 + c^2 + d^2 = 770 := by
  sorry

#check max_sum_of_squares

end max_sum_of_squares_l3402_340255


namespace stealth_fighter_most_suitable_for_census_l3402_340231

/-- Represents a survey option -/
structure SurveyOption where
  name : String
  population_size : Nat
  feasibility_of_comprehensive_testing : Nat

/-- Defines the criteria for a survey to be suitable for a comprehensive survey (census) -/
def is_suitable_for_census (s : SurveyOption) : Prop :=
  s.population_size ≤ 1000 ∧ 
  s.importance_of_individual ≥ 9 ∧ 
  s.feasibility_of_comprehensive_testing ≥ 9

/-- The four survey options -/
def survey_options : List SurveyOption := [
  { name := "Car crash resistance", population_size := 10000, importance_of_individual := 5, feasibility_of_comprehensive_testing := 2 },
  { name := "Traffic regulation awareness", population_size := 1000000, importance_of_individual := 3, feasibility_of_comprehensive_testing := 1 },
  { name := "Light bulb service life", population_size := 100000, importance_of_individual := 2, feasibility_of_comprehensive_testing := 3 },
  { name := "Stealth fighter components", population_size := 100, importance_of_individual := 10, feasibility_of_comprehensive_testing := 10 }
]

/-- Theorem stating that the stealth fighter components survey is the most suitable for a comprehensive survey -/
theorem stealth_fighter_most_suitable_for_census :
  ∃ (s : SurveyOption), s ∈ survey_options ∧ 
  s.name = "Stealth fighter components" ∧
  is_suitable_for_census s ∧
  ∀ (t : SurveyOption), t ∈ survey_options → t.name ≠ "Stealth fighter components" → ¬(is_suitable_for_census t) :=
sorry

end stealth_fighter_most_suitable_for_census_l3402_340231


namespace divisibility_by_3804_l3402_340265

theorem divisibility_by_3804 (n : ℕ+) :
  ∃ k : ℤ, (n.val^3 - n.val : ℤ) * (5^(8*n.val+4) + 3^(4*n.val+2)) = 3804 * k :=
by sorry

end divisibility_by_3804_l3402_340265


namespace type_B_completion_time_l3402_340207

/-- The time (in hours) it takes for a type R machine to complete the job -/
def time_R : ℝ := 5

/-- The time (in hours) it takes for 2 type R machines and 3 type B machines working together to complete the job -/
def time_combined : ℝ := 1.2068965517241381

/-- The time (in hours) it takes for a type B machine to complete the job -/
def time_B : ℝ := 7

theorem type_B_completion_time : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |time_B - (3 * time_combined) / (1 / time_combined - 2 / time_R)| < ε :=
sorry

end type_B_completion_time_l3402_340207


namespace total_balls_count_l3402_340286

-- Define the number of boxes
def num_boxes : ℕ := 3

-- Define the number of balls in each box
def balls_per_box : ℕ := 5

-- Theorem to prove
theorem total_balls_count : num_boxes * balls_per_box = 15 := by
  sorry

end total_balls_count_l3402_340286


namespace correct_system_l3402_340223

/-- Represents the length of rope needed to go around the tree once. -/
def y : ℝ := sorry

/-- Represents the total length of the rope. -/
def x : ℝ := sorry

/-- The condition that when the rope goes around the tree 3 times, there will be an extra 5 feet of rope left. -/
axiom three_wraps : 3 * y + 5 = x

/-- The condition that when the rope goes around the tree 4 times, there will be 2 feet less of rope left. -/
axiom four_wraps : 4 * y - 2 = x

/-- Theorem stating that the system of equations correctly represents the problem. -/
theorem correct_system : (3 * y + 5 = x) ∧ (4 * y - 2 = x) := by sorry

end correct_system_l3402_340223


namespace prism_volume_l3402_340261

/-- The volume of a right rectangular prism with face areas 10, 15, and 18 square inches is 30√3 cubic inches. -/
theorem prism_volume (l w h : ℝ) (h1 : l * w = 10) (h2 : w * h = 15) (h3 : l * h = 18) :
  l * w * h = 30 * Real.sqrt 3 := by
  sorry

end prism_volume_l3402_340261


namespace grunters_win_all_games_l3402_340288

/-- The probability of the Grunters winning a single game -/
def p : ℚ := 3/4

/-- The number of games in the series -/
def n : ℕ := 6

/-- The theorem stating the probability of the Grunters winning all games -/
theorem grunters_win_all_games : p ^ n = 729/4096 := by
  sorry

end grunters_win_all_games_l3402_340288


namespace area_of_triangle_PF1F2_l3402_340245

-- Define the ellipse C1
def C1 (x y : ℝ) : Prop := x^2/6 + y^2/2 = 1

-- Define the hyperbola C2
def C2 (x y : ℝ) : Prop := x^2/3 - y^2 = 1

-- Define the foci F1 and F2
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define a point P that satisfies both C1 and C2
def P : ℝ × ℝ := sorry

-- Assume P is on both C1 and C2
axiom P_on_C1 : C1 P.1 P.2
axiom P_on_C2 : C2 P.1 P.2

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Define the area of a triangle given three points
def triangle_area (p1 p2 p3 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem area_of_triangle_PF1F2 : 
  triangle_area P F1 F2 = Real.sqrt 2 := sorry

end area_of_triangle_PF1F2_l3402_340245


namespace xy_reciprocal_inequality_l3402_340250

theorem xy_reciprocal_inequality (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : (1 + x) * (1 + y) = 2) : x * y + 1 / (x * y) ≥ 6 := by
  sorry

end xy_reciprocal_inequality_l3402_340250


namespace dogwood_tree_count_l3402_340263

/-- The number of dogwood trees in the park after planting and removal operations --/
def final_tree_count (initial_trees : ℕ) (planted_today : ℕ) (planted_tomorrow : ℕ) 
                     (removed_today : ℕ) (workers : ℕ) : ℕ :=
  initial_trees + planted_today + planted_tomorrow - removed_today

theorem dogwood_tree_count : 
  let initial_trees := 7
  let planted_today := 5
  let planted_tomorrow := 4
  let removed_today := 3
  let workers := 8
  final_tree_count initial_trees planted_today planted_tomorrow removed_today workers = 13 := by
  sorry

end dogwood_tree_count_l3402_340263


namespace smallest_number_with_million_divisors_l3402_340259

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

theorem smallest_number_with_million_divisors (N : ℕ) :
  divisor_count N = 1000000 →
  N ≥ 2^9 * (3 * 5 * 7 * 11 * 13)^4 :=
by sorry

end smallest_number_with_million_divisors_l3402_340259


namespace sum_reciprocal_n_n_plus_three_l3402_340267

open Real

/-- The sum of the infinite series Σ(1/(n(n+3))) from n=1 to infinity equals 11/18 -/
theorem sum_reciprocal_n_n_plus_three : 
  ∑' n : ℕ+, (1 : ℝ) / (n * (n + 3)) = 11/18 := by sorry

end sum_reciprocal_n_n_plus_three_l3402_340267


namespace childrens_bikes_count_l3402_340287

theorem childrens_bikes_count (regular_bikes : ℕ) (regular_wheels : ℕ) (childrens_wheels : ℕ) (total_wheels : ℕ) :
  regular_bikes = 7 →
  regular_wheels = 2 →
  childrens_wheels = 4 →
  total_wheels = 58 →
  regular_bikes * regular_wheels + childrens_wheels * (total_wheels - regular_bikes * regular_wheels) / childrens_wheels = 11 :=
by
  sorry

#check childrens_bikes_count

end childrens_bikes_count_l3402_340287


namespace cloak_change_theorem_l3402_340204

/-- Represents the cost and change for buying an invisibility cloak -/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the number of silver coins received as change when buying a cloak with gold coins -/
def silver_change_for_gold_payment (t1 t2 : CloakTransaction) (gold_paid : ℕ) : ℕ :=
  sorry

/-- Theorem stating the correct change in silver coins when buying a cloak with 14 gold coins -/
theorem cloak_change_theorem (t1 t2 : CloakTransaction) 
  (h1 : t1 = { silver_paid := 20, gold_change := 4 })
  (h2 : t2 = { silver_paid := 15, gold_change := 1 }) :
  silver_change_for_gold_payment t1 t2 14 = 10 := by
  sorry

end cloak_change_theorem_l3402_340204


namespace complement_of_N_l3402_340229

-- Define the universal set M
def M : Set Nat := {1, 2, 3, 4, 5}

-- Define the set N
def N : Set Nat := {2, 4}

-- State the theorem
theorem complement_of_N : (M \ N) = {1, 3, 5} := by
  sorry

end complement_of_N_l3402_340229


namespace cofactor_sum_l3402_340299

/-- The algebraic cofactor function of the element 7 in the given determinant -/
def f (x a : ℝ) : ℝ := -x^2 - a*x + 2

/-- The theorem stating that if the solution set of f(x) > 0 is (-1, b), then a + b = 1 -/
theorem cofactor_sum (a b : ℝ) : 
  (∀ x, f x a > 0 ↔ -1 < x ∧ x < b) → a + b = 1 := by
  sorry

end cofactor_sum_l3402_340299


namespace rhombus_perimeter_l3402_340239

/-- Given a rhombus with diagonals of length 10 and 24, its perimeter is 52. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 10) (h2 : d2 = 24) : 
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 52 := by
sorry

end rhombus_perimeter_l3402_340239


namespace largest_n_divisibility_n_890_divisibility_n_890_largest_l3402_340208

theorem largest_n_divisibility : ∀ n : ℕ, n > 890 → ¬(n + 10 ∣ n^3 + 100) :=
by
  sorry

theorem n_890_divisibility : (890 + 10 ∣ 890^3 + 100) :=
by
  sorry

theorem n_890_largest : ∀ n : ℕ, n > 0 → (n + 10 ∣ n^3 + 100) → n ≤ 890 :=
by
  sorry

end largest_n_divisibility_n_890_divisibility_n_890_largest_l3402_340208


namespace choose_four_from_fifteen_l3402_340237

theorem choose_four_from_fifteen (n : ℕ) (k : ℕ) : Nat.choose 15 4 = 1365 := by
  sorry

end choose_four_from_fifteen_l3402_340237


namespace smallest_number_divisibility_l3402_340228

theorem smallest_number_divisibility (x : ℕ) : 
  (∀ y : ℕ, y < 1572 → ¬((y + 3) % 9 = 0 ∧ (y + 3) % 35 = 0 ∧ (y + 3) % 25 = 0 ∧ (y + 3) % 21 = 0)) ∧
  ((1572 + 3) % 9 = 0 ∧ (1572 + 3) % 35 = 0 ∧ (1572 + 3) % 25 = 0 ∧ (1572 + 3) % 21 = 0) :=
by
  sorry

end smallest_number_divisibility_l3402_340228


namespace tan_squared_to_sin_squared_l3402_340297

noncomputable def f (x : ℝ) : ℝ :=
  1 / ((x / (x - 1)))

theorem tan_squared_to_sin_squared (t : ℝ) (h1 : 0 ≤ t) (h2 : t ≤ π/2) :
  f (Real.tan t ^ 2) = Real.sin t ^ 2 :=
by
  sorry

end tan_squared_to_sin_squared_l3402_340297


namespace remainder_10_pow_23_minus_7_mod_6_l3402_340230

theorem remainder_10_pow_23_minus_7_mod_6 :
  (10^23 - 7) % 6 = 3 := by sorry

end remainder_10_pow_23_minus_7_mod_6_l3402_340230


namespace parabola_equation_l3402_340201

/-- Given a parabola with focus F(0, p/2) where p > 0, if its directrix intersects 
    the hyperbola x^2 - y^2 = 6 at points M and N such that triangle MNF is 
    a right-angled triangle, then the equation of the parabola is x^2 = 4√2y -/
theorem parabola_equation (p : ℝ) (M N : ℝ × ℝ) 
  (h_p : p > 0)
  (h_hyperbola : M.1^2 - M.2^2 = 6 ∧ N.1^2 - N.2^2 = 6)
  (h_right_triangle : (M.1 - 0)^2 + (M.2 - p/2)^2 = p^2 ∧ 
                      (N.1 - 0)^2 + (N.2 - p/2)^2 = p^2) :
  ∃ (x y : ℝ), x^2 = 4 * Real.sqrt 2 * y := by
  sorry

end parabola_equation_l3402_340201


namespace egg_production_increase_l3402_340240

theorem egg_production_increase (last_year_production this_year_production : ℕ) 
  (h1 : last_year_production = 1416)
  (h2 : this_year_production = 4636) :
  this_year_production - last_year_production = 3220 := by
  sorry

end egg_production_increase_l3402_340240


namespace special_tetrahedron_edges_l3402_340256

/-- A tetrahedron with congruent triangular faces, each having one 60° angle,
    inscribed in a sphere of diameter 23 cm. -/
structure SpecialTetrahedron where
  -- Edge lengths
  a : ℕ
  b : ℕ
  c : ℕ
  -- Conditions
  congruent_faces : a^2 + b^2 - a*b/2 = c^2
  circumsphere_diameter : a^2 + b^2 + c^2 = 2 * 23^2

/-- The edge lengths of the special tetrahedron are 16 cm, 19 cm, and 21 cm. -/
theorem special_tetrahedron_edges :
  ∃ (t : SpecialTetrahedron), t.a = 16 ∧ t.b = 21 ∧ t.c = 19 :=
by sorry

end special_tetrahedron_edges_l3402_340256


namespace inverse_difference_l3402_340295

theorem inverse_difference (a : ℝ) (h : a + a⁻¹ = 6) : a - a⁻¹ = 4 * Real.sqrt 2 ∨ a - a⁻¹ = -4 * Real.sqrt 2 := by
  sorry

end inverse_difference_l3402_340295


namespace union_M_N_equals_M_l3402_340252

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 + p.2 = 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 0}

-- State the theorem
theorem union_M_N_equals_M : M ∪ N = M := by sorry

end union_M_N_equals_M_l3402_340252


namespace simplify_exponential_fraction_l3402_340277

theorem simplify_exponential_fraction (n : ℕ) :
  (3^(n+4) - 3*(3^n)) / (3*(3^(n+3))) = 26 / 9 := by
  sorry

end simplify_exponential_fraction_l3402_340277


namespace bernoulli_misplacement_problem_l3402_340220

def D : ℕ → ℕ
  | 0 => 0
  | 1 => 0
  | 2 => 1
  | (n + 1) => n * (D n + D (n - 1))

theorem bernoulli_misplacement_problem :
  (D 4 : ℚ) / 24 = 3 / 8 ∧
  (6 * D 5 : ℚ) / 720 = 11 / 30 := by
  sorry


end bernoulli_misplacement_problem_l3402_340220


namespace reappearance_line_is_lcm_l3402_340296

/-- The cycle length of the letter sequence -/
def letter_cycle_length : ℕ := 8

/-- The cycle length of the digit sequence -/
def digit_cycle_length : ℕ := 4

/-- The line number where the original sequences reappear -/
def reappearance_line : ℕ := 8

/-- Theorem stating that the reappearance line is the least common multiple of the cycle lengths -/
theorem reappearance_line_is_lcm :
  reappearance_line = Nat.lcm letter_cycle_length digit_cycle_length :=
by sorry

end reappearance_line_is_lcm_l3402_340296


namespace quadratic_function_characterization_l3402_340221

/-- A function satisfying the given functional equation -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≠ y → (deriv f) ((x + y) / 2) = (f y - f x) / (y - x)

/-- The theorem stating that any differentiable function satisfying the functional equation
    is a quadratic function -/
theorem quadratic_function_characterization (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
    (h : SatisfiesFunctionalEquation f) :
    ∃ a b c : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x + c := by
  sorry

end quadratic_function_characterization_l3402_340221


namespace sin_cos_identity_l3402_340282

theorem sin_cos_identity (x : ℝ) (h : Real.sin (x + π/3) = 1/3) :
  Real.sin (5*π/3 - x) - Real.cos (2*x - π/3) = 4/9 := by
  sorry

end sin_cos_identity_l3402_340282


namespace white_space_area_is_31_l3402_340293

/-- Represents the dimensions of a rectangular board -/
structure Board :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the area of a board -/
def boardArea (b : Board) : ℕ := b.width * b.height

/-- Represents the area covered by each letter -/
structure LetterAreas :=
  (C : ℕ)
  (O : ℕ)
  (D : ℕ)
  (E : ℕ)

/-- Calculates the total area covered by all letters -/
def totalLetterArea (l : LetterAreas) : ℕ := l.C + l.O + l.D + l.E

/-- The main theorem stating the white space area -/
theorem white_space_area_is_31 (board : Board) (letters : LetterAreas) : 
  board.width = 4 ∧ board.height = 18 ∧ 
  letters.C = 8 ∧ letters.O = 10 ∧ letters.D = 10 ∧ letters.E = 13 →
  boardArea board - totalLetterArea letters = 31 := by
  sorry


end white_space_area_is_31_l3402_340293
