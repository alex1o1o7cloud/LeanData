import Mathlib

namespace perimeter_difference_is_zero_l3272_327226

/-- The perimeter of a rectangle -/
def rectangle_perimeter (length width : ℕ) : ℕ :=
  2 * (length + width)

/-- The perimeter of Shape 1 -/
def shape1_perimeter : ℕ :=
  rectangle_perimeter 4 3

/-- The perimeter of Shape 2 -/
def shape2_perimeter : ℕ :=
  rectangle_perimeter 6 1

/-- The positive difference between the perimeters of Shape 1 and Shape 2 -/
def perimeter_difference : ℕ :=
  Int.natAbs (shape1_perimeter - shape2_perimeter)

theorem perimeter_difference_is_zero : perimeter_difference = 0 := by
  sorry

end perimeter_difference_is_zero_l3272_327226


namespace part1_part2_l3272_327294

-- Define the propositions p and q
def p (x a : ℝ) : Prop := (x - 3*a) / (a - 2*x) ≥ 0 ∧ a > 0

def q (x : ℝ) : Prop := 2*x^2 - 7*x + 6 < 0

-- Part 1
theorem part1 (x : ℝ) : 
  p x 1 ∧ q x → 3/2 < x ∧ x < 2 := by sorry

-- Part 2
theorem part2 (a : ℝ) : 
  (∀ x, ¬(p x a) → ¬(q x)) ∧ 
  (∃ x, ¬(q x) ∧ p x a) → 
  2/3 ≤ a ∧ a ≤ 3 := by sorry

end part1_part2_l3272_327294


namespace quadratic_function_k_value_l3272_327210

/-- Quadratic function g(x) = ax^2 + bx + c -/
def g (a b c : ℤ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_function_k_value (a b c : ℤ) :
  g a b c 2 = 0 →
  90 < g a b c 9 ∧ g a b c 9 < 100 →
  120 < g a b c 10 ∧ g a b c 10 < 130 →
  (∃ k : ℤ, 7000 * k < g a b c 150 ∧ g a b c 150 < 7000 * (k + 1)) →
  (∃ k : ℤ, 7000 * k < g a b c 150 ∧ g a b c 150 < 7000 * (k + 1) ∧ k = 6) :=
by sorry

end quadratic_function_k_value_l3272_327210


namespace max_binder_price_l3272_327225

/-- Proves that the maximum whole-dollar price of a binder is $7 given the conditions --/
theorem max_binder_price (total_money : ℕ) (num_binders : ℕ) (entrance_fee : ℕ) (tax_rate : ℚ) : 
  total_money = 160 →
  num_binders = 18 →
  entrance_fee = 5 →
  tax_rate = 8 / 100 →
  ∃ (price : ℕ), price = 7 ∧ 
    price = ⌊(total_money - entrance_fee) / ((1 + tax_rate) * num_binders)⌋ ∧
    ∀ (p : ℕ), p > price → 
      p * num_binders * (1 + tax_rate) + entrance_fee > total_money :=
by
  sorry

#check max_binder_price

end max_binder_price_l3272_327225


namespace square_sum_implies_product_zero_l3272_327248

theorem square_sum_implies_product_zero (n : ℝ) :
  (n - 2022)^2 + (2023 - n)^2 = 1 → (2022 - n) * (n - 2023) = 0 := by
  sorry

end square_sum_implies_product_zero_l3272_327248


namespace marble_ratio_l3272_327290

/-- Represents the number of marbles each person has -/
structure Marbles where
  atticus : ℕ
  jensen : ℕ
  cruz : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  3 * (m.atticus + m.jensen + m.cruz) = 60 ∧
  m.atticus = 4 ∧
  m.cruz = 8

/-- The theorem stating the ratio of Atticus's marbles to Jensen's marbles -/
theorem marble_ratio (m : Marbles) :
  marble_conditions m → m.atticus * 2 = m.jensen := by
  sorry

#check marble_ratio

end marble_ratio_l3272_327290


namespace jake_not_dropping_coffee_percentage_l3272_327230

-- Define the probabilities
def trip_probability : ℝ := 0.4
def drop_coffee_when_tripping_probability : ℝ := 0.25

-- Theorem to prove
theorem jake_not_dropping_coffee_percentage :
  1 - (trip_probability * drop_coffee_when_tripping_probability) = 0.9 :=
by sorry

end jake_not_dropping_coffee_percentage_l3272_327230


namespace simplify_fraction_l3272_327205

theorem simplify_fraction : (144 : ℚ) / 12672 = 1 / 88 := by sorry

end simplify_fraction_l3272_327205


namespace quadratic_inequality_l3272_327293

/-- The quadratic function f(x) = ax^2 + bx + c -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The solution set of ax^2 + bx + c > 0 is {x | -2 < x < 4} -/
def solution_set (a b c : ℝ) : Set ℝ := {x | -2 < x ∧ x < 4}

theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x > 0) :
  f a b c 2 > f a b c (-1) ∧ f a b c (-1) > f a b c 5 := by
  sorry

end quadratic_inequality_l3272_327293


namespace line_intersection_x_axis_l3272_327221

/-- A line passing through two points (8, 2) and (4, 6) intersects the x-axis at (10, 0) -/
theorem line_intersection_x_axis :
  let p1 : ℝ × ℝ := (8, 2)
  let p2 : ℝ × ℝ := (4, 6)
  let m : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)
  let b : ℝ := p1.2 - m * p1.1
  let line (x : ℝ) : ℝ := m * x + b
  ∃ x : ℝ, line x = 0 ∧ x = 10
:= by sorry

end line_intersection_x_axis_l3272_327221


namespace unique_prime_root_equation_l3272_327243

theorem unique_prime_root_equation :
  ∀ p q n : ℕ,
    Prime p → Prime q → n > 0 →
    (p + q : ℝ) ^ (1 / n : ℝ) = p - q →
    p = 5 ∧ q = 3 ∧ n = 3 := by
  sorry

end unique_prime_root_equation_l3272_327243


namespace wall_area_calculation_l3272_327223

/-- The area of a rectangular wall with width and height of 4 feet is 16 square feet. -/
theorem wall_area_calculation : 
  ∀ (width height area : ℝ), 
  width = 4 → 
  height = 4 → 
  area = width * height → 
  area = 16 :=
by sorry

end wall_area_calculation_l3272_327223


namespace min_orchard_space_l3272_327247

/-- The space required for planting trees in an orchard. -/
def orchard_space (apple apricot plum : ℕ) : ℕ :=
  apple^2 + 5*apricot + plum^3

/-- The minimum space required for planting 10 trees, including at least one of each type. -/
theorem min_orchard_space :
  ∃ (apple apricot plum : ℕ),
    apple + apricot + plum = 10 ∧
    apple ≥ 1 ∧ apricot ≥ 1 ∧ plum ≥ 1 ∧
    ∀ (a b c : ℕ),
      a + b + c = 10 →
      a ≥ 1 → b ≥ 1 → c ≥ 1 →
      orchard_space apple apricot plum ≤ orchard_space a b c ∧
      orchard_space apple apricot plum = 37 :=
by sorry

end min_orchard_space_l3272_327247


namespace nell_initial_cards_l3272_327242

/-- The number of cards Nell gave away -/
def cards_given_away : ℕ := 276

/-- The number of cards Nell has left -/
def cards_left : ℕ := 252

/-- Nell's initial number of cards -/
def initial_cards : ℕ := cards_given_away + cards_left

theorem nell_initial_cards : initial_cards = 528 := by
  sorry

end nell_initial_cards_l3272_327242


namespace average_speed_round_trip_l3272_327216

theorem average_speed_round_trip (speed_xy speed_yx : ℝ) (h1 : speed_xy = 43) (h2 : speed_yx = 34) :
  (2 * speed_xy * speed_yx) / (speed_xy + speed_yx) = 38 := by
  sorry

end average_speed_round_trip_l3272_327216


namespace prob_green_ball_is_five_ninths_l3272_327222

structure Container where
  red_balls : ℕ
  green_balls : ℕ

def total_balls (c : Container) : ℕ := c.red_balls + c.green_balls

def prob_green (c : Container) : ℚ :=
  c.green_balls / (total_balls c)

def containers : List Container := [
  ⟨8, 4⟩,  -- Container I
  ⟨2, 4⟩,  -- Container II
  ⟨2, 4⟩   -- Container III
]

theorem prob_green_ball_is_five_ninths :
  (containers.map prob_green).sum / containers.length = 5 / 9 := by
  sorry

end prob_green_ball_is_five_ninths_l3272_327222


namespace expression_value_l3272_327265

theorem expression_value : 6^3 - 4 * 6^2 + 4 * 6 - 1 = 95 := by
  sorry

end expression_value_l3272_327265


namespace cube_root_square_root_l3272_327270

theorem cube_root_square_root (x : ℝ) : (2 * x)^3 = 216 → (x + 6)^(1/2) = 3 ∨ (x + 6)^(1/2) = -3 := by
  sorry

end cube_root_square_root_l3272_327270


namespace packages_per_box_l3272_327299

/-- Given that Julie bought two boxes of standard paper, each package contains 250 sheets,
    25 sheets are used per newspaper, and 100 newspapers can be printed,
    prove that there are 5 packages in each box. -/
theorem packages_per_box (boxes : ℕ) (sheets_per_package : ℕ) (sheets_per_newspaper : ℕ) (total_newspapers : ℕ) :
  boxes = 2 →
  sheets_per_package = 250 →
  sheets_per_newspaper = 25 →
  total_newspapers = 100 →
  (boxes * sheets_per_package * (total_newspapers * sheets_per_newspaper / (boxes * sheets_per_package)) : ℚ) = total_newspapers * sheets_per_newspaper →
  total_newspapers * sheets_per_newspaper / (boxes * sheets_per_package) = 5 := by
  sorry

end packages_per_box_l3272_327299


namespace largest_number_value_l3272_327262

theorem largest_number_value (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (h3 : a + b + c = 100) (h4 : c = b + 10) (h5 : b = a + 5) : c = 125/3 := by
  sorry

end largest_number_value_l3272_327262


namespace wire_cutting_problem_l3272_327278

theorem wire_cutting_problem :
  let total_length : ℕ := 102
  let piece_length_1 : ℕ := 15
  let piece_length_2 : ℕ := 12
  ∀ x y : ℕ, piece_length_1 * x + piece_length_2 * y = total_length →
    (x = 2 ∧ y = 6) ∨ (x = 6 ∧ y = 1) :=
by sorry

end wire_cutting_problem_l3272_327278


namespace total_initial_money_l3272_327282

/-- Represents the money redistribution game between three friends --/
structure MoneyRedistribution where
  amy_initial : ℝ
  jan_initial : ℝ
  toy_initial : ℝ
  amy_final : ℝ
  jan_final : ℝ
  toy_final : ℝ

/-- The theorem stating the total initial amount of money --/
theorem total_initial_money (game : MoneyRedistribution) 
  (h1 : game.amy_initial = 50)
  (h2 : game.toy_initial = 50)
  (h3 : game.amy_final = game.amy_initial)
  (h4 : game.toy_final = game.toy_initial)
  (h5 : game.amy_final = 2 * (2 * (game.amy_initial - (game.jan_initial + game.toy_initial))))
  (h6 : game.jan_final = 2 * (2 * game.jan_initial - (2 * game.toy_initial + (game.amy_initial - (game.jan_initial + game.toy_initial)))))
  (h7 : game.toy_final = 2 * game.toy_initial - (game.amy_final + game.jan_final - (2 * game.toy_initial + 2 * (game.amy_initial - (game.jan_initial + game.toy_initial)))))
  : game.amy_initial + game.jan_initial + game.toy_initial = 187.5 := by
  sorry

end total_initial_money_l3272_327282


namespace sum_of_roots_2016_l3272_327273

/-- The function f(x) = x^2 - 2016x + 2015 -/
def f (x : ℝ) : ℝ := x^2 - 2016*x + 2015

/-- Theorem: If f(a) = f(b) = c for distinct a and b, then a + b = 2016 -/
theorem sum_of_roots_2016 (a b c : ℝ) (ha : f a = c) (hb : f b = c) (hab : a ≠ b) : a + b = 2016 := by
  sorry

end sum_of_roots_2016_l3272_327273


namespace min_value_z3_l3272_327257

open Complex

theorem min_value_z3 (z₁ z₂ z₃ : ℂ) 
  (h_im : (z₁ / z₂).im ≠ 0 ∧ (z₁ / z₂).re = 0)
  (h_mag_z1 : abs z₁ = 1)
  (h_mag_z2 : abs z₂ = 1)
  (h_sum : abs (z₁ + z₂ + z₃) = 1) :
  abs z₃ ≥ Real.sqrt 2 - 1 := by
sorry

end min_value_z3_l3272_327257


namespace negative_two_inequality_l3272_327224

theorem negative_two_inequality (a b : ℝ) (h : a > b) : -2 * a < -2 * b := by
  sorry

end negative_two_inequality_l3272_327224


namespace angle_sum_in_cyclic_quad_l3272_327286

-- Define the quadrilateral ABCD and point E
variable (A B C D E : Point)

-- Define the cyclic property of quadrilateral ABCD
def is_cyclic_quad (A B C D : Point) : Prop := sorry

-- Define angle measure
def angle_measure (P Q R : Point) : ℝ := sorry

-- Theorem statement
theorem angle_sum_in_cyclic_quad 
  (h_cyclic : is_cyclic_quad A B C D)
  (h_angle_A : angle_measure B A C = 40)
  (h_equal_angles : angle_measure C E D = angle_measure E C D) :
  angle_measure A B C + angle_measure A D C = 160 := by
  sorry

end angle_sum_in_cyclic_quad_l3272_327286


namespace school_purchase_cost_l3272_327207

/-- The total cost of purchasing sweaters and sports shirts -/
def total_cost (sweater_price : ℕ) (sweater_quantity : ℕ) 
               (shirt_price : ℕ) (shirt_quantity : ℕ) : ℕ :=
  sweater_price * sweater_quantity + shirt_price * shirt_quantity

/-- Theorem stating that the total cost for the given quantities and prices is 5400 yuan -/
theorem school_purchase_cost : 
  total_cost 98 25 59 50 = 5400 := by
  sorry

end school_purchase_cost_l3272_327207


namespace geometric_sequence_minimum_value_l3272_327212

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem statement -/
theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geo : geometric_sequence a)
  (h_2016 : a 2016 = a 2015 + 2 * a 2014)
  (h_mn : ∃ m n : ℕ, a m * a n = 16 * (a 1)^2) :
  ∃ m n : ℕ, (4 / m + 1 / n : ℝ) ≥ 3/2 ∧
    ∀ k l : ℕ, (4 / k + 1 / l : ℝ) ≥ 3/2 :=
sorry

end geometric_sequence_minimum_value_l3272_327212


namespace inscribed_sphere_radius_l3272_327219

/-- A triangular pyramid with specific properties -/
structure SpecialPyramid where
  -- Base side lengths
  a : ℝ
  b : ℝ
  c : ℝ
  -- Condition: base side lengths are √85, √58, and √45
  ha : a = Real.sqrt 85
  hb : b = Real.sqrt 58
  hc : c = Real.sqrt 45
  -- Condition: lateral edges are mutually perpendicular
  lateral_edges_perpendicular : Bool

/-- The sphere inscribed in the special pyramid -/
structure InscribedSphere (p : SpecialPyramid) where
  -- The radius of the sphere
  radius : ℝ
  -- Condition: The sphere touches all lateral faces
  touches_all_faces : Bool
  -- Condition: The center of the sphere lies on the base
  center_on_base : Bool

/-- The main theorem stating the radius of the inscribed sphere -/
theorem inscribed_sphere_radius (p : SpecialPyramid) (s : InscribedSphere p) :
  s.touches_all_faces ∧ s.center_on_base ∧ p.lateral_edges_perpendicular →
  s.radius = 14 / 9 := by
  sorry

end inscribed_sphere_radius_l3272_327219


namespace problem_solution_l3272_327214

theorem problem_solution (x : ℝ) : 0.8 * x - 20 = 60 → x = 100 := by
  sorry

end problem_solution_l3272_327214


namespace linear_function_properties_l3272_327269

-- Define the linear function
def f (x : ℝ) : ℝ := x + 2

-- Theorem stating the properties of the function
theorem linear_function_properties :
  (f 1 = 3) ∧ 
  (f (-2) = 0) ∧ 
  (∃ x > 2, f x ≥ 4) ∧
  (∀ x y, f x = y → (x > 0 → y > 0)) :=
by sorry

#check linear_function_properties

end linear_function_properties_l3272_327269


namespace inequality_proof_l3272_327259

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y * z * ((x + y + z) / 3)) ^ (1/4) ≤ (((x + y) / 2) * ((y + z) / 2) * ((z + x) / 2)) ^ (1/3) :=
sorry

end inequality_proof_l3272_327259


namespace max_value_fraction_l3272_327234

theorem max_value_fraction (x y : ℝ) (hx : -6 ≤ x ∧ x ≤ -3) (hy : 1 ≤ y ∧ y ≤ 5) :
  (∀ x' y', -6 ≤ x' ∧ x' ≤ -3 → 1 ≤ y' ∧ y' ≤ 5 → (x' + y') / x' ≤ (x + y) / x) →
  (x + y) / x = 2/3 :=
sorry

end max_value_fraction_l3272_327234


namespace rafael_remaining_hours_l3272_327260

def hours_worked : ℕ := 18
def hourly_rate : ℕ := 20
def total_earnings : ℕ := 760

theorem rafael_remaining_hours : 
  (total_earnings - hours_worked * hourly_rate) / hourly_rate = 20 := by
  sorry

end rafael_remaining_hours_l3272_327260


namespace quadrilateral_angle_measure_l3272_327228

theorem quadrilateral_angle_measure (W X Y Z : ℝ) : 
  W > 0 ∧ X > 0 ∧ Y > 0 ∧ Z > 0 →
  W = 3 * X ∧ W = 4 * Y ∧ W = 6 * Z →
  W + X + Y + Z = 360 →
  W = 1440 / 7 := by
sorry

end quadrilateral_angle_measure_l3272_327228


namespace problem_solution_l3272_327220

theorem problem_solution : 2.017 * 2016 - 10.16 * 201.7 = 2017 := by
  sorry

end problem_solution_l3272_327220


namespace sum_of_integers_l3272_327292

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 6) (h2 : a * b = 272) : a + b = 34 := by
  sorry

end sum_of_integers_l3272_327292


namespace canoe_kayak_difference_is_five_l3272_327217

/-- Represents the rental information for canoes and kayaks --/
structure RentalInfo where
  canoe_cost : ℕ
  kayak_cost : ℕ
  canoe_kayak_ratio : Rat
  total_revenue : ℕ

/-- Calculates the difference between canoes and kayaks rented --/
def canoe_kayak_difference (info : RentalInfo) : ℕ :=
  let kayaks := (info.total_revenue : ℚ) * 3 / (11 * 4 + 16 * 3)
  let canoes := kayaks * info.canoe_kayak_ratio
  (canoes - kayaks).ceil.toNat

/-- Theorem stating the difference between canoes and kayaks rented --/
theorem canoe_kayak_difference_is_five (info : RentalInfo) 
  (h1 : info.canoe_cost = 11)
  (h2 : info.kayak_cost = 16)
  (h3 : info.canoe_kayak_ratio = 4 / 3)
  (h4 : info.total_revenue = 460) :
  canoe_kayak_difference info = 5 := by
  sorry

#eval canoe_kayak_difference { 
  canoe_cost := 11, 
  kayak_cost := 16, 
  canoe_kayak_ratio := 4 / 3, 
  total_revenue := 460 
}

end canoe_kayak_difference_is_five_l3272_327217


namespace angle_with_complement_quarter_supplement_l3272_327240

theorem angle_with_complement_quarter_supplement (x : ℝ) :
  (90 - x = (1 / 4) * (180 - x)) → x = 60 := by
  sorry

end angle_with_complement_quarter_supplement_l3272_327240


namespace repeating_decimal_equals_fraction_l3272_327279

/-- The repeating decimal 0.37246̄ expressed as a rational number -/
def repeating_decimal : ℚ := 
  37 / 100 + (246 / 999900)

/-- The target fraction -/
def target_fraction : ℚ := 3718740 / 999900

/-- Theorem stating that the repeating decimal 0.37246̄ is equal to 3718740/999900 -/
theorem repeating_decimal_equals_fraction : 
  repeating_decimal = target_fraction := by sorry

end repeating_decimal_equals_fraction_l3272_327279


namespace evaluate_expression_l3272_327237

theorem evaluate_expression : 2001^3 - 2000 * 2001^2 - 2000^2 * 2001 + 2 * 2000^3 = 24008004001 := by
  sorry

end evaluate_expression_l3272_327237


namespace train_speed_problem_l3272_327276

/-- Proves that given two trains traveling towards each other from cities 100 miles apart,
    with one train traveling at 45 mph, if they meet after 1.33333333333 hours,
    then the speed of the other train must be 30 mph. -/
theorem train_speed_problem (distance : ℝ) (speed_train1 : ℝ) (time : ℝ) (speed_train2 : ℝ) : 
  distance = 100 →
  speed_train1 = 45 →
  time = 1.33333333333 →
  distance = speed_train1 * time + speed_train2 * time →
  speed_train2 = 30 := by
sorry

end train_speed_problem_l3272_327276


namespace raymonds_dimes_proof_l3272_327238

/-- The number of dimes Raymond has left after spending at the arcade -/
def raymonds_remaining_dimes : ℕ :=
  let initial_amount : ℕ := 250 -- $2.50 in cents
  let petes_spent : ℕ := 20 -- 4 nickels * 5 cents
  let total_spent : ℕ := 200
  let raymonds_spent : ℕ := total_spent - petes_spent
  let raymonds_remaining : ℕ := initial_amount - raymonds_spent
  raymonds_remaining / 10 -- divide by 10 cents per dime

theorem raymonds_dimes_proof :
  raymonds_remaining_dimes = 7 := by
  sorry

end raymonds_dimes_proof_l3272_327238


namespace michaels_fish_count_l3272_327258

/-- Given Michael's initial fish count and the number of fish Ben gives him,
    prove that the total number of fish Michael has now is equal to the sum of these two quantities. -/
theorem michaels_fish_count (initial : Real) (given : Real) :
  initial + given = initial + given :=
by sorry

end michaels_fish_count_l3272_327258


namespace problem_solution_l3272_327200

theorem problem_solution (y : ℝ) (h1 : y > 0) (h2 : y * y / 100 = 9) : y = 30 := by
  sorry

end problem_solution_l3272_327200


namespace crackers_distribution_l3272_327295

theorem crackers_distribution (total_crackers : ℕ) (num_friends : ℕ) (crackers_per_friend : ℕ) :
  total_crackers = 36 →
  num_friends = 6 →
  total_crackers = num_friends * crackers_per_friend →
  crackers_per_friend = 6 :=
by sorry

end crackers_distribution_l3272_327295


namespace minimum_additional_games_l3272_327253

def initial_games : ℕ := 3
def initial_wins : ℕ := 2
def target_percentage : ℚ := 9/10

def winning_percentage (additional_games : ℕ) : ℚ :=
  (initial_wins + additional_games) / (initial_games + additional_games)

theorem minimum_additional_games :
  ∃ N : ℕ, (∀ n : ℕ, n < N → winning_percentage n < target_percentage) ∧
            winning_percentage N ≥ target_percentage ∧
            N = 7 :=
by sorry

end minimum_additional_games_l3272_327253


namespace find_d_l3272_327284

-- Define the functions f and g
def f (c : ℝ) (x : ℝ) : ℝ := 5 * x + c
def g (c : ℝ) (x : ℝ) : ℝ := c * x - 3

-- State the theorem
theorem find_d (c : ℝ) (d : ℝ) : 
  (∀ x, f c (g c x) = -15 * x + d) → d = -18 := by
  sorry

end find_d_l3272_327284


namespace smallest_common_pet_count_l3272_327213

theorem smallest_common_pet_count : ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), x > 1 ∧ 2 ∣ m ∧ x ∣ m) → 
    n ≤ m) ∧ 
  (∃ (x : ℕ), x > 1 ∧ 2 ∣ n ∧ x ∣ n) ∧
  n = 4 := by
  sorry

end smallest_common_pet_count_l3272_327213


namespace pattern_theorem_l3272_327251

/-- Function to create a number with the first n digits of 123456... -/
def firstNDigits (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (firstNDigits (n-1)) * 10 + n

/-- Function to create a number with n ones -/
def nOnes (n : ℕ) : ℕ :=
  if n = 0 then 0
  else (nOnes (n-1)) * 10 + 1

/-- Theorem stating the pattern observed in the problem -/
theorem pattern_theorem (n : ℕ) : 
  (firstNDigits n) * 9 + (n + 1) = nOnes (n + 1) :=
by sorry

end pattern_theorem_l3272_327251


namespace juanita_dessert_cost_l3272_327252

/-- Calculates the cost of Juanita's dessert given the prices of individual items --/
def dessert_cost (brownie_price ice_cream_price syrup_price nuts_price : ℚ) : ℚ :=
  brownie_price + 2 * ice_cream_price + 2 * syrup_price + nuts_price

/-- Proves that Juanita's dessert costs $7.00 given the prices of individual items --/
theorem juanita_dessert_cost :
  dessert_cost 2.5 1 0.5 1.5 = 7 := by
  sorry

#eval dessert_cost 2.5 1 0.5 1.5

end juanita_dessert_cost_l3272_327252


namespace min_value_of_expression_l3272_327261

/-- Given a function f(x) = x² + 2√a x - b + 1, where a and b are positive real numbers,
    and f(x) has only one zero, the minimum value of 1/a + 2a/(b+1) is 5/2. -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + 2 * Real.sqrt a * x - b + 1 = 0) → 
  (∀ a' b', a' > 0 → b' > 0 → 
    (∃! x, x^2 + 2 * Real.sqrt a' * x - b' + 1 = 0) → 
    1 / a + 2 * a / (b + 1) ≤ 1 / a' + 2 * a' / (b' + 1)) ∧
  (∃ a₀ b₀, a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃! x, x^2 + 2 * Real.sqrt a₀ * x - b₀ + 1 = 0) ∧
    1 / a₀ + 2 * a₀ / (b₀ + 1) = 5 / 2) :=
by sorry

end min_value_of_expression_l3272_327261


namespace transport_theorem_l3272_327254

-- Define the capacity of a worker per hour
def worker_capacity : ℝ := 30

-- Define the capacity of a robot per hour
def robot_capacity : ℝ := 450

-- Define the number of robots
def num_robots : ℕ := 3

-- Define the total amount to be transported
def total_amount : ℝ := 3600

-- Define the time limit
def time_limit : ℝ := 2

-- Define the function to calculate the minimum number of additional workers
def min_additional_workers : ℕ := 15

theorem transport_theorem :
  -- Condition 1: Robot carries 420kg more than a worker
  (robot_capacity = worker_capacity + 420) →
  -- Condition 2: Time for robot to carry 900kg equals time for 10 workers to carry 600kg
  (900 / robot_capacity = 600 / (10 * worker_capacity)) →
  -- Condition 3 & 4 are implicitly used in the conclusion
  -- Conclusion 1: Robot capacity is 450kg per hour
  (robot_capacity = 450) ∧
  -- Conclusion 2: Worker capacity is 30kg per hour
  (worker_capacity = 30) ∧
  -- Conclusion 3: Minimum additional workers needed is 15
  (min_additional_workers = 15 ∧
   robot_capacity * num_robots * time_limit + worker_capacity * min_additional_workers * time_limit ≥ total_amount ∧
   ∀ n : ℕ, n < 15 → robot_capacity * num_robots * time_limit + worker_capacity * n * time_limit < total_amount) :=
by sorry

end transport_theorem_l3272_327254


namespace matrix_inverse_proof_l3272_327296

theorem matrix_inverse_proof : 
  let A : Matrix (Fin 2) (Fin 2) ℝ := !![7, 5; 3, 2]
  let A_inv : Matrix (Fin 2) (Fin 2) ℝ := !![-2, 5; 3, -7]
  A * A_inv = 1 ∧ A_inv * A = 1 := by
  sorry

end matrix_inverse_proof_l3272_327296


namespace pet_store_kittens_l3272_327249

theorem pet_store_kittens (initial : ℕ) (final : ℕ) (new : ℕ) : 
  initial = 6 → final = 9 → new = final - initial → new = 3 := by
  sorry

end pet_store_kittens_l3272_327249


namespace square_side_length_l3272_327233

theorem square_side_length (perimeter : ℝ) (h : perimeter = 17.8) :
  perimeter / 4 = 4.45 := by sorry

end square_side_length_l3272_327233


namespace joshua_skittles_l3272_327274

/-- The number of friends Joshua has -/
def num_friends : ℕ := 5

/-- The number of Skittles each friend would get if Joshua shares them equally -/
def skittles_per_friend : ℕ := 8

/-- The total number of Skittles Joshua has -/
def total_skittles : ℕ := num_friends * skittles_per_friend

/-- Theorem: Joshua has 40 Skittles -/
theorem joshua_skittles : total_skittles = 40 := by
  sorry

end joshua_skittles_l3272_327274


namespace divisible_by_eight_l3272_327227

theorem divisible_by_eight (b n : ℕ) (h1 : Even b) (h2 : b > 0) (h3 : n > 1)
  (h4 : ∃ k : ℕ, (b^n - 1) / (b - 1) = k^2) : 
  8 ∣ b := by
  sorry

end divisible_by_eight_l3272_327227


namespace xy_plus_y_squared_l3272_327202

theorem xy_plus_y_squared (x y : ℝ) (h : x * (x + y) = x^2 + 3*y + 12) :
  x*y + y^2 = y^2 + 3*y + 12 := by
  sorry

end xy_plus_y_squared_l3272_327202


namespace arithmetic_sequence_sum_l3272_327236

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  a : ℤ  -- First term
  d : ℤ  -- Common difference

/-- Get the nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.a + seq.d * (n - 1)

/-- Sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  n * (2 * seq.a + seq.d * (n - 1)) / 2

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) :
  seq.nthTerm 7 = 4 ∧ seq.nthTerm 8 = 10 ∧ seq.nthTerm 9 = 16 →
  seq.sumFirstN 5 = -100 := by
  sorry


end arithmetic_sequence_sum_l3272_327236


namespace ned_trips_theorem_l3272_327283

/-- The number of trays Ned can carry in one trip -/
def trays_per_trip : ℕ := 8

/-- The number of trays on the first table -/
def trays_table1 : ℕ := 27

/-- The number of trays on the second table -/
def trays_table2 : ℕ := 5

/-- The total number of trays Ned needs to pick up -/
def total_trays : ℕ := trays_table1 + trays_table2

/-- The number of trips Ned will make -/
def num_trips : ℕ := (total_trays + trays_per_trip - 1) / trays_per_trip

theorem ned_trips_theorem : num_trips = 4 := by
  sorry

end ned_trips_theorem_l3272_327283


namespace farmer_cow_count_farmer_has_52_cows_l3272_327266

/-- The number of cows a farmer has, given milk production data. -/
theorem farmer_cow_count : ℕ :=
  let milk_per_cow_per_day : ℕ := 5
  let days_per_week : ℕ := 7
  let total_milk_per_week : ℕ := 1820
  let milk_per_cow_per_week : ℕ := milk_per_cow_per_day * days_per_week
  total_milk_per_week / milk_per_cow_per_week

/-- Proof that the farmer has 52 cows. -/
theorem farmer_has_52_cows : farmer_cow_count = 52 := by
  sorry

end farmer_cow_count_farmer_has_52_cows_l3272_327266


namespace min_value_reciprocal_sum_l3272_327288

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : m * n > 0) :
  1/m + 1/n ≥ 4 := by
sorry

end min_value_reciprocal_sum_l3272_327288


namespace platform_length_calculation_l3272_327289

/-- Given a train of length 300 meters that crosses a platform in 51 seconds
    and a signal pole in 18 seconds, prove that the length of the platform
    is approximately 550.17 meters. -/
theorem platform_length_calculation (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 51)
  (h3 : pole_crossing_time = 18) :
  ∃ (platform_length : ℝ), abs (platform_length - 550.17) < 0.01 :=
by sorry

end platform_length_calculation_l3272_327289


namespace power_equality_l3272_327291

theorem power_equality (y x : ℕ) (h1 : 9^y = 3^x) (h2 : y = 6) : x = 12 := by
  sorry

end power_equality_l3272_327291


namespace sum_abcd_equals_negative_fourteen_thirds_l3272_327201

theorem sum_abcd_equals_negative_fourteen_thirds 
  (a b c d : ℚ) 
  (h : a + 2 = b + 3 ∧ b + 3 = c + 4 ∧ c + 4 = d + 5 ∧ d + 5 = a + b + c + d + 7) : 
  a + b + c + d = -14/3 := by
sorry

end sum_abcd_equals_negative_fourteen_thirds_l3272_327201


namespace hyperbola_eccentricity_l3272_327281

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is c/a, where c² = a² + b² --/
theorem hyperbola_eccentricity (a b : ℝ) (h : 0 < a ∧ 0 < b) :
  let c := Real.sqrt (a^2 + b^2)
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) →
  c / a = 5 / 3 :=
by sorry

end hyperbola_eccentricity_l3272_327281


namespace unique_solution_l3272_327215

def is_valid_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

def satisfies_equation (Θ : ℕ) : Prop :=
  is_valid_digit Θ ∧ 
  (198 : ℚ) / Θ = (40 : ℚ) + 2 * Θ

theorem unique_solution : 
  ∃! Θ : ℕ, satisfies_equation Θ ∧ Θ = 4 :=
sorry

end unique_solution_l3272_327215


namespace smallest_value_complex_sum_l3272_327267

theorem smallest_value_complex_sum (x y z : ℕ) (θ : ℂ) 
  (hxyz : x < y ∧ y < z)
  (hθ4 : θ^4 = 1)
  (hθ_neq_1 : θ ≠ 1) :
  ∃ (w : ℕ), w > 0 ∧ ∀ (a b c : ℕ) (ϕ : ℂ),
    a < b ∧ b < c → ϕ^4 = 1 → ϕ ≠ 1 →
    Complex.abs (↑x + ↑y * θ + ↑z * θ^3) ≤ Complex.abs (↑a + ↑b * ϕ + ↑c * ϕ^3) ∧
    Complex.abs (↑x + ↑y * θ + ↑z * θ^3) = Real.sqrt (↑w) :=
sorry

end smallest_value_complex_sum_l3272_327267


namespace original_triangle_area_l3272_327211

/-- Given a triangle whose dimensions are quadrupled to form a new triangle with area 144 square feet,
    prove that the area of the original triangle is 9 square feet. -/
theorem original_triangle_area (original_area new_area : ℝ) : 
  (∀ side, new_area = (4 * side)^2 / 2 * (original_area / (side^2 / 2))) → 
  new_area = 144 → 
  original_area = 9 := by
  sorry

end original_triangle_area_l3272_327211


namespace martha_turtles_l3272_327255

theorem martha_turtles (martha_turtles : ℕ) (marion_turtles : ℕ) : 
  marion_turtles = martha_turtles + 20 →
  martha_turtles + marion_turtles = 100 →
  martha_turtles = 40 := by
sorry

end martha_turtles_l3272_327255


namespace complex_equation_solution_l3272_327264

theorem complex_equation_solution (a : ℝ) : 
  (1 : ℂ) + a * I = I * (2 - I) → a = 2 := by
  sorry

end complex_equation_solution_l3272_327264


namespace third_circle_properties_l3272_327275

/-- Given two concentric circles with radii 10 and 20 units, prove that a third circle
    with area equal to the shaded area between the two concentric circles has a radius
    of 10√3 and a circumference of 20√3π. -/
theorem third_circle_properties (r₁ r₂ r₃ : ℝ) (h₁ : r₁ = 10) (h₂ : r₂ = 20)
    (h₃ : π * r₃^2 = π * r₂^2 - π * r₁^2) :
  r₃ = 10 * Real.sqrt 3 ∧ 2 * π * r₃ = 20 * Real.sqrt 3 * π := by
  sorry

#check third_circle_properties

end third_circle_properties_l3272_327275


namespace count_99_is_stone_10_l3272_327280

/-- Represents the number of stones in the circle -/
def num_stones : ℕ := 14

/-- Represents the length of a full counting cycle -/
def cycle_length : ℕ := 2 * num_stones - 1

/-- Maps a count to its corresponding stone number -/
def count_to_stone (count : ℕ) : ℕ :=
  let adjusted_count := (count - 1) % cycle_length + 1
  if adjusted_count ≤ num_stones
  then adjusted_count
  else 2 * num_stones - adjusted_count

/-- Theorem stating that the 99th count corresponds to the 10th stone -/
theorem count_99_is_stone_10 : count_to_stone 99 = 10 := by
  sorry

end count_99_is_stone_10_l3272_327280


namespace geometric_series_first_term_l3272_327263

theorem geometric_series_first_term 
  (a r : ℝ) 
  (sum_condition : a / (1 - r) = 12)
  (sum_squares_condition : a^2 / (1 - r^2) = 54) :
  a = 72 / 11 := by
  sorry

end geometric_series_first_term_l3272_327263


namespace vector_parallel_k_l3272_327298

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (2, -3)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), v.1 * w.2 = t * v.2 * w.1

theorem vector_parallel_k (k : ℝ) :
  parallel ((k * a.1 - b.1, k * a.2 - b.2) : ℝ × ℝ) (a.1 + 3 * b.1, a.2 + 3 * b.2) →
  k = -1/3 :=
sorry

end vector_parallel_k_l3272_327298


namespace smallest_height_of_special_triangle_l3272_327239

/-- Given a scalene triangle with integer side lengths a, b, c satisfying 
    the relation (a^2/c) - (a-c)^2 = (b^2/c) - (b-c)^2, 
    the smallest height of the triangle is 12/5. -/
theorem smallest_height_of_special_triangle (a b c : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hscalene : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hrelation : (a^2 : ℚ)/c - (a-c)^2 = (b^2 : ℚ)/c - (b-c)^2) :
  ∃ h : ℚ, h = 12/5 ∧ h = min (2 * (a * b) / (2 * a)) (min (2 * (b * c) / (2 * b)) (2 * (a * c) / (2 * c))) :=
sorry

end smallest_height_of_special_triangle_l3272_327239


namespace geometric_series_relation_l3272_327209

/-- Given real numbers c and d satisfying an infinite geometric series condition,
    prove that another related infinite geometric series equals 5/7. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c/d) / (1 - 1/d) = 5) : 
    (c/(c+2*d)) / (1 - 1/(c+2*d)) = 5/7 := by
  sorry

end geometric_series_relation_l3272_327209


namespace quadratic_function_properties_l3272_327204

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 1

-- Define the function g
def g (a b k x : ℝ) : ℝ := f a b x - k * x

-- State the theorem
theorem quadratic_function_properties (a b : ℝ) (ha : a ≠ 0) :
  (f a b (-1) = 0) →
  (∀ x : ℝ, f a b x ≥ 0) →
  (a = 1 ∧ b = 2) ∧
  (∀ k : ℝ, (∀ x ∈ Set.Icc (-2) 2, Monotone (g a b k)) ↔ k ≤ -2 ∨ k ≥ 6) :=
by sorry

end quadratic_function_properties_l3272_327204


namespace trapezoid_cd_length_l3272_327206

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  /-- Length of side BD -/
  bd : ℝ
  /-- Angle DBA in radians -/
  angle_dba : ℝ
  /-- Angle BDC in radians -/
  angle_bdc : ℝ
  /-- Ratio of BC to AD -/
  ratio_bc_ad : ℝ
  /-- AD is parallel to BC -/
  ad_parallel_bc : True
  /-- BD equals 3 -/
  bd_eq_three : bd = 3
  /-- Angle DBA equals 30 degrees (π/6 radians) -/
  angle_dba_eq_thirty_deg : angle_dba = Real.pi / 6
  /-- Angle BDC equals 60 degrees (π/3 radians) -/
  angle_bdc_eq_sixty_deg : angle_bdc = Real.pi / 3
  /-- Ratio of BC to AD is 7:4 -/
  ratio_bc_ad_eq_seven_four : ratio_bc_ad = 7 / 4

/-- Theorem: In the given trapezoid, CD equals 9/4 -/
theorem trapezoid_cd_length (t : Trapezoid) : ∃ cd : ℝ, cd = 9 / 4 := by
  sorry

end trapezoid_cd_length_l3272_327206


namespace negation_of_forall_positive_power_of_two_l3272_327245

theorem negation_of_forall_positive_power_of_two (P : ℝ → Prop) :
  (¬ ∀ x > 0, 2^x > 0) ↔ (∃ x > 0, 2^x ≤ 0) :=
by sorry

end negation_of_forall_positive_power_of_two_l3272_327245


namespace council_arrangements_l3272_327285

/-- The number of distinct arrangements of chairs and stools around a round table -/
def distinct_arrangements (chairs : ℕ) (stools : ℕ) : ℕ :=
  Nat.choose (chairs + stools - 1) (stools - 1)

/-- Theorem: There are 55 distinct arrangements of 9 chairs and 3 stools around a round table -/
theorem council_arrangements :
  distinct_arrangements 9 3 = 55 := by
  sorry

end council_arrangements_l3272_327285


namespace equal_roots_quadratic_l3272_327256

theorem equal_roots_quadratic (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x - 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y - 2 * y + 12 = 0 → y = x) ↔ 
  (k = 10 ∨ k = -14) := by
sorry

end equal_roots_quadratic_l3272_327256


namespace special_pyramid_volume_l3272_327287

/-- Represents a pyramid with an equilateral triangle base and isosceles right triangle lateral faces -/
structure SpecialPyramid where
  base_side_length : ℝ
  is_equilateral_base : base_side_length > 0
  is_isosceles_right_lateral : True

/-- Calculates the volume of the special pyramid -/
def volume (p : SpecialPyramid) : ℝ :=
  sorry

/-- Theorem stating that the volume of the special pyramid with base side length 2 is √2/3 -/
theorem special_pyramid_volume :
  ∀ (p : SpecialPyramid), p.base_side_length = 2 → volume p = Real.sqrt 2 / 3 :=
  sorry

end special_pyramid_volume_l3272_327287


namespace simplify_expression_l3272_327241

theorem simplify_expression (a b c : ℝ) 
  (h : Real.sqrt (a - 5) + (b - 3)^2 = Real.sqrt (c - 4) + Real.sqrt (4 - c)) :
  Real.sqrt c / (Real.sqrt a - Real.sqrt b) = Real.sqrt 5 + Real.sqrt 3 := by
  sorry

end simplify_expression_l3272_327241


namespace polygon_sides_from_interior_angle_l3272_327218

theorem polygon_sides_from_interior_angle (interior_angle : ℝ) (n : ℕ) :
  interior_angle = 108 →
  (n : ℝ) * (180 - interior_angle) = 360 →
  n = 5 := by
  sorry

end polygon_sides_from_interior_angle_l3272_327218


namespace triangle_theorem_l3272_327297

-- Define a triangle with sides a, b, c opposite to angles A, B, C
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given equation
def satisfies_equation (t : Triangle) : Prop :=
  (t.b + t.c) / (2 * t.a * t.b * t.c) + (Real.cos t.B + Real.cos t.C - 2) / (t.b^2 + t.c^2 - t.a^2) = 0

-- Define the arithmetic sequence property
def is_arithmetic_sequence (t : Triangle) : Prop :=
  t.b + t.c = 2 * t.a

-- Define the additional conditions
def has_specific_area_and_cosA (t : Triangle) : Prop :=
  t.a * t.b * Real.sin t.C / 2 = 15 * Real.sqrt 7 / 4 ∧ Real.cos t.A = 9/16

-- State the theorem
theorem triangle_theorem (t : Triangle) :
  satisfies_equation t →
  is_arithmetic_sequence t ∧
  (has_specific_area_and_cosA t → t.a = 5 * Real.sqrt 6 / 6) :=
by sorry

end triangle_theorem_l3272_327297


namespace distinct_polygons_count_l3272_327271

/-- The number of points marked on the circle -/
def n : ℕ := 12

/-- The total number of possible subsets of n points -/
def total_subsets : ℕ := 2^n

/-- The number of subsets that cannot form polygons (0, 1, or 2 points) -/
def non_polygon_subsets : ℕ := (n.choose 0) + (n.choose 1) + (n.choose 2)

/-- The number of distinct convex polygons with three or more sides -/
def num_polygons : ℕ := total_subsets - non_polygon_subsets

theorem distinct_polygons_count :
  num_polygons = 4017 :=
by sorry

end distinct_polygons_count_l3272_327271


namespace prime_factors_count_four_equals_two_squared_seven_is_prime_eleven_is_prime_l3272_327235

/-- The total number of prime factors in the expression (4)^11 × (7)^5 × (11)^2 -/
def totalPrimeFactors : ℕ := 29

/-- The exponent of 4 in the expression -/
def exponent4 : ℕ := 11

/-- The exponent of 7 in the expression -/
def exponent7 : ℕ := 5

/-- The exponent of 11 in the expression -/
def exponent11 : ℕ := 2

theorem prime_factors_count :
  totalPrimeFactors = 2 * exponent4 + exponent7 + exponent11 := by
  sorry

/-- 4 is equal to 2^2 -/
theorem four_equals_two_squared : (4 : ℕ) = 2^2 := by
  sorry

/-- 7 is a prime number -/
theorem seven_is_prime : Nat.Prime 7 := by
  sorry

/-- 11 is a prime number -/
theorem eleven_is_prime : Nat.Prime 11 := by
  sorry

end prime_factors_count_four_equals_two_squared_seven_is_prime_eleven_is_prime_l3272_327235


namespace animal_food_cost_l3272_327231

/-- The total weekly cost for both animals' food -/
def total_weekly_cost : ℕ := 30

/-- The weekly cost of rabbit food -/
def rabbit_weekly_cost : ℕ := 12

/-- The number of weeks Julia has had the rabbit -/
def rabbit_weeks : ℕ := 5

/-- The number of weeks Julia has had the parrot -/
def parrot_weeks : ℕ := 3

/-- The total amount Julia has spent on animal food -/
def total_spent : ℕ := 114

theorem animal_food_cost :
  total_weekly_cost = rabbit_weekly_cost + (total_spent - rabbit_weekly_cost * rabbit_weeks) / parrot_weeks :=
by sorry

end animal_food_cost_l3272_327231


namespace triangle_sum_equals_58_l3272_327268

/-- The triangle operation that takes three numbers and returns the sum of their squares -/
def triangle (a b c : ℝ) : ℝ := a^2 + b^2 + c^2

/-- Theorem stating that the sum of triangle(2,3,6) and triangle(1,2,2) equals 58 -/
theorem triangle_sum_equals_58 : triangle 2 3 6 + triangle 1 2 2 = 58 := by
  sorry

end triangle_sum_equals_58_l3272_327268


namespace ratio_is_one_to_two_l3272_327232

/-- Represents a co-ed softball team -/
structure CoedSoftballTeam where
  men : ℕ
  women : ℕ
  total_players : ℕ
  women_more_than_men : women = men + 5
  total_is_sum : total_players = men + women

/-- The ratio of men to women in a co-ed softball team -/
def ratio_men_to_women (team : CoedSoftballTeam) : ℚ × ℚ :=
  (team.men, team.women)

theorem ratio_is_one_to_two (team : CoedSoftballTeam) 
    (h : team.total_players = 15) : 
    ratio_men_to_women team = (1, 2) := by
  sorry

#check ratio_is_one_to_two

end ratio_is_one_to_two_l3272_327232


namespace unique_integer_factorial_division_l3272_327229

theorem unique_integer_factorial_division : 
  ∃! n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (∃ k : ℕ, k * (Nat.factorial n)^(n + 2) = Nat.factorial (n^2)) :=
by sorry

end unique_integer_factorial_division_l3272_327229


namespace square_land_equation_l3272_327246

theorem square_land_equation (a p : ℝ) (h1 : p = 36) : 
  (∃ s : ℝ, s > 0 ∧ a = s^2 ∧ p = 4*s) → 
  (5*a = 10*p + 45) := by
sorry

end square_land_equation_l3272_327246


namespace network_paths_count_l3272_327272

-- Define the network structure
structure Network where
  hasDirectPath : (Char × Char) → Prop
  
-- Define the number of paths between two points
def numPaths (net : Network) (start finish : Char) : ℕ := sorry

-- Theorem statement
theorem network_paths_count (net : Network) :
  (net.hasDirectPath ('E', 'B')) →
  (net.hasDirectPath ('F', 'B')) →
  (net.hasDirectPath ('F', 'A')) →
  (net.hasDirectPath ('M', 'A')) →
  (net.hasDirectPath ('M', 'B')) →
  (net.hasDirectPath ('M', 'E')) →
  (net.hasDirectPath ('M', 'F')) →
  (net.hasDirectPath ('A', 'C')) →
  (net.hasDirectPath ('A', 'D')) →
  (net.hasDirectPath ('B', 'A')) →
  (net.hasDirectPath ('B', 'C')) →
  (net.hasDirectPath ('B', 'N')) →
  (net.hasDirectPath ('C', 'N')) →
  (net.hasDirectPath ('D', 'N')) →
  (numPaths net 'M' 'N' = 16) :=
by sorry

end network_paths_count_l3272_327272


namespace ellipse_and_outer_point_properties_l3272_327208

/-- Definition of an ellipse C with given properties -/
structure Ellipse :=
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a^2 - b^2 = 5)
  (h4 : a = 3)

/-- Definition of a point P outside the ellipse -/
structure OuterPoint (C : Ellipse) :=
  (x₀ y₀ : ℝ)
  (h5 : x₀^2 / C.a^2 + y₀^2 / C.b^2 > 1)

/-- Theorem stating the properties of the ellipse and outer point -/
theorem ellipse_and_outer_point_properties (C : Ellipse) (P : OuterPoint C) :
  (∀ x y, x^2 / 9 + y^2 / 4 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (P.x₀^2 + P.y₀^2 = 13) :=
sorry

end ellipse_and_outer_point_properties_l3272_327208


namespace problem_solution_l3272_327203

theorem problem_solution (p q r u v w : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hu : u > 0) (hv : v > 0) (hw : w > 0)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p*u + q*v + r*w = 56) :
  (p + q + r) / (u + v + w) = 7/8 := by
sorry

end problem_solution_l3272_327203


namespace tangent_line_to_parabola_l3272_327244

/-- The value of d for which the line y = 3x + d is tangent to the parabola y² = 12x -/
theorem tangent_line_to_parabola : ∃! d : ℝ,
  ∀ x y : ℝ, (y = 3 * x + d ∧ y^2 = 12 * x) →
  (∃! x₀ y₀ : ℝ, y₀ = 3 * x₀ + d ∧ y₀^2 = 12 * x₀) :=
by sorry

end tangent_line_to_parabola_l3272_327244


namespace chess_group_size_l3272_327250

-- Define the number of players in the chess group
def num_players : ℕ := 30

-- Define the total number of games played
def total_games : ℕ := 435

-- Theorem stating that the number of players is correct given the conditions
theorem chess_group_size :
  (num_players.choose 2 = total_games) ∧ (num_players > 0) := by
  sorry

end chess_group_size_l3272_327250


namespace line_vector_proof_l3272_327277

def line_vector (t : ℝ) : ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 1 = (2, 5) ∧ line_vector 4 = (5, -7)) →
  line_vector (-3) = (-2, 21) := by sorry

end line_vector_proof_l3272_327277
