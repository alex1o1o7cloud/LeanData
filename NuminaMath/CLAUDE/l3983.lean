import Mathlib

namespace min_value_tan_sum_l3983_398333

/-- For any acute-angled triangle ABC, the expression 
    3 tan B tan C + 2 tan A tan C + tan A tan B 
    is always greater than or equal to 6 + 2√3 + 2√2 + 2√6 -/
theorem min_value_tan_sum (A B C : ℝ) (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
    (h_triangle : A + B + C = π) : 
  3 * Real.tan B * Real.tan C + 2 * Real.tan A * Real.tan C + Real.tan A * Real.tan B 
    ≥ 6 + 2 * Real.sqrt 3 + 2 * Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end min_value_tan_sum_l3983_398333


namespace justina_tallest_l3983_398337

-- Define a type for people
inductive Person : Type
  | Gisa : Person
  | Henry : Person
  | Ivan : Person
  | Justina : Person
  | Katie : Person

-- Define a height function
variable (height : Person → ℝ)

-- Define the conditions
axiom gisa_taller_than_henry : height Person.Gisa > height Person.Henry
axiom gisa_shorter_than_justina : height Person.Gisa < height Person.Justina
axiom ivan_taller_than_katie : height Person.Ivan > height Person.Katie
axiom ivan_shorter_than_gisa : height Person.Ivan < height Person.Gisa

-- Theorem to prove
theorem justina_tallest : 
  ∀ p : Person, height Person.Justina ≥ height p :=
sorry

end justina_tallest_l3983_398337


namespace age_difference_l3983_398311

theorem age_difference (albert_age mary_age betty_age : ℕ) : 
  albert_age = 2 * mary_age →
  albert_age = 4 * betty_age →
  betty_age = 4 →
  albert_age - mary_age = 8 := by
  sorry

end age_difference_l3983_398311


namespace ladder_slide_l3983_398366

theorem ladder_slide (ladder_length : ℝ) (initial_distance : ℝ) (slip_distance : ℝ) :
  ladder_length = 30 →
  initial_distance = 8 →
  slip_distance = 6 →
  let initial_height : ℝ := Real.sqrt (ladder_length^2 - initial_distance^2)
  let new_height : ℝ := initial_height - slip_distance
  let new_distance : ℝ := Real.sqrt (ladder_length^2 - new_height^2)
  new_distance - initial_distance = 10 := by
  sorry

end ladder_slide_l3983_398366


namespace trains_passing_time_l3983_398396

theorem trains_passing_time (train_length : ℝ) (train_speed : ℝ) : 
  train_length = 500 →
  train_speed = 30 →
  (2 * train_length) / (2 * train_speed * (5/18)) = 60 :=
by
  sorry

#check trains_passing_time

end trains_passing_time_l3983_398396


namespace holmium166_neutron_proton_difference_l3983_398347

/-- Properties of Holmium-166 isotope -/
structure Holmium166 where
  mass_number : ℕ
  proton_number : ℕ
  mass_number_eq : mass_number = 166
  proton_number_eq : proton_number = 67

/-- Theorem: The difference between neutrons and protons in Holmium-166 is 32 -/
theorem holmium166_neutron_proton_difference (ho : Holmium166) :
  ho.mass_number - ho.proton_number - ho.proton_number = 32 := by
  sorry

#check holmium166_neutron_proton_difference

end holmium166_neutron_proton_difference_l3983_398347


namespace cube_root_minus_square_root_plus_power_l3983_398306

theorem cube_root_minus_square_root_plus_power : 
  ((-2 : ℝ)^3)^(1/3) - Real.sqrt 4 + (Real.sqrt 3)^0 = -3 := by sorry

end cube_root_minus_square_root_plus_power_l3983_398306


namespace expression_is_equation_l3983_398340

/-- Definition of an equation -/
def is_equation (e : Prop) : Prop :=
  ∃ (x : ℝ) (f g : ℝ → ℝ), e = (f x = g x)

/-- The expression 2x - 1 = 3 -/
def expression : Prop :=
  ∃ x : ℝ, 2 * x - 1 = 3

/-- Theorem: The given expression is an equation -/
theorem expression_is_equation : is_equation expression :=
sorry

end expression_is_equation_l3983_398340


namespace sphere_sum_l3983_398315

theorem sphere_sum (x y z : ℝ) : 
  x^2 + y^2 + z^2 - 2*x + 4*y - 6*z + 14 = 0 → x + y + z = 2 := by
  sorry

end sphere_sum_l3983_398315


namespace union_of_A_and_B_is_reals_l3983_398303

-- Define sets A and B
def A : Set ℝ := {x | 4 * x - 3 > 0}
def B : Set ℝ := {x | x - 6 < 0}

-- State the theorem
theorem union_of_A_and_B_is_reals : A ∪ B = Set.univ := by sorry

end union_of_A_and_B_is_reals_l3983_398303


namespace lexus_cars_sold_l3983_398364

def total_cars : ℕ := 300

def audi_percent : ℚ := 10 / 100
def toyota_percent : ℚ := 25 / 100
def bmw_percent : ℚ := 15 / 100
def acura_percent : ℚ := 30 / 100

def other_brands_percent : ℚ := audi_percent + toyota_percent + bmw_percent + acura_percent

def lexus_percent : ℚ := 1 - other_brands_percent

theorem lexus_cars_sold : 
  ⌊(lexus_percent * total_cars : ℚ)⌋ = 60 := by
  sorry

end lexus_cars_sold_l3983_398364


namespace percentage_problem_l3983_398362

theorem percentage_problem (x : ℝ) : (27 / x = 45 / 100) → x = 60 := by
  sorry

end percentage_problem_l3983_398362


namespace total_pages_is_281_l3983_398360

/-- Calculates the total number of pages read over two months given Janine's reading habits --/
def total_pages_read : ℕ :=
  let last_month_books := 3 + 2
  let last_month_pages := 3 * 12 + 2 * 15
  let this_month_books := 2 * last_month_books
  let this_month_pages := 1 * 20 + 4 * 25 + 2 * 30 + 1 * 35
  last_month_pages + this_month_pages

/-- Proves that the total number of pages read over two months is 281 --/
theorem total_pages_is_281 : total_pages_read = 281 := by
  sorry

end total_pages_is_281_l3983_398360


namespace ronaldo_age_l3983_398370

theorem ronaldo_age (ronnie_age_last_year : ℕ) (ronaldo_age_last_year : ℕ) 
  (h1 : ronnie_age_last_year * 7 = ronaldo_age_last_year * 6)
  (h2 : (ronnie_age_last_year + 5) * 8 = (ronaldo_age_last_year + 5) * 7) :
  ronaldo_age_last_year + 1 = 36 := by
  sorry

end ronaldo_age_l3983_398370


namespace simplify_fraction_l3983_398319

theorem simplify_fraction (a : ℝ) : 
  (1 + a^2 / (1 + 2*a)) / ((1 + a) / (1 + 2*a)) = 1 + a :=
by sorry

end simplify_fraction_l3983_398319


namespace perpendicular_unit_vector_l3983_398309

theorem perpendicular_unit_vector (a : ℝ × ℝ) (v : ℝ × ℝ) : 
  a = (1, 1) → 
  v = (-Real.sqrt 2 / 2, Real.sqrt 2 / 2) → 
  (a.1 * v.1 + a.2 * v.2 = 0) ∧ 
  (v.1^2 + v.2^2 = 1) := by
  sorry

end perpendicular_unit_vector_l3983_398309


namespace point_distance_range_l3983_398344

/-- Given points A(0,1) and B(0,4), and a point P on the line 2x-y+m=0 such that |PA| = 1/2|PB|,
    the range of values for m is -2√5 ≤ m ≤ 2√5. -/
theorem point_distance_range (m : ℝ) : 
  (∃ (x y : ℝ), 2*x - y + m = 0 ∧ 
    (x^2 + (y-1)^2)^(1/2) = 1/2 * (x^2 + (y-4)^2)^(1/2)) 
  ↔ -2 * Real.sqrt 5 ≤ m ∧ m ≤ 2 * Real.sqrt 5 :=
by sorry

end point_distance_range_l3983_398344


namespace sqrt_2_power_n_equals_64_l3983_398349

theorem sqrt_2_power_n_equals_64 (n : ℕ) : Real.sqrt (2^n) = 64 → n = 12 := by
  sorry

end sqrt_2_power_n_equals_64_l3983_398349


namespace gummy_bear_spending_percentage_l3983_398328

-- Define the given constants
def hourly_rate : ℚ := 12.5
def hours_worked : ℕ := 40
def tax_rate : ℚ := 0.2
def remaining_money : ℚ := 340

-- Define the function to calculate the percentage spent on gummy bears
def gummy_bear_percentage (rate : ℚ) (hours : ℕ) (tax : ℚ) (remaining : ℚ) : ℚ :=
  let gross_pay := rate * hours
  let net_pay := gross_pay * (1 - tax)
  let spent_on_gummy_bears := net_pay - remaining
  (spent_on_gummy_bears / net_pay) * 100

-- Theorem statement
theorem gummy_bear_spending_percentage :
  gummy_bear_percentage hourly_rate hours_worked tax_rate remaining_money = 15 :=
sorry

end gummy_bear_spending_percentage_l3983_398328


namespace min_value_theorem_l3983_398304

theorem min_value_theorem (x y : ℝ) (hx : x > 1) (hy : y > 1) 
  (h : x * y - 2 * x - y + 1 = 0) : 
  ∀ z w : ℝ, z > 1 → w > 1 → z * w - 2 * z - w + 1 = 0 → 
  (3 / 2) * x^2 + y^2 ≤ (3 / 2) * z^2 + w^2 := by
sorry

end min_value_theorem_l3983_398304


namespace mitzi_bowling_score_l3983_398316

/-- Proves that given three bowlers with an average score of 106, where one bowler scores 120 
    and another scores 85, the third bowler's score must be 113. -/
theorem mitzi_bowling_score (average_score gretchen_score beth_score : ℕ) 
    (h1 : average_score = 106)
    (h2 : gretchen_score = 120)
    (h3 : beth_score = 85) : 
  ∃ mitzi_score : ℕ, mitzi_score = 113 ∧ 
    (gretchen_score + beth_score + mitzi_score) / 3 = average_score :=
by
  sorry


end mitzi_bowling_score_l3983_398316


namespace identical_pairs_imply_x_equals_four_l3983_398342

-- Define the binary operation ★
def star (a b c d : ℤ) : ℤ × ℤ := (a - 2*c, b + 2*d)

-- Theorem statement
theorem identical_pairs_imply_x_equals_four :
  ∀ x y : ℤ, star 2 (-4) 1 (-3) = star x y 2 1 → x = 4 := by
  sorry

end identical_pairs_imply_x_equals_four_l3983_398342


namespace inequality_solution_l3983_398388

theorem inequality_solution (x : ℝ) : 
  3 - 1 / (3 * x + 4) < 5 ↔ x < -7/6 ∨ x > -4/3 :=
sorry

end inequality_solution_l3983_398388


namespace sand_amount_l3983_398391

/-- The amount of gravel bought by the company in tons -/
def gravel : ℝ := 5.91

/-- The total amount of material bought by the company in tons -/
def total_material : ℝ := 14.02

/-- The amount of sand bought by the company in tons -/
def sand : ℝ := total_material - gravel

theorem sand_amount : sand = 8.11 := by
  sorry

end sand_amount_l3983_398391


namespace quadratic_inequality_empty_set_l3983_398380

theorem quadratic_inequality_empty_set (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - a * x + 1 ≥ 0) ↔ 0 ≤ a ∧ a ≤ 4 := by sorry

end quadratic_inequality_empty_set_l3983_398380


namespace new_barbell_cost_l3983_398351

/-- The cost of a new barbell that is 30% more expensive than an old barbell priced at $250 is $325. -/
theorem new_barbell_cost (old_price : ℝ) (new_price : ℝ) : 
  old_price = 250 →
  new_price = old_price * 1.3 →
  new_price = 325 := by
  sorry

end new_barbell_cost_l3983_398351


namespace ice_cream_parlor_distance_l3983_398352

/-- The distance to the ice cream parlor satisfies the equation relating to Rita's canoe trip --/
theorem ice_cream_parlor_distance :
  ∃ D : ℝ, (D / (3 - 2)) + (D / (9 + 4)) = 8 - 0.25 := by
  sorry

end ice_cream_parlor_distance_l3983_398352


namespace geometric_arithmetic_sequence_l3983_398356

theorem geometric_arithmetic_sequence :
  ∃ (a b c : ℝ),
    a > 0 ∧ b > 0 ∧ c > 0 ∧  -- Positive numbers
    a = 2 ∧  -- First number is 2
    b / a = c / b ∧  -- Geometric sequence
    (b + 4 - a = c - (b + 4)) ∧  -- Arithmetic sequence when 4 is added to b
    a = 2 ∧ b = 6 ∧ c = 18 :=  -- The solution
by
  sorry


end geometric_arithmetic_sequence_l3983_398356


namespace identify_geometric_bodies_l3983_398367

/-- Represents the possible geometric bodies --/
inductive GeometricBody
  | TriangularPrism
  | QuadrangularPyramid
  | Cone
  | Frustum
  | TriangularFrustum
  | TriangularPyramid

/-- Represents a view of a geometric body --/
structure View where
  -- We'll assume some properties of the view, but won't define them explicitly
  dummy : Unit

/-- A function that determines if a set of views corresponds to a specific geometric body --/
def viewsMatchBody (views : List View) (body : GeometricBody) : Bool :=
  sorry -- The actual implementation would depend on how we define views

/-- The theorem stating that given the correct views, we can identify the four bodies --/
theorem identify_geometric_bodies 
  (views1 views2 views3 views4 : List View) 
  (h1 : viewsMatchBody views1 GeometricBody.TriangularPrism)
  (h2 : viewsMatchBody views2 GeometricBody.QuadrangularPyramid)
  (h3 : viewsMatchBody views3 GeometricBody.Cone)
  (h4 : viewsMatchBody views4 GeometricBody.Frustum) :
  ∃ (bodies : List GeometricBody), 
    bodies = [GeometricBody.TriangularPrism, 
              GeometricBody.QuadrangularPyramid, 
              GeometricBody.Cone, 
              GeometricBody.Frustum] ∧
    (∀ (views : List View), 
      views ∈ [views1, views2, views3, views4] → 
      ∃ (body : GeometricBody), body ∈ bodies ∧ viewsMatchBody views body) :=
by
  sorry


end identify_geometric_bodies_l3983_398367


namespace snack_packs_needed_l3983_398379

def trail_mix_pack_size : ℕ := 6
def granola_bar_pack_size : ℕ := 8
def fruit_cup_pack_size : ℕ := 4
def total_people : ℕ := 18

def min_packs_needed (pack_size : ℕ) (people : ℕ) : ℕ :=
  (people + pack_size - 1) / pack_size

theorem snack_packs_needed :
  (min_packs_needed trail_mix_pack_size total_people = 3) ∧
  (min_packs_needed granola_bar_pack_size total_people = 3) ∧
  (min_packs_needed fruit_cup_pack_size total_people = 5) :=
by sorry

end snack_packs_needed_l3983_398379


namespace jessie_weight_loss_l3983_398318

/-- Calculates the weight loss given initial and current weights -/
def weight_loss (initial_weight current_weight : ℕ) : ℕ :=
  initial_weight - current_weight

/-- Proves that Jessie's weight loss is 7 kilograms -/
theorem jessie_weight_loss : weight_loss 74 67 = 7 := by
  sorry

end jessie_weight_loss_l3983_398318


namespace beaus_sons_correct_number_of_sons_l3983_398386

theorem beaus_sons (sons_age_today : ℕ) (beaus_age_today : ℕ) : ℕ :=
  let sons_age_three_years_ago := sons_age_today - 3
  let beaus_age_three_years_ago := beaus_age_today - 3
  let num_sons := beaus_age_three_years_ago / sons_age_three_years_ago
  num_sons

theorem correct_number_of_sons : beaus_sons 16 42 = 3 := by
  sorry

end beaus_sons_correct_number_of_sons_l3983_398386


namespace fathers_catch_l3983_398301

/-- The number of fishes Hazel caught -/
def hazel_catch : ℕ := 48

/-- The total number of fishes caught by Hazel and her father -/
def total_catch : ℕ := 94

/-- Hazel's father's catch is the difference between the total catch and Hazel's catch -/
theorem fathers_catch (hazel_catch : ℕ) (total_catch : ℕ) : 
  total_catch - hazel_catch = 46 :=
by sorry

end fathers_catch_l3983_398301


namespace right_triangle_validity_and_area_l3983_398338

theorem right_triangle_validity_and_area :
  ∀ (a b c : ℝ),
  a = 5 ∧ c = 13 ∧ a < b ∧ b < c →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 30 :=
sorry

end right_triangle_validity_and_area_l3983_398338


namespace hexagon_to_rhombus_l3983_398326

/-- A regular hexagon -/
structure RegularHexagon where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A rhombus -/
structure Rhombus where
  side_length : ℝ
  side_length_pos : side_length > 0

/-- A part of the hexagon after cutting -/
structure HexagonPart where
  area : ℝ
  area_pos : area > 0

/-- Function to cut a regular hexagon into three parts -/
def cut_hexagon (h : RegularHexagon) : (HexagonPart × HexagonPart × HexagonPart) :=
  sorry

/-- Function to form a rhombus from three hexagon parts -/
def form_rhombus (p1 p2 p3 : HexagonPart) : Rhombus :=
  sorry

/-- Theorem stating that a regular hexagon can be cut into three parts that form a rhombus -/
theorem hexagon_to_rhombus (h : RegularHexagon) :
  ∃ (p1 p2 p3 : HexagonPart), 
    let (p1', p2', p3') := cut_hexagon h
    p1 = p1' ∧ p2 = p2' ∧ p3 = p3' ∧
    ∃ (r : Rhombus), r = form_rhombus p1 p2 p3 :=
  sorry

end hexagon_to_rhombus_l3983_398326


namespace banknote_probability_l3983_398327

/-- Represents a bag of banknotes -/
structure Bag :=
  (ten : ℕ)    -- Number of ten-yuan banknotes
  (five : ℕ)   -- Number of five-yuan banknotes
  (one : ℕ)    -- Number of one-yuan banknotes

/-- Calculate the total value of banknotes in a bag -/
def bagValue (b : Bag) : ℕ :=
  10 * b.ten + 5 * b.five + b.one

/-- Calculate the number of ways to choose 2 items from n items -/
def choose2 (n : ℕ) : ℕ :=
  n * (n - 1) / 2

/-- The probability of drawing at least one 5-yuan note from bag B -/
def probAtLeastOne5 (b : Bag) : ℚ :=
  (choose2 b.five.succ + b.five * b.one) / choose2 (b.five + b.one)

theorem banknote_probability :
  let bagA : Bag := ⟨2, 0, 3⟩
  let bagB : Bag := ⟨0, 4, 3⟩
  let totalDraws := choose2 (bagValue bagA) * choose2 (bagValue bagB)
  let favorableDraws := choose2 bagA.one * (choose2 bagB.five.succ + bagB.five * bagB.one)
  (favorableDraws : ℚ) / totalDraws = 9 / 35 := by
  sorry

end banknote_probability_l3983_398327


namespace angle_in_fourth_quadrant_l3983_398368

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : (Real.sin α) / (Real.tan α) > 0) 
  (h2 : (Real.tan α) / (Real.cos α) < 0) : 
  0 < α ∧ α < π / 2 ∧ Real.sin α < 0 ∧ Real.cos α > 0 :=
sorry

end angle_in_fourth_quadrant_l3983_398368


namespace value_of_a_l3983_398334

-- Define set A
def A : Set ℝ := {x | x^2 ≠ 1}

-- Define set B
def B (a : ℝ) : Set ℝ := {x | x = a}

-- Theorem statement
theorem value_of_a (a : ℝ) (h : B a ⊆ A) : a = 1 ∨ a = -1 := by
  sorry

end value_of_a_l3983_398334


namespace special_house_additional_profit_l3983_398336

/-- The additional profit made by building and selling a special house compared to a regular house -/
theorem special_house_additional_profit
  (C : ℝ)  -- Regular house construction cost
  (regular_selling_price : ℝ)
  (special_selling_price : ℝ)
  (h1 : regular_selling_price = 350000)
  (h2 : special_selling_price = 1.8 * regular_selling_price)
  : (special_selling_price - (C + 200000)) - (regular_selling_price - C) = 80000 := by
  sorry

#check special_house_additional_profit

end special_house_additional_profit_l3983_398336


namespace soccer_league_games_l3983_398358

theorem soccer_league_games (n : ℕ) (h : n = 12) : 
  (n * (n - 1)) / 2 = 66 := by
  sorry

#check soccer_league_games

end soccer_league_games_l3983_398358


namespace perfect_square_binomial_condition_l3983_398394

/-- A quadratic expression is a perfect square binomial if it can be written as (px + q)^2 for some real p and q -/
def IsPerfectSquareBinomial (f : ℝ → ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x, f x = (p * x + q)^2

/-- Given that 9x^2 - 27x + a is a perfect square binomial, prove that a = 20.25 -/
theorem perfect_square_binomial_condition (a : ℝ) 
  (h : IsPerfectSquareBinomial (fun x ↦ 9*x^2 - 27*x + a)) : 
  a = 20.25 := by
  sorry

end perfect_square_binomial_condition_l3983_398394


namespace don_buys_150_from_shop_A_l3983_398378

/-- The number of bottles Don buys from each shop -/
structure BottlePurchase where
  total : ℕ
  shopA : ℕ
  shopB : ℕ
  shopC : ℕ

/-- Don's bottle purchase satisfies the given conditions -/
def valid_purchase (p : BottlePurchase) : Prop :=
  p.total = 550 ∧ p.shopB = 180 ∧ p.shopC = 220 ∧ p.total = p.shopA + p.shopB + p.shopC

/-- Theorem: Don buys 150 bottles from Shop A -/
theorem don_buys_150_from_shop_A (p : BottlePurchase) (h : valid_purchase p) : p.shopA = 150 := by
  sorry

end don_buys_150_from_shop_A_l3983_398378


namespace optimal_washing_effect_l3983_398381

/-- Represents the laundry scenario with given parameters -/
structure LaundryScenario where
  tub_capacity : ℝ
  clothes_weight : ℝ
  initial_detergent_scoops : ℕ
  scoop_weight : ℝ
  optimal_ratio : ℝ

/-- Calculates the optimal amount of detergent and water to add -/
def optimal_addition (scenario : LaundryScenario) : ℝ × ℝ :=
  sorry

/-- Theorem stating that the calculated optimal addition achieves the desired washing effect -/
theorem optimal_washing_effect (scenario : LaundryScenario) 
  (h1 : scenario.tub_capacity = 20)
  (h2 : scenario.clothes_weight = 5)
  (h3 : scenario.initial_detergent_scoops = 2)
  (h4 : scenario.scoop_weight = 0.02)
  (h5 : scenario.optimal_ratio = 0.004) :
  let (added_detergent, added_water) := optimal_addition scenario
  (added_detergent = 0.02) ∧ 
  (added_water = 14.94) ∧
  (scenario.tub_capacity = scenario.clothes_weight + added_water + added_detergent + scenario.initial_detergent_scoops * scenario.scoop_weight) ∧
  (added_detergent + scenario.initial_detergent_scoops * scenario.scoop_weight = scenario.optimal_ratio * (added_water + scenario.initial_detergent_scoops * scenario.scoop_weight)) :=
by
  sorry


end optimal_washing_effect_l3983_398381


namespace cost_of_green_pill_l3983_398374

/-- Prove that the cost of one green pill is $20 -/
theorem cost_of_green_pill (treatment_duration : ℕ) (daily_green_pills : ℕ) (daily_pink_pills : ℕ) 
  (total_cost : ℕ) : ℕ :=
by
  sorry

#check cost_of_green_pill 3 1 1 819

end cost_of_green_pill_l3983_398374


namespace scientific_notation_of_1_5_million_l3983_398348

/-- Expresses 1.5 million in scientific notation -/
theorem scientific_notation_of_1_5_million :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 1500000 = a * (10 : ℝ) ^ n ∧ a = 1.5 ∧ n = 6 :=
by
  sorry

end scientific_notation_of_1_5_million_l3983_398348


namespace x_fourth_minus_inverse_fourth_l3983_398321

theorem x_fourth_minus_inverse_fourth (x : ℝ) (h : x - 1/x = 5) : x^4 - 1/x^4 = 527 := by
  sorry

end x_fourth_minus_inverse_fourth_l3983_398321


namespace rectangle_area_l3983_398330

/-- Given a rectangle with diagonal length x and length three times its width, 
    prove that its area is (3/10)x^2 -/
theorem rectangle_area (x : ℝ) (h : x > 0) : 
  ∃ w : ℝ, w > 0 ∧ 
    w^2 + (3*w)^2 = x^2 ∧ 
    3 * w^2 = (3/10) * x^2 := by
  sorry

end rectangle_area_l3983_398330


namespace power_three_1234_mod_5_l3983_398357

theorem power_three_1234_mod_5 : 3^1234 % 5 = 4 := by
  sorry

end power_three_1234_mod_5_l3983_398357


namespace bonus_pool_ratio_l3983_398399

theorem bonus_pool_ratio (P : ℕ) (k : ℕ) (h1 : P % 5 = 2) (h2 : (k * P) % 5 = 1) :
  k = 3 :=
sorry

end bonus_pool_ratio_l3983_398399


namespace median_is_106_l3983_398300

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The list where each integer n (1 ≤ n ≤ 150) appears n times -/
def special_list : List ℕ := sorry

/-- The length of the special list -/
def special_list_length : ℕ := sum_to_n 150

/-- The median index of the special list -/
def median_index : ℕ := special_list_length / 2 + 1

theorem median_is_106 : 
  ∃ (l : List ℕ), l = special_list ∧ l.length = special_list_length ∧ 
  (∀ n, 1 ≤ n ∧ n ≤ 150 → (l.count n = n)) ∧
  (l.nthLe (median_index - 1) sorry = 106) :=
sorry

end median_is_106_l3983_398300


namespace bus_arrival_probability_l3983_398365

-- Define the probability of the bus arriving on time
def p : ℚ := 3/5

-- Define the probability of the bus not arriving on time
def q : ℚ := 1 - p

-- Define the function to calculate the probability of exactly k successes in n trials
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k) * p^k * (1 - p)^(n - k)

-- State the theorem
theorem bus_arrival_probability :
  binomial_probability 3 2 p + binomial_probability 3 3 p = 81/125 := by
  sorry

end bus_arrival_probability_l3983_398365


namespace number_of_boys_l3983_398372

/-- The number of boys in a class, given the average weights and number of students. -/
theorem number_of_boys (avg_weight_boys : ℝ) (avg_weight_class : ℝ) (total_students : ℕ)
  (num_girls : ℕ) (avg_weight_girls : ℝ)
  (h1 : avg_weight_boys = 48)
  (h2 : avg_weight_class = 45)
  (h3 : total_students = 25)
  (h4 : num_girls = 15)
  (h5 : avg_weight_girls = 40.5) :
  total_students - num_girls = 10 := by
  sorry

#check number_of_boys

end number_of_boys_l3983_398372


namespace binomial_inequality_l3983_398320

theorem binomial_inequality (n : ℤ) (x : ℝ) (h : x > 0) : (1 + x)^n ≥ 1 + n * x := by
  sorry

end binomial_inequality_l3983_398320


namespace sixteen_seats_painting_ways_l3983_398343

def paintingWays (n : ℕ) : ℕ := 
  let rec a : ℕ → ℕ
    | 0 => 1
    | 1 => 1
    | i + 1 => (List.range ((i + 1) / 2 + 1)).foldl (λ sum j => sum + a (i - 2 * j)) 0
  2 * a n

theorem sixteen_seats_painting_ways :
  paintingWays 16 = 1686 := by sorry

end sixteen_seats_painting_ways_l3983_398343


namespace unique_interior_point_is_centroid_l3983_398387

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A triangle defined by three lattice points -/
structure LatticeTriangle where
  A : LatticePoint
  B : LatticePoint
  C : LatticePoint

/-- Predicate to check if a point is inside a triangle -/
def isInside (P : LatticePoint) (T : LatticeTriangle) : Prop :=
  sorry

/-- Predicate to check if a point is on the boundary of a triangle -/
def isOnBoundary (P : LatticePoint) (T : LatticeTriangle) : Prop :=
  sorry

/-- The centroid of a triangle -/
def centroid (T : LatticeTriangle) : LatticePoint :=
  sorry

/-- Main theorem -/
theorem unique_interior_point_is_centroid (T : LatticeTriangle) (P : LatticePoint) :
  (∀ Q : LatticePoint, isOnBoundary Q T → (Q = T.A ∨ Q = T.B ∨ Q = T.C)) →
  isInside P T →
  (∀ Q : LatticePoint, isInside Q T → Q = P) →
  P = centroid T :=
by sorry

end unique_interior_point_is_centroid_l3983_398387


namespace sum_of_fractions_l3983_398324

theorem sum_of_fractions : 
  (1 / 15 : ℚ) + (2 / 15 : ℚ) + (3 / 15 : ℚ) + (4 / 15 : ℚ) + 
  (5 / 15 : ℚ) + (6 / 15 : ℚ) + (7 / 15 : ℚ) + (8 / 15 : ℚ) + 
  (30 / 15 : ℚ) = 4 + (2 / 5 : ℚ) := by
sorry

end sum_of_fractions_l3983_398324


namespace at_least_two_fever_probability_l3983_398377

def vaccine_fever_prob : ℝ := 0.80

def num_people : ℕ := 3

def at_least_two_fever_prob : ℝ := 
  (Nat.choose num_people 2) * (vaccine_fever_prob ^ 2) * (1 - vaccine_fever_prob) +
  vaccine_fever_prob ^ num_people

theorem at_least_two_fever_probability :
  at_least_two_fever_prob = 0.896 := by sorry

end at_least_two_fever_probability_l3983_398377


namespace pascal_diagonal_sum_equals_fibonacci_l3983_398341

/-- Sum of numbers in the n-th diagonal of Pascal's triangle -/
def b (n : ℕ) : ℕ := sorry

/-- n-th term of the Fibonacci sequence -/
def a : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => a (n + 1) + a n

/-- Theorem stating that b_n equals a_n for all n -/
theorem pascal_diagonal_sum_equals_fibonacci (n : ℕ) : b n = a n := by
  sorry

end pascal_diagonal_sum_equals_fibonacci_l3983_398341


namespace area_between_concentric_circles_l3983_398350

/-- The area between two concentric circles, where a chord of length 100 units
    is tangent to the smaller circle, is equal to 2500π square units. -/
theorem area_between_concentric_circles (R r : ℝ) : 
  R > r → r > 0 → R^2 - r^2 = 2500 → π * (R^2 - r^2) = 2500 * π := by
  sorry

#check area_between_concentric_circles

end area_between_concentric_circles_l3983_398350


namespace carla_marbles_l3983_398376

/-- The number of marbles Carla bought -/
def marbles_bought : ℕ := 134

/-- The number of marbles Carla has now -/
def marbles_now : ℕ := 187

/-- The number of marbles Carla started with -/
def marbles_start : ℕ := marbles_now - marbles_bought

theorem carla_marbles : marbles_start = 53 := by
  sorry

end carla_marbles_l3983_398376


namespace negative_square_opposite_l3983_398325

-- Define opposite numbers
def opposite (a b : ℤ) : Prop := a = -b

-- Theorem statement
theorem negative_square_opposite : opposite (-2^2) ((-2)^2) := by
  sorry

end negative_square_opposite_l3983_398325


namespace prob_different_given_alone_is_half_l3983_398354

/-- The number of people visiting tourist spots -/
def num_people : ℕ := 3

/-- The number of tourist spots -/
def num_spots : ℕ := 3

/-- The number of ways person A can visit a spot alone -/
def ways_A_alone : ℕ := num_spots * (num_spots - 1) * (num_spots - 1)

/-- The number of ways all three people can visit different spots -/
def ways_all_different : ℕ := num_spots * (num_spots - 1) * (num_spots - 2)

/-- The probability that three people visit different spots given that one person visits a spot alone -/
def prob_different_given_alone : ℚ := ways_all_different / ways_A_alone

theorem prob_different_given_alone_is_half : prob_different_given_alone = 1 / 2 := by
  sorry

end prob_different_given_alone_is_half_l3983_398354


namespace triangle_inradius_l3983_398339

/-- The inradius of a triangle with side lengths 7, 24, and 25 is 3 -/
theorem triangle_inradius (a b c : ℝ) (h1 : a = 7) (h2 : b = 24) (h3 : c = 25) :
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area / s = 3 := by sorry

end triangle_inradius_l3983_398339


namespace sinusoidal_amplitude_l3983_398302

/-- Given a sinusoidal function y = a * sin(b * x + c) + d where a, b, c, and d are positive constants,
    if the function oscillates between 5 and -3, then a = 4. -/
theorem sinusoidal_amplitude (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
  (h_osc : ∀ x, -3 ≤ a * Real.sin (b * x + c) + d ∧ a * Real.sin (b * x + c) + d ≤ 5) :
  a = 4 := by
sorry

end sinusoidal_amplitude_l3983_398302


namespace lemonade_stand_lemons_cost_l3983_398397

/-- Proves that the amount spent on lemons is $10 given the lemonade stand conditions --/
theorem lemonade_stand_lemons_cost (sugar_cost cups_cost : ℕ) 
  (cups_sold price_per_cup : ℕ) (profit : ℕ) :
  sugar_cost = 5 →
  cups_cost = 3 →
  cups_sold = 21 →
  price_per_cup = 4 →
  profit = 66 →
  ∃ (lemons_cost : ℕ),
    lemons_cost = 10 ∧
    profit = cups_sold * price_per_cup - (lemons_cost + sugar_cost + cups_cost) :=
by sorry

end lemonade_stand_lemons_cost_l3983_398397


namespace infinitely_many_solutions_l3983_398371

theorem infinitely_many_solutions (b : ℝ) : 
  (∀ x, 5 * (3 * x - b) = 3 * (5 * x + 15)) ↔ b = -9 := by sorry

end infinitely_many_solutions_l3983_398371


namespace repeating_decimal_to_fraction_l3983_398322

/-- Expresses 0.3̄56 as a rational number -/
theorem repeating_decimal_to_fraction : 
  ∃ (n d : ℕ), d ≠ 0 ∧ (0.3 + (56 : ℚ) / 99 / 10) = (n : ℚ) / d := by
  sorry

end repeating_decimal_to_fraction_l3983_398322


namespace calculation_one_l3983_398363

theorem calculation_one : 6.8 - (-4.2) + (-4) * (-3) = 23 := by
  sorry

end calculation_one_l3983_398363


namespace root_equation_value_l3983_398361

theorem root_equation_value (m : ℝ) : m^2 + m - 1 = 0 → 2023 - m^2 - m = 2022 := by
  sorry

end root_equation_value_l3983_398361


namespace smallest_prime_pair_l3983_398332

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

theorem smallest_prime_pair : 
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ q = 13 * p + 2 ∧ 
  (∀ (p' : ℕ), is_prime p' ∧ p' < p → ¬(is_prime (13 * p' + 2))) ∧
  p = 3 ∧ q = 41 := by
sorry

end smallest_prime_pair_l3983_398332


namespace solution_set_a_2_range_of_a_l3983_398390

-- Define the function f
def f (a x : ℝ) := |x - a| + |2*x - 2|

-- Theorem 1: Solution set of f(x) > 2 when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f 2 x > 2} = {x : ℝ | x < 2/3 ∨ x > 2} :=
sorry

-- Theorem 2: Range of a when f(x) ≥ 2 for all x
theorem range_of_a :
  ∀ a : ℝ, (∀ x : ℝ, f a x ≥ 2) ↔ (a ≤ -1 ∨ a ≥ 3) :=
sorry

end solution_set_a_2_range_of_a_l3983_398390


namespace min_value_of_f_l3983_398398

/-- The quadratic function f(x) = x^2 + 6x + 8 -/
def f (x : ℝ) : ℝ := x^2 + 6*x + 8

/-- The minimum value of f(x) is -1 -/
theorem min_value_of_f :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = -1 :=
sorry

end min_value_of_f_l3983_398398


namespace fundraiser_customers_l3983_398353

/-- The number of customers who participated in the fundraiser -/
def num_customers : ℕ := 40

/-- The restaurant's donation ratio -/
def restaurant_ratio : ℚ := 2 / 10

/-- The average donation per customer -/
def avg_donation : ℚ := 3

/-- The total donation by the restaurant -/
def total_restaurant_donation : ℚ := 24

/-- Theorem stating that the number of customers is correct given the conditions -/
theorem fundraiser_customers :
  restaurant_ratio * (↑num_customers * avg_donation) = total_restaurant_donation :=
by sorry

end fundraiser_customers_l3983_398353


namespace min_reciprocal_sum_l3983_398314

theorem min_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : m + n = 1) :
  (1 / m + 1 / n) ≥ 4 ∧ ∃ m n, m > 0 ∧ n > 0 ∧ m + n = 1 ∧ 1 / m + 1 / n = 4 :=
sorry

end min_reciprocal_sum_l3983_398314


namespace lucky_point_properties_l3983_398359

/-- Definition of a lucky point -/
def is_lucky_point (m n x y : ℝ) : Prop :=
  2 * m = 4 + n ∧ x = m - 1 ∧ y = (n + 2) / 2

theorem lucky_point_properties :
  -- Part 1: When m = 2, the lucky point is (1, 1)
  (∃ n : ℝ, is_lucky_point 2 n 1 1) ∧
  -- Part 2: Point (3, 3) is a lucky point
  (∃ m n : ℝ, is_lucky_point m n 3 3) ∧
  -- Part 3: If (a, 2a-1) is a lucky point, then it's in the first quadrant
  (∀ a m n : ℝ, is_lucky_point m n a (2*a-1) → a > 0 ∧ 2*a-1 > 0) :=
by sorry

end lucky_point_properties_l3983_398359


namespace lcm_36_65_l3983_398346

theorem lcm_36_65 : Nat.lcm 36 65 = 2340 := by
  sorry

end lcm_36_65_l3983_398346


namespace negation_of_ln_positive_l3983_398385

theorem negation_of_ln_positive :
  (¬ ∀ x : ℝ, x > 0 → Real.log x > 0) ↔ (∃ x : ℝ, x > 0 ∧ Real.log x ≤ 0) :=
by sorry

end negation_of_ln_positive_l3983_398385


namespace opposite_absolute_values_sum_l3983_398305

theorem opposite_absolute_values_sum (a b : ℝ) : 
  |a - 2| = -|b + 3| → a + b = -1 := by
  sorry

end opposite_absolute_values_sum_l3983_398305


namespace alberts_earnings_increase_l3983_398313

theorem alberts_earnings_increase (E : ℝ) (P : ℝ) : 
  (1.26 * E = 693) → 
  ((1 + P) * E = 660) →
  P = 0.2 := by
sorry

end alberts_earnings_increase_l3983_398313


namespace number_exceeding_fraction_l3983_398382

theorem number_exceeding_fraction : 
  ∀ x : ℚ, x = (3/8) * x + 20 → x = 32 := by
sorry

end number_exceeding_fraction_l3983_398382


namespace central_position_theorem_l3983_398312

/-- Represents a row of stones -/
def StoneRow := List Bool

/-- An action changes the color of neighboring stones of a black stone -/
def action (row : StoneRow) (pos : Nat) : StoneRow :=
  sorry

/-- Checks if all stones in the row are black -/
def allBlack (row : StoneRow) : Prop :=
  sorry

/-- Checks if a given initial position can lead to all black stones -/
def canMakeAllBlack (initialPos : Nat) (totalStones : Nat) : Prop :=
  sorry

theorem central_position_theorem :
  ∀ initialPos : Nat,
    initialPos ≤ 2009 →
    canMakeAllBlack initialPos 2009 ↔ initialPos = 1005 :=
  sorry

end central_position_theorem_l3983_398312


namespace ezekiel_new_shoes_l3983_398375

/-- The number of pairs of shoes Ezekiel bought -/
def pairs_bought : ℕ := 3

/-- The number of shoes in each pair -/
def shoes_per_pair : ℕ := 2

/-- The total number of new shoes Ezekiel has now -/
def total_new_shoes : ℕ := pairs_bought * shoes_per_pair

theorem ezekiel_new_shoes : total_new_shoes = 6 := by
  sorry

end ezekiel_new_shoes_l3983_398375


namespace three_planes_division_l3983_398369

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields or axioms here
  dummy : Unit

/-- The number of parts that a set of planes divides 3D space into -/
def num_parts (planes : Set Plane3D) : ℕ := sorry

theorem three_planes_division :
  ∀ (p1 p2 p3 : Plane3D),
  4 ≤ num_parts {p1, p2, p3} ∧ num_parts {p1, p2, p3} ≤ 8 := by
  sorry

end three_planes_division_l3983_398369


namespace common_divisors_product_l3983_398393

theorem common_divisors_product (list : List Int) : 
  list = [48, 64, -18, 162, 144] →
  ∃ (a b c d e : Nat), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
    (∀ x ∈ list, a ∣ x.natAbs) ∧
    (∀ x ∈ list, b ∣ x.natAbs) ∧
    (∀ x ∈ list, c ∣ x.natAbs) ∧
    (∀ x ∈ list, d ∣ x.natAbs) ∧
    (∀ x ∈ list, e ∣ x.natAbs) ∧
    a * b * c * d * e = 108 :=
by sorry

end common_divisors_product_l3983_398393


namespace print_shop_price_difference_l3983_398335

def print_shop_x_price : ℚ := 120 / 100
def print_shop_y_price : ℚ := 170 / 100
def number_of_copies : ℕ := 40

theorem print_shop_price_difference :
  number_of_copies * print_shop_y_price - number_of_copies * print_shop_x_price = 20 := by
  sorry

end print_shop_price_difference_l3983_398335


namespace pencil_pen_difference_l3983_398392

-- Define the given conditions
def paige_pencils_home : ℕ := 15
def paige_pens_backpack : ℕ := 7

-- Define the theorem
theorem pencil_pen_difference : 
  paige_pencils_home - paige_pens_backpack = 8 := by
  sorry


end pencil_pen_difference_l3983_398392


namespace tan_70_cos_10_identity_l3983_398384

theorem tan_70_cos_10_identity : 
  Real.tan (70 * π / 180) * Real.cos (10 * π / 180) * (Real.sqrt 3 * Real.tan (20 * π / 180) - 1) = -1 := by
  sorry

end tan_70_cos_10_identity_l3983_398384


namespace greater_number_proof_l3983_398329

theorem greater_number_proof (x y : ℝ) (sum_eq : x + y = 36) (diff_eq : x - y = 12) : 
  max x y = 24 := by
sorry

end greater_number_proof_l3983_398329


namespace expression_evaluation_l3983_398323

theorem expression_evaluation :
  let x : ℚ := 1/2
  6 * x^2 - (2*x + 1) * (3*x - 2) + (x + 3) * (x - 3) = -25/4 := by
  sorry

end expression_evaluation_l3983_398323


namespace line_equation_through_point_with_inclination_l3983_398310

/-- The equation of a line passing through (-2, 3) with a 45° angle of inclination -/
theorem line_equation_through_point_with_inclination :
  ∃ (m b : ℝ), 
    (∀ x y : ℝ, y = m * x + b ↔ (x + 2) = (y - 3)) ∧ 
    m = Real.tan (45 * π / 180) ∧
    (x - y + 5 = 0) = (y = m * x + b) := by
  sorry

end line_equation_through_point_with_inclination_l3983_398310


namespace fraction_simplification_l3983_398389

theorem fraction_simplification : (5 * 6 - 3 * 4) / (6 + 3) = 2 := by
  sorry

end fraction_simplification_l3983_398389


namespace B_subset_complement_A_iff_m_in_range_A_intersect_B_nonempty_iff_m_in_range_A_union_B_eq_A_iff_m_in_range_l3983_398308

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | x^2 - 3*x + 2 > 0}
def B (m : ℝ) : Set ℝ := {x : ℝ | m*x^2 - (m+2)*x + 2 < 0}

-- State the theorems
theorem B_subset_complement_A_iff_m_in_range :
  ∀ m : ℝ, B m ⊆ (Set.univ \ A) ↔ m ∈ Set.Icc 1 2 := by sorry

theorem A_intersect_B_nonempty_iff_m_in_range :
  ∀ m : ℝ, (A ∩ B m).Nonempty ↔ m ∈ Set.Iic 1 ∪ Set.Ioi 2 := by sorry

theorem A_union_B_eq_A_iff_m_in_range :
  ∀ m : ℝ, A ∪ B m = A ↔ m ∈ Set.Ici 2 := by sorry

end B_subset_complement_A_iff_m_in_range_A_intersect_B_nonempty_iff_m_in_range_A_union_B_eq_A_iff_m_in_range_l3983_398308


namespace least_subtraction_for_divisibility_by_two_l3983_398373

theorem least_subtraction_for_divisibility_by_two : 
  ∃ (n : ℕ), n = 1 ∧ 
  (∀ m : ℕ, (9671 - m) % 2 = 0 → m ≥ n) ∧
  (9671 - n) % 2 = 0 := by
  sorry

end least_subtraction_for_divisibility_by_two_l3983_398373


namespace phi_minus_phi_squared_l3983_398331

theorem phi_minus_phi_squared (Φ φ : ℝ) : 
  Φ ≠ φ → Φ^2 = Φ + 1 → φ^2 = φ + 1 → (Φ - φ)^2 = 5 := by
  sorry

end phi_minus_phi_squared_l3983_398331


namespace distance_between_points_l3983_398383

theorem distance_between_points (a b c : ℝ) : 
  Real.sqrt ((a - (a + 3))^2 + (b - (b + 7))^2 + (c - (c + 1))^2) = Real.sqrt 59 := by
  sorry

end distance_between_points_l3983_398383


namespace coin_difference_l3983_398345

theorem coin_difference (total_coins quarters : ℕ) 
  (h1 : total_coins = 77)
  (h2 : quarters = 29) : 
  total_coins - quarters = 48 := by
  sorry

end coin_difference_l3983_398345


namespace unique_solution_system_l3983_398317

theorem unique_solution_system (x y : ℝ) :
  (2 * x - 3 * abs y = 1 ∧ abs x + 2 * y = 4) ↔ (x = 2 ∧ y = 1) :=
by sorry

end unique_solution_system_l3983_398317


namespace james_container_capacity_l3983_398395

/-- Represents the capacity of different container types and their quantities --/
structure ContainerInventory where
  largeCaskCapacity : ℕ
  barrelCapacity : ℕ
  smallCaskCapacity : ℕ
  glassBottleCapacity : ℕ
  clayJugCapacity : ℕ
  barrelCount : ℕ
  largeCaskCount : ℕ
  smallCaskCount : ℕ
  glassBottleCount : ℕ
  clayJugCount : ℕ

/-- Calculates the total capacity of all containers --/
def totalCapacity (inv : ContainerInventory) : ℕ :=
  inv.barrelCapacity * inv.barrelCount +
  inv.largeCaskCapacity * inv.largeCaskCount +
  inv.smallCaskCapacity * inv.smallCaskCount +
  inv.glassBottleCapacity * inv.glassBottleCount +
  inv.clayJugCapacity * inv.clayJugCount

/-- Theorem stating that James' total container capacity is 318 gallons --/
theorem james_container_capacity :
  ∀ (inv : ContainerInventory),
    inv.largeCaskCapacity = 20 →
    inv.barrelCapacity = 2 * inv.largeCaskCapacity + 3 →
    inv.smallCaskCapacity = inv.largeCaskCapacity / 2 →
    inv.glassBottleCapacity = inv.smallCaskCapacity / 10 →
    inv.clayJugCapacity = 3 * inv.glassBottleCapacity →
    inv.barrelCount = 4 →
    inv.largeCaskCount = 3 →
    inv.smallCaskCount = 5 →
    inv.glassBottleCount = 12 →
    inv.clayJugCount = 8 →
    totalCapacity inv = 318 :=
by
  sorry


end james_container_capacity_l3983_398395


namespace student_club_distribution_l3983_398307

-- Define the number of students and clubs
def num_students : ℕ := 5
def num_clubs : ℕ := 3

-- Define a function to calculate the number of ways to distribute students into clubs
def distribute_students (students : ℕ) (clubs : ℕ) : ℕ :=
  sorry

-- Theorem statement
theorem student_club_distribution :
  distribute_students num_students num_clubs = 150 :=
sorry

end student_club_distribution_l3983_398307


namespace algebraic_expression_value_l3983_398355

theorem algebraic_expression_value (x : ℝ) : 
  2 * x^2 + 3 * x + 7 = 8 → 4 * x^2 + 6 * x - 9 = -7 := by
  sorry

end algebraic_expression_value_l3983_398355
