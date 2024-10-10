import Mathlib

namespace right_triangle_area_l2889_288974

theorem right_triangle_area (a b : ℝ) (h1 : a = 40) (h2 : b = 42) :
  (1 / 2 : ℝ) * a * b = 840 :=
by sorry

end right_triangle_area_l2889_288974


namespace line_perpendicular_to_parallel_plane_l2889_288943

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)

-- State the theorem
theorem line_perpendicular_to_parallel_plane
  (m : Line) (α β : Plane)
  (h1 : perpendicular m α)
  (h2 : parallel α β) :
  perpendicular m β :=
sorry

end line_perpendicular_to_parallel_plane_l2889_288943


namespace base_a_equations_l2889_288998

/-- Converts a base-10 number to base-a --/
def toBaseA (n : ℕ) (a : ℕ) : ℕ := sorry

/-- Converts a base-a number to base-10 --/
def fromBaseA (n : ℕ) (a : ℕ) : ℕ := sorry

theorem base_a_equations (a : ℕ) :
  (toBaseA 375 a + toBaseA 596 a = toBaseA (9 * a + fromBaseA 12 10) a) ∧
  (fromBaseA 12 10 = 12) ∧
  (toBaseA 697 a + toBaseA 226 a = toBaseA (9 * a + fromBaseA 13 10) a) ∧
  (fromBaseA 13 10 = 13) →
  a = 14 := by sorry

end base_a_equations_l2889_288998


namespace inequality_solution_range_l2889_288956

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 + 2 * a * x + 2 ≥ 0) → 0 ≤ a ∧ a ≤ 2 := by
  sorry

end inequality_solution_range_l2889_288956


namespace unqualified_pieces_l2889_288937

theorem unqualified_pieces (total_products : ℕ) (pass_rate : ℚ) : 
  total_products = 400 → pass_rate = 98 / 100 → 
  ↑total_products * (1 - pass_rate) = 8 := by
  sorry

end unqualified_pieces_l2889_288937


namespace consecutive_points_ratio_l2889_288902

/-- Given five consecutive points on a line, prove the ratio of distances -/
theorem consecutive_points_ratio (a b c d e : ℝ) : 
  (b - a = 5) → 
  (c - a = 11) → 
  (e - d = 7) → 
  (e - a = 20) → 
  (c - b) / (d - c) = 3 := by
  sorry

end consecutive_points_ratio_l2889_288902


namespace loaves_sold_is_one_l2889_288926

/-- Represents the baker's sales and prices --/
structure BakerSales where
  usual_pastries : ℕ
  usual_bread : ℕ
  today_pastries : ℕ
  pastry_price : ℚ
  bread_price : ℚ
  sales_difference : ℚ

/-- Calculates the number of loaves of bread sold today --/
def loaves_sold_today (s : BakerSales) : ℚ :=
  ((s.usual_pastries * s.pastry_price + s.usual_bread * s.bread_price) -
   (s.today_pastries * s.pastry_price + s.sales_difference)) / s.bread_price

/-- Theorem stating that the number of loaves sold today is 1 --/
theorem loaves_sold_is_one (s : BakerSales)
  (h1 : s.usual_pastries = 20)
  (h2 : s.usual_bread = 10)
  (h3 : s.today_pastries = 14)
  (h4 : s.pastry_price = 2)
  (h5 : s.bread_price = 4)
  (h6 : s.sales_difference = 48) :
  loaves_sold_today s = 1 := by
  sorry

#eval loaves_sold_today {
  usual_pastries := 20,
  usual_bread := 10,
  today_pastries := 14,
  pastry_price := 2,
  bread_price := 4,
  sales_difference := 48
}

end loaves_sold_is_one_l2889_288926


namespace equation_solution_l2889_288924

theorem equation_solution : ∃! x : ℚ, (x^2 + 2*x + 3) / (x + 4) = x + 5 := by
  sorry

end equation_solution_l2889_288924


namespace isosceles_base_length_isosceles_x_bounds_l2889_288929

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  /-- Length of the equal sides -/
  x : ℝ
  /-- Length of the base -/
  y : ℝ
  /-- The triangle has perimeter 20 -/
  perimeter_eq : 2 * x + y = 20
  /-- The equal sides are longer than 5 and shorter than 10 -/
  x_bounds : 5 < x ∧ x < 10

/-- The base length of an isosceles triangle with perimeter 20 is 20 - 2x -/
theorem isosceles_base_length (t : IsoscelesTriangle) : t.y = 20 - 2 * t.x := by
  sorry

/-- The base length formula is valid only when 5 < x < 10 -/
theorem isosceles_x_bounds (t : IsoscelesTriangle) : 5 < t.x ∧ t.x < 10 := by
  sorry

end isosceles_base_length_isosceles_x_bounds_l2889_288929


namespace intersection_M_N_l2889_288911

def M : Set ℝ := {x : ℝ | -2 < x ∧ x < 3}
def N : Set ℝ := {-2, -1, 0, 1}

theorem intersection_M_N : M ∩ N = {-1, 0, 1} := by sorry

end intersection_M_N_l2889_288911


namespace expression_simplification_l2889_288947

theorem expression_simplification (x : ℝ) : 
  x * (x * (x * (x - 3) - 5) + 11) + 2 = x^4 - 3*x^3 - 5*x^2 + 11*x + 2 := by
  sorry

end expression_simplification_l2889_288947


namespace det_trig_matrix_zero_l2889_288900

theorem det_trig_matrix_zero (a b : ℝ) : 
  let M : Matrix (Fin 3) (Fin 3) ℝ := ![![1, Real.sin (a - b), Real.sin a],
                                       ![Real.sin (a - b), 1, Real.sin b],
                                       ![Real.sin a, Real.sin b, 1]]
  Matrix.det M = 0 := by
  sorry

end det_trig_matrix_zero_l2889_288900


namespace solve_equation_for_m_l2889_288948

theorem solve_equation_for_m : ∃ m : ℝ, (m - 5)^3 = (1/27)⁻¹ ∧ m = 8 := by sorry

end solve_equation_for_m_l2889_288948


namespace cloth_cost_price_l2889_288958

theorem cloth_cost_price
  (meters_sold : ℕ)
  (selling_price : ℕ)
  (loss_per_meter : ℕ)
  (h1 : meters_sold = 450)
  (h2 : selling_price = 18000)
  (h3 : loss_per_meter = 5) :
  (selling_price + meters_sold * loss_per_meter) / meters_sold = 45 := by
sorry

end cloth_cost_price_l2889_288958


namespace special_function_value_l2889_288978

/-- A function satisfying f(xy) = f(x)/y for all positive real numbers x and y -/
def special_function (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value (f : ℝ → ℝ) 
  (h : special_function f) (h1000 : f 1000 = 2) : f 750 = 8/3 := by
  sorry

end special_function_value_l2889_288978


namespace quadratic_distinct_roots_range_l2889_288957

theorem quadratic_distinct_roots_range (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 3*x + 2*m = 0 ∧ y^2 - 3*y + 2*m = 0) ↔ m < 9/8 := by
  sorry

end quadratic_distinct_roots_range_l2889_288957


namespace product_inequality_l2889_288910

theorem product_inequality (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_one : x + y + z = 1) : 
  (1 - x) * (1 - y) * (1 - z) ≥ 8 * x * y * z := by
  sorry

end product_inequality_l2889_288910


namespace f_monotonicity_and_extrema_l2889_288967

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem f_monotonicity_and_extrema :
  (∀ x y, -2 < x → x < y → f x < f y) ∧
  (∀ x y, x < y → y < -2 → f y < f x) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, f x ≤ f 0) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, -1 / Real.exp 2 ≤ f x) ∧
  f 0 = 1 ∧
  f (-2) = -1 / Real.exp 2 := by sorry

end f_monotonicity_and_extrema_l2889_288967


namespace sphere_speed_l2889_288913

-- Define constants
def Q : Real := -20e-6
def q : Real := 50e-6
def AB : Real := 2
def AC : Real := 3
def m : Real := 0.2
def g : Real := 10
def k : Real := 9e9

-- Define the theorem
theorem sphere_speed (BC : Real) (v : Real) 
  (h1 : BC^2 = AC^2 - AB^2)  -- Pythagorean theorem
  (h2 : v^2 = (2/m) * (k*Q*q * (1/AB - 1/BC) + m*g*AB)) : -- Energy conservation
  v = 5 := by
sorry

end sphere_speed_l2889_288913


namespace uncool_relatives_count_l2889_288920

/-- Given a club with the following characteristics:
  * 50 total people
  * 25 people with cool dads
  * 28 people with cool moms
  * 10 people with cool siblings
  * 15 people with both cool dads and cool moms
  * 5 people with both cool dads and cool siblings
  * 7 people with both cool moms and cool siblings
  * 3 people with cool dads, cool moms, and cool siblings
Prove that the number of people with all uncool relatives is 11. -/
theorem uncool_relatives_count (total : Nat) (cool_dad : Nat) (cool_mom : Nat) (cool_sibling : Nat)
    (cool_dad_and_mom : Nat) (cool_dad_and_sibling : Nat) (cool_mom_and_sibling : Nat)
    (cool_all : Nat) (h1 : total = 50) (h2 : cool_dad = 25) (h3 : cool_mom = 28)
    (h4 : cool_sibling = 10) (h5 : cool_dad_and_mom = 15) (h6 : cool_dad_and_sibling = 5)
    (h7 : cool_mom_and_sibling = 7) (h8 : cool_all = 3) :
  total - (cool_dad + cool_mom + cool_sibling - cool_dad_and_mom - cool_dad_and_sibling
           - cool_mom_and_sibling + cool_all) = 11 := by
  sorry

end uncool_relatives_count_l2889_288920


namespace ratio_of_sum_and_difference_l2889_288918

theorem ratio_of_sum_and_difference (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 5 * (a - b)) : a / b = 3 / 2 := by
  sorry

end ratio_of_sum_and_difference_l2889_288918


namespace smallest_positive_d_l2889_288930

theorem smallest_positive_d : ∃ d : ℝ, d > 0 ∧
  (5 * Real.sqrt 5)^2 + (d + 5)^2 = (5 * d)^2 ∧
  ∀ d' : ℝ, d' > 0 → (5 * Real.sqrt 5)^2 + (d' + 5)^2 = (5 * d')^2 → d ≤ d' :=
by sorry

end smallest_positive_d_l2889_288930


namespace apartment_renovation_is_credence_good_decision_is_difficult_l2889_288979

-- Define the types
structure Service where
  name : String
  is_credence_good : Bool
  has_info_asymmetry : Bool
  quality_hard_to_assess : Bool

-- Define the apartment renovation service
def apartment_renovation : Service where
  name := "Complete Apartment Renovation"
  is_credence_good := true
  has_info_asymmetry := true
  quality_hard_to_assess := true

-- Theorem statement
theorem apartment_renovation_is_credence_good :
  apartment_renovation.is_credence_good ∧
  apartment_renovation.has_info_asymmetry ∧
  apartment_renovation.quality_hard_to_assess :=
by sorry

-- Define the provider types
inductive Provider
| ConstructionCompany
| PrivateRepairCrew

-- Define a function to represent the decision-making process
def choose_provider (service : Service) : Provider → Bool
| Provider.ConstructionCompany => true  -- Simplified for demonstration
| Provider.PrivateRepairCrew => false   -- Simplified for demonstration

-- Theorem about the difficulty of the decision
theorem decision_is_difficult (service : Service) :
  ∃ (p1 p2 : Provider), p1 ≠ p2 ∧ choose_provider service p1 = choose_provider service p2 :=
by sorry

end apartment_renovation_is_credence_good_decision_is_difficult_l2889_288979


namespace childrens_cookbook_cost_l2889_288969

theorem childrens_cookbook_cost 
  (dictionary_cost : ℕ)
  (dinosaur_book_cost : ℕ)
  (saved_amount : ℕ)
  (additional_amount_needed : ℕ)
  (h1 : dictionary_cost = 11)
  (h2 : dinosaur_book_cost = 19)
  (h3 : saved_amount = 8)
  (h4 : additional_amount_needed = 29) :
  saved_amount + additional_amount_needed - (dictionary_cost + dinosaur_book_cost) = 7 := by
  sorry

end childrens_cookbook_cost_l2889_288969


namespace jessica_expense_increase_l2889_288904

-- Define Jessica's monthly expenses last year
def last_year_rent : ℝ := 1000
def last_year_food : ℝ := 200
def last_year_car_insurance : ℝ := 100
def last_year_utilities : ℝ := 50
def last_year_healthcare : ℝ := 150

-- Define the increase rates
def rent_increase_rate : ℝ := 0.3
def food_increase_rate : ℝ := 0.5
def car_insurance_increase_rate : ℝ := 2
def utilities_increase_rate : ℝ := 0.2
def healthcare_increase_rate : ℝ := 1

-- Define the theorem
theorem jessica_expense_increase :
  let this_year_rent := last_year_rent * (1 + rent_increase_rate)
  let this_year_food := last_year_food * (1 + food_increase_rate)
  let this_year_car_insurance := last_year_car_insurance * (1 + car_insurance_increase_rate)
  let this_year_utilities := last_year_utilities * (1 + utilities_increase_rate)
  let this_year_healthcare := last_year_healthcare * (1 + healthcare_increase_rate)
  let last_year_total := last_year_rent + last_year_food + last_year_car_insurance + last_year_utilities + last_year_healthcare
  let this_year_total := this_year_rent + this_year_food + this_year_car_insurance + this_year_utilities + this_year_healthcare
  (this_year_total - last_year_total) * 12 = 9120 :=
by sorry


end jessica_expense_increase_l2889_288904


namespace add_1457_minutes_to_3pm_l2889_288965

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  let newHours := totalMinutes / 60 % 24
  let newMinutes := totalMinutes % 60
  { hours := newHours, minutes := newMinutes }

/-- Theorem: Adding 1457 minutes to 3:00 p.m. results in 3:17 p.m. the next day -/
theorem add_1457_minutes_to_3pm (initial : Time) (final : Time) :
  initial = { hours := 15, minutes := 0 } →
  final = addMinutes initial 1457 →
  final = { hours := 15, minutes := 17 } :=
by sorry

end add_1457_minutes_to_3pm_l2889_288965


namespace promotion_difference_l2889_288936

/-- Represents a shoe promotion strategy -/
inductive Promotion
  | A  -- Buy one pair, get second pair half price
  | B  -- Buy one pair, get $15 off second pair

/-- Calculates the total cost of two pairs of shoes under a given promotion -/
def calculateCost (p : Promotion) (price1 : ℕ) (price2 : ℕ) : ℕ :=
  match p with
  | Promotion.A => price1 + price2 / 2
  | Promotion.B => price1 + price2 - 15

/-- Theorem stating the difference in cost between Promotion B and A -/
theorem promotion_difference :
  ∀ (price1 price2 : ℕ),
  price1 = 50 →
  price2 = 40 →
  calculateCost Promotion.B price1 price2 - calculateCost Promotion.A price1 price2 = 5 := by
  sorry


end promotion_difference_l2889_288936


namespace simple_interest_rate_l2889_288914

/-- Calculate the interest rate given principal, time, and simple interest -/
theorem simple_interest_rate (principal time simple_interest : ℝ) :
  principal > 0 ∧ time > 0 ∧ simple_interest > 0 →
  (simple_interest * 100) / (principal * time) = 9 ∧
  principal = 8965 ∧ time = 5 ∧ simple_interest = 4034.25 :=
by sorry

end simple_interest_rate_l2889_288914


namespace greatest_divisor_with_remainders_l2889_288963

theorem greatest_divisor_with_remainders : 
  let a := 150 - 50
  let b := 230 - 5
  let c := 175 - 25
  Nat.gcd a (Nat.gcd b c) = 25 := by
  sorry

end greatest_divisor_with_remainders_l2889_288963


namespace triangles_in_3x7_rectangle_l2889_288931

/-- The number of small triangles created by cutting a rectangle --/
def num_triangles (length width : ℕ) : ℕ :=
  let total_squares := length * width
  let corner_squares := 4
  let cut_squares := total_squares - corner_squares
  let triangles_per_square := 4
  cut_squares * triangles_per_square

/-- Theorem stating the number of triangles for a 3x7 rectangle --/
theorem triangles_in_3x7_rectangle :
  num_triangles 3 7 = 68 := by
  sorry

end triangles_in_3x7_rectangle_l2889_288931


namespace election_vote_difference_l2889_288992

theorem election_vote_difference 
  (total_votes : ℕ) 
  (winner_votes second_votes third_votes fourth_votes : ℕ) 
  (h_total : total_votes = 979)
  (h_candidates : winner_votes + second_votes + third_votes + fourth_votes = total_votes)
  (h_second : winner_votes = second_votes + 53)
  (h_third : winner_votes = third_votes + 79)
  (h_fourth : fourth_votes = 199) :
  winner_votes - fourth_votes = 105 := by
sorry

end election_vote_difference_l2889_288992


namespace theo_daily_consumption_l2889_288955

/-- Represents the daily water consumption of the three siblings -/
structure SiblingWaterConsumption where
  theo : ℕ
  mason : ℕ
  roxy : ℕ

/-- The total water consumption of the siblings over a week -/
def weeklyConsumption (swc : SiblingWaterConsumption) : ℕ :=
  7 * (swc.theo + swc.mason + swc.roxy)

/-- Theorem stating Theo's daily water consumption -/
theorem theo_daily_consumption :
  ∃ (swc : SiblingWaterConsumption),
    swc.mason = 7 ∧
    swc.roxy = 9 ∧
    weeklyConsumption swc = 168 ∧
    swc.theo = 8 := by
  sorry

end theo_daily_consumption_l2889_288955


namespace cube_has_eight_vertices_l2889_288934

/-- A cube is a three-dimensional solid object with six square faces -/
structure Cube where
  -- We don't need to define the internal structure of a cube for this problem

/-- The number of vertices in a cube -/
def num_vertices (c : Cube) : ℕ := 8

/-- Theorem: A cube has 8 vertices -/
theorem cube_has_eight_vertices (c : Cube) : num_vertices c = 8 := by
  sorry

end cube_has_eight_vertices_l2889_288934


namespace parabola_line_distance_l2889_288925

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Line structure -/
structure Line where
  k : ℝ

/-- Problem statement -/
theorem parabola_line_distance (parab : Parabola) (l : Line) : 
  (parab.p / 2 = 4) →
  (∃ x y : ℝ, x^2 = 2 * parab.p * y ∧ y = l.k * (x + 1)) →
  (∀ x y : ℝ, x^2 = 2 * parab.p * y ∧ y = l.k * (x + 1) → x = -1) →
  let focus_distance := dist (0, parab.p / 2) (x, l.k * (x + 1))
  (∃ x : ℝ, focus_distance = 1 ∨ focus_distance = 4 ∨ focus_distance = Real.sqrt 17) :=
by sorry

end parabola_line_distance_l2889_288925


namespace partnership_profit_l2889_288916

/-- Represents the profit distribution in a partnership --/
structure Partnership where
  a_investment : ℕ  -- A's investment
  b_investment : ℕ  -- B's investment
  a_period : ℕ      -- A's investment period
  b_period : ℕ      -- B's investment period
  b_profit : ℕ      -- B's profit

/-- Calculates the total profit of the partnership --/
def total_profit (p : Partnership) : ℕ :=
  let ratio := (p.a_investment * p.a_period) / (p.b_investment * p.b_period)
  p.b_profit * (ratio + 1)

/-- Theorem stating the total profit for the given partnership conditions --/
theorem partnership_profit (p : Partnership) 
  (h1 : p.a_investment = 3 * p.b_investment) 
  (h2 : p.a_period = 2 * p.b_period)
  (h3 : p.b_profit = 3000) : 
  total_profit p = 21000 := by
  sorry

#eval total_profit { a_investment := 3, b_investment := 1, a_period := 2, b_period := 1, b_profit := 3000 }

end partnership_profit_l2889_288916


namespace dog_food_consumption_l2889_288906

/-- The amount of dog food eaten by two dogs per day, given that each dog eats 0.125 scoop per day. -/
theorem dog_food_consumption (dog1_consumption dog2_consumption : ℝ) 
  (h1 : dog1_consumption = 0.125)
  (h2 : dog2_consumption = 0.125) : 
  dog1_consumption + dog2_consumption = 0.25 := by
  sorry

end dog_food_consumption_l2889_288906


namespace team_selection_ways_l2889_288908

-- Define the number of teachers and students
def num_teachers : ℕ := 5
def num_students : ℕ := 10

-- Define the function to calculate the number of ways to select one person from a group
def select_one (n : ℕ) : ℕ := n

-- Define the function to calculate the total number of ways to form a team
def total_ways : ℕ := select_one num_teachers * select_one num_students

-- Theorem statement
theorem team_selection_ways : total_ways = 50 := by
  sorry

end team_selection_ways_l2889_288908


namespace rectangle_diagonal_after_expansion_l2889_288983

/-- Given a rectangle with width 10 meters and area 150 square meters,
    if its length is increased such that the new area is 3.7 times the original area,
    then the length of the diagonal of the new rectangle is approximately 56.39 meters. -/
theorem rectangle_diagonal_after_expansion (ε : ℝ) (ε_pos : ε > 0) :
  ∃ (original_length new_length diagonal : ℝ),
    original_length > 0 ∧
    new_length > 0 ∧
    diagonal > 0 ∧
    10 * original_length = 150 ∧
    10 * new_length = 3.7 * 150 ∧
    diagonal ^ 2 = 10 ^ 2 + new_length ^ 2 ∧
    |diagonal - 56.39| < ε :=
by sorry

end rectangle_diagonal_after_expansion_l2889_288983


namespace largest_number_game_l2889_288901

theorem largest_number_game (a b c d : ℤ) 
  (eq1 : (a + b + c) / 3 + d = 17)
  (eq2 : (a + b + d) / 3 + c = 21)
  (eq3 : (a + c + d) / 3 + b = 23)
  (eq4 : (b + c + d) / 3 + a = 29) :
  max a (max b (max c d)) = 21 := by
  sorry

end largest_number_game_l2889_288901


namespace triangle_isosceles_l2889_288921

/-- A triangle with side lengths satisfying a specific equation is isosceles. -/
theorem triangle_isosceles (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_triangle : (a - c)^2 + (a - c) * b = 0) : a = c := by
  sorry

end triangle_isosceles_l2889_288921


namespace line_condition_l2889_288933

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the form y = kx + b -/
structure Line where
  k : ℝ
  b : ℝ

/-- Checks if two points are on the same side of a line x - y + 2 = 0 -/
def sameSideOfLine (p1 p2 : Point) : Prop :=
  (p1.x - p1.y + 2) * (p2.x - p2.y + 2) > 0

/-- The origin point (0, 0) -/
def origin : Point := ⟨0, 0⟩

/-- A point on the line y = kx + b -/
def pointOnLine (l : Line) (x : ℝ) : Point :=
  ⟨x, l.k * x + l.b⟩

theorem line_condition (l : Line) : 
  (∀ x : ℝ, sameSideOfLine (pointOnLine l x) origin) → 
  l.k = 1 ∧ l.b < 2 := by
  sorry

end line_condition_l2889_288933


namespace heart_shaped_chocolate_weight_l2889_288993

/-- Represents the weight of a chocolate bar -/
def chocolate_bar_weight (whole_squares : ℕ) (triangles : ℕ) (square_weight : ℕ) : ℕ :=
  whole_squares * square_weight + triangles * (square_weight / 2)

/-- Theorem stating the weight of the heart-shaped chocolate bar -/
theorem heart_shaped_chocolate_weight :
  chocolate_bar_weight 32 16 6 = 240 := by
  sorry

end heart_shaped_chocolate_weight_l2889_288993


namespace zeros_of_f_l2889_288949

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 - 9

-- Theorem stating that the zeros of f(x) are ±3
theorem zeros_of_f :
  ∀ x : ℝ, f x = 0 ↔ x = 3 ∨ x = -3 :=
by
  sorry

end zeros_of_f_l2889_288949


namespace pentagon_area_sum_l2889_288964

theorem pentagon_area_sum (a b : ℤ) (h1 : 0 < b) (h2 : b < a) :
  let F := (a, b)
  let G := (b, a)
  let H := (-b, a)
  let I := (-b, -a)
  let J := (-a, -b)
  let pentagon_area := (a^2 : ℝ) + 3 * (a * b)
  pentagon_area = 550 → a + b = 24 := by
  sorry

end pentagon_area_sum_l2889_288964


namespace inequality_preservation_l2889_288990

theorem inequality_preservation (x y : ℝ) (h : x > y) : 2 * x + 1 > 2 * y + 1 := by
  sorry

end inequality_preservation_l2889_288990


namespace divisor_power_expression_l2889_288968

theorem divisor_power_expression (k : ℕ) : 
  (30 ^ k : ℕ) ∣ 929260 → 3 ^ k - k ^ 3 = 1 := by
  sorry

end divisor_power_expression_l2889_288968


namespace alternating_arrangements_count_alternating_arrangements_proof_l2889_288922

/-- The number of ways to arrange 2 men and 2 women in a row,
    such that no two men or two women are adjacent. -/
def alternating_arrangements : ℕ := 8

/-- The number of men in the arrangement. -/
def num_men : ℕ := 2

/-- The number of women in the arrangement. -/
def num_women : ℕ := 2

/-- Theorem stating that the number of alternating arrangements
    of 2 men and 2 women is 8. -/
theorem alternating_arrangements_count :
  alternating_arrangements = 8 ∧
  num_men = 2 ∧
  num_women = 2 := by
  sorry

/-- Proof that the number of alternating arrangements is correct. -/
theorem alternating_arrangements_proof :
  alternating_arrangements = 2 * (Nat.factorial num_men) * (Nat.factorial num_women) := by
  sorry

end alternating_arrangements_count_alternating_arrangements_proof_l2889_288922


namespace parallel_line_x_coordinate_l2889_288988

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Checks if two points form a line parallel to the y-axis -/
def parallelToYAxis (p q : Point) : Prop :=
  p.x = q.x

/-- The theorem statement -/
theorem parallel_line_x_coordinate 
  (a : ℝ) 
  (P : Point) 
  (Q : Point) 
  (h1 : P = ⟨a, -5⟩) 
  (h2 : Q = ⟨4, 3⟩) 
  (h3 : parallelToYAxis P Q) : 
  a = 4 := by
  sorry

end parallel_line_x_coordinate_l2889_288988


namespace students_per_classroom_l2889_288942

/-- Given a school trip scenario, calculate the number of students per classroom. -/
theorem students_per_classroom
  (num_classrooms : ℕ)
  (seats_per_bus : ℕ)
  (num_buses : ℕ)
  (h1 : num_classrooms = 67)
  (h2 : seats_per_bus = 6)
  (h3 : num_buses = 737) :
  (num_buses * seats_per_bus) / num_classrooms = 66 :=
by sorry

end students_per_classroom_l2889_288942


namespace circle_equation_is_correct_l2889_288960

/-- A circle with center on the x-axis, radius 2, and passing through (1, 2) -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ
  passes_through : ℝ × ℝ
  center_on_x_axis : center.2 = 0
  radius_is_2 : radius = 2
  passes_through_point : passes_through = (1, 2)

/-- The equation of the circle -/
def circle_equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + y^2 = c.radius^2

theorem circle_equation_is_correct (c : Circle) :
  circle_equation c = λ x y ↦ (x - 1)^2 + y^2 = 4 := by
sorry

end circle_equation_is_correct_l2889_288960


namespace coefficient_implies_a_value_l2889_288962

theorem coefficient_implies_a_value (a : ℝ) : 
  (5 / 2) * a^3 = 20 → a = 2 := by
  sorry

end coefficient_implies_a_value_l2889_288962


namespace honey_barrel_problem_l2889_288938

theorem honey_barrel_problem (total_weight : ℝ) (half_removed_weight : ℝ) 
  (h1 : total_weight = 56)
  (h2 : half_removed_weight = 34) :
  ∃ (honey_weight barrel_weight : ℝ),
    honey_weight = 44 ∧
    barrel_weight = 12 ∧
    honey_weight + barrel_weight = total_weight ∧
    honey_weight / 2 + barrel_weight = half_removed_weight := by
  sorry

end honey_barrel_problem_l2889_288938


namespace ratio_p_to_r_l2889_288952

theorem ratio_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4)
  (h2 : r / s = 3 / 2)
  (h3 : s / q = 1 / 5) :
  p / r = 25 / 6 := by
  sorry

end ratio_p_to_r_l2889_288952


namespace determinant_theorem_l2889_288961

theorem determinant_theorem (a b c d : ℝ) : 
  a * d - b * c = -3 → 
  (a + 2*b) * d - (2*b - d) * (3*c) = -3 - 5*b*c + 2*b*d + 3*c*d := by
sorry

end determinant_theorem_l2889_288961


namespace complement_of_35_degrees_l2889_288976

theorem complement_of_35_degrees :
  ∀ α : Real,
  α = 35 →
  90 - α = 55 :=
by
  sorry

end complement_of_35_degrees_l2889_288976


namespace convergence_of_difference_series_l2889_288975

open Topology
open Real

-- Define a monotonic sequence
def IsMonotonic (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, n ≤ m → a n ≤ a m ∨ ∀ n m : ℕ, n ≤ m → a m ≤ a n

-- Define the theorem
theorem convergence_of_difference_series (a : ℕ → ℝ) 
  (h_monotonic : IsMonotonic a) 
  (h_converge : Summable a) :
  Summable (fun n => n • (a n - a (n + 1))) :=
sorry

end convergence_of_difference_series_l2889_288975


namespace matrix_determinant_and_fraction_sum_l2889_288971

theorem matrix_determinant_and_fraction_sum (p q r : ℝ) :
  let M := ![![p, 2*q, r],
             ![q, r, p],
             ![r, p, q]]
  Matrix.det M = 0 →
  (p / (2*q + r) + 2*q / (p + r) + r / (p + q) = -4) ∨
  (p / (2*q + r) + 2*q / (p + r) + r / (p + q) = 11/6) :=
by sorry

end matrix_determinant_and_fraction_sum_l2889_288971


namespace stating_two_students_math_course_l2889_288995

/-- The number of students -/
def num_students : ℕ := 4

/-- The number of courses -/
def num_courses : ℕ := 4

/-- The number of students who should choose mathematics -/
def math_students : ℕ := 2

/-- The number of remaining courses after mathematics -/
def remaining_courses : ℕ := 3

/-- Function to calculate the number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := 
  Nat.choose n k

/-- 
Theorem stating that the number of ways in which exactly two out of four students 
can choose a mathematics tutoring course, while the other two choose from three 
remaining courses, is equal to 54.
-/
theorem two_students_math_course : 
  (choose num_students math_students) * (remaining_courses^(num_students - math_students)) = 54 := by
  sorry

end stating_two_students_math_course_l2889_288995


namespace triangle_construction_theorem_l2889_288915

-- Define a triangle as a structure with three points
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define a function to check if a point is on a line segment
def isOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop := sorry

-- Define a function to check if two line segments are parallel
def areParallel (A B : ℝ × ℝ) (C D : ℝ × ℝ) : Prop := sorry

-- Define a function to count valid triangles
def countValidTriangles (ABC X'Y'Z' : Triangle) : ℕ := sorry

-- Define a function to count valid triangles including extensions
def countValidTrianglesWithExtensions (ABC X'Y'Z' : Triangle) : ℕ := sorry

theorem triangle_construction_theorem (ABC X'Y'Z' : Triangle) :
  (countValidTriangles ABC X'Y'Z' = 2 ∨
   countValidTriangles ABC X'Y'Z' = 1 ∨
   countValidTriangles ABC X'Y'Z' = 0) ∧
  countValidTrianglesWithExtensions ABC X'Y'Z' = 6 := by
  sorry

end triangle_construction_theorem_l2889_288915


namespace greatest_partition_number_l2889_288954

/-- A partition of the positive integers into k subsets -/
def Partition (k : ℕ) := Fin k → Set ℕ

/-- The property that a partition satisfies the sum condition for all n ≥ 15 -/
def SatisfiesSumCondition (p : Partition k) : Prop :=
  ∀ (n : ℕ) (i : Fin k), n ≥ 15 →
    ∃ (x y : ℕ), x ∈ p i ∧ y ∈ p i ∧ x ≠ y ∧ x + y = n

/-- The main theorem stating that 3 is the greatest k satisfying the conditions -/
theorem greatest_partition_number :
  (∃ (p : Partition 3), SatisfiesSumCondition p) ∧
  (∀ k > 3, ¬∃ (p : Partition k), SatisfiesSumCondition p) :=
sorry

end greatest_partition_number_l2889_288954


namespace min_dwarves_at_risk_l2889_288996

/-- Represents the color of a hat -/
inductive HatColor
| Black
| White

/-- Represents a dwarf with a hat -/
structure Dwarf :=
  (hat : HatColor)

/-- A line of dwarves -/
def DwarfLine := List Dwarf

/-- The probability of guessing correctly for a single dwarf -/
def guessProb : ℚ := 1/2

/-- The minimum number of dwarves at risk given a strategy -/
def minRisk (p : ℕ) (strategy : DwarfLine → ℕ) : ℕ :=
  min p (strategy (List.replicate p (Dwarf.mk HatColor.Black)))

theorem min_dwarves_at_risk (p : ℕ) (h : p > 0) :
  ∃ (strategy : DwarfLine → ℕ), minRisk p strategy = 1 :=
sorry

end min_dwarves_at_risk_l2889_288996


namespace ice_cream_frozen_yoghurt_cost_difference_l2889_288951

/-- Calculates the difference in cost between ice cream and frozen yoghurt purchases -/
theorem ice_cream_frozen_yoghurt_cost_difference :
  let ice_cream_cartons : ℕ := 10
  let frozen_yoghurt_cartons : ℕ := 4
  let ice_cream_price : ℕ := 4
  let frozen_yoghurt_price : ℕ := 1
  let ice_cream_total_cost := ice_cream_cartons * ice_cream_price
  let frozen_yoghurt_total_cost := frozen_yoghurt_cartons * frozen_yoghurt_price
  ice_cream_total_cost - frozen_yoghurt_total_cost = 36 :=
by sorry

end ice_cream_frozen_yoghurt_cost_difference_l2889_288951


namespace geometric_sequence_single_digit_numbers_l2889_288997

theorem geometric_sequence_single_digit_numbers :
  ∃! (a b c : ℕ),
    a < 10 ∧ b < 10 ∧ c < 10 ∧
    ∃ (q : ℚ),
      b = a * q ∧
      (10 * a + c : ℚ) = a * q^2 ∧
      (10 * c + b : ℚ) = a * q^3 ∧
      a = 1 ∧ b = 4 ∧ c = 6 :=
by sorry

end geometric_sequence_single_digit_numbers_l2889_288997


namespace apple_lovers_count_l2889_288991

/-- The number of people who like apple -/
def apple_lovers : ℕ := 40

/-- The number of people who like orange and mango but dislike apple -/
def orange_mango_lovers : ℕ := 7

/-- The number of people who like mango and apple and dislike orange -/
def mango_apple_lovers : ℕ := 10

/-- The number of people who like all three fruits -/
def all_fruit_lovers : ℕ := 4

/-- Theorem stating that the number of people who like apple is 40 -/
theorem apple_lovers_count : apple_lovers = 40 := by
  sorry

end apple_lovers_count_l2889_288991


namespace min_value_sqrt_inequality_l2889_288953

theorem min_value_sqrt_inequality :
  ∃ (a : ℝ), (∀ (x y : ℝ), x > 0 ∧ y > 0 → Real.sqrt x + Real.sqrt y ≤ a * Real.sqrt (x + y)) ∧
  (∀ (b : ℝ), (∀ (x y : ℝ), x > 0 ∧ y > 0 → Real.sqrt x + Real.sqrt y ≤ b * Real.sqrt (x + y)) → a ≤ b) ∧
  a = Real.sqrt 2 :=
by sorry

end min_value_sqrt_inequality_l2889_288953


namespace inequality_solution_l2889_288907

theorem inequality_solution (a : ℝ) (x : ℝ) : 
  (x + 1) * ((a - 1) * x - 1) > 0 ↔ 
    (a < 0 ∧ -1 < x ∧ x < 1 / (a - 1)) ∨
    (0 < a ∧ a < 1 ∧ 1 / (a - 1) < x ∧ x < -1) ∨
    (a = 1 ∧ x < -1) ∨
    (a > 1 ∧ (x < -1 ∨ x > 1 / (a - 1))) :=
sorry

end inequality_solution_l2889_288907


namespace chocolate_milk_probability_l2889_288950

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- number of days
  let k : ℕ := 5  -- number of successful days
  let p : ℚ := 3/4  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end chocolate_milk_probability_l2889_288950


namespace applicants_theorem_l2889_288959

/-- The number of applicants with less than 4 years' experience and no degree -/
def applicants_less_exp_no_degree (total : ℕ) (exp : ℕ) (deg : ℕ) (exp_and_deg : ℕ) : ℕ :=
  total - (exp + deg - exp_and_deg)

theorem applicants_theorem (total : ℕ) (exp : ℕ) (deg : ℕ) (exp_and_deg : ℕ)
  (h_total : total = 30)
  (h_exp : exp = 10)
  (h_deg : deg = 18)
  (h_exp_and_deg : exp_and_deg = 9) :
  applicants_less_exp_no_degree total exp deg exp_and_deg = 11 :=
by
  sorry

#eval applicants_less_exp_no_degree 30 10 18 9

end applicants_theorem_l2889_288959


namespace alice_bob_race_difference_l2889_288987

/-- The time difference between two runners finishing a race -/
def race_time_difference (alice_speed bob_speed race_distance : ℝ) : ℝ :=
  bob_speed * race_distance - alice_speed * race_distance

/-- Theorem stating the time difference between Alice and Bob in a 12-mile race -/
theorem alice_bob_race_difference :
  race_time_difference 7 9 12 = 24 := by
  sorry

end alice_bob_race_difference_l2889_288987


namespace percentage_increase_proof_l2889_288973

def initial_earnings : ℝ := 40
def new_earnings : ℝ := 60

theorem percentage_increase_proof :
  (new_earnings - initial_earnings) / initial_earnings * 100 = 50 := by
  sorry

end percentage_increase_proof_l2889_288973


namespace x_equals_y_at_half_l2889_288935

theorem x_equals_y_at_half (t : ℝ) : 
  let x := 1 - 4 * t
  let y := 2 * t - 2
  t = 0.5 → x = y := by sorry

end x_equals_y_at_half_l2889_288935


namespace coffee_blend_price_l2889_288903

/-- Given two blends of coffee, prove the price of the first blend -/
theorem coffee_blend_price 
  (price_blend2 : ℝ) 
  (total_weight : ℝ) 
  (total_price_per_pound : ℝ) 
  (weight_blend2 : ℝ) 
  (h1 : price_blend2 = 8) 
  (h2 : total_weight = 20) 
  (h3 : total_price_per_pound = 8.4) 
  (h4 : weight_blend2 = 12) : 
  ∃ (price_blend1 : ℝ), price_blend1 = 9 := by
sorry

end coffee_blend_price_l2889_288903


namespace farrah_matchsticks_l2889_288972

/-- Calculates the total number of matchsticks given the number of boxes, matchboxes per box, and sticks per matchbox. -/
def total_matchsticks (x y z : ℕ) : ℕ := x * y * z

/-- Theorem stating that for the given values, the total number of matchsticks is 300,000. -/
theorem farrah_matchsticks :
  let x : ℕ := 10
  let y : ℕ := 50
  let z : ℕ := 600
  total_matchsticks x y z = 300000 := by
  sorry

end farrah_matchsticks_l2889_288972


namespace inverse_variation_cube_and_sqrt_l2889_288966

/-- Given that x^3 varies inversely with √x, prove that y = 1/16384 when x = 64, 
    given that y = 16 when x = 4 -/
theorem inverse_variation_cube_and_sqrt (y : ℝ → ℝ) :
  (∀ x : ℝ, x > 0 → ∃ k : ℝ, y x * (x^3 * x.sqrt) = k) →
  y 4 = 16 →
  y 64 = 1 / 16384 := by
sorry

end inverse_variation_cube_and_sqrt_l2889_288966


namespace fraction_scaling_l2889_288985

theorem fraction_scaling (a b : ℝ) :
  (2*a + 2*b) / ((2*a)^2 + (2*b)^2) = (1/2) * ((a + b) / (a^2 + b^2)) :=
by sorry

end fraction_scaling_l2889_288985


namespace first_reduction_percentage_l2889_288994

theorem first_reduction_percentage (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 0.81 → x = 10 := by
  sorry

end first_reduction_percentage_l2889_288994


namespace valid_triples_equal_solution_set_l2889_288946

def is_valid_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a ≤ b ∧ b ≤ c ∧ a^2 + b^2 + c^2 = 2005

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(23, 24, 30), (12, 30, 31), (9, 18, 40), (9, 30, 32), (4, 15, 42), (15, 22, 36), (4, 30, 33)}

theorem valid_triples_equal_solution_set :
  {(a, b, c) | is_valid_triple a b c} = solution_set :=
by sorry

end valid_triples_equal_solution_set_l2889_288946


namespace sqrt_sum_squares_irrational_l2889_288940

/-- Given two consecutive even integers and their sum, the square root of the sum of their squares is irrational -/
theorem sqrt_sum_squares_irrational (x : ℤ) : 
  let a : ℤ := 2 * x
  let b : ℤ := 2 * x + 2
  let c : ℤ := a + b
  let D : ℤ := a^2 + b^2 + c^2
  Irrational (Real.sqrt (D : ℝ)) := by
sorry


end sqrt_sum_squares_irrational_l2889_288940


namespace unit_vector_d_l2889_288919

def d : ℝ × ℝ := (12, -5)

theorem unit_vector_d :
  let magnitude := Real.sqrt (d.1 ^ 2 + d.2 ^ 2)
  (d.1 / magnitude, d.2 / magnitude) = (12 / 13, -5 / 13) := by
  sorry

end unit_vector_d_l2889_288919


namespace f_is_quadratic_l2889_288944

-- Define a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the function y = x(2x - 1)
def f (x : ℝ) : ℝ := x * (2 * x - 1)

-- Theorem statement
theorem f_is_quadratic : is_quadratic f := by
  sorry

end f_is_quadratic_l2889_288944


namespace sum_of_sequences_l2889_288912

def sequence1 : List Nat := [2, 12, 22, 32, 42]
def sequence2 : List Nat := [10, 20, 30, 40, 50]

theorem sum_of_sequences : (sequence1.sum + sequence2.sum = 260) := by
  sorry

end sum_of_sequences_l2889_288912


namespace polynomial_division_theorem_l2889_288917

theorem polynomial_division_theorem (x : ℝ) :
  x^5 - x^4 + x^3 - 9 = (x - 1) * (x^4 - x^3 + x^2 - x + 1) + (-9) := by
  sorry

end polynomial_division_theorem_l2889_288917


namespace black_balls_count_l2889_288909

theorem black_balls_count (white_balls : ℕ) (prob_white : ℚ) : 
  white_balls = 5 →
  prob_white = 5 / 11 →
  ∃ (total_balls : ℕ), 
    (prob_white = white_balls / total_balls) ∧
    (total_balls - white_balls = 6) :=
by sorry

end black_balls_count_l2889_288909


namespace composition_equality_l2889_288980

def f (a b x : ℝ) : ℝ := a * x + b
def g (c d x : ℝ) : ℝ := c * x + d

theorem composition_equality (a b c d : ℝ) :
  (∀ x, f a b (g c d x) = g c d (f a b x)) ↔ b * (1 - c) - d * (1 - a) = 0 := by
  sorry

end composition_equality_l2889_288980


namespace x_squared_plus_reciprocal_l2889_288970

theorem x_squared_plus_reciprocal (x : ℝ) (h : x^4 + 1/x^4 = 240) : 
  x^2 + 1/x^2 = Real.sqrt 242 := by
sorry

end x_squared_plus_reciprocal_l2889_288970


namespace M_intersect_N_l2889_288982

def M : Set ℝ := {x : ℝ | -4 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | x^2 - x - 6 < 0}

theorem M_intersect_N : M ∩ N = {x : ℝ | -2 < x ∧ x < 2} := by sorry

end M_intersect_N_l2889_288982


namespace zeros_in_square_of_near_power_of_ten_l2889_288977

theorem zeros_in_square_of_near_power_of_ten : 
  ∃ n : ℕ, (10^12 - 3)^2 = n * 10^11 ∧ n % 10 ≠ 0 :=
by sorry

end zeros_in_square_of_near_power_of_ten_l2889_288977


namespace product_inequality_l2889_288941

theorem product_inequality (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0) (h : A * B * C = 1) :
  (A - 1 + 1/B) * (B - 1 + 1/C) * (C - 1 + 1/A) ≤ 1 := by
  sorry

end product_inequality_l2889_288941


namespace galya_number_puzzle_l2889_288928

theorem galya_number_puzzle (k : ℝ) (N : ℝ) : 
  (((k * N + N) / N) - N = k - 7729) → N = 7730 :=
by
  sorry

end galya_number_puzzle_l2889_288928


namespace power_multiplication_l2889_288999

theorem power_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end power_multiplication_l2889_288999


namespace cookie_problem_indeterminate_l2889_288939

/-- Represents the number of cookies Paco had and ate --/
structure CookieCount where
  initialSweet : ℕ
  initialSalty : ℕ
  eatenSweet : ℕ
  eatenSalty : ℕ

/-- Represents the conditions of the cookie problem --/
def CookieProblem (c : CookieCount) : Prop :=
  c.initialSalty = 6 ∧
  c.eatenSweet = 20 ∧
  c.eatenSalty = 34 ∧
  c.eatenSalty = c.eatenSweet + 14

theorem cookie_problem_indeterminate :
  ∀ (c : CookieCount), CookieProblem c →
    (c.initialSweet ≥ 20 ∧
     ∀ (n : ℕ), n ≥ 20 → ∃ (c' : CookieCount), CookieProblem c' ∧ c'.initialSweet = n) :=
by sorry

#check cookie_problem_indeterminate

end cookie_problem_indeterminate_l2889_288939


namespace additional_bottles_needed_l2889_288927

def medium_bottle_capacity : ℕ := 50
def giant_bottle_capacity : ℕ := 750
def bottles_already_owned : ℕ := 3

theorem additional_bottles_needed : 
  (giant_bottle_capacity / medium_bottle_capacity) - bottles_already_owned = 12 := by
  sorry

end additional_bottles_needed_l2889_288927


namespace granger_grocery_bill_l2889_288905

/-- Calculates the total cost of a grocery shopping trip -/
def total_cost (spam_price peanut_butter_price bread_price : ℕ) 
               (spam_quantity peanut_butter_quantity bread_quantity : ℕ) : ℕ :=
  spam_price * spam_quantity + 
  peanut_butter_price * peanut_butter_quantity + 
  bread_price * bread_quantity

/-- Proves that Granger's grocery bill is $59 -/
theorem granger_grocery_bill : 
  total_cost 3 5 2 12 3 4 = 59 := by
  sorry

end granger_grocery_bill_l2889_288905


namespace minimum_value_implies_a_l2889_288923

/-- The function f(x) = x^3 + 3ax^2 - 6ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x^2 - 6*a*x + 2

/-- The derivative of f(x) with respect to x -/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 6*a*x - 6*a

theorem minimum_value_implies_a (a : ℝ) :
  ∃ x₀ : ℝ, x₀ > 1 ∧ x₀ < 3 ∧
  (∀ x : ℝ, f a x ≥ f a x₀) ∧
  (f_derivative a x₀ = 0) →
  a = -2 := by sorry

end minimum_value_implies_a_l2889_288923


namespace ones_digit_of_19_power_l2889_288945

theorem ones_digit_of_19_power (n : ℕ) : 19^(19 * (13^13)) ≡ 9 [ZMOD 10] := by
  sorry

end ones_digit_of_19_power_l2889_288945


namespace white_surface_area_fraction_l2889_288989

theorem white_surface_area_fraction (cube_edge : ℕ) (total_unit_cubes : ℕ) 
  (white_unit_cubes : ℕ) (black_unit_cubes : ℕ) :
  cube_edge = 4 →
  total_unit_cubes = 64 →
  white_unit_cubes = 48 →
  black_unit_cubes = 16 →
  white_unit_cubes + black_unit_cubes = total_unit_cubes →
  (black_unit_cubes : ℚ) * 3 / (6 * cube_edge^2 : ℚ) = 1/2 →
  (6 * cube_edge^2 - black_unit_cubes * 3 : ℚ) / (6 * cube_edge^2 : ℚ) = 1/2 := by
  sorry

#check white_surface_area_fraction

end white_surface_area_fraction_l2889_288989


namespace constant_term_binomial_expansion_l2889_288986

/-- The constant term in the expansion of (1/√x - x^2)^10 is 45 -/
theorem constant_term_binomial_expansion :
  let f (x : ℝ) := (1 / Real.sqrt x - x^2)^10
  ∃ (c : ℝ), ∀ (x : ℝ), x ≠ 0 → f x = c + x * (f x - c) ∧ c = 45 :=
sorry

end constant_term_binomial_expansion_l2889_288986


namespace area_of_triangle_l2889_288932

-- Define the triangle ABC and point P
structure Triangle :=
  (A B C P : ℝ × ℝ)

-- Define the properties of the triangle
def isRightTriangle (t : Triangle) : Prop :=
  -- Right angle at B
  sorry

def pointOnHypotenuse (t : Triangle) : Prop :=
  -- P is on AC
  sorry

def angleABP (t : Triangle) : ℝ :=
  -- Angle ABP in radians
  sorry

def lengthAP (t : Triangle) : ℝ :=
  -- Length of AP
  sorry

def lengthCP (t : Triangle) : ℝ :=
  -- Length of CP
  sorry

def areaABC (t : Triangle) : ℝ :=
  -- Area of triangle ABC
  sorry

-- Theorem statement
theorem area_of_triangle (t : Triangle) 
  (h1 : isRightTriangle t)
  (h2 : pointOnHypotenuse t)
  (h3 : angleABP t = π / 6)  -- 30° in radians
  (h4 : lengthAP t = 2)
  (h5 : lengthCP t = 1) :
  areaABC t = 9 / 5 :=
by
  sorry

end area_of_triangle_l2889_288932


namespace complex_expression_evaluation_l2889_288981

theorem complex_expression_evaluation :
  let i : ℂ := Complex.I
  ((2 + i) * (3 + i)) / (1 + i) = 5 := by sorry

end complex_expression_evaluation_l2889_288981


namespace ellipse_triangle_perimeter_l2889_288984

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 25 + y^2 / 16 = 1

-- Define the foci
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point on the ellipse
def point_on_ellipse (p : ℝ × ℝ) : Prop :=
  ellipse p.1 p.2

-- Define collinearity
def collinear (p q r : ℝ × ℝ) : Prop := sorry

-- Define the distance between two points
def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h₁ : point_on_ellipse A) 
  (h₂ : point_on_ellipse B) 
  (h₃ : collinear A B F₂) :
  distance F₁ A + distance F₁ B + distance A B = 20 := 
sorry

end ellipse_triangle_perimeter_l2889_288984
