import Mathlib

namespace marks_trees_l4130_413018

theorem marks_trees (current_trees planted_trees : ℕ) :
  current_trees = 13 → planted_trees = 12 →
  current_trees + planted_trees = 25 := by
  sorry

end marks_trees_l4130_413018


namespace intersection_point_coordinates_l4130_413096

/-- Given a triangle ABC, this theorem proves the position of point Q
    based on the given ratios of points G and H on the sides of the triangle. -/
theorem intersection_point_coordinates (A B C G H Q : ℝ × ℝ) : 
  (∃ t : ℝ, G = (1 - t) • A + t • B ∧ t = 2/5) →
  (∃ s : ℝ, H = (1 - s) • B + s • C ∧ s = 3/4) →
  (∃ r : ℝ, Q = (1 - r) • A + r • G) →
  (∃ u : ℝ, Q = (1 - u) • C + u • H) →
  Q = (3/8) • A + (1/4) • B + (3/8) • C :=
by sorry

end intersection_point_coordinates_l4130_413096


namespace units_digit_of_7_to_2083_l4130_413079

theorem units_digit_of_7_to_2083 : 7^2083 % 10 = 3 := by
  sorry

end units_digit_of_7_to_2083_l4130_413079


namespace equation_solution_l4130_413038

theorem equation_solution : ∃! x : ℝ, 5 + 3.5 * x = 2.5 * x - 25 :=
  by
    use -30
    constructor
    · -- Prove that x = -30 satisfies the equation
      sorry
    · -- Prove uniqueness
      sorry

end equation_solution_l4130_413038


namespace long_furred_dogs_count_l4130_413037

/-- Represents the characteristics of dogs in a kennel --/
structure KennelData where
  total_dogs : ℕ
  brown_dogs : ℕ
  neither_long_furred_nor_brown : ℕ
  long_furred_brown : ℕ

/-- Calculates the number of dogs with long fur in the kennel --/
def long_furred_dogs (data : KennelData) : ℕ :=
  data.long_furred_brown + (data.total_dogs - data.brown_dogs - data.neither_long_furred_nor_brown)

/-- Theorem stating the number of dogs with long fur in the specific kennel scenario --/
theorem long_furred_dogs_count (data : KennelData) 
  (h1 : data.total_dogs = 45)
  (h2 : data.brown_dogs = 30)
  (h3 : data.neither_long_furred_nor_brown = 8)
  (h4 : data.long_furred_brown = 19) :
  long_furred_dogs data = 26 := by
  sorry

#eval long_furred_dogs { total_dogs := 45, brown_dogs := 30, neither_long_furred_nor_brown := 8, long_furred_brown := 19 }

end long_furred_dogs_count_l4130_413037


namespace total_cost_of_hats_l4130_413097

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of weeks John can wear a different hat each day -/
def weeks_of_different_hats : ℕ := 2

/-- The cost of each hat in dollars -/
def cost_per_hat : ℕ := 50

/-- Theorem: The total cost of John's hats is $700 -/
theorem total_cost_of_hats : 
  weeks_of_different_hats * days_per_week * cost_per_hat = 700 := by
  sorry

end total_cost_of_hats_l4130_413097


namespace log_difference_equals_negative_three_l4130_413044

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_difference_equals_negative_three :
  log10 4 - log10 4000 = -3 := by
  sorry

end log_difference_equals_negative_three_l4130_413044


namespace arrangements_with_A_not_first_is_48_l4130_413021

/-- The number of ways to arrange 3 people from 5, including A and B, with A not at the head -/
def arrangements_with_A_not_first (total_people : ℕ) (selected_people : ℕ) : ℕ :=
  (total_people * (total_people - 1) * (total_people - 2)) -
  ((total_people - 1) * (total_people - 2))

/-- Theorem stating that the number of arrangements with A not at the head is 48 -/
theorem arrangements_with_A_not_first_is_48 :
  arrangements_with_A_not_first 5 3 = 48 := by
  sorry

#eval arrangements_with_A_not_first 5 3

end arrangements_with_A_not_first_is_48_l4130_413021


namespace u_2002_equals_2_l4130_413071

def f (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 1
  | 3 => 3
  | 4 => 5
  | 5 => 2
  | _ => 0  -- Default case for completeness

def u : ℕ → ℕ
  | 0 => 4
  | n + 1 => f (u n)

theorem u_2002_equals_2 : u 2002 = 2 := by
  sorry

end u_2002_equals_2_l4130_413071


namespace quadratic_solution_average_l4130_413020

theorem quadratic_solution_average (c : ℝ) :
  c < 3 →  -- Condition for real and distinct solutions
  let equation := fun x : ℝ => 3 * x^2 - 6 * x + c
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation x₁ = 0 ∧ equation x₂ = 0 ∧ (x₁ + x₂) / 2 = 1 :=
by
  sorry


end quadratic_solution_average_l4130_413020


namespace quadrilateral_offset_l4130_413032

theorem quadrilateral_offset (diagonal : ℝ) (offset1 : ℝ) (offset2 : ℝ) (area : ℝ) :
  diagonal = 10 →
  offset1 = 3 →
  area = 50 →
  area = (1 / 2) * diagonal * offset1 + (1 / 2) * diagonal * offset2 →
  offset2 = 7 := by
  sorry

end quadrilateral_offset_l4130_413032


namespace tess_distance_graph_l4130_413053

-- Define the triangular block
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define Tess's position as a function of time
def tessPosition (t : ℝ) (tri : Triangle) : ℝ × ℝ :=
  sorry

-- Define the straight-line distance from A to Tess's position
def distanceFromA (t : ℝ) (tri : Triangle) : ℝ :=
  sorry

-- Define the properties of the distance function
def isRisingThenFalling (f : ℝ → ℝ) : Prop :=
  sorry

def peaksAtB (f : ℝ → ℝ) (tri : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem tess_distance_graph (tri : Triangle) :
  isRisingThenFalling (fun t => distanceFromA t tri) ∧
  peaksAtB (fun t => distanceFromA t tri) tri :=
sorry

end tess_distance_graph_l4130_413053


namespace marla_horse_purchase_l4130_413011

/-- The number of bottle caps equivalent to one lizard -/
def bottlecaps_per_lizard : ℕ := 8

/-- The number of lizards equivalent to 5 gallons of water -/
def lizards_per_five_gallons : ℕ := 3

/-- The number of gallons of water equivalent to one horse -/
def gallons_per_horse : ℕ := 80

/-- The number of bottle caps Marla can scavenge per day -/
def daily_scavenge : ℕ := 20

/-- The number of bottle caps Marla pays per night for food and shelter -/
def daily_expense : ℕ := 4

/-- The number of days it takes Marla to collect enough bottle caps to buy a horse -/
def days_to_buy_horse : ℕ := 24

theorem marla_horse_purchase :
  days_to_buy_horse * (daily_scavenge - daily_expense) =
  (gallons_per_horse * lizards_per_five_gallons * bottlecaps_per_lizard) / 5 :=
by sorry

end marla_horse_purchase_l4130_413011


namespace arccos_of_one_eq_zero_l4130_413082

theorem arccos_of_one_eq_zero : Real.arccos 1 = 0 := by sorry

end arccos_of_one_eq_zero_l4130_413082


namespace H_div_G_equals_two_l4130_413017

-- Define the equation as a function
def equation (G H x : ℝ) : Prop :=
  G / (x + 5) + H / (x^2 - 4*x) = (x^2 - 2*x + 10) / (x^3 + x^2 - 20*x)

-- Define the theorem
theorem H_div_G_equals_two :
  ∀ G H : ℤ,
  (∀ x : ℝ, x ≠ -5 ∧ x ≠ 0 ∧ x ≠ 4 → equation G H x) →
  (H : ℝ) / (G : ℝ) = 2 := by
  sorry


end H_div_G_equals_two_l4130_413017


namespace lcm_of_given_numbers_l4130_413087

theorem lcm_of_given_numbers : 
  Nat.lcm 24 (Nat.lcm 30 (Nat.lcm 40 (Nat.lcm 50 60))) = 600 := by sorry

end lcm_of_given_numbers_l4130_413087


namespace train_speed_problem_l4130_413050

/-- Proves that the speed of the faster train is 31.25 km/hr given the problem conditions. -/
theorem train_speed_problem (v : ℝ) (h1 : v > 25) : 
  v = 31.25 ∧ 
  ∃ (t : ℝ), t > 0 ∧ 
    v * t + 25 * t = 630 ∧ 
    v * t = 25 * t + 70 := by
  sorry

end train_speed_problem_l4130_413050


namespace point_A_final_position_l4130_413046

-- Define the initial position of point A
def initial_position : Set ℤ := {-5, 5}

-- Define the movement function
def move (start : ℤ) (left : ℤ) (right : ℤ) : ℤ := start - left + right

-- Theorem statement
theorem point_A_final_position :
  ∀ start ∈ initial_position,
  move start 2 6 = -1 ∨ move start 2 6 = 9 := by
sorry

end point_A_final_position_l4130_413046


namespace problem_solution_l4130_413066

def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a + 1)*x + (a^2 - 5) = 0}

theorem problem_solution :
  (∀ a : ℝ, A ∩ B a = {2} → a = -1 ∨ a = -3) ∧
  (∀ a : ℝ, A ∪ B a = A → a ≤ -3) := by
  sorry

end problem_solution_l4130_413066


namespace sum_of_circle_areas_is_14pi_l4130_413003

/-- Represents a circle with a center point and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Represents a right triangle with side lengths 3, 4, and 5 -/
structure RightTriangle where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  side_ab : Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 3
  side_bc : Real.sqrt ((b.1 - c.1)^2 + (b.2 - c.2)^2) = 4
  side_ca : Real.sqrt ((c.1 - a.1)^2 + (c.2 - a.2)^2) = 5
  right_angle : (a.1 - b.1) * (c.1 - b.1) + (a.2 - b.2) * (c.2 - b.2) = 0

/-- Checks if two circles are externally tangent -/
def areExternallyTangent (c1 c2 : Circle) : Prop :=
  Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2) = c1.radius + c2.radius

/-- Theorem: The sum of the areas of three mutually externally tangent circles
    centered at the vertices of a 3-4-5 right triangle is 14π -/
theorem sum_of_circle_areas_is_14pi (t : RightTriangle) 
    (c1 : Circle) (c2 : Circle) (c3 : Circle)
    (h1 : c1.center = t.a) (h2 : c2.center = t.b) (h3 : c3.center = t.c)
    (h4 : areExternallyTangent c1 c2)
    (h5 : areExternallyTangent c2 c3)
    (h6 : areExternallyTangent c3 c1) :
    π * (c1.radius^2 + c2.radius^2 + c3.radius^2) = 14 * π := by
  sorry


end sum_of_circle_areas_is_14pi_l4130_413003


namespace servant_payment_is_40_l4130_413041

/-- Calculates the cash payment to a servant who leaves early -/
def servant_cash_payment (total_yearly_salary : ℚ) (turban_price : ℚ) (months_worked : ℚ) : ℚ :=
  (total_yearly_salary * (months_worked / 12)) - turban_price

/-- Proof that the servant receives Rs. 40 in cash -/
theorem servant_payment_is_40 :
  let total_yearly_salary : ℚ := 200
  let turban_price : ℚ := 110
  let months_worked : ℚ := 9
  servant_cash_payment total_yearly_salary turban_price months_worked = 40 := by
sorry

#eval servant_cash_payment 200 110 9

end servant_payment_is_40_l4130_413041


namespace f_2011_value_l4130_413062

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem f_2011_value 
  (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_period : ∀ x, f (x + 2) = -f x) 
  (h_def : ∀ x ∈ Set.Ioo 0 2, f x = 2 * x^2) : 
  f 2011 = -2 := by
  sorry

end f_2011_value_l4130_413062


namespace log_simplification_l4130_413064

theorem log_simplification (p q r s t z : ℝ) (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hz : z > 0) :
  Real.log (p / q) + Real.log (q / r) + Real.log (r / s) - Real.log ((p * t) / (s * z)) = Real.log (z / t) := by
  sorry

end log_simplification_l4130_413064


namespace sum_of_squares_theorem_l4130_413014

theorem sum_of_squares_theorem (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 2)
  (h2 : a^2 / x^2 + b^2 / y^2 + c^2 / z^2 = 1) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 := by
  sorry

end sum_of_squares_theorem_l4130_413014


namespace milk_ratio_l4130_413035

/-- Given a cafeteria that sells two types of milk (regular and chocolate),
    this theorem proves the ratio of chocolate to regular milk cartons sold. -/
theorem milk_ratio (total : ℕ) (regular : ℕ) 
    (h1 : total = 24) 
    (h2 : regular = 3) : 
    (total - regular) / regular = 7 := by
  sorry

#check milk_ratio

end milk_ratio_l4130_413035


namespace pies_sold_in_week_l4130_413051

/-- The number of pies sold daily -/
def daily_sales : ℕ := 8

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pies sold in a week -/
def weekly_sales : ℕ := daily_sales * days_in_week

theorem pies_sold_in_week : weekly_sales = 56 := by
  sorry

end pies_sold_in_week_l4130_413051


namespace no_solution_when_p_is_seven_l4130_413033

theorem no_solution_when_p_is_seven (p : ℝ) : 
  (∀ x : ℝ, x ≠ 4 ∧ x ≠ 8 → (x - 3) / (x - 4) ≠ (x - p) / (x - 8)) ↔ p = 7 :=
by sorry

end no_solution_when_p_is_seven_l4130_413033


namespace weight_difference_l4130_413072

theorem weight_difference (w_a w_b w_c w_d w_e : ℝ) : 
  (w_a + w_b + w_c) / 3 = 84 →
  (w_a + w_b + w_c + w_d) / 4 = 80 →
  (w_b + w_c + w_d + w_e) / 4 = 79 →
  w_a = 80 →
  w_e > w_d →
  w_e - w_d = 8 := by
sorry


end weight_difference_l4130_413072


namespace solve_proportion_l4130_413092

theorem solve_proportion (y : ℝ) (h : 9 / (y^2) = y / 81) : y = 9 := by
  sorry

end solve_proportion_l4130_413092


namespace toucan_count_l4130_413000

/-- Given that there are initially 2 toucans on a tree limb and 1 more toucan joins them,
    prove that the total number of toucans is 3. -/
theorem toucan_count (initial : ℕ) (joined : ℕ) (h1 : initial = 2) (h2 : joined = 1) :
  initial + joined = 3 := by
  sorry

end toucan_count_l4130_413000


namespace quadratic_solution_difference_squared_zero_l4130_413091

theorem quadratic_solution_difference_squared_zero :
  ∀ a b : ℝ,
  (5 * a^2 - 30 * a + 45 = 0) →
  (5 * b^2 - 30 * b + 45 = 0) →
  (a - b)^2 = 0 := by
sorry

end quadratic_solution_difference_squared_zero_l4130_413091


namespace negation_of_root_existence_l4130_413025

theorem negation_of_root_existence :
  ¬(∀ a : ℝ, a > 0 → a ≠ 1 → ∃ x : ℝ, a^x - x - a = 0) ↔
  (∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, a^x - x - a ≠ 0) :=
by sorry

end negation_of_root_existence_l4130_413025


namespace fraction_simplification_l4130_413099

theorem fraction_simplification (a b : ℝ) (h : a ≠ b) : (2 * a) / (2 * b) = a / b := by
  sorry

end fraction_simplification_l4130_413099


namespace probability_x_gt_5y_l4130_413009

/-- The probability of selecting a point (x,y) from a rectangle with vertices
    (0,0), (2020,0), (2020,2021), and (0,2021) such that x > 5y is 101/1011. -/
theorem probability_x_gt_5y : 
  let rectangle_area := 2020 * 2021
  let triangle_area := (1 / 2) * 2020 * 404
  triangle_area / rectangle_area = 101 / 1011 := by
sorry

end probability_x_gt_5y_l4130_413009


namespace find_n_l4130_413084

theorem find_n (x y n : ℝ) (h1 : (7 * x + 2 * y) / (x - n * y) = 23) (h2 : x / (2 * y) = 3 / 2) : n = 2 := by
  sorry

end find_n_l4130_413084


namespace probability_score_3_points_l4130_413024

/-- The probability of hitting target A -/
def prob_hit_A : ℚ := 3/4

/-- The probability of hitting target B -/
def prob_hit_B : ℚ := 2/3

/-- The score for hitting target A -/
def score_hit_A : ℤ := 1

/-- The score for missing target A -/
def score_miss_A : ℤ := -1

/-- The score for hitting target B -/
def score_hit_B : ℤ := 2

/-- The score for missing target B -/
def score_miss_B : ℤ := 0

/-- The number of shots at target B -/
def shots_B : ℕ := 2

theorem probability_score_3_points : 
  (prob_hit_A * shots_B * prob_hit_B * (1 - prob_hit_B) + 
   (1 - prob_hit_A) * prob_hit_B^shots_B) = 4/9 := by
  sorry

end probability_score_3_points_l4130_413024


namespace millie_bracelets_l4130_413089

/-- The number of bracelets Millie had initially -/
def initial_bracelets : ℕ := 9

/-- The number of bracelets Millie lost -/
def lost_bracelets : ℕ := 2

/-- The number of bracelets Millie has left -/
def remaining_bracelets : ℕ := initial_bracelets - lost_bracelets

theorem millie_bracelets : remaining_bracelets = 7 := by
  sorry

end millie_bracelets_l4130_413089


namespace paper_completion_days_l4130_413006

theorem paper_completion_days (total_pages : ℕ) (pages_per_day : ℕ) (days : ℕ) : 
  total_pages = 81 → pages_per_day = 27 → days * pages_per_day = total_pages → days = 3 := by
  sorry

end paper_completion_days_l4130_413006


namespace triangle_shape_l4130_413095

theorem triangle_shape (a b c : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_eq : a^2 + 2*b^2 = 2*b*(a+c) - c^2) : a = b ∧ b = c := by
  sorry

end triangle_shape_l4130_413095


namespace siblings_ages_sum_l4130_413023

theorem siblings_ages_sum (x y z : ℕ+) 
  (h1 : y = x + 1)
  (h2 : x * y * z = 96) :
  x + y + z = 15 := by
sorry

end siblings_ages_sum_l4130_413023


namespace perpendicular_vectors_m_value_l4130_413085

/-- Two vectors in R² -/
def Vector2 := ℝ × ℝ

/-- Dot product of two vectors in R² -/
def dot_product (v w : Vector2) : ℝ :=
  v.1 * w.1 + v.2 * w.2

/-- Perpendicularity of two vectors in R² -/
def perpendicular (v w : Vector2) : Prop :=
  dot_product v w = 0

theorem perpendicular_vectors_m_value :
  ∀ m : ℝ,
  let a : Vector2 := (1, 2)
  let b : Vector2 := (m, 1)
  perpendicular a b → m = -2 := by
sorry

end perpendicular_vectors_m_value_l4130_413085


namespace M₄_is_mutually_orthogonal_l4130_413026

/-- A set M is a mutually orthogonal point set if for all (x₁, y₁) in M,
    there exists (x₂, y₂) in M such that x₁x₂ + y₁y₂ = 0 -/
def MutuallyOrthogonalPointSet (M : Set (ℝ × ℝ)) : Prop :=
  ∀ p₁ ∈ M, ∃ p₂ ∈ M, p₁.1 * p₂.1 + p₁.2 * p₂.2 = 0

/-- The set M₄ defined as {(x, y) | y = sin(x) + 1} -/
def M₄ : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = Real.sin p.1 + 1}

/-- Theorem stating that M₄ is a mutually orthogonal point set -/
theorem M₄_is_mutually_orthogonal : MutuallyOrthogonalPointSet M₄ := by
  sorry

end M₄_is_mutually_orthogonal_l4130_413026


namespace jude_matchbox_vehicles_l4130_413045

/-- Calculates the total number of matchbox vehicles Jude buys given the specified conditions -/
theorem jude_matchbox_vehicles :
  let car_cost : ℕ := 10
  let truck_cost : ℕ := 15
  let helicopter_cost : ℕ := 20
  let total_caps : ℕ := 250
  let trucks_bought : ℕ := 5
  let caps_spent_on_trucks : ℕ := trucks_bought * truck_cost
  let remaining_caps : ℕ := total_caps - caps_spent_on_trucks
  let caps_for_cars : ℕ := (remaining_caps * 60) / 100
  let cars_bought : ℕ := caps_for_cars / car_cost
  let caps_left : ℕ := remaining_caps - (cars_bought * car_cost)
  let helicopters_bought : ℕ := caps_left / helicopter_cost
  trucks_bought + cars_bought + helicopters_bought = 18 :=
by sorry

end jude_matchbox_vehicles_l4130_413045


namespace left_focus_coordinates_l4130_413015

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / 2 - y^2 / 2 = 1

/-- The left focus of the hyperbola -/
def left_focus : ℝ × ℝ := (-2, 0)

/-- Theorem: The coordinates of the left focus of the given hyperbola are (-2,0) -/
theorem left_focus_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → left_focus = (-2, 0) := by
  sorry

end left_focus_coordinates_l4130_413015


namespace cayley_competition_certificates_l4130_413090

theorem cayley_competition_certificates (boys girls : ℕ) 
  (boys_percent girls_percent : ℚ) (h1 : boys = 30) (h2 : girls = 20) 
  (h3 : boys_percent = 1/10) (h4 : girls_percent = 1/5) : 
  (boys_percent * boys + girls_percent * girls) / (boys + girls) = 7/50 := by
  sorry

end cayley_competition_certificates_l4130_413090


namespace function_symmetry_range_l4130_413042

open Real

theorem function_symmetry_range (a : ℝ) : 
  (∃ x ∈ Set.Icc (1/ℯ) ℯ, a + 8 * log x = x^2 + 2) ↔ 
  a ∈ Set.Icc (6 - 8 * log 2) (10 + 1 / ℯ^2) :=
sorry

end function_symmetry_range_l4130_413042


namespace inequality_solution_l4130_413029

theorem inequality_solution (x : ℝ) : 
  (x^2 - 9) / ((x - 3)^2) < 0 ↔ -3 < x ∧ x < 3 ∧ x ≠ 3 :=
by sorry

end inequality_solution_l4130_413029


namespace tangent_point_coordinates_l4130_413001

/-- The curve function f(x) = x^4 + x -/
def f (x : ℝ) : ℝ := x^4 + x

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 4 * x^3 + 1

theorem tangent_point_coordinates :
  ∃ (x y : ℝ), f y = f x ∧ f' x = -3 → x = -1 ∧ y = 0 := by
  sorry

end tangent_point_coordinates_l4130_413001


namespace meet_once_l4130_413028

/-- Represents the meeting of Michael and the garbage truck -/
structure MeetingProblem where
  michaelSpeed : ℝ
  truckSpeed : ℝ
  pailDistance : ℝ
  truckStopTime : ℝ
  initialDistance : ℝ

/-- Calculates the number of meetings between Michael and the truck -/
def numberOfMeetings (p : MeetingProblem) : ℕ :=
  sorry

/-- The specific problem instance -/
def problemInstance : MeetingProblem :=
  { michaelSpeed := 4
  , truckSpeed := 12
  , pailDistance := 300
  , truckStopTime := 40
  , initialDistance := 300 }

/-- Theorem stating that Michael and the truck meet exactly once -/
theorem meet_once : numberOfMeetings problemInstance = 1 := by
  sorry

end meet_once_l4130_413028


namespace four_balls_three_boxes_l4130_413098

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ways_to_put_balls_in_boxes (n : ℕ) (k : ℕ) : ℕ := sorry

/-- Theorem: There are 8 ways to put 4 distinguishable balls into 3 indistinguishable boxes -/
theorem four_balls_three_boxes : ways_to_put_balls_in_boxes 4 3 = 8 := by
  sorry

end four_balls_three_boxes_l4130_413098


namespace roots_quadratic_sum_l4130_413077

theorem roots_quadratic_sum (a b : ℝ) : 
  (a^2 + 3*a - 4 = 0) → 
  (b^2 + 3*b - 4 = 0) → 
  (a^2 + 4*a + b - 3 = -2) := by
  sorry

end roots_quadratic_sum_l4130_413077


namespace max_areas_formula_max_areas_for_n_3_l4130_413005

/-- Represents a circular disk divided by radii and secant lines -/
structure DividedDisk where
  n : ℕ
  radii_count : ℕ
  secant_lines : ℕ
  h_positive : n > 0
  h_radii : radii_count = 2 * n
  h_secants : secant_lines = 2

/-- Calculates the maximum number of non-overlapping areas in a divided disk -/
def max_areas (d : DividedDisk) : ℕ :=
  4 * d.n + 4

/-- Theorem stating the maximum number of non-overlapping areas -/
theorem max_areas_formula (d : DividedDisk) :
  max_areas d = 4 * d.n + 4 :=
by sorry

/-- Specific case for n = 3 -/
theorem max_areas_for_n_3 :
  ∃ (d : DividedDisk), d.n = 3 ∧ max_areas d = 16 :=
by sorry

end max_areas_formula_max_areas_for_n_3_l4130_413005


namespace b_time_approx_l4130_413012

/-- The time it takes for A to complete the work alone -/
def a_time : ℝ := 20

/-- The time it takes for A and B to complete the work together -/
def ab_time : ℝ := 12.727272727272728

/-- The time it takes for B to complete the work alone -/
noncomputable def b_time : ℝ := (a_time * ab_time) / (a_time - ab_time)

/-- Theorem stating that B can complete the work in approximately 34.90909090909091 days -/
theorem b_time_approx : 
  ∃ ε > 0, |b_time - 34.90909090909091| < ε :=
sorry

end b_time_approx_l4130_413012


namespace company_fund_problem_l4130_413078

/-- Proves that the initial amount in the company fund was $950 given the problem conditions --/
theorem company_fund_problem (n : ℕ) : 
  (60 * n - 10 = 50 * n + 150) → 
  (60 * n - 10 = 950) :=
by
  sorry

end company_fund_problem_l4130_413078


namespace girls_boys_seating_arrangements_l4130_413049

theorem girls_boys_seating_arrangements (n : ℕ) (h : n = 5) : 
  (n.factorial * n.factorial : ℕ) = 14400 := by
  sorry

end girls_boys_seating_arrangements_l4130_413049


namespace largest_prime_common_factor_l4130_413081

def is_largest_prime_common_factor (n : ℕ) : Prop :=
  n.Prime ∧
  n ∣ 462 ∧
  n ∣ 385 ∧
  ∀ m : ℕ, m.Prime → m ∣ 462 → m ∣ 385 → m ≤ n

theorem largest_prime_common_factor :
  is_largest_prime_common_factor 7 := by sorry

end largest_prime_common_factor_l4130_413081


namespace grandchildren_gender_probability_l4130_413058

theorem grandchildren_gender_probability :
  let n : ℕ := 12  -- total number of grandchildren
  let p : ℚ := 1/2  -- probability of a grandchild being male (or female)
  let equal_prob := (n.choose (n/2)) / 2^n  -- probability of equal number of grandsons and granddaughters
  1 - equal_prob = 793/1024 := by
  sorry

end grandchildren_gender_probability_l4130_413058


namespace hyperbola_properties_l4130_413007

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 3 - y^2 / 6 = 1

-- Define the asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Define the point the hyperbola passes through
def passes_through : Prop := hyperbola 3 (-2 * Real.sqrt 3)

-- Define the intersection line
def intersection_line (x y : ℝ) : Prop := y = Real.sqrt 3 * (x - 3)

-- State the theorem
theorem hyperbola_properties :
  (∀ x y, asymptotes x y → hyperbola x y) ∧
  passes_through ∧
  (∃ A B : ℝ × ℝ,
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    intersection_line A.1 A.2 ∧
    intersection_line B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 16 * Real.sqrt 3) :=
by sorry

end hyperbola_properties_l4130_413007


namespace parabola_circle_separation_l4130_413074

/-- The range of 'a' for a parabola y^2 = 4ax with directrix separate from the circle x^2 + y^2 - 2y = 0 -/
theorem parabola_circle_separation (a : ℝ) : 
  (∀ x y : ℝ, y^2 = 4*a*x → x^2 + y^2 - 2*y ≠ 0) →
  (∀ x y : ℝ, x = a → x^2 + y^2 - 2*y ≠ 0) →
  a > 1 ∨ a < -1 :=
sorry

end parabola_circle_separation_l4130_413074


namespace quadratic_function_minimum_value_l4130_413002

/-- A quadratic function f(x) = ax² + bx + c with a ≠ 0 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0

/-- The function f(x) defined by the quadratic function -/
def f (q : QuadraticFunction) (x : ℝ) : ℝ :=
  q.a * x^2 + q.b * x + q.c

/-- The derivative of f(x) -/
def f' (q : QuadraticFunction) (x : ℝ) : ℝ :=
  2 * q.a * x + q.b

theorem quadratic_function_minimum_value (q : QuadraticFunction)
  (h1 : f' q 0 > 0)
  (h2 : ∀ x : ℝ, f q x ≥ 0) :
  2 ≤ (f q 1) / (f' q 0) :=
sorry

end quadratic_function_minimum_value_l4130_413002


namespace function_property_l4130_413059

theorem function_property (f : ℝ → ℝ) (h : ¬(∀ x > 0, f x > 0)) : ∃ x > 0, f x ≤ 0 := by
  sorry

end function_property_l4130_413059


namespace smallest_valid_tournament_l4130_413008

/-- A tournament is valid if for any two players, there exists a third player who beat both of them -/
def is_valid_tournament (k : ℕ) (tournament : Fin k → Fin k → Bool) : Prop :=
  k > 1 ∧
  (∀ i j, i ≠ j → tournament i j = !tournament j i) ∧
  (∀ i j, i ≠ j → ∃ m, m ≠ i ∧ m ≠ j ∧ tournament m i ∧ tournament m j)

/-- The smallest k for which a valid tournament exists is 7 -/
theorem smallest_valid_tournament : 
  (∃ k : ℕ, ∃ tournament : Fin k → Fin k → Bool, is_valid_tournament k tournament) ∧
  (∀ k : ℕ, k < 7 → ¬∃ tournament : Fin k → Fin k → Bool, is_valid_tournament k tournament) :=
sorry

end smallest_valid_tournament_l4130_413008


namespace largest_five_digit_with_product_15120_l4130_413022

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

theorem largest_five_digit_with_product_15120 :
  ∀ n : ℕ, 
    10000 ≤ n ∧ n < 100000 ∧ digit_product n = 15120 →
    n ≤ 98754 :=
by
  sorry

end largest_five_digit_with_product_15120_l4130_413022


namespace combined_tennis_preference_l4130_413047

theorem combined_tennis_preference (east_total : ℕ) (west_total : ℕ) 
  (east_tennis_percent : ℚ) (west_tennis_percent : ℚ) :
  east_total = 2000 →
  west_total = 2500 →
  east_tennis_percent = 22 / 100 →
  west_tennis_percent = 40 / 100 →
  (east_total * east_tennis_percent + west_total * west_tennis_percent) / 
  (east_total + west_total) = 32 / 100 :=
by sorry

end combined_tennis_preference_l4130_413047


namespace value_of_x_l4130_413080

theorem value_of_x : ∀ w y z x : ℤ,
  w = 90 →
  z = w + 15 →
  y = z - 3 →
  x = y + 7 →
  x = 109 := by
sorry

end value_of_x_l4130_413080


namespace lincoln_county_houses_l4130_413016

def original_houses : ℕ := 20817
def new_houses : ℕ := 97741

theorem lincoln_county_houses : original_houses + new_houses = 118558 := by
  sorry

end lincoln_county_houses_l4130_413016


namespace correct_quadratic_equation_l4130_413057

theorem correct_quadratic_equation 
  (b c : ℝ) 
  (h1 : 5 + 1 = -b) 
  (h2 : (-6) * (-4) = c) : 
  b = -10 ∧ c = 6 := by
sorry

end correct_quadratic_equation_l4130_413057


namespace bikers_meeting_time_l4130_413068

def biker1_time : ℕ := 12
def biker2_time : ℕ := 18
def biker3_time : ℕ := 24

theorem bikers_meeting_time :
  Nat.lcm (Nat.lcm biker1_time biker2_time) biker3_time = 72 := by
  sorry

end bikers_meeting_time_l4130_413068


namespace arithmetic_sequence_sum_l4130_413076

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 = 2 →
  a 2 + a 3 = 13 →
  a 4 + a 5 + a 6 = 42 := by
  sorry

end arithmetic_sequence_sum_l4130_413076


namespace sum_interior_angles_pentagon_l4130_413048

/-- The sum of interior angles of a polygon with n sides -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The sum of the interior angles of a pentagon is 540° -/
theorem sum_interior_angles_pentagon :
  sum_interior_angles pentagon_sides = 540 := by sorry

end sum_interior_angles_pentagon_l4130_413048


namespace max_leftover_candy_l4130_413039

theorem max_leftover_candy (x : ℕ) : ∃ (q r : ℕ), x = 10 * q + r ∧ r < 10 ∧ r ≤ 9 :=
sorry

end max_leftover_candy_l4130_413039


namespace range_of_x_l4130_413093

theorem range_of_x (a b c x : ℝ) : 
  a^2 + 2*b^2 + 3*c^2 = 6 →
  a + 2*b + 3*c > |x + 1| →
  -7 < x ∧ x < 5 :=
by sorry

end range_of_x_l4130_413093


namespace inscribed_square_area_ratio_l4130_413070

/-- The ratio of areas between an inscribed square and a larger square -/
theorem inscribed_square_area_ratio :
  let large_square_side : ℝ := 4
  let inscribed_square_horizontal_offset : ℝ := 1.5
  let inscribed_square_vertical_offset : ℝ := 4/3
  let inscribed_square_side : ℝ := large_square_side - 2 * inscribed_square_horizontal_offset
  let large_square_area : ℝ := large_square_side ^ 2
  let inscribed_square_area : ℝ := inscribed_square_side ^ 2
  inscribed_square_area / large_square_area = 1 / 16 := by
  sorry

end inscribed_square_area_ratio_l4130_413070


namespace cubic_equation_coefficient_l4130_413040

theorem cubic_equation_coefficient (a b : ℝ) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + 1 = (a * x - 1) * (x^2 - x - 1)) → 
  b = -2 := by
sorry

end cubic_equation_coefficient_l4130_413040


namespace squares_in_figure_50_l4130_413010

/-- The function representing the number of squares in the nth figure -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- Theorem stating that the 50th figure has 7651 squares -/
theorem squares_in_figure_50 :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37 → f 50 = 7651 := by
  sorry

#eval f 50  -- This will evaluate f(50) and should output 7651

end squares_in_figure_50_l4130_413010


namespace bankers_gain_example_l4130_413013

/-- Calculates the banker's gain given present worth, interest rate, and time period. -/
def bankers_gain (present_worth : ℝ) (interest_rate : ℝ) (years : ℕ) : ℝ :=
  present_worth * (1 + interest_rate) ^ years - present_worth

/-- Theorem stating that the banker's gain is 126 given the specific conditions. -/
theorem bankers_gain_example : bankers_gain 600 0.1 2 = 126 := by
  sorry

end bankers_gain_example_l4130_413013


namespace chocolate_distribution_l4130_413004

/-- Proves that each friend receives 24/7 pounds of chocolate given the initial conditions -/
theorem chocolate_distribution (total : ℚ) (initial_piles : ℕ) (friends : ℕ) : 
  total = 60 / 7 →
  initial_piles = 5 →
  friends = 2 →
  (total - (total / initial_piles)) / friends = 24 / 7 := by
sorry

#eval (60 / 7 : ℚ)
#eval (24 / 7 : ℚ)

end chocolate_distribution_l4130_413004


namespace tax_percentage_calculation_l4130_413054

theorem tax_percentage_calculation (initial_bars : ℕ) (remaining_bars : ℕ) : 
  initial_bars = 60 →
  remaining_bars = 27 →
  ∃ (tax_percentage : ℚ),
    tax_percentage = 10 ∧
    remaining_bars = (initial_bars * (1 - tax_percentage / 100) / 2).floor :=
by sorry

end tax_percentage_calculation_l4130_413054


namespace parabola_properties_l4130_413034

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 4 * a * x - 5 * a

theorem parabola_properties :
  ∀ a : ℝ, a ≠ 0 →
  -- 1. Intersections with x-axis
  (∃ x : ℝ, parabola a x = 0 ↔ x = -1 ∨ x = 5) ∧
  -- 2. Conditions for a = 1
  (a > 0 → (∀ m n : ℝ, parabola a m = n → m ≥ 0 → n ≥ -9) → a = 1 ∧ 
    ∀ x : ℝ, parabola 1 x = x^2 - 4*x - 5) ∧
  -- 3. Range of m for shifted parabola
  (∀ m : ℝ, m > 0 → 
    (∃ t : ℝ, -1/2 < t ∧ t < 5/2 ∧ parabola 1 t + m = 0) →
    11/4 < m ∧ m ≤ 9) :=
by sorry

end parabola_properties_l4130_413034


namespace no_prime_power_solution_l4130_413043

theorem no_prime_power_solution : 
  ¬ ∃ (p : ℕ) (x : ℕ) (k : ℕ), 
    Nat.Prime p ∧ x^5 + 2*x + 3 = p^k :=
sorry

end no_prime_power_solution_l4130_413043


namespace polynomial_factorization_l4130_413073

theorem polynomial_factorization (a : ℝ) : a^2 - 5*a - 6 = (a - 6) * (a + 1) := by
  sorry

end polynomial_factorization_l4130_413073


namespace different_color_probability_l4130_413052

def total_balls : ℕ := 5
def red_balls : ℕ := 3
def yellow_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem different_color_probability :
  (Nat.choose red_balls 1 * Nat.choose yellow_balls 1) / Nat.choose total_balls drawn_balls = 3 / 5 := by
  sorry

end different_color_probability_l4130_413052


namespace highest_score_is_174_l4130_413019

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  totalInnings : ℕ
  overallAverage : ℚ
  scoreDifference : ℕ
  averageExcludingExtremes : ℚ

/-- Calculates the highest score of a batsman given their statistics -/
def highestScore (stats : BatsmanStats) : ℚ :=
  let totalRuns := stats.overallAverage * stats.totalInnings
  let runsExcludingExtremes := stats.averageExcludingExtremes * (stats.totalInnings - 2)
  let sumExtremes := totalRuns - runsExcludingExtremes
  (sumExtremes + stats.scoreDifference) / 2

/-- Theorem stating that the highest score is 174 for the given statistics -/
theorem highest_score_is_174 (stats : BatsmanStats)
  (h1 : stats.totalInnings = 46)
  (h2 : stats.overallAverage = 60)
  (h3 : stats.scoreDifference = 140)
  (h4 : stats.averageExcludingExtremes = 58) :
  highestScore stats = 174 := by
  sorry

#eval highestScore {
  totalInnings := 46,
  overallAverage := 60,
  scoreDifference := 140,
  averageExcludingExtremes := 58
}

end highest_score_is_174_l4130_413019


namespace unique_solution_is_zero_l4130_413055

theorem unique_solution_is_zero :
  ∃! y : ℝ, y = 3 * (1 / y * (-y)) + 3 :=
by sorry

end unique_solution_is_zero_l4130_413055


namespace julia_played_with_33_kids_l4130_413027

/-- The number of kids Julia played with on Monday and Tuesday combined -/
def total_kids_monday_tuesday (monday : ℕ) (tuesday : ℕ) : ℕ :=
  monday + tuesday

/-- Proof that Julia played with 33 kids on Monday and Tuesday combined -/
theorem julia_played_with_33_kids : 
  total_kids_monday_tuesday 15 18 = 33 := by
  sorry

end julia_played_with_33_kids_l4130_413027


namespace sequence_floor_representation_l4130_413065

theorem sequence_floor_representation (a : Fin 1999 → ℕ) 
  (h : ∀ i j : Fin 1999, i + j < 1999 → a i + a 1 ≤ a (i + j) ∧ a (i + j) ≤ a i + a j + 1) :
  ∃ x : ℝ, ∀ n : Fin 1999, a n = ⌊n * x⌋ := by sorry

end sequence_floor_representation_l4130_413065


namespace odd_function_positive_range_l4130_413060

open Set

def isOdd (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem odd_function_positive_range
  (f : ℝ → ℝ)
  (hf_odd : isOdd f)
  (hf_neg_one : f (-1) = 0)
  (hf_deriv : ∀ x > 0, x * (deriv^[2] f x) - deriv f x > 0) :
  {x : ℝ | f x > 0} = Ioo (-1) 0 ∪ Ioi 1 := by sorry

end odd_function_positive_range_l4130_413060


namespace circle_equation_with_given_conditions_l4130_413094

/-- A circle with center (h, k) and radius r has the standard equation (x - h)² + (y - k)² = r² -/
def is_standard_circle_equation (h k r : ℝ) (f : ℝ → ℝ → Prop) :=
  ∀ x y, f x y ↔ (x - h)^2 + (y - k)^2 = r^2

/-- A point (x, y) lies on the line 2x - y = 3 -/
def lies_on_line (x y : ℝ) : Prop := 2*x - y = 3

/-- A circle is tangent to the x-axis if its distance to the x-axis equals its radius -/
def tangent_to_x_axis (h k r : ℝ) : Prop := |k| = r

/-- A circle is tangent to the y-axis if its distance to the y-axis equals its radius -/
def tangent_to_y_axis (h k r : ℝ) : Prop := |h| = r

theorem circle_equation_with_given_conditions :
  ∃ f : ℝ → ℝ → Prop,
    (∃ h k r : ℝ, 
      is_standard_circle_equation h k r f ∧
      lies_on_line h k ∧
      tangent_to_x_axis h k r ∧
      tangent_to_y_axis h k r) →
    (∀ x y, f x y ↔ ((x - 3)^2 + (y - 3)^2 = 9 ∨ (x - 1)^2 + (y + 1)^2 = 1)) :=
by sorry

end circle_equation_with_given_conditions_l4130_413094


namespace average_salary_of_all_employees_l4130_413088

/-- Calculates the average salary of all employees in an office -/
theorem average_salary_of_all_employees 
  (officer_salary : ℝ) 
  (non_officer_salary : ℝ) 
  (num_officers : ℕ) 
  (num_non_officers : ℕ) 
  (h1 : officer_salary = 420)
  (h2 : non_officer_salary = 110)
  (h3 : num_officers = 15)
  (h4 : num_non_officers = 450) :
  (officer_salary * num_officers + non_officer_salary * num_non_officers) / (num_officers + num_non_officers) = 120 :=
by
  sorry

#check average_salary_of_all_employees

end average_salary_of_all_employees_l4130_413088


namespace stephanie_store_visits_l4130_413075

/-- The number of times Stephanie went to the store last month -/
def store_visits : ℕ := 16 / 2

/-- The number of oranges Stephanie buys each time she goes to the store -/
def oranges_per_visit : ℕ := 2

/-- The total number of oranges Stephanie bought last month -/
def total_oranges : ℕ := 16

theorem stephanie_store_visits : store_visits = 8 :=
by
  sorry

end stephanie_store_visits_l4130_413075


namespace baseball_card_value_decrease_l4130_413056

theorem baseball_card_value_decrease : 
  ∀ (initial_value : ℝ), initial_value > 0 →
  let first_year_value := initial_value * (1 - 0.5)
  let second_year_value := first_year_value * (1 - 0.1)
  let total_decrease := (initial_value - second_year_value) / initial_value
  total_decrease = 0.55 := by
sorry

end baseball_card_value_decrease_l4130_413056


namespace max_x_minus_y_value_l4130_413030

theorem max_x_minus_y_value (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) :
  ∃ (max : ℝ), max = 2 / Real.sqrt 3 ∧ ∀ (a b : ℝ), 3 * (a^2 + b^2) = a + b → a - b ≤ max :=
sorry

end max_x_minus_y_value_l4130_413030


namespace largest_divisor_of_odd_product_l4130_413036

theorem largest_divisor_of_odd_product (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ (m : ℕ), m ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → m ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end largest_divisor_of_odd_product_l4130_413036


namespace no_regular_lattice_polygon_except_square_l4130_413067

/-- A lattice point in 2D space -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- A regular n-gon with vertices at lattice points -/
structure RegularLatticePolygon where
  n : ℕ
  vertices : Fin n → LatticePoint

/-- Predicate to check if a set of points forms a regular n-gon -/
def IsRegularPolygon (poly : RegularLatticePolygon) : Prop :=
  ∀ i j : Fin poly.n,
    (poly.vertices i).x ^ 2 + (poly.vertices i).y ^ 2 =
    (poly.vertices j).x ^ 2 + (poly.vertices j).y ^ 2

/-- Main theorem: No regular n-gon with vertices at lattice points exists for n ≠ 4 -/
theorem no_regular_lattice_polygon_except_square :
  ∀ n : ℕ, n ≠ 4 → ¬∃ (poly : RegularLatticePolygon), poly.n = n ∧ IsRegularPolygon poly :=
sorry

end no_regular_lattice_polygon_except_square_l4130_413067


namespace paint_calculation_l4130_413083

theorem paint_calculation (P : ℝ) 
  (h1 : (1/3 : ℝ) * P + (1/5 : ℝ) * (2/3 : ℝ) * P = 168) : 
  P = 360 :=
sorry

end paint_calculation_l4130_413083


namespace deanna_speed_l4130_413086

/-- Proves that given the conditions of Deanna's trip, her speed in the first 30 minutes was 90 km/h -/
theorem deanna_speed (v : ℝ) : 
  (v * (1/2) + (v + 20) * (1/2) = 100) → 
  v = 90 := by
  sorry

end deanna_speed_l4130_413086


namespace cornelia_triple_kilee_age_l4130_413069

/-- The number of years in the future when Cornelia will be three times as old as Kilee -/
def future_years : ℕ := 10

/-- Kilee's current age -/
def kilee_age : ℕ := 20

/-- Cornelia's current age -/
def cornelia_age : ℕ := 80

/-- Theorem stating that in 'future_years' years, Cornelia will be three times as old as Kilee -/
theorem cornelia_triple_kilee_age :
  cornelia_age + future_years = 3 * (kilee_age + future_years) :=
by sorry

end cornelia_triple_kilee_age_l4130_413069


namespace circle_radius_c_value_l4130_413061

theorem circle_radius_c_value :
  ∀ (c : ℝ),
  (∀ (x y : ℝ), x^2 + 8*x + y^2 + 2*y + c = 0 ↔ (x + 4)^2 + (y + 1)^2 = 36) →
  c = -19 := by
sorry

end circle_radius_c_value_l4130_413061


namespace chess_tournament_games_l4130_413031

/-- The number of games played in a chess tournament --/
def num_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a chess tournament with 30 players, where each player plays every other player exactly once,
    the total number of games played is 435. --/
theorem chess_tournament_games :
  num_games 30 = 435 := by
  sorry

#eval num_games 30  -- This will evaluate to 435

end chess_tournament_games_l4130_413031


namespace divisible_difference_exists_l4130_413063

theorem divisible_difference_exists (n : ℕ) (a : Fin (n + 1) → ℤ) :
  ∃ i j : Fin (n + 1), i ≠ j ∧ (n : ℤ) ∣ (a i - a j) := by
  sorry

end divisible_difference_exists_l4130_413063
