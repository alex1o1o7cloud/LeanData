import Mathlib

namespace hexagon_side_length_l2163_216395

/-- A regular hexagon with a point inside it -/
structure RegularHexagonWithPoint where
  /-- Side length of the hexagon -/
  side_length : ℝ
  /-- The point inside the hexagon -/
  point : ℝ × ℝ
  /-- First vertex of the hexagon -/
  vertex1 : ℝ × ℝ
  /-- Second vertex of the hexagon -/
  vertex2 : ℝ × ℝ
  /-- Third vertex of the hexagon -/
  vertex3 : ℝ × ℝ
  /-- The hexagon is regular -/
  regular : side_length > 0
  /-- The distance between the point and the first vertex is 1 -/
  dist1 : Real.sqrt ((point.1 - vertex1.1)^2 + (point.2 - vertex1.2)^2) = 1
  /-- The distance between the point and the second vertex is 1 -/
  dist2 : Real.sqrt ((point.1 - vertex2.1)^2 + (point.2 - vertex2.2)^2) = 1
  /-- The distance between the point and the third vertex is 2 -/
  dist3 : Real.sqrt ((point.1 - vertex3.1)^2 + (point.2 - vertex3.2)^2) = 2
  /-- The vertices are consecutive -/
  consecutive : Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = side_length ∧
                Real.sqrt ((vertex2.1 - vertex3.1)^2 + (vertex2.2 - vertex3.2)^2) = side_length

/-- The theorem stating that the side length of the hexagon is √3 -/
theorem hexagon_side_length (h : RegularHexagonWithPoint) : h.side_length = Real.sqrt 3 := by
  sorry

end hexagon_side_length_l2163_216395


namespace hyperbola_eccentricity_l2163_216386

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_positive : a > 0 ∧ b > 0

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the foci of a hyperbola -/
structure Foci where
  F₁ : Point
  F₂ : Point

/-- Checks if a point is on the hyperbola -/
def on_hyperbola (h : Hyperbola) (p : Point) : Prop :=
  (p.x^2 / h.a^2) - (p.y^2 / h.b^2) = 1

/-- Calculates the angle between three points -/
noncomputable def angle (p₁ p₂ p₃ : Point) : ℝ := sorry

/-- Calculates the eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := sorry

/-- Main theorem -/
theorem hyperbola_eccentricity (h : Hyperbola) (f : Foci) (p : Point) :
  on_hyperbola h p →
  angle f.F₁ p f.F₂ = Real.pi / 2 →
  2 * angle p f.F₁ f.F₂ = angle p f.F₂ f.F₁ →
  eccentricity h = Real.sqrt 3 + 1 := by sorry

end hyperbola_eccentricity_l2163_216386


namespace apples_in_refrigerator_l2163_216369

def initial_apples : ℕ := 62
def pie_apples : ℕ := initial_apples / 2
def muffin_apples : ℕ := 6

def refrigerator_apples : ℕ := initial_apples - pie_apples - muffin_apples

theorem apples_in_refrigerator : refrigerator_apples = 25 := by
  sorry

end apples_in_refrigerator_l2163_216369


namespace dangerous_animals_count_l2163_216393

/-- The number of crocodiles pointed out by the teacher -/
def num_crocodiles : ℕ := 22

/-- The number of alligators pointed out by the teacher -/
def num_alligators : ℕ := 23

/-- The number of vipers pointed out by the teacher -/
def num_vipers : ℕ := 5

/-- The total number of dangerous animals pointed out by the teacher -/
def total_dangerous_animals : ℕ := num_crocodiles + num_alligators + num_vipers

theorem dangerous_animals_count : total_dangerous_animals = 50 := by
  sorry

end dangerous_animals_count_l2163_216393


namespace bank_depositors_bound_l2163_216319

theorem bank_depositors_bound (total_deposits : ℝ) (probability_less_100 : ℝ) 
  (h_total : total_deposits = 20000)
  (h_prob : probability_less_100 = 0.8) :
  ∃ n : ℕ, n ≤ 1000 ∧ (total_deposits / n : ℝ) ≤ 100 / (1 - probability_less_100) := by
  sorry

end bank_depositors_bound_l2163_216319


namespace sufficient_not_necessary_condition_l2163_216329

theorem sufficient_not_necessary_condition (a b : ℝ) :
  (a^2 + b^2 = 0 → a * b = 0) ∧
  ∃ a b : ℝ, a * b = 0 ∧ a^2 + b^2 ≠ 0 :=
by sorry

end sufficient_not_necessary_condition_l2163_216329


namespace equation_solution_l2163_216335

theorem equation_solution (x y : ℝ) (h1 : x ≠ 0) (h2 : 2*x + y ≠ 0) 
  (h3 : (x + y) / x = y / (2*x + y)) : x = -y/2 := by
  sorry

end equation_solution_l2163_216335


namespace cos_105_degrees_l2163_216311

theorem cos_105_degrees : Real.cos (105 * π / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end cos_105_degrees_l2163_216311


namespace smallest_n_divisible_by_p_iff_p_minus_one_l2163_216388

theorem smallest_n_divisible_by_p_iff_p_minus_one : ∃ (n : ℕ), n = 1806 ∧
  (∀ (p : ℕ), Nat.Prime p → (p ∣ n ↔ (p - 1) ∣ n)) ∧
  (∀ (m : ℕ), m < n → ∃ (q : ℕ), Nat.Prime q ∧ ((q ∣ m ↔ (q - 1) ∣ m) → False)) :=
by sorry

end smallest_n_divisible_by_p_iff_p_minus_one_l2163_216388


namespace sqrt_sum_product_equals_twenty_l2163_216378

theorem sqrt_sum_product_equals_twenty : (Real.sqrt 8 + Real.sqrt (1/2)) * Real.sqrt 32 = 20 := by
  sorry

end sqrt_sum_product_equals_twenty_l2163_216378


namespace pythagorean_numbers_l2163_216324

theorem pythagorean_numbers (m : ℕ) (a b c : ℝ) : 
  m % 2 = 1 → 
  m > 1 → 
  a = (1/2 : ℝ) * m^2 - (1/2 : ℝ) → 
  c = (1/2 : ℝ) * m^2 + (1/2 : ℝ) → 
  a < c → 
  b < c → 
  a^2 + b^2 = c^2 → 
  b = m := by sorry

end pythagorean_numbers_l2163_216324


namespace y_equals_zero_l2163_216351

theorem y_equals_zero (x y : ℝ) : (x + y)^5 - x^5 + y = 0 → y = 0 := by
  sorry

end y_equals_zero_l2163_216351


namespace lemonade_recipe_l2163_216398

/-- Lemonade recipe problem -/
theorem lemonade_recipe (lemon_juice sugar water : ℚ) : 
  water = 3 * sugar →  -- Water is 3 times sugar
  sugar = 3 * lemon_juice →  -- Sugar is 3 times lemon juice
  lemon_juice = 4 →  -- Luka uses 4 cups of lemon juice
  water = 36 := by  -- The amount of water needed is 36 cups
sorry


end lemonade_recipe_l2163_216398


namespace soccer_ball_contribution_l2163_216339

theorem soccer_ball_contribution (k l m : ℝ) : 
  k ≥ 0 → l ≥ 0 → m ≥ 0 →
  k + l + m = 6 →
  2 * k ≤ l + m →
  2 * l ≤ k + m →
  2 * m ≤ k + l →
  k = 2 ∧ l = 2 ∧ m = 2 := by
sorry

end soccer_ball_contribution_l2163_216339


namespace at_least_one_hit_probability_l2163_216384

theorem at_least_one_hit_probability 
  (prob_A prob_B prob_C : ℝ) 
  (h_A : prob_A = 0.7) 
  (h_B : prob_B = 0.5) 
  (h_C : prob_C = 0.4) 
  (h_independent : True) -- Assumption of independence
  : 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C) = 0.91 := by
  sorry

end at_least_one_hit_probability_l2163_216384


namespace arithmetic_sequence_a20_l2163_216312

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a20 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 1 + a 3 + a 5 = 105 →
  a 2 + a 4 + a 6 = 99 →
  a 20 = 1 := by
sorry

end arithmetic_sequence_a20_l2163_216312


namespace snooker_ticket_difference_l2163_216382

theorem snooker_ticket_difference :
  ∀ (vip_price general_price : ℚ) 
    (total_tickets : ℕ) 
    (total_cost : ℚ) 
    (vip_count general_count : ℕ),
  vip_price = 45 →
  general_price = 20 →
  total_tickets = 320 →
  total_cost = 7500 →
  vip_count + general_count = total_tickets →
  vip_price * vip_count + general_price * general_count = total_cost →
  general_count - vip_count = 232 := by
sorry

end snooker_ticket_difference_l2163_216382


namespace sqrt_4_4_times_9_2_l2163_216322

theorem sqrt_4_4_times_9_2 : Real.sqrt (4^4 * 9^2) = 144 := by
  sorry

end sqrt_4_4_times_9_2_l2163_216322


namespace sum_of_roots_squared_diff_eq_sum_of_roots_eq_fourteen_l2163_216367

theorem sum_of_roots_squared_diff_eq (a c : ℝ) : 
  (∀ x : ℝ, (x - a)^2 = c) → (∃ x₁ x₂ : ℝ, (x₁ - a)^2 = c ∧ (x₂ - a)^2 = c ∧ x₁ + x₂ = 2 * a) :=
by sorry

theorem sum_of_roots_eq_fourteen : 
  (∃ x₁ x₂ : ℝ, (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 ∧ x₁ + x₂ = 14) :=
by sorry

end sum_of_roots_squared_diff_eq_sum_of_roots_eq_fourteen_l2163_216367


namespace semicircle_overlap_width_l2163_216308

/-- Given a rectangle with two semicircles drawn inside, where each semicircle
    has a radius of 5 cm and the rectangle height is 8 cm, the width of the
    overlap between the semicircles is 6 cm. -/
theorem semicircle_overlap_width (r : ℝ) (h : ℝ) (w : ℝ) :
  r = 5 →
  h = 8 →
  w = 2 * Real.sqrt (r^2 - (h/2)^2) →
  w = 6 := by
  sorry

#check semicircle_overlap_width

end semicircle_overlap_width_l2163_216308


namespace family_age_problem_l2163_216383

theorem family_age_problem (family_size : ℕ) (current_average_age : ℝ) (youngest_age : ℝ) :
  family_size = 5 →
  current_average_age = 20 →
  youngest_age = 10 →
  let total_age : ℝ := family_size * current_average_age
  let other_members_age : ℝ := total_age - youngest_age
  let age_reduction : ℝ := (family_size - 1) * youngest_age
  let total_age_at_birth : ℝ := other_members_age - age_reduction
  let average_age_at_birth : ℝ := total_age_at_birth / family_size
  average_age_at_birth = 10 := by
sorry

end family_age_problem_l2163_216383


namespace kendra_shirts_l2163_216360

/-- Represents the number of shirts Kendra needs for a two-week period --/
def shirts_needed : ℕ :=
  let weekday_shirts := 5
  let club_shirts := 3
  let saturday_shirt := 1
  let sunday_shirts := 2
  let weekly_shirts := weekday_shirts + club_shirts + saturday_shirt + sunday_shirts
  2 * weekly_shirts

/-- Theorem stating that Kendra needs 22 shirts for a two-week period --/
theorem kendra_shirts : shirts_needed = 22 := by
  sorry

end kendra_shirts_l2163_216360


namespace parking_probability_l2163_216385

/-- Represents a parking lot -/
structure ParkingLot where
  totalSpaces : ℕ
  occupiedSpaces : ℕ

/-- Calculates the probability of finding a specified number of adjacent empty spaces -/
def probabilityOfAdjacentEmptySpaces (lot : ParkingLot) (requiredSpaces : ℕ) : ℚ :=
  sorry

theorem parking_probability (lot : ParkingLot) :
  lot.totalSpaces = 20 →
  lot.occupiedSpaces = 14 →
  probabilityOfAdjacentEmptySpaces lot 3 = 19/25 :=
by sorry

end parking_probability_l2163_216385


namespace binomial_coefficient_15_l2163_216389

theorem binomial_coefficient_15 (n : ℕ) (h1 : n > 0) 
  (h2 : Nat.choose n 2 = 15) : n = 6 := by
  sorry

end binomial_coefficient_15_l2163_216389


namespace count_valid_parallelograms_l2163_216301

/-- A point in the coordinate plane with integer coordinates -/
structure IntPoint where
  x : ℤ
  y : ℤ

/-- A parallelogram defined by four points in the coordinate plane -/
structure Parallelogram where
  O : IntPoint
  A : IntPoint
  B : IntPoint
  C : IntPoint

/-- The center of a parallelogram -/
def center (p : Parallelogram) : IntPoint :=
  { x := (p.O.x + p.B.x) / 2,
    y := (p.O.y + p.B.y) / 2 }

/-- Check if a parallelogram satisfies the given conditions -/
def validParallelogram (p : Parallelogram) : Prop :=
  p.O = { x := 0, y := 0 } ∧
  center p = { x := 19, y := 15 } ∧
  p.A.x > 0 ∧ p.A.y > 0 ∧
  p.B.x > 0 ∧ p.B.y > 0 ∧
  p.C.x > 0 ∧ p.C.y > 0

/-- Two parallelograms are considered equivalent if they have the same set of vertices -/
def equivalentParallelograms (p1 p2 : Parallelogram) : Prop :=
  (p1.O = p2.O ∧ p1.A = p2.A ∧ p1.B = p2.B ∧ p1.C = p2.C) ∨
  (p1.O = p2.O ∧ p1.A = p2.C ∧ p1.B = p2.B ∧ p1.C = p2.A)

theorem count_valid_parallelograms :
  ∃ (s : Finset Parallelogram),
    (∀ p ∈ s, validParallelogram p) ∧
    (∀ p, validParallelogram p → ∃ q ∈ s, equivalentParallelograms p q) ∧
    s.card = 126 :=
sorry

end count_valid_parallelograms_l2163_216301


namespace trig_identity_l2163_216364

open Real

theorem trig_identity (α : ℝ) : 
  (1 / sin (-α) - sin (π + α)) / (1 / cos (3*π - α) + cos (2*π - α)) = 1 / tan α^3 := by
  sorry

end trig_identity_l2163_216364


namespace farmer_duck_sales_l2163_216358

/-- A farmer sells ducks and chickens, buys a wheelbarrow, and resells it. -/
theorem farmer_duck_sales
  (duck_price : ℕ)
  (chicken_price : ℕ)
  (chicken_count : ℕ)
  (duck_count : ℕ)
  (wheelbarrow_profit : ℕ)
  (h1 : duck_price = 10)
  (h2 : chicken_price = 8)
  (h3 : chicken_count = 5)
  (h4 : wheelbarrow_profit = 60)
  (h5 : (duck_price * duck_count + chicken_price * chicken_count) / 2 = wheelbarrow_profit / 2) :
  duck_count = 2 := by
sorry


end farmer_duck_sales_l2163_216358


namespace upper_limit_of_set_D_l2163_216397

def is_prime (n : ℕ) : Prop := sorry

def set_D (upper_bound : ℕ) : Set ℕ :=
  {n : ℕ | 10 < n ∧ n ≤ upper_bound ∧ is_prime n}

theorem upper_limit_of_set_D (upper_bound : ℕ) :
  (∃ (a b : ℕ), a ∈ set_D upper_bound ∧ b ∈ set_D upper_bound ∧ b - a = 12) →
  (∃ (max : ℕ), max ∈ set_D upper_bound ∧ ∀ (x : ℕ), x ∈ set_D upper_bound → x ≤ max) →
  (∃ (max : ℕ), max ∈ set_D upper_bound ∧ ∀ (x : ℕ), x ∈ set_D upper_bound → x ≤ max ∧ max = 23) :=
by sorry

end upper_limit_of_set_D_l2163_216397


namespace min_value_fraction_l2163_216368

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (∀ x y : ℝ, 0 < x → 0 < y → x + y = 2 → 1/a + a/(8*b) ≤ 1/x + x/(8*y)) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + y = 2 ∧ 1/x + x/(8*y) = 1) :=
sorry

end min_value_fraction_l2163_216368


namespace number_problem_l2163_216325

theorem number_problem : 
  ∃ x : ℚ, (30 / 100 : ℚ) * x = (25 / 100 : ℚ) * 40 ∧ x = 100 / 3 := by
  sorry

end number_problem_l2163_216325


namespace nested_bracket_value_l2163_216313

def bracket (a b c : ℚ) : ℚ :=
  if c ≠ 0 then (a + b) / c else 0

theorem nested_bracket_value :
  bracket (bracket 30 45 75) (bracket 4 2 6) (bracket 12 18 30) = 2 :=
by sorry

end nested_bracket_value_l2163_216313


namespace line_through_point_l2163_216343

/-- A line contains a point if the point's coordinates satisfy the line's equation. -/
def line_contains_point (m : ℚ) (x y : ℚ) : Prop :=
  2 - m * x = -4 * y

/-- The theorem states that the line 2 - mx = -4y contains the point (5, -2) when m = -6/5. -/
theorem line_through_point (m : ℚ) :
  line_contains_point m 5 (-2) ↔ m = -6/5 := by
  sorry

end line_through_point_l2163_216343


namespace cylinder_height_relationship_l2163_216348

theorem cylinder_height_relationship (r₁ h₁ r₂ h₂ : ℝ) :
  r₁ > 0 ∧ h₁ > 0 ∧ r₂ > 0 ∧ h₂ > 0 →
  r₂ = 1.2 * r₁ →
  π * r₁^2 * h₁ = π * r₂^2 * h₂ →
  h₁ = 1.44 * h₂ := by
sorry

end cylinder_height_relationship_l2163_216348


namespace ellipse_existence_and_uniqueness_l2163_216300

/-- A structure representing a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A structure representing a line in a 2D plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A structure representing an ellipse in a 2D plane -/
structure Ellipse where
  center : Point
  semiMajorAxis : ℝ
  semiMinorAxis : ℝ
  rotation : ℝ

/-- Function to check if two lines are perpendicular -/
def arePerpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Function to check if a point lies on an ellipse -/
def pointOnEllipse (p : Point) (e : Ellipse) : Prop :=
  sorry

/-- Function to check if an ellipse has its axes on given lines -/
def ellipseAxesOnLines (e : Ellipse) (l1 l2 : Line) : Prop :=
  sorry

/-- Theorem stating the existence and uniqueness of ellipses -/
theorem ellipse_existence_and_uniqueness 
  (l1 l2 : Line) (p1 p2 : Point) 
  (h_perp : arePerpendicular l1 l2) :
  (p1 ≠ p2 → ∃! e : Ellipse, pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ ellipseAxesOnLines e l1 l2) ∧
  (p1 = p2 → ∃ e : Ellipse, pointOnEllipse p1 e ∧ pointOnEllipse p2 e ∧ ellipseAxesOnLines e l1 l2) :=
sorry

end ellipse_existence_and_uniqueness_l2163_216300


namespace factor_of_100140001_l2163_216352

theorem factor_of_100140001 : ∃ (n : ℕ), 
  8000 < n ∧ 
  n < 9000 ∧ 
  100140001 % n = 0 :=
by
  use 8221
  sorry

end factor_of_100140001_l2163_216352


namespace possible_a_values_l2163_216387

theorem possible_a_values (a : ℝ) : 
  (∃ x ∈ Set.Icc 0 5, x^2 - 6*x + 2 - a > 0) →
  (a = 0 ∨ a = 1) :=
by sorry

end possible_a_values_l2163_216387


namespace boxes_with_neither_l2163_216375

/-- Given a set of boxes with markers and stickers, calculate the number of boxes
    containing neither markers nor stickers. -/
theorem boxes_with_neither (total : ℕ) (markers : ℕ) (stickers : ℕ) (both : ℕ)
    (h_total : total = 15)
    (h_markers : markers = 9)
    (h_stickers : stickers = 5)
    (h_both : both = 4) :
    total - (markers + stickers - both) = 5 := by
  sorry

end boxes_with_neither_l2163_216375


namespace intersection_of_M_and_complement_of_N_l2163_216347

-- Define the universal set U
def U : Set ℕ := {0, 1, 2, 4, 6, 8}

-- Define set M
def M : Set ℕ := {0, 4, 6}

-- Define set N
def N : Set ℕ := {0, 1, 6}

-- Theorem statement
theorem intersection_of_M_and_complement_of_N :
  M ∩ (U \ N) = {4} := by sorry

end intersection_of_M_and_complement_of_N_l2163_216347


namespace integral_sqrt_minus_2x_l2163_216346

theorem integral_sqrt_minus_2x : 
  ∫ x in (0:ℝ)..1, (Real.sqrt (1 - (x - 1)^2) - 2*x) = π/4 - 1 := by sorry

end integral_sqrt_minus_2x_l2163_216346


namespace notebook_cost_l2163_216394

/-- The cost of a purchase given the number of notebooks, number of pencils, and total paid -/
def purchase_cost (notebooks : ℕ) (pencils : ℕ) (total_paid : ℚ) : ℚ := total_paid

/-- The theorem stating the cost of each notebook -/
theorem notebook_cost :
  ∀ (notebook_price pencil_price : ℚ),
    purchase_cost 5 4 20 - 3.5 = 5 * notebook_price + 4 * pencil_price →
    purchase_cost 2 2 7 = 2 * notebook_price + 2 * pencil_price →
    notebook_price = 2.5 := by
  sorry

end notebook_cost_l2163_216394


namespace aluminum_carbonate_weight_l2163_216357

/-- The atomic weight of Aluminum in g/mol -/
def Al_weight : ℝ := 26.98

/-- The atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.01

/-- The atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 16.00

/-- The molecular formula of aluminum carbonate -/
structure AluminumCarbonate where
  Al : Fin 2
  CO3 : Fin 3

/-- Calculate the molecular weight of aluminum carbonate -/
def molecular_weight (ac : AluminumCarbonate) : ℝ :=
  2 * Al_weight + 3 * C_weight + 9 * O_weight

/-- Theorem: The molecular weight of aluminum carbonate is 233.99 g/mol -/
theorem aluminum_carbonate_weight :
  ∀ ac : AluminumCarbonate, molecular_weight ac = 233.99 := by
  sorry

end aluminum_carbonate_weight_l2163_216357


namespace sector_central_angle_l2163_216306

/-- Theorem: Given a circular sector with arc length 3 and radius 2, the central angle is 3/2 radians. -/
theorem sector_central_angle (l : ℝ) (r : ℝ) (θ : ℝ) 
  (hl : l = 3) (hr : r = 2) (hθ : l = r * θ) : θ = 3/2 := by
  sorry

end sector_central_angle_l2163_216306


namespace archies_sod_area_l2163_216340

/-- Calculates the area of sod needed for a rectangular backyard with a rectangular shed. -/
def area_of_sod_needed (backyard_length backyard_width shed_length shed_width : ℝ) : ℝ :=
  backyard_length * backyard_width - shed_length * shed_width

/-- Theorem: The area of sod needed for Archie's backyard is 245 square yards. -/
theorem archies_sod_area :
  area_of_sod_needed 20 13 3 5 = 245 := by
  sorry

end archies_sod_area_l2163_216340


namespace smallest_multiple_of_90_with_128_divisors_l2163_216320

-- Define the number of divisors function
def num_divisors (n : ℕ) : ℕ := sorry

-- Define the property of being a multiple of 90
def is_multiple_of_90 (n : ℕ) : Prop := ∃ k : ℕ, n = 90 * k

-- Define the main theorem
theorem smallest_multiple_of_90_with_128_divisors :
  ∃ n : ℕ, 
    (∀ m : ℕ, m < n → ¬(is_multiple_of_90 m ∧ num_divisors m = 128)) ∧
    is_multiple_of_90 n ∧
    num_divisors n = 128 ∧
    n / 90 = 1728 := by sorry

end smallest_multiple_of_90_with_128_divisors_l2163_216320


namespace otimes_self_otimes_self_l2163_216307

/-- Definition of the ⊗ operation -/
def otimes (x y : ℝ) : ℝ := x^3 + x^2 - y

/-- Theorem: For any real number a, a ⊗ (a ⊗ a) = a -/
theorem otimes_self_otimes_self (a : ℝ) : otimes a (otimes a a) = a := by
  sorry

end otimes_self_otimes_self_l2163_216307


namespace unique_solution_range_l2163_216310

theorem unique_solution_range (a : ℝ) : 
  (∃! x : ℝ, 1 < x ∧ x < 3 ∧ Real.log (x - 1) + Real.log (3 - x) = Real.log (x - a)) ↔ 
  (3/4 ≤ a ∧ a < 3) :=
sorry

end unique_solution_range_l2163_216310


namespace gcd_property_l2163_216336

theorem gcd_property (a : ℕ) (h : ∀ n : ℤ, (Int.gcd (a * n + 1) (2 * n + 1) = 1)) :
  (∀ n : ℤ, Int.gcd (a - 2) (2 * n + 1) = 1) ∧
  (a = 1 ∨ ∃ m : ℕ, a = 2 + 2^m) := by
  sorry

end gcd_property_l2163_216336


namespace regular_octagon_interior_angle_l2163_216371

/-- The number of sides in an octagon -/
def octagon_sides : ℕ := 8

/-- The sum of interior angles of a polygon with n sides -/
def interior_angle_sum (n : ℕ) : ℝ := 180 * (n - 2)

/-- Theorem: Each interior angle of a regular octagon measures 135 degrees -/
theorem regular_octagon_interior_angle :
  (interior_angle_sum octagon_sides) / octagon_sides = 135 := by
  sorry

end regular_octagon_interior_angle_l2163_216371


namespace virginia_friends_l2163_216315

/-- The number of friends Virginia gave Sweettarts to -/
def num_friends (total : ℕ) (per_person : ℕ) : ℕ :=
  (total / per_person) - 1

/-- Proof that Virginia gave Sweettarts to 3 friends -/
theorem virginia_friends :
  num_friends 13 3 = 3 :=
by sorry

end virginia_friends_l2163_216315


namespace watermelon_pricing_l2163_216361

/-- Represents the number of watermelons each brother brought --/
structure Watermelons :=
  (elder : ℕ)
  (second : ℕ)
  (youngest : ℕ)

/-- Represents the number of watermelons sold in the morning --/
structure MorningSales :=
  (elder : ℕ)
  (second : ℕ)
  (youngest : ℕ)

/-- Theorem: Given the conditions, prove that the morning price was 3.75 yuan and the afternoon price was 1.25 yuan --/
theorem watermelon_pricing
  (w : Watermelons)
  (m : MorningSales)
  (h1 : w.elder = 10)
  (h2 : w.second = 16)
  (h3 : w.youngest = 26)
  (h4 : m.elder ≤ w.elder)
  (h5 : m.second ≤ w.second)
  (h6 : m.youngest ≤ w.youngest)
  (h7 : ∃ (morning_price afternoon_price : ℚ),
    morning_price > afternoon_price ∧
    afternoon_price > 0 ∧
    morning_price * m.elder + afternoon_price * (w.elder - m.elder) = 35 ∧
    morning_price * m.second + afternoon_price * (w.second - m.second) = 35 ∧
    morning_price * m.youngest + afternoon_price * (w.youngest - m.youngest) = 35) :
  ∃ (morning_price afternoon_price : ℚ),
    morning_price = 3.75 ∧ afternoon_price = 1.25 := by
  sorry

end watermelon_pricing_l2163_216361


namespace log_xy_value_l2163_216344

theorem log_xy_value (x y : ℝ) (h1 : Real.log (x * y^2) = 2) (h2 : Real.log (x^3 * y) = 3) :
  Real.log (x * y) = 7/5 := by
  sorry

end log_xy_value_l2163_216344


namespace james_annual_training_hours_l2163_216374

/-- Represents James' training schedule and calculates his total training hours in a year --/
def jamesTrainingHours : ℕ :=
  let weeklyHours : ℕ := 3 * 2 * 4 + 2 * (3 + 5)  -- Weekly training hours
  let totalWeeks : ℕ := 52  -- Weeks in a year
  let holidayWeeks : ℕ := 1  -- Week off for holidays
  let missedDays : ℕ := 10  -- Additional missed days
  let trainingDaysPerWeek : ℕ := 5  -- Number of training days per week
  let effectiveTrainingWeeks : ℕ := totalWeeks - holidayWeeks - (missedDays / trainingDaysPerWeek)
  weeklyHours * effectiveTrainingWeeks

/-- Theorem stating that James trains for 1960 hours in a year --/
theorem james_annual_training_hours :
  jamesTrainingHours = 1960 := by
  sorry

end james_annual_training_hours_l2163_216374


namespace bakery_chairs_l2163_216345

/-- The number of chairs in a bakery -/
def total_chairs (indoor_tables outdoor_tables chairs_per_indoor_table chairs_per_outdoor_table : ℕ) : ℕ :=
  indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table

/-- Proof that the total number of chairs in the bakery is 60 -/
theorem bakery_chairs :
  total_chairs 8 12 3 3 = 60 := by
  sorry

end bakery_chairs_l2163_216345


namespace dress_price_problem_l2163_216338

/-- 
Given a dress with an original price x, if Barb buys it for (x/2 - 10) dollars 
and saves 80 dollars, then x = 140.
-/
theorem dress_price_problem (x : ℝ) 
  (h1 : x - (x / 2 - 10) = 80) : x = 140 := by
  sorry

end dress_price_problem_l2163_216338


namespace f_local_min_at_neg_one_f_two_extrema_iff_l2163_216373

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x * (x - a * Real.exp x)

-- Theorem 1: When a = 0, f has a local minimum at x = -1
theorem f_local_min_at_neg_one :
  ∃ δ > 0, ∀ x, |x - (-1)| < δ ∧ x ≠ -1 → f 0 x > f 0 (-1) :=
sorry

-- Theorem 2: f has two different extremum points iff 0 < a < 1/2
theorem f_two_extrema_iff (a : ℝ) :
  (∃ x₁ x₂, x₁ < x₂ ∧ 
    (∀ h, 0 < h → f a (x₁ - h) > f a x₁ ∧ f a (x₁ + h) > f a x₁) ∧
    (∀ h, 0 < h → f a (x₂ - h) < f a x₂ ∧ f a (x₂ + h) < f a x₂))
  ↔ 0 < a ∧ a < 1/2 :=
sorry

end

end f_local_min_at_neg_one_f_two_extrema_iff_l2163_216373


namespace rectangular_to_spherical_conversion_l2163_216341

/-- Conversion from rectangular to spherical coordinates -/
theorem rectangular_to_spherical_conversion
  (x y z : ℝ)
  (h_x : x = 3 * Real.sqrt 2)
  (h_y : y = -3)
  (h_z : z = 5)
  (h_rho_pos : 0 < Real.sqrt (x^2 + y^2 + z^2))
  (h_theta_range : 0 ≤ 2 * Real.pi - Real.arctan (1 / Real.sqrt 2) ∧ 
                   2 * Real.pi - Real.arctan (1 / Real.sqrt 2) < 2 * Real.pi)
  (h_phi_range : 0 ≤ Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2)) ∧ 
                 Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2)) ≤ Real.pi) :
  (Real.sqrt (x^2 + y^2 + z^2),
   2 * Real.pi - Real.arctan (1 / Real.sqrt 2),
   Real.arccos (z / Real.sqrt (x^2 + y^2 + z^2))) =
  (Real.sqrt 52, 2 * Real.pi - Real.arctan (1 / Real.sqrt 2), Real.arccos (5 / Real.sqrt 52)) := by
  sorry

#check rectangular_to_spherical_conversion

end rectangular_to_spherical_conversion_l2163_216341


namespace intersection_M_N_l2163_216380

def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {-3, -2, -1, 0, 1}

theorem intersection_M_N :
  M ∩ N = {-2, -1, 0} := by sorry

end intersection_M_N_l2163_216380


namespace july_husband_age_l2163_216328

def hannah_age_then : ℕ := 6
def years_passed : ℕ := 20

theorem july_husband_age :
  ∀ (july_age_then : ℕ),
  hannah_age_then = 2 * july_age_then →
  (july_age_then + years_passed + 2 = 25) :=
by
  sorry

end july_husband_age_l2163_216328


namespace certain_number_value_l2163_216372

theorem certain_number_value : ∃ x : ℝ, 0.65 * 40 = (4/5) * x + 6 ∧ x = 25 := by
  sorry

end certain_number_value_l2163_216372


namespace is_circle_center_l2163_216304

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 4*y - 4 = 0

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (1, 2)

/-- Theorem stating that the given point is the center of the circle -/
theorem is_circle_center :
  ∀ (x y : ℝ), circle_equation x y ↔ (x - circle_center.1)^2 + (y - circle_center.2)^2 = 9 :=
by sorry

end is_circle_center_l2163_216304


namespace distance_to_center_squared_l2163_216309

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the square of the distance between two points -/
def distanceSquared (p1 p2 : Point) : ℝ :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Theorem: The square of the distance from B to the center of the circle is 50 -/
theorem distance_to_center_squared (O A B C : Point) : 
  O.x = 0 ∧ O.y = 0 →  -- Center at origin
  distanceSquared O A = 100 →  -- A is on the circle
  distanceSquared O C = 100 →  -- C is on the circle
  distanceSquared A B = 64 →  -- AB = 8
  distanceSquared B C = 9 →  -- BC = 3
  (B.x - A.x) * (C.y - B.y) = (B.y - A.y) * (C.x - B.x) →  -- ABC is a right angle
  distanceSquared O B = 50 := by
  sorry


end distance_to_center_squared_l2163_216309


namespace inequality_theorem_l2163_216355

theorem inequality_theorem (x₁ x₂ y₁ y₂ z₁ z₂ : ℝ) 
  (hx₁ : x₁ > 0) (hx₂ : x₂ > 0) 
  (hy₁ : y₁ > 0) (hy₂ : y₂ > 0)
  (hz₁ : x₁ * y₁ - z₁^2 > 0) (hz₂ : x₂ * y₂ - z₂^2 > 0) :
  ((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2)⁻¹ ≤ (x₁ * y₁ - z₁^2)⁻¹ + (x₂ * y₂ - z₂^2)⁻¹ ∧
  (((x₁ + x₂) * (y₁ + y₂) - (z₁ + z₂)^2)⁻¹ = (x₁ * y₁ - z₁^2)⁻¹ + (x₂ * y₂ - z₂^2)⁻¹ ↔ 
   x₁ = x₂ ∧ y₁ = y₂ ∧ z₁ = z₂) :=
by sorry

end inequality_theorem_l2163_216355


namespace max_absolute_value_of_z_l2163_216321

theorem max_absolute_value_of_z (a b c z : ℂ) 
  (h1 : Complex.abs a = Complex.abs b)
  (h2 : Complex.abs b = Complex.abs c)
  (h3 : Complex.abs a > 0)
  (h4 : a * z^2 + b * z + c = 0) :
  Complex.abs z ≤ (1 + Real.sqrt 5) / 2 := by sorry

end max_absolute_value_of_z_l2163_216321


namespace quadratic_function_properties_l2163_216392

def f (x : ℝ) := -2 * x^2 + 12 * x - 10

theorem quadratic_function_properties :
  f 1 = 0 ∧ f 5 = 0 ∧ f 3 = 8 := by
  sorry

end quadratic_function_properties_l2163_216392


namespace min_sum_of_squares_l2163_216303

def S : Finset Int := {-7, -5, -3, -2, 2, 4, 6, 13}

theorem min_sum_of_squares (a b c d e f g h : Int) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧
                b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧
                c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧
                d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧
                e ≠ f ∧ e ≠ g ∧ e ≠ h ∧
                f ≠ g ∧ f ≠ h ∧
                g ≠ h)
  (h_in_S : a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ e ∈ S ∧ f ∈ S ∧ g ∈ S ∧ h ∈ S) :
  (a + b + c + d)^2 + (e + f + g + h)^2 ≥ 34 ∧ 
  ∃ (a' b' c' d' e' f' g' h' : Int), 
    (a' ∈ S ∧ b' ∈ S ∧ c' ∈ S ∧ d' ∈ S ∧ e' ∈ S ∧ f' ∈ S ∧ g' ∈ S ∧ h' ∈ S) ∧
    (a' ≠ b' ∧ a' ≠ c' ∧ a' ≠ d' ∧ a' ≠ e' ∧ a' ≠ f' ∧ a' ≠ g' ∧ a' ≠ h' ∧
     b' ≠ c' ∧ b' ≠ d' ∧ b' ≠ e' ∧ b' ≠ f' ∧ b' ≠ g' ∧ b' ≠ h' ∧
     c' ≠ d' ∧ c' ≠ e' ∧ c' ≠ f' ∧ c' ≠ g' ∧ c' ≠ h' ∧
     d' ≠ e' ∧ d' ≠ f' ∧ d' ≠ g' ∧ d' ≠ h' ∧
     e' ≠ f' ∧ e' ≠ g' ∧ e' ≠ h' ∧
     f' ≠ g' ∧ f' ≠ h' ∧
     g' ≠ h') ∧
    (a' + b' + c' + d')^2 + (e' + f' + g' + h')^2 = 34 :=
by sorry

end min_sum_of_squares_l2163_216303


namespace compare_values_l2163_216342

theorem compare_values : 
  let a := (4 : ℝ) ^ (1/4 : ℝ)
  let b := (27 : ℝ) ^ (1/3 : ℝ)
  let c := (16 : ℝ) ^ (1/8 : ℝ)
  let d := (81 : ℝ) ^ (1/2 : ℝ)
  (d > a ∧ d > b ∧ d > c) ∧ 
  (b > a ∧ b > c) :=
by sorry

end compare_values_l2163_216342


namespace infinite_geometric_series_first_term_l2163_216318

/-- For an infinite geometric series with common ratio 1/4 and sum 80, the first term is 60. -/
theorem infinite_geometric_series_first_term : 
  ∀ (a : ℝ), 
  (∑' n, a * (1/4)^n) = 80 → 
  a = 60 := by
sorry

end infinite_geometric_series_first_term_l2163_216318


namespace honey_production_l2163_216305

theorem honey_production (bees : ℕ) (days : ℕ) (honey_per_bee : ℝ) :
  bees = 70 → days = 70 → honey_per_bee = 1 →
  bees * honey_per_bee = 70 := by
sorry

end honey_production_l2163_216305


namespace function_equality_l2163_216302

theorem function_equality (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = x + f (f y)) : 
  ∀ x : ℝ, f x = x := by
sorry

end function_equality_l2163_216302


namespace cafeteria_pie_problem_l2163_216350

/-- Given a cafeteria with initial apples, apples handed out, and number of pies made,
    calculate the number of apples used for each pie. -/
def apples_per_pie (initial_apples : ℕ) (apples_handed_out : ℕ) (num_pies : ℕ) : ℕ :=
  (initial_apples - apples_handed_out) / num_pies

/-- Theorem stating that given 47 initial apples, 27 apples handed out, and 5 pies made,
    the number of apples used for each pie is 4. -/
theorem cafeteria_pie_problem :
  apples_per_pie 47 27 5 = 4 := by
  sorry

end cafeteria_pie_problem_l2163_216350


namespace initial_songs_count_l2163_216370

/-- 
Given an album where:
- Each song is 3 minutes long
- Adding 10 more songs will make the total listening time 105 minutes
Prove that the initial number of songs in the album is 25.
-/
theorem initial_songs_count (song_duration : ℕ) (additional_songs : ℕ) (total_duration : ℕ) :
  song_duration = 3 →
  additional_songs = 10 →
  total_duration = 105 →
  ∃ (initial_songs : ℕ), song_duration * (initial_songs + additional_songs) = total_duration ∧ initial_songs = 25 :=
by sorry

end initial_songs_count_l2163_216370


namespace max_candy_count_l2163_216365

/-- The number of candy pieces Frankie got -/
def frankies_candy : ℕ := 74

/-- The additional candy pieces Max got compared to Frankie -/
def extra_candy : ℕ := 18

/-- The number of candy pieces Max got -/
def maxs_candy : ℕ := frankies_candy + extra_candy

theorem max_candy_count : maxs_candy = 92 := by
  sorry

end max_candy_count_l2163_216365


namespace postman_speed_calculation_postman_speed_is_30_l2163_216399

/-- Calculates the downhill average speed of a postman's round trip, given the following conditions:
  * The route length is 5 miles each way
  * The uphill delivery takes 2 hours
  * The uphill average speed is 4 miles per hour
  * The overall average speed for the round trip is 6 miles per hour
  * There's an extra 15 minutes (0.25 hours) delay on the return trip due to rain
-/
theorem postman_speed_calculation (route_length : ℝ) (uphill_time : ℝ) (uphill_speed : ℝ) 
  (overall_speed : ℝ) (rain_delay : ℝ) : ℝ :=
  let downhill_speed := 
    route_length / (((2 * route_length) / overall_speed) - uphill_time - rain_delay)
  30

/-- The main theorem that proves the downhill speed is 30 mph given the specific conditions -/
theorem postman_speed_is_30 : 
  postman_speed_calculation 5 2 4 6 0.25 = 30 := by
  sorry

end postman_speed_calculation_postman_speed_is_30_l2163_216399


namespace properties_dependency_l2163_216390

-- Define a type for geometric figures
inductive GeometricFigure
| Square
| Rectangle

-- Define properties for geometric figures
def hasEqualSides (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => true
  | GeometricFigure.Rectangle => true

def hasRightAngles (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => true
  | GeometricFigure.Rectangle => true

-- Define dependency of properties
def arePropertiesDependent (f : GeometricFigure) : Prop :=
  match f with
  | GeometricFigure.Square => hasEqualSides f ↔ hasRightAngles f
  | GeometricFigure.Rectangle => ¬(hasEqualSides f ↔ hasRightAngles f)

-- Theorem statement
theorem properties_dependency :
  arePropertiesDependent GeometricFigure.Square ∧
  ¬(arePropertiesDependent GeometricFigure.Rectangle) :=
sorry

end properties_dependency_l2163_216390


namespace count_subsets_correct_l2163_216337

/-- Given a natural number n, this function returns the number of two-tuples (X, Y) 
    of subsets of {1, 2, ..., n} such that max X > min Y -/
def count_subsets (n : ℕ) : ℕ := 
  2^(2*n) - (n+1) * 2^n

/-- Theorem stating that count_subsets gives the correct number of two-tuples -/
theorem count_subsets_correct (n : ℕ) : 
  count_subsets n = (Finset.powerset (Finset.range n)).card * 
                    (Finset.powerset (Finset.range n)).card - 
                    (Finset.filter (fun p : Finset ℕ × Finset ℕ => 
                      p.1.max ≤ p.2.min) 
                      ((Finset.powerset (Finset.range n)).product 
                       (Finset.powerset (Finset.range n)))).card :=
  sorry

#eval count_subsets 3  -- Example usage

end count_subsets_correct_l2163_216337


namespace quadratic_equation_roots_l2163_216323

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 + m * x = -2 ∧ x = 2) →
  (∃ y : ℝ, 3 * y^2 + m * y = -2 ∧ y = 1/3) :=
by sorry

end quadratic_equation_roots_l2163_216323


namespace residue_of_12_pow_2040_mod_19_l2163_216396

theorem residue_of_12_pow_2040_mod_19 :
  (12 : ℤ) ^ 2040 ≡ 7 [ZMOD 19] := by
  sorry

end residue_of_12_pow_2040_mod_19_l2163_216396


namespace equal_digit_probability_l2163_216379

/-- The number of sides on each die -/
def num_sides : ℕ := 20

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The number of one-digit outcomes on a die -/
def one_digit_outcomes : ℕ := 9

/-- The number of two-digit outcomes on a die -/
def two_digit_outcomes : ℕ := 11

/-- The probability of rolling an equal number of one-digit and two-digit numbers with 5 20-sided dice -/
theorem equal_digit_probability : 
  (Nat.choose num_dice (num_dice / 2) *
   (one_digit_outcomes ^ (num_dice / 2) * two_digit_outcomes ^ (num_dice - num_dice / 2))) /
  (num_sides ^ num_dice) = 539055 / 1600000 := by
sorry

end equal_digit_probability_l2163_216379


namespace wheels_in_garage_l2163_216362

theorem wheels_in_garage : 
  let bicycles : ℕ := 9
  let cars : ℕ := 16
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_car : ℕ := 4
  bicycles * wheels_per_bicycle + cars * wheels_per_car = 82 :=
by sorry

end wheels_in_garage_l2163_216362


namespace probability_no_red_square_l2163_216366

/-- Represents a coloring of a 4-by-4 grid -/
def Coloring := Fin 4 → Fin 4 → Bool

/-- Returns true if the coloring contains a 2-by-2 square of red squares -/
def has_red_square (c : Coloring) : Bool :=
  ∃ i j, i < 3 ∧ j < 3 ∧ 
    c i j ∧ c i (j+1) ∧ c (i+1) j ∧ c (i+1) (j+1)

/-- The probability of a square being red -/
def p_red : ℚ := 1/2

/-- The total number of possible colorings -/
def total_colorings : ℕ := 2^16

/-- The number of colorings without a 2-by-2 red square -/
def valid_colorings : ℕ := 40512

theorem probability_no_red_square :
  (valid_colorings : ℚ) / total_colorings = 315 / 512 :=
sorry

end probability_no_red_square_l2163_216366


namespace sqrt_product_simplification_l2163_216331

theorem sqrt_product_simplification (x : ℝ) (hx : x > 0) :
  Real.sqrt (48 * x) * Real.sqrt (3 * x) * Real.sqrt (50 * x) = 60 * x * Real.sqrt (2 * x) :=
by sorry

end sqrt_product_simplification_l2163_216331


namespace new_person_weight_l2163_216359

/-- 
Given a group of 8 people where one person weighing 65 kg is replaced by a new person,
and the average weight of the group increases by 2.5 kg, prove that the weight of the new person is 85 kg.
-/
theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  leaving_weight = 65 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + leaving_weight = 85 :=
by
  sorry

end new_person_weight_l2163_216359


namespace min_value_theorem_l2163_216381

-- Define the quadratic inequality solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x, (a * x^2 + 2 * x + b > 0) ↔ (x ≠ -1/a)

-- Define the theorem
theorem min_value_theorem (a b : ℝ) (h1 : solution_set a b) (h2 : a > b) :
  ∃ min_val : ℝ, min_val = 2 * Real.sqrt 2 ∧
  ∀ x : ℝ, (a^2 + b^2) / (a - b) ≥ min_val :=
by sorry

end min_value_theorem_l2163_216381


namespace shortest_line_on_square_pyramid_l2163_216333

/-- The shortest line on the lateral faces of a square pyramid -/
theorem shortest_line_on_square_pyramid (a m : ℝ) (ha : a > 0) (hm : m > 0) (h_eq : a = m) :
  let x := Real.sqrt (2 * a^2)
  let m₁ := Real.sqrt (x^2 - (a/2)^2)
  2 * a * m₁ / x = 80 * Real.sqrt (5/6) :=
by sorry

end shortest_line_on_square_pyramid_l2163_216333


namespace closest_to_95_l2163_216349

def options : List ℝ := [90, 92, 95, 98, 100]

theorem closest_to_95 :
  let product := 2.1 * (45.5 - 0.25)
  ∀ x ∈ options, |product - 95| ≤ |product - x| :=
by
  sorry

end closest_to_95_l2163_216349


namespace quadratic_inequality_l2163_216353

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_inequality (a b c : ℝ) (h1 : a > 0) 
  (h2 : ∀ x : ℝ, f a b c (2 + x) = f a b c (2 - x)) :
  f a b c 2 < f a b c 1 ∧ f a b c 1 < f a b c 4 := by
  sorry

end quadratic_inequality_l2163_216353


namespace n_to_b_equals_eight_l2163_216316

theorem n_to_b_equals_eight :
  let n : ℝ := 2 ^ (1/4)
  let b : ℝ := 12.000000000000002
  n ^ b = 8 := by
sorry

end n_to_b_equals_eight_l2163_216316


namespace project_duration_l2163_216363

theorem project_duration (x : ℝ) : 
  (1 / (x - 6) = 1.4 * (1 / x)) → x = 21 :=
by
  sorry

end project_duration_l2163_216363


namespace geometric_progression_ratio_l2163_216376

theorem geometric_progression_ratio (x y z r : ℝ) 
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h2 : x * (2 * y - z) ≠ y * (2 * z - x))
  (h3 : y * (2 * z - x) ≠ z * (2 * x - y))
  (h4 : x * (2 * y - z) ≠ z * (2 * x - y))
  (h5 : ∃ (a : ℝ), a ≠ 0 ∧ 
    x * (2 * y - z) = a ∧ 
    y * (2 * z - x) = a * r ∧ 
    z * (2 * x - y) = a * r^2) :
  r^2 + r + 1 = 0 :=
sorry

end geometric_progression_ratio_l2163_216376


namespace sum_of_digits_product_72_sevens_72_fives_l2163_216377

/-- Represents a number consisting of n repetitions of a single digit --/
def repeatedDigit (d : Nat) (n : Nat) : Nat :=
  d * (10^n - 1) / 9

/-- Calculates the sum of digits in a natural number --/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem to be proved --/
theorem sum_of_digits_product_72_sevens_72_fives :
  sumOfDigits (repeatedDigit 7 72 * repeatedDigit 5 72) = 576 := by
  sorry

end sum_of_digits_product_72_sevens_72_fives_l2163_216377


namespace function_inequality_implies_a_greater_than_one_l2163_216330

/-- Given functions f and g, prove that if for any x₁ in [-1, 2], 
    there exists an x₂ in [0, 2] such that f(x₁) > g(x₂), then a > 1 -/
theorem function_inequality_implies_a_greater_than_one (a : ℝ) : 
  (∀ x₁ ∈ Set.Icc (-1 : ℝ) 2, ∃ x₂ ∈ Set.Icc (0 : ℝ) 2, x₁^2 > 2^x₂ - a) → 
  a > 1 := by
  sorry

#check function_inequality_implies_a_greater_than_one

end function_inequality_implies_a_greater_than_one_l2163_216330


namespace sum_squares_product_l2163_216326

theorem sum_squares_product (m n : ℝ) (h : m + n = -2) : 5*m^2 + 5*n^2 + 10*m*n = 20 := by
  sorry

end sum_squares_product_l2163_216326


namespace ball_problem_proof_l2163_216314

/-- Represents the arrangement of 8 balls with specific conditions -/
def arrangement_count : ℕ := 576

/-- Represents the number of ways to take out 4 balls ensuring each color is taken -/
def takeout_count : ℕ := 40

/-- Represents the number of ways to divide 8 balls into three groups, each with at least 2 balls -/
def division_count : ℕ := 490

/-- Total number of balls -/
def total_balls : ℕ := 8

/-- Number of black balls -/
def black_balls : ℕ := 4

/-- Number of red balls -/
def red_balls : ℕ := 2

/-- Number of yellow balls -/
def yellow_balls : ℕ := 2

theorem ball_problem_proof :
  (total_balls = black_balls + red_balls + yellow_balls) ∧
  (arrangement_count = 576) ∧
  (takeout_count = 40) ∧
  (division_count = 490) := by
  sorry

end ball_problem_proof_l2163_216314


namespace power_two_greater_than_sum_of_powers_l2163_216354

theorem power_two_greater_than_sum_of_powers (n : ℕ) (x : ℝ) 
  (h1 : n ≥ 2) (h2 : |x| < 1) : 
  (2 : ℝ) ^ n > (1 - x) ^ n + (1 + x) ^ n := by
  sorry

end power_two_greater_than_sum_of_powers_l2163_216354


namespace largest_sum_is_994_l2163_216356

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- The sum of the given configuration -/
def sum (x y : Digit) : ℕ := 113 * x.val + 10 * y.val

/-- The largest possible 3-digit sum for the given configuration -/
def largest_sum : ℕ := 994

theorem largest_sum_is_994 (x y z : Digit) (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z) :
  sum x y ≤ largest_sum ∧
  ∃ (a b : Digit), sum a b = largest_sum ∧ a ≠ b :=
sorry

end largest_sum_is_994_l2163_216356


namespace compound_interest_calculation_l2163_216332

/-- Calculates the final amount after two years of compound interest with different rates each year. -/
def final_amount (initial_amount : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount_after_first_year := initial_amount * (1 + rate1)
  amount_after_first_year * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, the final amount after two years is as calculated. -/
theorem compound_interest_calculation :
  final_amount 5460 0.04 0.05 = 5962.32 := by
  sorry

#eval final_amount 5460 0.04 0.05

end compound_interest_calculation_l2163_216332


namespace furniture_assembly_time_l2163_216327

/-- Given the number of chairs and tables, and the time spent on each piece,
    calculate the total time taken to assemble all furniture. -/
theorem furniture_assembly_time 
  (num_chairs : ℕ) 
  (num_tables : ℕ) 
  (time_per_piece : ℕ) 
  (h1 : num_chairs = 4) 
  (h2 : num_tables = 2) 
  (h3 : time_per_piece = 8) : 
  (num_chairs + num_tables) * time_per_piece = 48 := by
  sorry

end furniture_assembly_time_l2163_216327


namespace welders_left_correct_l2163_216391

/-- The number of welders who started working on another project --/
def welders_left : ℕ := 12

/-- The initial number of welders --/
def initial_welders : ℕ := 36

/-- The number of days to complete the order with all welders --/
def initial_days : ℕ := 5

/-- The number of additional days needed after some welders left --/
def additional_days : ℕ := 6

/-- The rate at which each welder works --/
def welder_rate : ℝ := 1

/-- The total work to be done --/
def total_work : ℝ := initial_welders * initial_days * welder_rate

theorem welders_left_correct :
  (initial_welders - welders_left) * (additional_days * welder_rate) =
  total_work - (initial_welders * welder_rate) := by sorry

end welders_left_correct_l2163_216391


namespace system_solution_l2163_216317

theorem system_solution : 
  ∃! (x y : ℝ), x + y = 8 ∧ 2*x - y = 7 ∧ x = 5 ∧ y = 3 := by
sorry

end system_solution_l2163_216317


namespace cube_root_difference_l2163_216334

theorem cube_root_difference : (8 : ℝ) ^ (1/3) - (343 : ℝ) ^ (1/3) = -5 := by sorry

end cube_root_difference_l2163_216334
