import Mathlib

namespace hair_color_cost_l257_25766

/-- Calculates the cost of each box of hair color based on Maddie's beauty store purchase. -/
theorem hair_color_cost (palette_price : ℝ) (lipstick_price : ℝ) (total_paid : ℝ)
  (palette_count : ℕ) (lipstick_count : ℕ) (hair_color_count : ℕ) :
  palette_price = 15 →
  lipstick_price = 2.5 →
  palette_count = 3 →
  lipstick_count = 4 →
  hair_color_count = 3 →
  total_paid = 67 →
  (total_paid - (palette_price * palette_count + lipstick_price * lipstick_count)) / hair_color_count = 4 :=
by sorry

end hair_color_cost_l257_25766


namespace kaleb_fair_expense_l257_25706

/-- Calculates the total cost of rides given the number of tickets used and the cost per ticket -/
def total_cost (tickets_used : ℕ) (cost_per_ticket : ℕ) : ℕ :=
  tickets_used * cost_per_ticket

theorem kaleb_fair_expense :
  let initial_tickets : ℕ := 6
  let ferris_wheel_cost : ℕ := 2
  let bumper_cars_cost : ℕ := 1
  let roller_coaster_cost : ℕ := 2
  let ticket_price : ℕ := 9
  let total_tickets_used : ℕ := ferris_wheel_cost + bumper_cars_cost + roller_coaster_cost
  total_cost total_tickets_used ticket_price = 45 := by
  sorry

#eval total_cost 5 9

end kaleb_fair_expense_l257_25706


namespace monotonic_quadratic_function_l257_25779

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + 1

-- Define monotonicity on an interval
def MonotonicOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

-- State the theorem
theorem monotonic_quadratic_function (a : ℝ) :
  MonotonicOn (f a) 1 2 → a ≤ -2 ∨ a ≥ -1 := by
  sorry

end monotonic_quadratic_function_l257_25779


namespace johns_money_ratio_l257_25767

theorem johns_money_ratio (days_in_april : Nat) (sundays : Nat) (daily_earnings : Nat) 
  (book_expense : Nat) (money_left : Nat) : 
  days_in_april = 30 →
  sundays = 4 →
  daily_earnings = 10 →
  book_expense = 50 →
  money_left = 160 →
  (days_in_april - sundays) * daily_earnings - book_expense - money_left = book_expense := by
  sorry

end johns_money_ratio_l257_25767


namespace game_collection_proof_l257_25799

theorem game_collection_proof (games_from_friend games_from_garage_sale total_good_games : ℕ) :
  let total_games := games_from_friend + games_from_garage_sale
  let non_working_games := total_games - total_good_games
  total_good_games = total_games - non_working_games :=
by
  sorry

end game_collection_proof_l257_25799


namespace fraction_of_120_l257_25721

theorem fraction_of_120 : (1 / 2 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 120 = 10 / 3 := by
  sorry

end fraction_of_120_l257_25721


namespace intersection_nonempty_implies_a_geq_neg_one_l257_25758

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a}

-- State the theorem
theorem intersection_nonempty_implies_a_geq_neg_one (a : ℝ) :
  (A ∩ B a).Nonempty → a ≥ -1 := by
  sorry

end intersection_nonempty_implies_a_geq_neg_one_l257_25758


namespace zainab_works_two_hours_per_day_l257_25705

/-- Represents Zainab's flyer distribution job --/
structure FlyerJob where
  hourly_rate : ℕ
  days_per_week : ℕ
  total_weeks : ℕ
  total_earnings : ℕ

/-- Calculates the number of hours worked per day --/
def hours_per_day (job : FlyerJob) : ℚ :=
  job.total_earnings / (job.hourly_rate * job.days_per_week * job.total_weeks)

/-- Theorem stating that Zainab works 2 hours per day --/
theorem zainab_works_two_hours_per_day :
  let job := FlyerJob.mk 2 3 4 96
  hours_per_day job = 2 := by sorry

end zainab_works_two_hours_per_day_l257_25705


namespace total_spent_is_157_l257_25781

-- Define the initial amount given to each person
def initial_amount : ℕ := 250

-- Define Pete's spending
def pete_spending : ℕ := 20 + 30 + 50 + 5

-- Define Raymond's remaining money
def raymond_remaining : ℕ := 70 + 100 + 25 + 3

-- Theorem to prove
theorem total_spent_is_157 : 
  pete_spending + (initial_amount - raymond_remaining) = 157 := by
  sorry


end total_spent_is_157_l257_25781


namespace complex_sum_equals_two_l257_25742

def z : ℂ := 1 - Complex.I

theorem complex_sum_equals_two : (2 / z) + z = 2 := by sorry

end complex_sum_equals_two_l257_25742


namespace remaining_balloons_l257_25724

def initial_balloons : ℕ := 30
def given_balloons : ℕ := 16

theorem remaining_balloons : initial_balloons - given_balloons = 14 := by
  sorry

end remaining_balloons_l257_25724


namespace triangle_theorem_triangle_area_l257_25782

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C) :
  t.b^2 = t.a * t.c :=
sorry

/-- The area theorem when a = 2c = 2 -/
theorem triangle_area (t : Triangle)
  (h1 : t.A + t.B + t.C = Real.pi)
  (h2 : Real.sin t.B * (Real.tan t.A + Real.tan t.C) = Real.tan t.A * Real.tan t.C)
  (h3 : t.a = 2 * t.c)
  (h4 : t.a = 2) :
  (1/2) * t.a * t.c * Real.sin t.B = Real.sqrt 7 / 4 :=
sorry

end triangle_theorem_triangle_area_l257_25782


namespace complex_product_magnitude_l257_25741

theorem complex_product_magnitude : 
  Complex.abs ((-6 * Real.sqrt 3 + 6 * Complex.I) * (2 * Real.sqrt 2 - 2 * Complex.I)) = 24 * Real.sqrt 3 := by
  sorry

end complex_product_magnitude_l257_25741


namespace work_increase_with_absence_l257_25733

theorem work_increase_with_absence (p : ℕ) (W : ℝ) (h : p > 0) :
  let original_work := W / p
  let remaining_workers := (3 : ℝ) / 4 * p
  let new_work := W / remaining_workers
  new_work - original_work = (1 : ℝ) / 3 * original_work :=
by sorry

end work_increase_with_absence_l257_25733


namespace vertex_of_quadratic_l257_25776

/-- The quadratic function f(x) = x^2 - 2x -/
def f (x : ℝ) : ℝ := x^2 - 2*x

/-- The vertex of f(x) -/
def vertex : ℝ × ℝ := (1, -1)

theorem vertex_of_quadratic :
  (∀ x : ℝ, f x ≥ f (vertex.1)) ∧ f (vertex.1) = vertex.2 := by
  sorry

end vertex_of_quadratic_l257_25776


namespace line_inclination_is_30_degrees_l257_25709

-- Define the line equation
def line_equation (x y : ℝ) : Prop := x - Real.sqrt 3 * y - 2 = 0

-- Define the angle of inclination
def angle_of_inclination (α : ℝ) : Prop := 
  ∃ (k : ℝ), k = Real.sqrt 3 / 3 ∧ k = Real.tan α

-- Theorem statement
theorem line_inclination_is_30_degrees : 
  ∃ (α : ℝ), angle_of_inclination α ∧ α = π / 6 :=
sorry

end line_inclination_is_30_degrees_l257_25709


namespace swap_result_l257_25725

def swap_values (x y : ℕ) : ℕ × ℕ :=
  let t := x
  let x := y
  let y := t
  (x, y)

theorem swap_result : swap_values 5 6 = (6, 5) := by
  sorry

end swap_result_l257_25725


namespace triangle_side_length_l257_25704

theorem triangle_side_length (A B C : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let cos_C := (AB^2 + AC^2 - BC^2) / (2 * AB * AC)
  AB = Real.sqrt 5 ∧ AC = 5 ∧ cos_C = 9/10 →
  BC = 4 ∨ BC = 5 :=
by sorry

end triangle_side_length_l257_25704


namespace water_for_reaction_l257_25768

/-- Represents the balanced chemical reaction between Ammonium chloride and Water -/
structure BalancedReaction where
  nh4cl : ℕ  -- moles of Ammonium chloride
  h2o : ℕ    -- moles of Water
  hcl : ℕ    -- moles of Hydrochloric acid
  nh4oh : ℕ  -- moles of Ammonium hydroxide
  balanced : nh4cl = h2o ∧ nh4cl = hcl ∧ nh4cl = nh4oh

/-- The amount of water required for the given reaction -/
def water_required (r : BalancedReaction) : ℕ := r.h2o

/-- Theorem stating that 2 moles of water are required for the given reaction -/
theorem water_for_reaction (r : BalancedReaction) 
  (h1 : r.nh4cl = 2) 
  (h2 : r.hcl = 2) 
  (h3 : r.nh4oh = 2) : 
  water_required r = 2 := by
  sorry


end water_for_reaction_l257_25768


namespace expanded_lattice_equilateral_triangles_l257_25778

/-- Represents a point in the triangular lattice --/
structure LatticePoint where
  x : ℚ
  y : ℚ

/-- The set of all points in the expanded lattice --/
def ExpandedLattice : Set LatticePoint :=
  sorry

/-- Checks if three points form an equilateral triangle --/
def IsEquilateralTriangle (p1 p2 p3 : LatticePoint) : Prop :=
  sorry

/-- Counts the number of equilateral triangles in the expanded lattice --/
def CountEquilateralTriangles (lattice : Set LatticePoint) : ℕ :=
  sorry

/-- Main theorem: The number of equilateral triangles in the expanded lattice is 14 --/
theorem expanded_lattice_equilateral_triangles :
  CountEquilateralTriangles ExpandedLattice = 14 :=
sorry

end expanded_lattice_equilateral_triangles_l257_25778


namespace max_gcd_consecutive_b_terms_l257_25772

def b (n : ℕ) : ℕ := n.factorial + 2 * n

theorem max_gcd_consecutive_b_terms :
  ∃ (m : ℕ), m = 3 ∧ 
  (∀ (n : ℕ), n ≥ 1 → Nat.gcd (b n) (b (n + 1)) ≤ m) ∧
  (∃ (k : ℕ), k ≥ 1 ∧ Nat.gcd (b k) (b (k + 1)) = m) :=
sorry

end max_gcd_consecutive_b_terms_l257_25772


namespace square_rectangle_area_relation_l257_25727

theorem square_rectangle_area_relation :
  ∃ (x₁ x₂ : ℝ),
    (x₁ - 5) * (x₁ + 6) = 3 * (x₁ - 4)^2 ∧
    (x₂ - 5) * (x₂ + 6) = 3 * (x₂ - 4)^2 ∧
    x₁ ≠ x₂ ∧
    x₁ + x₂ = 25/2 := by
  sorry

end square_rectangle_area_relation_l257_25727


namespace solution_set_is_open_interval_l257_25749

-- Define a decreasing function f
def is_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f y < f x

-- Define the function f passing through specific points
def f_passes_through (f : ℝ → ℝ) : Prop :=
  f 0 = 3 ∧ f 3 = -1

-- Define the solution set
def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x : ℝ | |f (x + 1) - 1| < 2}

theorem solution_set_is_open_interval
  (f : ℝ → ℝ)
  (h_decreasing : is_decreasing f)
  (h_passes_through : f_passes_through f) :
  solution_set f = Set.Ioo (-1 : ℝ) 2 :=
sorry

end solution_set_is_open_interval_l257_25749


namespace parallel_vectors_m_value_l257_25755

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (-1, 2) and b = (2, m), then m = -4. -/
theorem parallel_vectors_m_value (a b : ℝ × ℝ) (m : ℝ) :
  a = (-1, 2) →
  b = (2, m) →
  (∃ (k : ℝ), b = k • a) →
  m = -4 := by
sorry

end parallel_vectors_m_value_l257_25755


namespace pats_calculation_l257_25774

theorem pats_calculation (x : ℝ) : 
  (x / 4 - 18 = 12) → (400 < 4*x + 18 ∧ 4*x + 18 < 600) := by
  sorry

end pats_calculation_l257_25774


namespace range_of_a_l257_25791

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def N (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem range_of_a (a : ℝ) : M ⊆ N a → a ∈ Set.Iic (-1) := by
  sorry

end range_of_a_l257_25791


namespace meiosis_fertilization_importance_l257_25750

structure ReproductiveProcess where
  meiosis : Bool
  fertilization : Bool

structure BiologicalImportance where
  chromosome_maintenance : Bool
  organism_biology : Bool

structure GenerationalEffect where
  somatic_cell_chromosomes : Bool
  heredity : Bool
  variation : Bool

/-- Given that meiosis and fertilization are important for maintaining constant
    chromosome numbers in species and crucial for the biology of organisms,
    prove that they are crucial for maintaining constant chromosome numbers in
    somatic cells of successive generations and are important for heredity and variation. -/
theorem meiosis_fertilization_importance
  (process : ReproductiveProcess)
  (importance : BiologicalImportance)
  (h1 : process.meiosis ∧ process.fertilization)
  (h2 : importance.chromosome_maintenance)
  (h3 : importance.organism_biology) :
  ∃ (effect : GenerationalEffect),
    effect.somatic_cell_chromosomes ∧
    effect.heredity ∧
    effect.variation :=
sorry

end meiosis_fertilization_importance_l257_25750


namespace train_length_proof_l257_25738

-- Define the given parameters
def train_speed : Real := 45 -- km/hr
def platform_length : Real := 180 -- meters
def time_to_pass : Real := 43.2 -- seconds

-- Define the theorem
theorem train_length_proof :
  let speed_ms : Real := train_speed * 1000 / 3600 -- Convert km/hr to m/s
  let total_distance : Real := speed_ms * time_to_pass
  let train_length : Real := total_distance - platform_length
  train_length = 360 := by
  sorry

end train_length_proof_l257_25738


namespace pencil_color_fraction_l257_25700

theorem pencil_color_fraction (total_length : ℝ) (green_fraction : ℝ) (white_fraction : ℝ) :
  total_length = 2 →
  green_fraction = 7 / 10 →
  white_fraction = 1 / 2 →
  (total_length - green_fraction * total_length) / 2 = 
  (1 - white_fraction) * (total_length - green_fraction * total_length) :=
by sorry

end pencil_color_fraction_l257_25700


namespace favorite_season_fall_l257_25707

theorem favorite_season_fall (total_students : ℕ) (winter_angle spring_angle : ℝ) :
  total_students = 600 →
  winter_angle = 90 →
  spring_angle = 60 →
  (total_students : ℝ) * (360 - winter_angle - spring_angle - 180) / 360 = 50 :=
by sorry

end favorite_season_fall_l257_25707


namespace decreasing_quadratic_implies_m_geq_neg_one_l257_25769

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 6

-- State the theorem
theorem decreasing_quadratic_implies_m_geq_neg_one (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ ≤ -1 → f m x₁ > f m x₂) →
  m ≥ -1 :=
by sorry

end decreasing_quadratic_implies_m_geq_neg_one_l257_25769


namespace butterflies_in_garden_l257_25759

theorem butterflies_in_garden (initial : ℕ) (flew_away : ℕ) (remaining : ℕ) : 
  initial = 9 → 
  flew_away = initial / 3 → 
  remaining = initial - flew_away → 
  remaining = 6 := by
sorry

end butterflies_in_garden_l257_25759


namespace intersection_A_B_l257_25757

def A : Set ℝ := {x | x^2 - x - 2 < 0}
def B : Set ℝ := {-2, -1, 0, 1, 2}

theorem intersection_A_B : A ∩ B = {0, 1} := by sorry

end intersection_A_B_l257_25757


namespace cos_double_angle_fourth_quadrant_l257_25716

/-- Prove that for an angle in the fourth quadrant, if the sum of coordinates of its terminal point on the unit circle is -1/3, then cos 2θ = -√17/9 -/
theorem cos_double_angle_fourth_quadrant (θ : ℝ) (x₀ y₀ : ℝ) :
  (π < θ ∧ θ < 2*π) →  -- θ is in the fourth quadrant
  x₀^2 + y₀^2 = 1 →    -- point (x₀, y₀) is on the unit circle
  x₀ = Real.cos θ →    -- x₀ is the cosine of θ
  y₀ = Real.sin θ →    -- y₀ is the sine of θ
  x₀ + y₀ = -1/3 →     -- sum of coordinates is -1/3
  Real.cos (2*θ) = -Real.sqrt 17 / 9 := by
sorry

end cos_double_angle_fourth_quadrant_l257_25716


namespace tetrahedron_inference_is_logical_l257_25787

/-- Represents the concept of logical reasoning -/
def LogicalReasoning : Type := Unit

/-- Represents the concept of analogical reasoning -/
def AnalogicalReasoning : Type := Unit

/-- Represents the act of inferring properties of a spatial tetrahedron from a plane triangle -/
def InferTetrahedronFromTriangle : Type := Unit

/-- Analogical reasoning is a type of logical reasoning -/
axiom analogical_is_logical : AnalogicalReasoning → LogicalReasoning

/-- Inferring tetrahedron properties from triangle properties is analogical reasoning -/
axiom tetrahedron_inference_is_analogical : InferTetrahedronFromTriangle → AnalogicalReasoning

/-- Theorem: Inferring properties of a spatial tetrahedron from properties of a plane triangle
    is a kind of logical reasoning -/
theorem tetrahedron_inference_is_logical : InferTetrahedronFromTriangle → LogicalReasoning := by
  sorry

end tetrahedron_inference_is_logical_l257_25787


namespace calculation_proof_l257_25794

theorem calculation_proof :
  ((-5/6 + 2/3) / (-7/12) * (7/2) = 1) ∧
  ((1 - 1/6) * (-3) - (-11/6) / (-22/3) = -11/4) := by
  sorry

end calculation_proof_l257_25794


namespace cubic_factorization_l257_25723

theorem cubic_factorization (x : ℝ) : x^3 - 4*x^2 + 4*x = x*(x-2)^2 := by
  sorry

end cubic_factorization_l257_25723


namespace election_winner_percentage_l257_25763

theorem election_winner_percentage (total_votes : ℕ) (vote_majority : ℕ) (winning_percentage : ℚ) : 
  total_votes = 400 →
  vote_majority = 160 →
  winning_percentage = 70 / 100 →
  (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = vote_majority :=
by sorry

end election_winner_percentage_l257_25763


namespace gcd_654327_543216_l257_25792

theorem gcd_654327_543216 : Nat.gcd 654327 543216 = 1 := by
  sorry

end gcd_654327_543216_l257_25792


namespace sofie_total_distance_l257_25789

/-- Represents the side lengths of the pentagon-shaped track in meters -/
def track_sides : List ℕ := [25, 35, 20, 40, 30]

/-- Calculates the perimeter of the track in meters -/
def track_perimeter : ℕ := track_sides.sum

/-- The number of initial laps Sofie runs -/
def initial_laps : ℕ := 2

/-- The number of additional laps Sofie runs -/
def additional_laps : ℕ := 5

/-- Theorem stating the total distance Sofie runs -/
theorem sofie_total_distance :
  initial_laps * track_perimeter + additional_laps * track_perimeter = 1050 := by
  sorry

end sofie_total_distance_l257_25789


namespace train_crossing_time_l257_25739

/-- Time taken for a train to cross a man running in the same direction --/
theorem train_crossing_time (train_length : ℝ) (train_speed : ℝ) (man_speed : ℝ) : 
  train_length = 450 →
  train_speed = 60 * 1000 / 3600 →
  man_speed = 6 * 1000 / 3600 →
  (train_length / (train_speed - man_speed)) = 30 := by
  sorry

end train_crossing_time_l257_25739


namespace mildred_weight_l257_25717

/-- Mildred's weight problem -/
theorem mildred_weight (carol_weight : ℕ) (weight_difference : ℕ) 
  (h1 : carol_weight = 9)
  (h2 : weight_difference = 50) :
  carol_weight + weight_difference = 59 := by
  sorry

end mildred_weight_l257_25717


namespace increasing_g_implies_m_bound_l257_25773

open Real

theorem increasing_g_implies_m_bound (m : ℝ) :
  (∀ x > 2, Monotone (fun x => (x - m) * (exp x - x) - exp x + x^2 + x)) →
  m ≤ (2 * exp 2 + 1) / (exp 2 - 1) :=
by sorry

end increasing_g_implies_m_bound_l257_25773


namespace orange_boxes_l257_25798

theorem orange_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (h1 : total_oranges = 42) (h2 : oranges_per_box = 6) :
  total_oranges / oranges_per_box = 7 :=
by sorry

end orange_boxes_l257_25798


namespace max_value_of_f_l257_25740

def f (a b : ℕ) : ℚ :=
  (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b)

theorem max_value_of_f :
  ∀ a b : ℕ,
  a ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ) →
  b ∈ ({2, 3, 4, 5, 6, 7, 8} : Set ℕ) →
  f a b ≤ 89 / 287 :=
by sorry

end max_value_of_f_l257_25740


namespace inequality_proof_l257_25703

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (1 + x / y)^3 + (1 + y / x)^3 ≥ 16 := by
  sorry

end inequality_proof_l257_25703


namespace sketch_finalization_orders_l257_25747

/-- Represents the order of sketches in the stack -/
def sketchOrder : List Nat := [2, 4, 1, 3, 5, 7, 6, 10, 9, 8]

/-- Represents the sketches completed before lunch -/
def completedSketches : List Nat := [8, 4]

/-- Calculates the number of possible orders for finalizing remaining sketches -/
def possibleOrders (order : List Nat) (completed : List Nat) : Nat :=
  sorry

theorem sketch_finalization_orders :
  possibleOrders sketchOrder completedSketches = 64 := by
  sorry

end sketch_finalization_orders_l257_25747


namespace class_size_problem_l257_25751

theorem class_size_problem (total : ℕ) : 
  (total / 3 : ℚ) + 26 = total → total = 39 :=
by sorry

end class_size_problem_l257_25751


namespace solvable_iff_edge_start_l257_25732

/-- Represents a cell on the 4x4 board -/
inductive Cell
| corner : Cell
| edge : Cell
| center : Cell

/-- Represents the state of the board -/
structure Board :=
(empty_cell : Cell)
(stones : Nat)

/-- Defines a valid move on the board -/
def valid_move (b : Board) : Prop :=
  b.stones > 1 ∧ ∃ (new_empty : Cell), new_empty ≠ b.empty_cell

/-- Defines the final state with one stone -/
def final_state (b : Board) : Prop :=
  b.stones = 1

/-- The main theorem to prove -/
theorem solvable_iff_edge_start :
  ∀ (b : Board),
    (b.empty_cell = Cell.edge ∧ b.stones = 15) ↔
    (∃ (b_final : Board), 
      final_state b_final ∧
      (∃ (moves : Nat), ∀ (i : Nat), i < moves → valid_move (Board.mk b.empty_cell (b.stones - i)))) :=
sorry

end solvable_iff_edge_start_l257_25732


namespace prism_volume_l257_25711

/-- Represents the dimensions of a rectangular prism -/
structure PrismDimensions where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines the properties of our specific rectangular prism -/
def RectangularPrism (d : PrismDimensions) : Prop :=
  d.x * d.y = 18 ∧ 
  d.y * d.z = 12 ∧ 
  d.x * d.z = 8 ∧
  d.y = 2 * min d.x d.z

theorem prism_volume (d : PrismDimensions) 
  (h : RectangularPrism d) : d.x * d.y * d.z = 16 := by
  sorry

#check prism_volume

end prism_volume_l257_25711


namespace perfect_square_condition_l257_25722

theorem perfect_square_condition (a : ℕ) : a ≥ 1 → (∃ k : ℕ, 1 - 8 * 3^a + 2^(a+2) * (2^a - 1) = k^2) ↔ a = 3 ∨ a = 5 := by
  sorry

end perfect_square_condition_l257_25722


namespace distribute_5_4_l257_25737

/-- The number of ways to distribute n distinct items into k identical bags, allowing empty bags. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 5 distinct items into 4 identical bags, allowing empty bags, is 36. -/
theorem distribute_5_4 : distribute 5 4 = 36 := by sorry

end distribute_5_4_l257_25737


namespace jonathan_calorie_deficit_l257_25785

/-- Jonathan's calorie consumption and burning schedule --/
structure CalorieSchedule where
  regularDailyIntake : ℕ
  saturdayIntake : ℕ
  dailyBurn : ℕ

/-- Calculate the weekly caloric deficit --/
def weeklyCalorieDeficit (schedule : CalorieSchedule) : ℕ :=
  7 * schedule.dailyBurn - (6 * schedule.regularDailyIntake + schedule.saturdayIntake)

/-- Theorem stating Jonathan's weekly caloric deficit --/
theorem jonathan_calorie_deficit :
  let schedule : CalorieSchedule := {
    regularDailyIntake := 2500,
    saturdayIntake := 3500,
    dailyBurn := 3000
  }
  weeklyCalorieDeficit schedule = 2500 := by
  sorry


end jonathan_calorie_deficit_l257_25785


namespace inequality_system_solution_l257_25743

theorem inequality_system_solution (x : ℤ) : 
  (2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1) ↔ x ∈ ({3, 4, 5} : Set ℤ) := by
  sorry

end inequality_system_solution_l257_25743


namespace rect_to_polar_conversion_l257_25729

/-- Conversion from rectangular to polar coordinates -/
theorem rect_to_polar_conversion :
  ∀ (x y : ℝ), x = 2 * Real.sqrt 2 ∧ y = 2 * Real.sqrt 2 →
  ∃ (r θ : ℝ), r > 0 ∧ 0 ≤ θ ∧ θ < 2 * π ∧
  r = 4 ∧ θ = π / 4 ∧
  x = r * Real.cos θ ∧ y = r * Real.sin θ :=
by sorry


end rect_to_polar_conversion_l257_25729


namespace evaluate_expression_l257_25718

theorem evaluate_expression : (4^150 * 9^152) / 6^301 = 27 / 2 := by
  sorry

end evaluate_expression_l257_25718


namespace min_value_sum_reciprocals_l257_25730

theorem min_value_sum_reciprocals (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z = 1) : 
  (1 / x + 4 / y + 9 / z) ≥ 36 := by
sorry

end min_value_sum_reciprocals_l257_25730


namespace negation_of_existential_proposition_l257_25710

theorem negation_of_existential_proposition :
  (¬ ∃ n : ℕ, n + 10 / n < 4) ↔ (∀ n : ℕ, n + 10 / n ≥ 4) := by sorry

end negation_of_existential_proposition_l257_25710


namespace max_value_trig_expression_l257_25736

theorem max_value_trig_expression :
  ∀ x y z : ℝ,
  (Real.sin (3 * x) + Real.sin (2 * y) + Real.sin z) *
  (Real.cos (3 * x) + Real.cos (2 * y) + Real.cos z) ≤ 4.5 ∧
  ∃ a b c : ℝ,
  (Real.sin (3 * a) + Real.sin (2 * b) + Real.sin c) *
  (Real.cos (3 * a) + Real.cos (2 * b) + Real.cos c) = 4.5 :=
by sorry

end max_value_trig_expression_l257_25736


namespace fraction_power_product_l257_25713

theorem fraction_power_product : (1 / 3 : ℚ)^4 * (1 / 5 : ℚ) = 1 / 405 := by
  sorry

end fraction_power_product_l257_25713


namespace base_conversion_2546_to_base5_l257_25754

/-- Converts a base 5 number to base 10 --/
def base5ToBase10 (a b c d : Nat) : Nat :=
  a * 5^3 + b * 5^2 + c * 5^1 + d * 5^0

/-- Theorem stating that 2546 (base 10) is equal to 4141 (base 5) --/
theorem base_conversion_2546_to_base5 :
  base5ToBase10 4 1 4 1 = 2546 := by
  sorry


end base_conversion_2546_to_base5_l257_25754


namespace class_division_theorem_l257_25753

theorem class_division_theorem :
  ∀ (x : ℕ),
  x ≤ 26 ∧ x ≤ 30 →
  x - (24 - (30 - x)) = 6 :=
by
  sorry

end class_division_theorem_l257_25753


namespace bill_pot_stacking_l257_25731

/-- Calculates the total number of pots that can be stacked given the vertical stack size, 
    number of stacks per shelf, and number of shelves. -/
def total_pots (vertical_stack : ℕ) (stacks_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  vertical_stack * stacks_per_shelf * num_shelves

/-- Proves that given the specific conditions of Bill's pot stacking problem, 
    the total number of pots is 60. -/
theorem bill_pot_stacking : total_pots 5 3 4 = 60 := by
  sorry

end bill_pot_stacking_l257_25731


namespace added_amount_after_doubling_and_tripling_l257_25790

theorem added_amount_after_doubling_and_tripling (x y : ℝ) : x = 5 → 3 * (2 * x + y) = 75 → y = 15 := by
  sorry

end added_amount_after_doubling_and_tripling_l257_25790


namespace rectangle_length_proof_l257_25795

/-- The length of each identical rectangle forming PQRS, rounded to the nearest integer -/
def rectangle_length : ℕ :=
  37

theorem rectangle_length_proof (area_PQRS : ℝ) (num_rectangles : ℕ) (PQ_ratio : ℝ) (RS_ratio : ℝ) :
  area_PQRS = 6000 →
  num_rectangles = 6 →
  PQ_ratio = 4 →
  RS_ratio = 3 →
  rectangle_length = 37 := by
  sorry

#check rectangle_length_proof

end rectangle_length_proof_l257_25795


namespace hugo_mountain_elevation_l257_25761

/-- The elevation of Hugo's mountain in feet -/
def hugo_elevation : ℝ := 10000

/-- The elevation of Boris' mountain in feet -/
def boris_elevation : ℝ := hugo_elevation - 2500

theorem hugo_mountain_elevation :
  (3 * hugo_elevation = 4 * boris_elevation) ∧
  (boris_elevation = hugo_elevation - 2500) →
  hugo_elevation = 10000 := by
  sorry

end hugo_mountain_elevation_l257_25761


namespace sets_and_domains_l257_25797

-- Define the sets A, B, and C
def A : Set ℝ := {x | |x - 1| ≥ 1}
def B : Set ℝ := {x | x < -1 ∨ x ≥ 1}
def C (a : ℝ) : Set ℝ := {x | 2*a < x ∧ x < a + 1}

-- State the theorem
theorem sets_and_domains (a : ℝ) (h : a < 1) :
  (A ∩ B = {x | x < -1 ∨ x ≥ 2}) ∧
  ((Set.univ \ (A ∪ B)) = {x | 0 < x ∧ x < 1}) ∧
  (C a ⊆ B → (a ≤ -2 ∨ (1/2 ≤ a ∧ a < 1))) :=
by sorry

end sets_and_domains_l257_25797


namespace stratified_sampling_correct_sizes_l257_25796

def total_population : ℕ := 300
def top_class_size : ℕ := 30
def experimental_class_size : ℕ := 90
def regular_class_size : ℕ := 180
def total_sample_size : ℕ := 30

def stratum_sample_size (stratum_size : ℕ) : ℕ :=
  (stratum_size * total_sample_size) / total_population

theorem stratified_sampling_correct_sizes :
  stratum_sample_size top_class_size = 3 ∧
  stratum_sample_size experimental_class_size = 9 ∧
  stratum_sample_size regular_class_size = 18 :=
by sorry

end stratified_sampling_correct_sizes_l257_25796


namespace root_count_relationship_l257_25760

-- Define the number of real roots for each equation
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry

-- State the theorem
theorem root_count_relationship : a > c ∧ c > b := by sorry

end root_count_relationship_l257_25760


namespace library_books_existence_l257_25770

theorem library_books_existence :
  ∃ (r p c b : ℕ), 
    r > 3000 ∧
    r = p + c + b ∧
    3 * c = 2 * p ∧
    4 * b = 3 * c :=
by sorry

end library_books_existence_l257_25770


namespace percent_problem_l257_25746

theorem percent_problem (x : ℝ) : (24 / x = 30 / 100) → x = 80 := by
  sorry

end percent_problem_l257_25746


namespace lucy_money_ratio_l257_25756

theorem lucy_money_ratio : 
  ∀ (initial_amount spent remaining : ℚ),
    initial_amount = 30 →
    remaining = 15 →
    spent + remaining = initial_amount * (2/3) →
    spent / (initial_amount * (2/3)) = 1/4 := by
  sorry

end lucy_money_ratio_l257_25756


namespace combined_height_is_twelve_l257_25793

/-- The height of Chiquita in feet -/
def chiquita_height : ℝ := 5

/-- The height difference between Mr. Martinez and Chiquita in feet -/
def height_difference : ℝ := 2

/-- The height of Mr. Martinez in feet -/
def martinez_height : ℝ := chiquita_height + height_difference

/-- The combined height of Mr. Martinez and Chiquita in feet -/
def combined_height : ℝ := chiquita_height + martinez_height

theorem combined_height_is_twelve : combined_height = 12 := by
  sorry

end combined_height_is_twelve_l257_25793


namespace custom_mult_example_l257_25745

/-- Custom multiplication operation for fractions -/
def custom_mult (m n p q : ℚ) : ℚ := m * p * (2 * q / n)

/-- Theorem stating that (6/5) * (3/4) = 144/5 under the custom multiplication -/
theorem custom_mult_example : custom_mult 6 5 3 4 = 144 / 5 := by
  sorry

end custom_mult_example_l257_25745


namespace quincy_peter_picture_difference_l257_25702

theorem quincy_peter_picture_difference :
  ∀ (peter_pictures randy_pictures quincy_pictures total_pictures : ℕ),
    peter_pictures = 8 →
    randy_pictures = 5 →
    total_pictures = 41 →
    total_pictures = peter_pictures + randy_pictures + quincy_pictures →
    quincy_pictures - peter_pictures = 20 := by
  sorry

end quincy_peter_picture_difference_l257_25702


namespace gcd_factorial_eight_and_factorial_six_squared_l257_25771

theorem gcd_factorial_eight_and_factorial_six_squared : Nat.gcd (Nat.factorial 8) ((Nat.factorial 6)^2) = 1920 := by
  sorry

end gcd_factorial_eight_and_factorial_six_squared_l257_25771


namespace garment_fraction_theorem_l257_25764

theorem garment_fraction_theorem (bikini_fraction trunks_fraction : ℝ) 
  (h1 : bikini_fraction = 0.38)
  (h2 : trunks_fraction = 0.25) : 
  bikini_fraction + trunks_fraction = 0.63 := by
  sorry

end garment_fraction_theorem_l257_25764


namespace pen_price_proof_l257_25734

/-- Represents the regular price of a pen in dollars -/
def regular_price : ℝ := 2

/-- Represents the total number of pens bought -/
def total_pens : ℕ := 20

/-- Represents the total cost paid by the customer in dollars -/
def total_cost : ℝ := 30

/-- Represents the number of pens at regular price -/
def regular_price_pens : ℕ := 10

/-- Represents the number of pens at half price -/
def half_price_pens : ℕ := 10

theorem pen_price_proof :
  regular_price * regular_price_pens + 
  (regular_price / 2) * half_price_pens = total_cost ∧
  regular_price_pens + half_price_pens = total_pens := by
  sorry

end pen_price_proof_l257_25734


namespace passing_percentage_l257_25719

def max_marks : ℕ := 800
def obtained_marks : ℕ := 175
def failed_by : ℕ := 89

theorem passing_percentage :
  (((obtained_marks + failed_by : ℚ) / max_marks) * 100).floor = 33 := by
  sorry

end passing_percentage_l257_25719


namespace pyramid_scheme_characterization_l257_25701

/-- Represents a financial scheme -/
structure FinancialScheme where
  returns : ℝ
  information_completeness : ℝ
  advertising_aggressiveness : ℝ

/-- Defines the average market return -/
def average_market_return : ℝ := sorry

/-- Defines the threshold for complete information -/
def complete_information_threshold : ℝ := sorry

/-- Defines the threshold for aggressive advertising -/
def aggressive_advertising_threshold : ℝ := sorry

/-- Determines if a financial scheme is a pyramid scheme -/
def is_pyramid_scheme (scheme : FinancialScheme) : Prop :=
  scheme.returns > average_market_return ∧
  scheme.information_completeness < complete_information_threshold ∧
  scheme.advertising_aggressiveness > aggressive_advertising_threshold

theorem pyramid_scheme_characterization (scheme : FinancialScheme) :
  is_pyramid_scheme scheme ↔
    scheme.returns > average_market_return ∧
    scheme.information_completeness < complete_information_threshold ∧
    scheme.advertising_aggressiveness > aggressive_advertising_threshold := by
  sorry

end pyramid_scheme_characterization_l257_25701


namespace min_tangent_length_l257_25735

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define the symmetry line
def symmetry_line (a b x y : ℝ) : Prop := 2*a*x + b*y + 6 = 0

-- Define the tangent point
def tangent_point (a b : ℝ) : Prop := ∃ x y : ℝ, circle_C x y ∧ symmetry_line a b x y

-- Theorem statement
theorem min_tangent_length (a b : ℝ) : 
  tangent_point a b → 
  (∃ t : ℝ, t ≥ 0 ∧ 
    (∀ s : ℝ, s ≥ 0 → 
      (∃ x y : ℝ, circle_C x y ∧ (x - a)^2 + (y - b)^2 = s^2) → 
      t ≤ s) ∧ 
    t = 4) := 
sorry

end min_tangent_length_l257_25735


namespace odd_function_value_l257_25728

theorem odd_function_value (m : ℝ) : 
  let f : ℝ → ℝ := λ x => x^(2-m)
  (∀ x ∈ Set.Icc (-3-m) (m^2-m), f (-x) = -f x) →
  f m = -1 := by
sorry

end odd_function_value_l257_25728


namespace quadratic_function_increasing_condition_l257_25726

/-- Given a quadratic function y = x^2 - 2mx + 5, if y increases as x increases when x > -1, then m ≤ -1 -/
theorem quadratic_function_increasing_condition (m : ℝ) : 
  (∀ x > -1, ∀ y > x, (y^2 - 2*m*y + 5) > (x^2 - 2*m*x + 5)) → m ≤ -1 :=
by sorry

end quadratic_function_increasing_condition_l257_25726


namespace tank_fill_time_with_leak_l257_25720

/-- Given a tank and two processes:
    1. Pipe A that can fill the tank in 6 hours
    2. A leak that can empty the full tank in 15 hours
    This theorem proves that it takes 10 hours for Pipe A to fill the tank with the leak present. -/
theorem tank_fill_time_with_leak (tank : ℝ) (pipe_a_rate : ℝ) (leak_rate : ℝ) : 
  pipe_a_rate = 1 / 6 →
  leak_rate = 1 / 15 →
  (pipe_a_rate - leak_rate)⁻¹ = 10 := by
  sorry

end tank_fill_time_with_leak_l257_25720


namespace wong_valentines_l257_25752

/-- The number of Valentines Mrs. Wong initially had -/
def initial_valentines : ℕ := 30

/-- The number of Valentines Mrs. Wong gave to her children -/
def given_valentines : ℕ := 8

/-- The number of Valentines Mrs. Wong has left -/
def remaining_valentines : ℕ := initial_valentines - given_valentines

theorem wong_valentines : remaining_valentines = 22 := by
  sorry

end wong_valentines_l257_25752


namespace sum_of_x_coordinates_is_two_l257_25765

theorem sum_of_x_coordinates_is_two :
  let f (x : ℝ) := |x^2 - 4*x + 3|
  let g (x : ℝ) := 7 - 2*x
  ∃ (x₁ x₂ : ℝ), (f x₁ = g x₁) ∧ (f x₂ = g x₂) ∧ (x₁ + x₂ = 2) ∧
    (∀ (x : ℝ), f x = g x → x = x₁ ∨ x = x₂) :=
by sorry

end sum_of_x_coordinates_is_two_l257_25765


namespace greatest_integer_difference_l257_25748

theorem greatest_integer_difference (x y : ℝ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  (⌊y⌋ - ⌈x⌉ : ℤ) ≤ 6 ∧ ∃ (x' y' : ℝ), 4 < x' ∧ x' < 8 ∧ 8 < y' ∧ y' < 12 ∧ (⌊y'⌋ - ⌈x'⌉ : ℤ) = 6 := by
  sorry

end greatest_integer_difference_l257_25748


namespace three_number_sum_l257_25775

theorem three_number_sum (a b c : ℝ) 
  (h1 : a ≤ b)
  (h2 : b ≤ c)
  (h3 : (a + b + c) / 3 = a + 7)
  (h4 : (a + b + c) / 3 = c - 20)
  (h5 : b = 8) : 
  a + b + c = 63 := by sorry

end three_number_sum_l257_25775


namespace plane_equation_proof_l257_25784

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a vector in 3D space -/
structure Vector3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Defines a plane in 3D space using a point and a normal vector -/
structure Plane where
  point : Point3D
  normal : Vector3D

/-- Checks if a given equation represents the plane defined by a point and normal vector -/
def is_plane_equation (p : Plane) (a b c d : ℝ) : Prop :=
  ∀ (x y z : ℝ),
    (a * x + b * y + c * z + d = 0) ↔
    (x - p.point.x) * p.normal.x + (y - p.point.y) * p.normal.y + (z - p.point.z) * p.normal.z = 0

/-- The main theorem: proving that x + 2y - z - 2 = 0 is the equation of the specified plane -/
theorem plane_equation_proof :
  let A : Point3D := ⟨1, 2, 3⟩
  let n : Vector3D := ⟨-1, -2, 1⟩
  let p : Plane := ⟨A, n⟩
  is_plane_equation p 1 2 (-1) (-2) := by sorry

end plane_equation_proof_l257_25784


namespace carpet_cost_l257_25708

/-- Calculates the total cost of carpeting a rectangular floor with square carpet tiles -/
theorem carpet_cost (floor_length floor_width carpet_side_length carpet_cost : ℝ) :
  floor_length = 6 ∧ 
  floor_width = 10 ∧ 
  carpet_side_length = 2 ∧ 
  carpet_cost = 15 →
  (floor_length * floor_width) / (carpet_side_length * carpet_side_length) * carpet_cost = 225 := by
  sorry

#check carpet_cost

end carpet_cost_l257_25708


namespace sum_of_ten_consecutive_squares_not_perfect_square_l257_25712

theorem sum_of_ten_consecutive_squares_not_perfect_square (x : ℤ) :
  ∃ (y : ℤ), 5 * (2 * x^2 + 10 * x + 29) ≠ y^2 := by
  sorry

end sum_of_ten_consecutive_squares_not_perfect_square_l257_25712


namespace percentage_unsold_books_l257_25715

def initial_stock : ℕ := 900
def monday_sales : ℕ := 75
def tuesday_sales : ℕ := 50
def wednesday_sales : ℕ := 64
def thursday_sales : ℕ := 78
def friday_sales : ℕ := 135

theorem percentage_unsold_books :
  let total_sales := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales
  let unsold_books := initial_stock - total_sales
  (unsold_books : ℚ) / initial_stock * 100 = 55.33 := by
  sorry

end percentage_unsold_books_l257_25715


namespace root_sum_theorem_l257_25780

-- Define the polynomial
def P (x r s t : ℝ) : ℝ := x^4 + r*x^2 + s*x + t

-- State the theorem
theorem root_sum_theorem (b r s t : ℝ) :
  (P (b - 6) r s t = 0) ∧ 
  (P (b - 5) r s t = 0) ∧ 
  (P (b - 4) r s t = 0) →
  r + t = -61 := by
  sorry


end root_sum_theorem_l257_25780


namespace cone_base_area_l257_25788

-- Define the cone
structure Cone where
  lateral_surface_area : ℝ
  base_radius : ℝ

-- Define the properties of the cone
def is_valid_cone (c : Cone) : Prop :=
  c.lateral_surface_area = 2 * Real.pi ∧
  c.lateral_surface_area = Real.pi * c.base_radius * c.base_radius

-- Theorem statement
theorem cone_base_area (c : Cone) (h : is_valid_cone c) :
  Real.pi * c.base_radius^2 = Real.pi := by
  sorry

end cone_base_area_l257_25788


namespace linear_equation_solution_l257_25714

theorem linear_equation_solution (x y : ℝ) : x - 3 * y = 4 ↔ x = 1 ∧ y = -1 := by
  sorry

end linear_equation_solution_l257_25714


namespace right_to_left_grouping_l257_25783

/-- Evaluates an expression using right-to-left grouping -/
def evaluateRightToLeft (a b c d : ℤ) : ℚ :=
  a / (b - c * d^2)

/-- The original expression as a function -/
def originalExpression (a b c d : ℤ) : ℚ :=
  a / b - c * d^2

theorem right_to_left_grouping (a b c d : ℤ) :
  evaluateRightToLeft a b c d = originalExpression a b c d :=
sorry

end right_to_left_grouping_l257_25783


namespace ln_inequality_l257_25777

theorem ln_inequality (x y a b : ℝ) 
  (hx : 0 < x) (hy : x < y) (hy1 : y < 1)
  (hb : 1 < b) (ha : b < a) : 
  (Real.log x) / b < (Real.log y) / a :=
sorry

end ln_inequality_l257_25777


namespace infinite_k_no_prime_sequence_l257_25786

theorem infinite_k_no_prime_sequence :
  ∃ (S : Set ℕ), Set.Infinite S ∧
    ∀ k ∈ S, ∃ (x : ℕ → ℕ),
      x 1 = 1 ∧
      x 2 = k + 2 ∧
      (∀ n, x (n + 2) = (k + 1) * x (n + 1) - x n) ∧
      ∀ n, ¬ Nat.Prime (x n) :=
sorry

end infinite_k_no_prime_sequence_l257_25786


namespace triangle_area_l257_25744

/-- Given an oblique triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    prove that its area is (5 * √3) / 4 under certain conditions. -/
theorem triangle_area (a b c A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  c * Real.sin A = Real.sqrt 3 * a * Real.cos C ∧  -- Given condition
  c = Real.sqrt 21 ∧  -- Given condition
  Real.sin C + Real.sin (B - A) = 5 * Real.sin (2 * A) →  -- Given condition
  (1 / 2) * a * b * Real.sin C = (5 * Real.sqrt 3) / 4 := by
  sorry

end triangle_area_l257_25744


namespace odd_and_even_implies_zero_range_even_function_abs_property_l257_25762

-- Define the concept of an odd function
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

-- Define the concept of an even function
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

-- Theorem 1: If a function is both odd and even, its range is {0}
theorem odd_and_even_implies_zero_range (f : ℝ → ℝ) 
  (h_odd : IsOdd f) (h_even : IsEven f) : 
  ∀ x, f x = 0 := by sorry

-- Theorem 2: If a function is even, then f(|x|) = f(x)
theorem even_function_abs_property (f : ℝ → ℝ) 
  (h_even : IsEven f) : 
  ∀ x, f (|x|) = f x := by sorry

end odd_and_even_implies_zero_range_even_function_abs_property_l257_25762
