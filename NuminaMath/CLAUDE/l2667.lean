import Mathlib

namespace investment_comparison_l2667_266755

def initial_aa : ℝ := 150
def initial_bb : ℝ := 120
def initial_cc : ℝ := 100

def year1_aa_change : ℝ := 1.15
def year1_bb_change : ℝ := 0.70
def year1_cc_change : ℝ := 1.00

def year2_aa_change : ℝ := 0.85
def year2_bb_change : ℝ := 1.20
def year2_cc_change : ℝ := 1.00

def year3_aa_change : ℝ := 1.10
def year3_bb_change : ℝ := 0.95
def year3_cc_change : ℝ := 1.05

def final_aa : ℝ := initial_aa * year1_aa_change * year2_aa_change * year3_aa_change
def final_bb : ℝ := initial_bb * year1_bb_change * year2_bb_change * year3_bb_change
def final_cc : ℝ := initial_cc * year1_cc_change * year2_cc_change * year3_cc_change

theorem investment_comparison : final_bb < final_cc ∧ final_cc < final_aa :=
sorry

end investment_comparison_l2667_266755


namespace class_ratio_proof_l2667_266767

theorem class_ratio_proof (eduardo_classes : ℕ) (total_classes : ℕ) 
  (h1 : eduardo_classes = 3)
  (h2 : total_classes = 9) :
  (total_classes - eduardo_classes) / eduardo_classes = 2 := by
sorry

end class_ratio_proof_l2667_266767


namespace hyperbola_eccentricity_l2667_266708

/-- The eccentricity of a hyperbola with specific conditions -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let l := {(x, y) : ℝ × ℝ | x - 2*y + 1 = 0}
  let asymptote_slope := b / a
  let line_slope := 1 / 2
  (asymptote_slope = 2 * line_slope / (1 - line_slope^2)) →
  Real.sqrt (1 + (b/a)^2) = 5/3 :=
by sorry

end hyperbola_eccentricity_l2667_266708


namespace train_crossing_time_l2667_266734

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 135 →
  train_speed_kmh = 54 →
  crossing_time = 9 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) :=
by
  sorry

#check train_crossing_time

end train_crossing_time_l2667_266734


namespace cube_sum_inequality_equality_iff_condition_l2667_266784

/-- For any pairwise distinct natural numbers a, b, and c, 
    (a³ + b³ + c³) / 3 ≥ abc + a + b + c holds. -/
theorem cube_sum_inequality (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 + b^3 + c^3) / 3 ≥ a * b * c + a + b + c :=
sorry

/-- Characterization of when equality holds in the cube sum inequality. -/
def equality_condition (a b c : ℕ) : Prop :=
  (a = b + 1 ∧ b = c + 1) ∨ 
  (b = a + 1 ∧ a = c + 1) ∨ 
  (c = a + 1 ∧ a = b + 1)

/-- The equality condition is necessary and sufficient for the cube sum inequality to be an equality. -/
theorem equality_iff_condition (a b c : ℕ) (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  (a^3 + b^3 + c^3) / 3 = a * b * c + a + b + c ↔ equality_condition a b c :=
sorry

end cube_sum_inequality_equality_iff_condition_l2667_266784


namespace income_calculation_l2667_266706

theorem income_calculation (income expenditure savings : ℕ) : 
  income * 4 = expenditure * 5 →
  income - expenditure = savings →
  savings = 3800 →
  income = 19000 := by
sorry

end income_calculation_l2667_266706


namespace hockey_league_season_games_l2667_266778

/-- The number of games played in a hockey league season -/
def hockey_league_games (n : ℕ) (m : ℕ) : ℕ :=
  n * (n - 1) / 2 * m

/-- Theorem: In a hockey league with 25 teams, where each team plays every other team 12 times,
    the total number of games played in the season is 3600. -/
theorem hockey_league_season_games :
  hockey_league_games 25 12 = 3600 := by
  sorry

end hockey_league_season_games_l2667_266778


namespace identical_projections_implies_sphere_or_cube_l2667_266794

/-- A solid is a three-dimensional object. -/
structure Solid :=
  (shape : Type)

/-- Orthographic projection is a method of representing a 3D object in 2D. -/
def orthographic_projection (s : Solid) : Type := sorry

/-- A solid has identical orthographic projections if all three standard views are the same. -/
def has_identical_projections (s : Solid) : Prop :=
  ∃ (view : orthographic_projection s), ∀ (p : orthographic_projection s), p = view

/-- The theorem states that if a solid has identical orthographic projections,
    it can be either a sphere or a cube. -/
theorem identical_projections_implies_sphere_or_cube (s : Solid) :
  has_identical_projections s → (s.shape = Sphere ∨ s.shape = Cube) :=
sorry

end identical_projections_implies_sphere_or_cube_l2667_266794


namespace complement_intersection_equals_specific_set_l2667_266714

def U : Set Nat := {1,2,3,4,5,6,7,8}
def S : Set Nat := {1,3,5}
def T : Set Nat := {3,6}

theorem complement_intersection_equals_specific_set :
  (U \ S) ∩ (U \ T) = {2,4,7,8} := by sorry

end complement_intersection_equals_specific_set_l2667_266714


namespace machine_a_production_rate_l2667_266774

/-- The number of sprockets produced by both machines -/
def total_sprockets : ℕ := 660

/-- The difference in production time between Machine A and Machine G -/
def time_difference : ℕ := 10

/-- The production rate of Machine G relative to Machine A -/
def g_to_a_ratio : ℚ := 11/10

/-- The production rate of Machine A in sprockets per hour -/
def machine_a_rate : ℚ := 6

theorem machine_a_production_rate :
  ∃ (machine_g_rate : ℚ) (time_g : ℚ),
    machine_g_rate = g_to_a_ratio * machine_a_rate ∧
    time_g * machine_g_rate = total_sprockets ∧
    (time_g + time_difference) * machine_a_rate = total_sprockets :=
by sorry

end machine_a_production_rate_l2667_266774


namespace equal_roots_when_m_is_negative_half_l2667_266782

theorem equal_roots_when_m_is_negative_half :
  let f (x m : ℝ) := (x * (x - 1) - (m^2 + m*x + 1)) / ((x - 1) * (m - 1)) - x / m
  ∀ x₁ x₂ : ℝ, f x₁ (-1/2) = 0 → f x₂ (-1/2) = 0 → x₁ = x₂ := by
  sorry

end equal_roots_when_m_is_negative_half_l2667_266782


namespace intersection_point_values_l2667_266780

theorem intersection_point_values (m n : ℚ) : 
  (1 / 2 : ℚ) * 1 + n = -2 → -- y = x/2 + n at x = 1
  m * 1 - 1 = -2 →          -- y = mx - 1 at x = 1
  m = -1 ∧ n = -5/2 := by
  sorry

end intersection_point_values_l2667_266780


namespace orange_juice_fraction_l2667_266746

-- Define the capacities of the pitchers
def pitcher1_capacity : ℚ := 800
def pitcher2_capacity : ℚ := 700

-- Define the fractions of orange juice in each pitcher
def pitcher1_juice_fraction : ℚ := 1/4
def pitcher2_juice_fraction : ℚ := 1/3

-- Calculate the amount of orange juice in each pitcher
def pitcher1_juice : ℚ := pitcher1_capacity * pitcher1_juice_fraction
def pitcher2_juice : ℚ := pitcher2_capacity * pitcher2_juice_fraction

-- Calculate the total amount of orange juice
def total_juice : ℚ := pitcher1_juice + pitcher2_juice

-- Calculate the total volume of the mixture
def total_volume : ℚ := pitcher1_capacity + pitcher2_capacity

-- Define the fraction of orange juice in the large container
def juice_fraction : ℚ := total_juice / total_volume

-- Theorem to prove
theorem orange_juice_fraction :
  juice_fraction = 433.33 / 1500 := by sorry

end orange_juice_fraction_l2667_266746


namespace square_area_from_vertices_l2667_266709

/-- The area of a square with adjacent vertices at (1,3) and (-4,6) is 34 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-4, 6)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  side_length^2 = 34 := by sorry

end square_area_from_vertices_l2667_266709


namespace fraction_inequality_l2667_266781

theorem fraction_inequality (a b x y : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0)
  (h1 : 1 / a > 1 / b) (h2 : x > y) : 
  x / (x + a) > y / (y + b) := by
sorry

end fraction_inequality_l2667_266781


namespace sin_C_value_max_area_l2667_266710

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = Real.sqrt 5 ∧ 2 * Real.sin t.A = t.a * Real.cos t.B

-- Theorem 1: If c = 2, then sin C = 2/3
theorem sin_C_value (t : Triangle) (h : triangle_conditions t) (hc : t.c = 2) :
  Real.sin t.C = 2/3 := by
  sorry

-- Theorem 2: Maximum area of triangle ABC
theorem max_area (t : Triangle) (h : triangle_conditions t) :
  ∃ (max_area : ℝ), max_area = (5 * Real.sqrt 5) / 4 ∧
  ∀ (actual_area : ℝ), actual_area = (1/2) * t.a * t.b * Real.sin t.C → actual_area ≤ max_area := by
  sorry

end sin_C_value_max_area_l2667_266710


namespace expression_simplification_l2667_266775

theorem expression_simplification (b c x : ℝ) (hb : b ≠ 1) (hc : c ≠ 1) (hbc : b ≠ c) :
  (x + 1)^2 / ((1 - b) * (1 - c)) + (x + b)^2 / ((b - 1) * (b - c)) + (x + c)^2 / ((c - 1) * (c - b)) = 1 := by
  sorry

end expression_simplification_l2667_266775


namespace sum_seven_consecutive_integers_l2667_266783

theorem sum_seven_consecutive_integers (m : ℤ) :
  m + (m + 1) + (m + 2) + (m + 3) + (m + 4) + (m + 5) + (m + 6) = 7 * m + 21 := by
  sorry

end sum_seven_consecutive_integers_l2667_266783


namespace rain_on_tuesday_l2667_266742

theorem rain_on_tuesday (rain_monday : ℝ) (rain_both : ℝ) (no_rain : ℝ)
  (h1 : rain_monday = 0.7)
  (h2 : rain_both = 0.4)
  (h3 : no_rain = 0.2) :
  ∃ rain_tuesday : ℝ,
    rain_tuesday = 0.5 ∧
    rain_monday + rain_tuesday - rain_both = 1 - no_rain :=
by
  sorry

end rain_on_tuesday_l2667_266742


namespace second_caterer_cheaper_at_least_people_l2667_266771

/-- Cost function for the first caterer -/
def cost1 (n : ℕ) : ℚ := 120 + 18 * n

/-- Cost function for the second caterer -/
def cost2 (n : ℕ) : ℚ := 250 + 15 * n

/-- The least number of people for which the second caterer is cheaper -/
def least_people : ℕ := 44

theorem second_caterer_cheaper_at_least_people :
  cost2 least_people < cost1 least_people ∧
  cost1 (least_people - 1) ≤ cost2 (least_people - 1) :=
by sorry

end second_caterer_cheaper_at_least_people_l2667_266771


namespace sum_max_min_xy_xz_yz_l2667_266747

/-- Given real numbers x, y, and z satisfying 5(x + y + z) = x^2 + y^2 + z^2,
    the sum of the maximum value of xy + xz + yz and 10 times its minimum value is 150. -/
theorem sum_max_min_xy_xz_yz (x y z : ℝ) (h : 5 * (x + y + z) = x^2 + y^2 + z^2) :
  ∃ (N n : ℝ),
    (∀ a b c : ℝ, 5 * (a + b + c) = a^2 + b^2 + c^2 →
      a * b + a * c + b * c ≤ N ∧ n ≤ a * b + a * c + b * c) ∧
    N + 10 * n = 150 := by
  sorry

end sum_max_min_xy_xz_yz_l2667_266747


namespace membership_change_l2667_266711

theorem membership_change (initial_members : ℝ) (h : initial_members > 0) :
  let fall_increase := 0.04
  let spring_decrease := 0.19
  let fall_members := initial_members * (1 + fall_increase)
  let spring_members := fall_members * (1 - spring_decrease)
  let total_change := (spring_members - initial_members) / initial_members
  total_change = -0.1576 := by
sorry

end membership_change_l2667_266711


namespace jessica_rent_last_year_l2667_266729

/-- Calculates Jessica's monthly rent last year given the increase in expenses --/
theorem jessica_rent_last_year (food_cost_last_year car_insurance_last_year : ℕ)
  (rent_increase_percent food_increase_percent : ℚ)
  (car_insurance_multiplier : ℕ)
  (total_yearly_increase : ℕ) :
  food_cost_last_year = 200 →
  car_insurance_last_year = 100 →
  rent_increase_percent = 30 / 100 →
  food_increase_percent = 50 / 100 →
  car_insurance_multiplier = 3 →
  total_yearly_increase = 7200 →
  ∃ (rent_last_year : ℕ),
    rent_last_year = 1000 ∧
    12 * ((1 + rent_increase_percent) * rent_last_year - rent_last_year +
         (1 + food_increase_percent) * food_cost_last_year - food_cost_last_year +
         car_insurance_multiplier * car_insurance_last_year - car_insurance_last_year) =
    total_yearly_increase :=
by sorry

end jessica_rent_last_year_l2667_266729


namespace power_sum_l2667_266776

theorem power_sum (a m n : ℝ) (hm : a^m = 3) (hn : a^n = 2) : a^(m+n) = 6 := by
  sorry

end power_sum_l2667_266776


namespace digital_earth_sharing_l2667_266743

/-- Represents the concept of Digital Earth -/
structure DigitalEarth where
  technology : Type
  data : Type
  sharing_method : Type

/-- Represents the internet as a sharing method -/
def Internet : Type := Unit

/-- Axiom: Digital Earth involves digital technology and Earth-related data -/
axiom digital_earth_components : ∀ (de : DigitalEarth), de.technology × de.data

/-- Theorem: Digital Earth can only achieve global information sharing through the internet -/
theorem digital_earth_sharing (de : DigitalEarth) : 
  de.sharing_method = Internet :=
sorry

end digital_earth_sharing_l2667_266743


namespace quadratic_has_minimum_l2667_266797

/-- Given a quadratic function f(x) = ax^2 + bx + c where c = -b^2/(4a) and a > 0,
    prove that the graph of y = f(x) has a minimum. -/
theorem quadratic_has_minimum (a b : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a * x^2 + b * x + (-b^2) / (4 * a)
  ∃ x₀, ∀ x, f x₀ ≤ f x :=
by sorry

end quadratic_has_minimum_l2667_266797


namespace arrangements_count_l2667_266712

/-- The number of people to be arranged -/
def num_people : ℕ := 5

/-- The number of positions available -/
def num_positions : ℕ := 4

/-- Function to calculate the number of arrangements -/
def calculate_arrangements (n_people : ℕ) (n_positions : ℕ) : ℕ :=
  -- Arrangements when A is selected (can't be in position A)
  (n_positions - 1) * (Nat.factorial (n_positions - 1)) +
  -- Arrangements when A is not selected
  (Nat.factorial n_positions)

/-- Theorem stating the number of arrangements -/
theorem arrangements_count :
  calculate_arrangements num_people num_positions = 42 :=
by sorry

end arrangements_count_l2667_266712


namespace deductive_reasoning_is_general_to_specific_l2667_266792

/-- Represents a form of reasoning --/
inductive ReasoningForm
  | GeneralToSpecific
  | SpecificToGeneral
  | GeneralToGeneral
  | SpecificToSpecific

/-- Definition of deductive reasoning --/
def deductive_reasoning : ReasoningForm := ReasoningForm.GeneralToSpecific

/-- Theorem stating that deductive reasoning is from general to specific --/
theorem deductive_reasoning_is_general_to_specific :
  deductive_reasoning = ReasoningForm.GeneralToSpecific := by sorry

end deductive_reasoning_is_general_to_specific_l2667_266792


namespace shortest_path_length_l2667_266779

/-- A regular octahedron with edge length 1 -/
structure RegularOctahedron where
  /-- The edge length of the octahedron is 1 -/
  edge_length : ℝ
  edge_length_eq : edge_length = 1

/-- A path on the surface of an octahedron -/
structure SurfacePath (o : RegularOctahedron) where
  /-- The length of the path -/
  length : ℝ
  /-- The path starts at a vertex -/
  starts_at_vertex : Bool
  /-- The path ends at the opposite vertex -/
  ends_at_opposite_vertex : Bool

/-- The theorem stating that the shortest path between opposite vertices has length 2 -/
theorem shortest_path_length (o : RegularOctahedron) : 
  ∃ (p : SurfacePath o), p.length = 2 ∧ 
  ∀ (q : SurfacePath o), q.starts_at_vertex ∧ q.ends_at_opposite_vertex → q.length ≥ p.length :=
sorry

end shortest_path_length_l2667_266779


namespace inequality_solution_l2667_266790

theorem inequality_solution (x : ℝ) : 
  (3 + 1 / (3 * x - 2) ≥ 5) ∧ (3 * x - 2 ≠ 0) → 
  x ∈ Set.Iio (2 / 3) ∪ Set.Ioc (2 / 3) (5 / 6) :=
by sorry

end inequality_solution_l2667_266790


namespace quadratic_minimum_l2667_266716

theorem quadratic_minimum (x : ℝ) : 
  3 * x^2 - 18 * x + 2023 ≥ 1996 ∧ ∃ y : ℝ, 3 * y^2 - 18 * y + 2023 = 1996 := by
  sorry

end quadratic_minimum_l2667_266716


namespace problem_statement_l2667_266798

-- Define the proposition p
def p : Prop := ∃ x₀ : ℝ, x₀ > 0 ∧ 3^x₀ + x₀ = 2016

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x| - a * x

-- Define the proposition q
def q : Prop := ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f a x = f a (-x)

-- State the theorem
theorem problem_statement : p ∧ ¬q := by sorry

end problem_statement_l2667_266798


namespace mean_cat_weight_l2667_266733

def cat_weights : List ℝ := [87, 90, 93, 95, 95, 98, 104, 106, 106, 107, 109, 110, 111, 112]

theorem mean_cat_weight :
  let n : ℕ := cat_weights.length
  let sum : ℝ := cat_weights.sum
  sum / n = 101.64 := by sorry

end mean_cat_weight_l2667_266733


namespace tangent_segment_equality_tangent_line_distance_equality_l2667_266738

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  slope : ℝ
  intercept : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define a tangent line to a circle
def is_tangent (l : Line) (c : Circle) : Prop :=
  sorry

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ :=
  sorry

-- Define the distance between two lines
def line_distance (l1 l2 : Line) : ℝ :=
  sorry

-- Define the point of tangency
def point_of_tangency (l : Line) (c : Circle) : Point :=
  sorry

theorem tangent_segment_equality (c1 c2 : Circle) (l1 l2 l3 l4 : Line) 
  (h1 : is_tangent l1 c1) (h2 : is_tangent l2 c1)
  (h3 : is_tangent l3 c2) (h4 : is_tangent l4 c2) :
  let p1 := point_of_tangency l1 c1
  let p2 := point_of_tangency l2 c1
  let p3 := point_of_tangency l3 c2
  let p4 := point_of_tangency l4 c2
  distance p1 p3 = distance p2 p4 :=
sorry

theorem tangent_line_distance_equality (c1 c2 : Circle) (l1 l2 l3 l4 : Line)
  (h1 : is_tangent l1 c1) (h2 : is_tangent l2 c1)
  (h3 : is_tangent l3 c2) (h4 : is_tangent l4 c2) :
  line_distance l1 l3 = line_distance l2 l4 :=
sorry

end tangent_segment_equality_tangent_line_distance_equality_l2667_266738


namespace steven_needs_three_more_seeds_l2667_266730

/-- The number of seeds needed for the assignment -/
def assignment_seeds : ℕ := 60

/-- The average number of seeds in an apple -/
def apple_seeds : ℕ := 6

/-- The average number of seeds in a pear -/
def pear_seeds : ℕ := 2

/-- The average number of seeds in a grape -/
def grape_seeds : ℕ := 3

/-- The number of apples Steven has -/
def steven_apples : ℕ := 4

/-- The number of pears Steven has -/
def steven_pears : ℕ := 3

/-- The number of grapes Steven has -/
def steven_grapes : ℕ := 9

/-- The number of additional seeds Steven needs -/
def additional_seeds_needed : ℕ := 3

theorem steven_needs_three_more_seeds :
  assignment_seeds - (steven_apples * apple_seeds + steven_pears * pear_seeds + steven_grapes * grape_seeds) = additional_seeds_needed := by
  sorry

end steven_needs_three_more_seeds_l2667_266730


namespace inscribed_circles_radii_sum_l2667_266773

theorem inscribed_circles_radii_sum (D : ℝ) (r₁ r₂ : ℝ) : 
  D = 23 → r₁ > 0 → r₂ > 0 → r₁ + r₂ = D / 2 := by
  sorry

end inscribed_circles_radii_sum_l2667_266773


namespace percentage_of_sum_l2667_266777

theorem percentage_of_sum (x y : ℝ) (P : ℝ) :
  (0.6 * (x - y) = (P / 100) * (x + y)) →
  (y = (1 / 3) * x) →
  P = 45 := by
sorry

end percentage_of_sum_l2667_266777


namespace jar_weight_percentage_l2667_266754

theorem jar_weight_percentage (jar_weight : ℝ) (full_beans_weight : ℝ) 
  (h1 : jar_weight = 0.2 * (jar_weight + full_beans_weight))
  (h2 : 0.5 * full_beans_weight = full_beans_weight / 2) :
  (jar_weight + full_beans_weight / 2) / (jar_weight + full_beans_weight) = 0.6 := by
  sorry

end jar_weight_percentage_l2667_266754


namespace new_rectangle_area_l2667_266731

theorem new_rectangle_area (a b : ℝ) (h : a > b) :
  let base := a^2 + b^2 + a
  let height := a^2 + b^2 - b
  base * height = a^4 + a^3 + 2*a^2*b^2 + a*b^3 - a*b + b^4 - b^3 - b^2 :=
by sorry

end new_rectangle_area_l2667_266731


namespace work_completion_workers_work_completion_workers_proof_l2667_266735

/-- Given a work that can be finished in 12 days by an initial group of workers,
    and is finished in 9 days after 10 more workers join,
    prove that the total number of workers after the addition is 40. -/
theorem work_completion_workers : ℕ → Prop :=
  λ initial_workers =>
    (initial_workers * 12 = (initial_workers + 10) * 9) →
    initial_workers + 10 = 40
  
#check work_completion_workers

/-- Proof of the theorem -/
theorem work_completion_workers_proof : ∃ n : ℕ, work_completion_workers n := by
  sorry

end work_completion_workers_work_completion_workers_proof_l2667_266735


namespace nell_initial_cards_l2667_266722

/-- The number of baseball cards Nell had initially -/
def initial_cards : ℕ := sorry

/-- The number of cards Jeff gave to Nell -/
def cards_from_jeff : ℝ := 276.0

/-- The total number of cards Nell has now -/
def total_cards : ℕ := 580

/-- Theorem stating that Nell's initial number of cards was 304 -/
theorem nell_initial_cards : 
  initial_cards = 304 :=
by
  sorry

end nell_initial_cards_l2667_266722


namespace initial_mean_calculation_l2667_266705

/-- Given 50 observations with an initial mean, if one observation is corrected
    from 23 to 34, and the new mean becomes 36.5, then the initial mean must be 36.28. -/
theorem initial_mean_calculation (n : ℕ) (initial_mean corrected_mean : ℝ) :
  n = 50 ∧
  corrected_mean = 36.5 ∧
  (n : ℝ) * initial_mean + (34 - 23) = n * corrected_mean →
  initial_mean = 36.28 := by
  sorry

end initial_mean_calculation_l2667_266705


namespace composite_divisors_theorem_l2667_266718

/-- A function that returns the set of proper divisors of a natural number -/
def proper_divisors (a : ℕ) : Set ℕ :=
  {d | d ∣ a ∧ 1 < d ∧ d < a}

/-- A function that increases each element of a set by 1 -/
def increase_by_one (S : Set ℕ) : Set ℕ :=
  {x + 1 | x ∈ S}

/-- The main theorem -/
theorem composite_divisors_theorem (n : ℕ) (h_composite : ¬ Prime n) :
  (∃ m : ℕ, increase_by_one (proper_divisors n) = proper_divisors m) ↔ n = 4 ∨ n = 8 := by
  sorry

#check composite_divisors_theorem

end composite_divisors_theorem_l2667_266718


namespace sqrt_inequality_l2667_266702

theorem sqrt_inequality (a : ℝ) (h : a ≥ 2) : 
  Real.sqrt (a + 1) - Real.sqrt a < Real.sqrt (a - 1) - Real.sqrt (a - 2) := by
  sorry

end sqrt_inequality_l2667_266702


namespace total_flowers_l2667_266728

theorem total_flowers (num_pots : ℕ) (flowers_per_pot : ℕ) (h1 : num_pots = 544) (h2 : flowers_per_pot = 32) :
  num_pots * flowers_per_pot = 17408 :=
by sorry

end total_flowers_l2667_266728


namespace prime_sum_112_l2667_266725

theorem prime_sum_112 :
  ∃ (S : Finset Nat), 
    (∀ p ∈ S, Nat.Prime p ∧ p > 10) ∧ 
    (S.sum id = 112) ∧ 
    (S.card = 6) := by
  sorry

end prime_sum_112_l2667_266725


namespace distance_from_P_to_y_axis_l2667_266703

def point_to_y_axis_distance (x y : ℝ) : ℝ := |x|

theorem distance_from_P_to_y_axis :
  let P : ℝ × ℝ := (-3, -4)
  point_to_y_axis_distance P.1 P.2 = 3 := by
  sorry

end distance_from_P_to_y_axis_l2667_266703


namespace product_xy_l2667_266789

theorem product_xy (x y : ℝ) (h1 : x - y = 6) (h2 : x^3 - y^3 = 72) : x * y = -8 := by
  sorry

end product_xy_l2667_266789


namespace x_seventh_plus_64x_squared_l2667_266739

theorem x_seventh_plus_64x_squared (x : ℝ) (h : x^3 + 4*x = 8) : x^7 + 64*x^2 = 128 := by
  sorry

end x_seventh_plus_64x_squared_l2667_266739


namespace beach_trip_time_l2667_266796

theorem beach_trip_time :
  let drive_time_one_way : ℝ := 2
  let total_drive_time : ℝ := 2 * drive_time_one_way
  let beach_time : ℝ := 2.5 * total_drive_time
  let total_trip_time : ℝ := total_drive_time + beach_time
  total_trip_time = 14 := by
  sorry

end beach_trip_time_l2667_266796


namespace female_height_calculation_l2667_266751

theorem female_height_calculation (total_avg : ℝ) (male_avg : ℝ) (ratio : ℝ) 
  (h1 : total_avg = 180)
  (h2 : male_avg = 185)
  (h3 : ratio = 2) :
  ∃ female_avg : ℝ, female_avg = 170 ∧ 
  (ratio * female_avg + male_avg) / (ratio + 1) = total_avg :=
by sorry

end female_height_calculation_l2667_266751


namespace trig_fraction_value_l2667_266707

theorem trig_fraction_value (α : Real) (h : Real.tan α = 2) :
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 1 := by
  sorry

end trig_fraction_value_l2667_266707


namespace monochromatic_rectangle_exists_l2667_266701

/-- A color type with four possible values -/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- A point in the grid -/
structure Point where
  x : Fin 5
  y : Fin 41

/-- A coloring of the grid -/
def Coloring := Point → Color

/-- A rectangle in the grid -/
structure Rectangle where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Predicate to check if four points form a valid rectangle with integer side lengths -/
def IsValidRectangle (r : Rectangle) : Prop :=
  (r.p1.x = r.p2.x ∧ r.p3.x = r.p4.x ∧ r.p1.y = r.p3.y ∧ r.p2.y = r.p4.y) ∨
  (r.p1.x = r.p3.x ∧ r.p2.x = r.p4.x ∧ r.p1.y = r.p2.y ∧ r.p3.y = r.p4.y)

/-- Main theorem: There exists a monochromatic rectangle with integer side lengths -/
theorem monochromatic_rectangle_exists (c : Coloring) : 
  ∃ (r : Rectangle), IsValidRectangle r ∧ 
    c r.p1 = c r.p2 ∧ c r.p2 = c r.p3 ∧ c r.p3 = c r.p4 := by
  sorry

end monochromatic_rectangle_exists_l2667_266701


namespace no_integer_solutions_l2667_266724

theorem no_integer_solutions :
  ¬ ∃ (x y z : ℤ), x^4 + y^4 + z^4 = 2*x^2*y^2 + 2*y^2*z^2 + 2*z^2*x^2 + 24 :=
by sorry

end no_integer_solutions_l2667_266724


namespace raffle_tickets_sold_l2667_266727

theorem raffle_tickets_sold (ticket_price : ℚ) (total_donations : ℚ) (total_raised : ℚ) :
  ticket_price = 2 →
  total_donations = 50 →
  total_raised = 100 →
  (total_raised - total_donations) / ticket_price = 25 := by
  sorry

end raffle_tickets_sold_l2667_266727


namespace thirteen_bead_necklace_l2667_266785

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

def arrangements (n : ℕ) : ℕ :=
  fibonacci (n + 2) - fibonacci (n - 2)

def circular_arrangements (n : ℕ) : ℕ :=
  (arrangements n - 1) / n + 1

theorem thirteen_bead_necklace :
  circular_arrangements 13 = 41 := by
  sorry

end thirteen_bead_necklace_l2667_266785


namespace inscribed_rhombus_radius_l2667_266720

/-- A rhombus inscribed in the intersection of two equal circles -/
structure InscribedRhombus where
  /-- The length of one diagonal of the rhombus -/
  diagonal1 : ℝ
  /-- The length of the other diagonal of the rhombus -/
  diagonal2 : ℝ
  /-- The radius of the circles -/
  radius : ℝ
  /-- The diagonals are positive -/
  diagonal1_pos : diagonal1 > 0
  diagonal2_pos : diagonal2 > 0
  /-- The radius is positive -/
  radius_pos : radius > 0
  /-- The relationship between the diagonals and the radius -/
  radius_eq : radius^2 = (radius - diagonal1/2)^2 + (diagonal2/2)^2

/-- The theorem stating that a rhombus with diagonals 12 and 6 inscribed in two equal circles implies the radius is 7.5 -/
theorem inscribed_rhombus_radius (r : InscribedRhombus) (h1 : r.diagonal1 = 6) (h2 : r.diagonal2 = 12) : 
  r.radius = 7.5 := by
  sorry

end inscribed_rhombus_radius_l2667_266720


namespace power_difference_equals_negative_sixteen_million_l2667_266759

theorem power_difference_equals_negative_sixteen_million : (3^4)^3 - (4^3)^4 = -16245775 := by
  sorry

end power_difference_equals_negative_sixteen_million_l2667_266759


namespace circle_center_l2667_266749

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + 6*x + y^2 - 8*y - 48 = 0

/-- The center of a circle given by its coordinates -/
def is_center (h k : ℝ) : Prop :=
  ∀ (x y : ℝ), circle_equation x y ↔ (x - h)^2 + (y - k)^2 = (x + h)^2 + (y - k)^2

theorem circle_center :
  is_center (-3) 4 :=
sorry

end circle_center_l2667_266749


namespace student_multiplication_problem_l2667_266757

theorem student_multiplication_problem (x : ℚ) : 
  (63 * x) - 142 = 110 → x = 4 := by
  sorry

end student_multiplication_problem_l2667_266757


namespace nesbitt_inequality_l2667_266704

theorem nesbitt_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
by sorry

end nesbitt_inequality_l2667_266704


namespace first_group_size_l2667_266748

/-- The number of days taken by the first group to complete the work -/
def days_first_group : ℝ := 28

/-- The number of men in the second group -/
def men_second_group : ℝ := 20

/-- The number of days taken by the second group to complete the work -/
def days_second_group : ℝ := 22.4

/-- The work done by a group is inversely proportional to the time taken -/
axiom work_time_inverse_proportion {men days : ℝ} : men * days = (men_second_group * days_second_group)

theorem first_group_size : ∃ (men : ℝ), men * days_first_group = men_second_group * days_second_group ∧ men = 16 := by
  sorry

end first_group_size_l2667_266748


namespace specific_prism_volume_l2667_266762

/-- Regular triangular prism inscribed in a sphere -/
structure InscribedPrism where
  -- Radius of the sphere
  R : ℝ
  -- Length of AD
  AD : ℝ
  -- Assertion that CD is a diameter
  is_diameter : Bool

/-- Volume of the inscribed prism -/
def prism_volume (p : InscribedPrism) : ℝ :=
  sorry

/-- Theorem: The volume of the specific inscribed prism is 48√15 -/
theorem specific_prism_volume :
  let p : InscribedPrism := {
    R := 6,
    AD := 4 * Real.sqrt 6,
    is_diameter := true
  }
  prism_volume p = 48 * Real.sqrt 15 := by
  sorry

end specific_prism_volume_l2667_266762


namespace cookie_sales_revenue_l2667_266788

theorem cookie_sales_revenue : 
  let chocolate_cookies : ℕ := 220
  let vanilla_cookies : ℕ := 70
  let chocolate_price : ℚ := 1
  let vanilla_price : ℚ := 2
  let chocolate_discount : ℚ := 0.1
  let sales_tax_rate : ℚ := 0.05
  
  let chocolate_revenue := chocolate_cookies * chocolate_price
  let chocolate_discount_amount := chocolate_revenue * chocolate_discount
  let discounted_chocolate_revenue := chocolate_revenue - chocolate_discount_amount
  let vanilla_revenue := vanilla_cookies * vanilla_price
  let total_revenue_before_tax := discounted_chocolate_revenue + vanilla_revenue
  let sales_tax := total_revenue_before_tax * sales_tax_rate
  let total_revenue_after_tax := total_revenue_before_tax + sales_tax
  
  total_revenue_after_tax = 354.90 := by sorry

end cookie_sales_revenue_l2667_266788


namespace equation_condition_l2667_266770

theorem equation_condition (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a < 10 ∧ b < 10 ∧ c < 10 ∧
  (20 * a + b) * (20 * a + c) = 400 * a * (a + 1) + 10 * b * c →
  b + c = 20 :=
sorry

end equation_condition_l2667_266770


namespace fenced_area_calculation_l2667_266753

/-- The area of a rectangle with two square cut-outs at opposite corners -/
def fencedArea (length width cutout1 cutout2 : ℝ) : ℝ :=
  length * width - cutout1^2 - cutout2^2

/-- Theorem stating that the area of the fenced region is 340 square feet -/
theorem fenced_area_calculation :
  fencedArea 20 18 4 2 = 340 := by
  sorry

end fenced_area_calculation_l2667_266753


namespace jacks_allowance_l2667_266715

/-- Calculates Jack's weekly allowance given the initial amount, number of weeks, and final amount in his piggy bank -/
def calculate_allowance (initial_amount : ℚ) (weeks : ℕ) (final_amount : ℚ) : ℚ :=
  2 * (final_amount - initial_amount) / weeks

/-- Proves that Jack's weekly allowance is $10 given the problem conditions -/
theorem jacks_allowance :
  let initial_amount : ℚ := 43
  let weeks : ℕ := 8
  let final_amount : ℚ := 83
  calculate_allowance initial_amount weeks final_amount = 10 := by
  sorry

#eval calculate_allowance 43 8 83

end jacks_allowance_l2667_266715


namespace a_in_range_l2667_266732

/-- A function f(x) = ax^2 + (a-3)x + 1 that is decreasing on [-1, +∞) -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + (a - 3) * x + 1

/-- The property that f is decreasing on [-1, +∞) -/
def is_decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, -1 ≤ x ∧ x < y → f a y < f a x

/-- The theorem stating that if f is decreasing on [-1, +∞), then a is in [-3, 0) -/
theorem a_in_range (a : ℝ) : is_decreasing_on_interval a → a ∈ Set.Icc (-3) 0 :=
sorry

end a_in_range_l2667_266732


namespace sum_of_digits_cube_n_nines_l2667_266768

/-- The sum of digits function for natural numbers -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The function that returns a number composed of n nines -/
def n_nines (n : ℕ) : ℕ := 10^n - 1

theorem sum_of_digits_cube_n_nines (n : ℕ) :
  sum_of_digits ((n_nines n)^3) = 18 * n := by sorry

end sum_of_digits_cube_n_nines_l2667_266768


namespace crayon_count_l2667_266700

theorem crayon_count (num_people : ℕ) (crayons_per_person : ℕ) (h1 : num_people = 3) (h2 : crayons_per_person = 8) : 
  num_people * crayons_per_person = 24 := by
  sorry

end crayon_count_l2667_266700


namespace at_least_one_square_l2667_266764

/-- Represents a rectangle with integer side lengths -/
structure Rectangle where
  width : Nat
  height : Nat
  width_gt_one : width > 1
  height_gt_one : height > 1

/-- Represents a division of a square into rectangles -/
structure SquareDivision where
  side_length : Nat
  rectangles : List Rectangle
  total_rectangles : rectangles.length = 17
  covers_square : (rectangles.map (λ r => r.width * r.height)).sum = side_length * side_length

theorem at_least_one_square (d : SquareDivision) (h : d.side_length = 10) :
  ∃ (r : Rectangle), r ∈ d.rectangles ∧ r.width = r.height := by
  sorry

end at_least_one_square_l2667_266764


namespace wild_weatherman_answers_l2667_266765

/-- Represents the format of the text --/
inductive TextFormat
  | Interview
  | Diary
  | NewsStory
  | Announcement

/-- Represents Sam Champion's childhood career aspiration --/
inductive ChildhoodAspiration
  | SpaceScientist
  | Weatherman
  | NewsReporter
  | Meteorologist

/-- Represents the state of present weather forecasting technology --/
structure WeatherForecastingTechnology where
  moreExact : Bool
  stillImperfect : Bool

/-- Represents the name of the study of weather science --/
inductive WeatherScienceName
  | Meteorology
  | Forecasting
  | Geography
  | EarthScience

/-- The main theorem statement --/
theorem wild_weatherman_answers 
  (text_format : TextFormat)
  (sam_aspiration : ChildhoodAspiration)
  (forecast_tech : WeatherForecastingTechnology)
  (weather_science : WeatherScienceName) :
  text_format = TextFormat.Interview ∧
  sam_aspiration = ChildhoodAspiration.NewsReporter ∧
  forecast_tech.moreExact = true ∧
  forecast_tech.stillImperfect = true ∧
  weather_science = WeatherScienceName.Meteorology :=
by sorry

end wild_weatherman_answers_l2667_266765


namespace painted_cube_theorem_l2667_266756

theorem painted_cube_theorem (n : ℕ) (h : n > 2) :
  6 * (n - 2)^2 = (n - 2)^3 ↔ n = 8 := by
  sorry

end painted_cube_theorem_l2667_266756


namespace additional_distance_at_faster_speed_l2667_266719

/-- Given a person walking at two different speeds for a fixed distance, 
    calculate the additional distance covered at the faster speed in the same time. -/
theorem additional_distance_at_faster_speed 
  (actual_speed : ℝ) 
  (faster_speed : ℝ) 
  (actual_distance : ℝ) 
  (h1 : actual_speed = 10)
  (h2 : faster_speed = 15)
  (h3 : actual_distance = 30)
  : (faster_speed * (actual_distance / actual_speed)) - actual_distance = 15 := by
  sorry

end additional_distance_at_faster_speed_l2667_266719


namespace product_of_numbers_l2667_266761

theorem product_of_numbers (x y : ℝ) (h1 : x + y = 24) (h2 : x^2 + y^2 = 404) : x * y = 86 := by
  sorry

end product_of_numbers_l2667_266761


namespace equal_area_line_slope_l2667_266799

/-- Represents a circle in 2D space -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The problem setup -/
def circles : List Circle := [
  { center := (10, 80), radius := 4 },
  { center := (13, 60), radius := 4 },
  { center := (15, 70), radius := 4 }
]

/-- A line passing through a given point -/
structure Line where
  slope : ℝ
  passesThrough : ℝ × ℝ

/-- Checks if a line divides the total area of circles equally -/
def dividesAreaEqually (l : Line) (cs : List Circle) : Prop := sorry

/-- The main theorem -/
theorem equal_area_line_slope :
  ∃ l : Line, l.passesThrough = (13, 60) ∧ 
    dividesAreaEqually l circles ∧ 
    abs l.slope = 5 := by sorry

end equal_area_line_slope_l2667_266799


namespace correct_life_insights_l2667_266787

/- Define the types of connections -/
inductive ConnectionType
  | Objective
  | Diverse
  | Inevitable
  | Conditional

/- Define the actions related to connections -/
inductive ConnectionAction
  | CannotAdjust
  | EstablishNew
  | EliminateAccidental
  | GraspConditions

/- Define a proposition that represents an insight about connections -/
structure ConnectionInsight where
  type : ConnectionType
  action : ConnectionAction

/- Define the function that determines if an insight is correct -/
def isCorrectInsight (insight : ConnectionInsight) : Prop :=
  (insight.type = ConnectionType.Diverse ∧ insight.action = ConnectionAction.EstablishNew) ∨
  (insight.type = ConnectionType.Conditional ∧ insight.action = ConnectionAction.GraspConditions)

/- The theorem to prove -/
theorem correct_life_insights :
  ∀ (insight : ConnectionInsight),
    isCorrectInsight insight ↔
      (insight.type = ConnectionType.Diverse ∧ insight.action = ConnectionAction.EstablishNew) ∨
      (insight.type = ConnectionType.Conditional ∧ insight.action = ConnectionAction.GraspConditions) :=
by sorry


end correct_life_insights_l2667_266787


namespace absolute_value_inequality_l2667_266766

theorem absolute_value_inequality (x : ℝ) : 
  (2 ≤ |x - 3| ∧ |x - 3| ≤ 6) ↔ ((-3 ≤ x ∧ x ≤ 1) ∨ (5 ≤ x ∧ x ≤ 9)) :=
by sorry

end absolute_value_inequality_l2667_266766


namespace intersection_M_N_l2667_266760

def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.log x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2 + 1}

theorem intersection_M_N : M ∩ N = Set.Ici 1 := by sorry

end intersection_M_N_l2667_266760


namespace and_or_sufficient_not_necessary_l2667_266795

theorem and_or_sufficient_not_necessary :
  (∃ p q : Prop, (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q)) :=
by sorry

end and_or_sufficient_not_necessary_l2667_266795


namespace billy_cherries_l2667_266793

theorem billy_cherries (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 74 → remaining = 2 → eaten = initial - remaining → eaten = 72 := by
  sorry

end billy_cherries_l2667_266793


namespace quarter_equals_two_eighths_l2667_266713

theorem quarter_equals_two_eighths : (1 : ℚ) / 4 = 1 / 8 + 1 / 8 := by
  sorry

end quarter_equals_two_eighths_l2667_266713


namespace hyperbola_focal_length_l2667_266726

/-- The focal length of a hyperbola with equation x²- y²/4 = 1 is 2√5 -/
theorem hyperbola_focal_length : 
  let h : Set ((ℝ × ℝ) → Prop) := {f | ∃ (x y : ℝ), f (x, y) ↔ x^2 - y^2/4 = 1}
  ∃ (f : (ℝ × ℝ) → Prop), f ∈ h ∧ 
    (∃ (a b c : ℝ), a^2 = 1 ∧ b^2 = 4 ∧ c^2 = a^2 + b^2 ∧ 2*c = 2*Real.sqrt 5) :=
by
  sorry

end hyperbola_focal_length_l2667_266726


namespace average_of_xyz_l2667_266752

theorem average_of_xyz (x y z : ℝ) : 
  x = 3 → y = 2 * x → z = 3 * y → (x + y + z) / 3 = 9 := by
  sorry

end average_of_xyz_l2667_266752


namespace intersection_of_A_and_B_l2667_266745

-- Define set A as the domain of y = lg x
def A : Set ℝ := {x | x > 0}

-- Define set B
def B : Set ℝ := {0, 1, 2}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {1, 2} := by sorry

end intersection_of_A_and_B_l2667_266745


namespace count_congruent_integers_l2667_266769

theorem count_congruent_integers (n : ℕ) (m : ℕ) (a : ℕ) (b : ℕ) : 
  (Finset.filter (fun x => x > 0 ∧ x < n ∧ x % m = a) (Finset.range n)).card = b + 1 :=
by
  sorry

#check count_congruent_integers 1500 13 7 114

end count_congruent_integers_l2667_266769


namespace no_valid_solution_l2667_266791

-- Define the equation
def equation (x : ℝ) : Prop :=
  (36 - x) - (14 - x) = 2 * ((36 - x) - (18 - x))

-- Theorem stating that there is no valid solution
theorem no_valid_solution : ¬∃ (x : ℝ), x ≥ 0 ∧ equation x :=
sorry

end no_valid_solution_l2667_266791


namespace sector_max_area_l2667_266740

/-- Given a sector with perimeter 4, its area is maximized when the central angle equals 2 -/
theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 4) :
  let α := l / r
  let area := (1 / 2) * r * l
  (∀ r' l', 2 * r' + l' = 4 → (1 / 2) * r' * l' ≤ area) →
  α = 2 :=
by sorry

end sector_max_area_l2667_266740


namespace complex_number_modulus_l2667_266786

theorem complex_number_modulus (a : ℝ) (i : ℂ) : 
  a < 0 → 
  i * i = -1 → 
  Complex.abs (a * i / (1 + 2 * i)) = Real.sqrt 5 → 
  a = -5 := by
sorry

end complex_number_modulus_l2667_266786


namespace largest_int_less_100_rem_4_div_7_l2667_266744

theorem largest_int_less_100_rem_4_div_7 : ∃ n : ℕ, n < 100 ∧ n % 7 = 4 ∧ ∀ m : ℕ, m < 100 → m % 7 = 4 → m ≤ n :=
by sorry

end largest_int_less_100_rem_4_div_7_l2667_266744


namespace floor_neg_sqrt_64_over_9_l2667_266721

theorem floor_neg_sqrt_64_over_9 : ⌊-Real.sqrt (64 / 9)⌋ = -3 := by
  sorry

end floor_neg_sqrt_64_over_9_l2667_266721


namespace soccer_team_points_l2667_266763

theorem soccer_team_points : ∀ (total_games wins losses draws : ℕ)
  (points_per_win points_per_draw points_per_loss : ℕ),
  total_games = 20 →
  wins = 14 →
  losses = 2 →
  draws = total_games - wins - losses →
  points_per_win = 3 →
  points_per_draw = 1 →
  points_per_loss = 0 →
  wins * points_per_win + draws * points_per_draw + losses * points_per_loss = 46 :=
by sorry

end soccer_team_points_l2667_266763


namespace car_speed_problem_l2667_266736

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time : ℝ) (new_speed : ℝ) :
  original_time = 12 →
  new_time = 4 →
  new_speed = 30 →
  distance = new_speed * new_time →
  distance = (distance / original_time) * original_time →
  distance / original_time = 10 := by
sorry

end car_speed_problem_l2667_266736


namespace percent_of_200_l2667_266758

theorem percent_of_200 : (25 / 100) * 200 = 50 := by sorry

end percent_of_200_l2667_266758


namespace factorial_calculation_l2667_266750

theorem factorial_calculation : (4 * Nat.factorial 6 + 32 * Nat.factorial 5) / Nat.factorial 7 = 4 / 3 := by
  sorry

end factorial_calculation_l2667_266750


namespace table_satisfies_conditions_l2667_266723

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def is_consecutive_prime_product (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ q = p + 2 ∧ n = p * q

def table : Matrix (Fin 4) (Fin 4) ℕ :=
  ![![2, 1, 8, 7],
    ![7, 3, 8, 7],
    ![7, 7, 4, 4],
    ![7, 8, 4, 4]]

theorem table_satisfies_conditions :
  (∀ i j, table i j < 10) ∧
  (∀ i, table i 0 ≠ 0) ∧
  (∃ p q : ℕ, is_prime p ∧ is_prime q ∧ 
    1000 * table 0 0 + 100 * table 0 1 + 10 * table 0 2 + table 0 3 = p^q) ∧
  (is_consecutive_prime_product 
    (1000 * table 1 0 + 100 * table 1 1 + 10 * table 1 2 + table 1 3)) ∧
  (is_perfect_square 
    (1000 * table 2 0 + 100 * table 2 1 + 10 * table 2 2 + table 2 3)) ∧
  ((1000 * table 3 0 + 100 * table 3 1 + 10 * table 3 2 + table 3 3) % 37 = 0) :=
by sorry

end table_satisfies_conditions_l2667_266723


namespace sufficient_not_necessary_condition_l2667_266772

theorem sufficient_not_necessary_condition :
  (∃ x : ℝ, x^2 + x = 0 ∧ x ≠ -1) ∧
  (∀ x : ℝ, x = -1 → x^2 + x = 0) := by
  sorry

end sufficient_not_necessary_condition_l2667_266772


namespace four_digit_sum_l2667_266737

/-- Given four distinct non-zero digits, the sum of all four-digit numbers formed using these digits without repetition is 73,326 if and only if the digits are 1, 2, 3, and 5. -/
theorem four_digit_sum (a b c d : ℕ) : 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0) →  -- non-zero digits
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- distinct digits
  (6 * (a + b + c + d) * 1111 = 73326) →  -- sum condition
  ({a, b, c, d} : Set ℕ) = {1, 2, 3, 5} :=
sorry

end four_digit_sum_l2667_266737


namespace point_in_second_quadrant_l2667_266741

theorem point_in_second_quadrant (θ : Real) (h : π/2 < θ ∧ θ < π) :
  let P := (Real.tan θ, Real.sin θ)
  P.1 < 0 ∧ P.2 > 0 :=
by sorry

end point_in_second_quadrant_l2667_266741


namespace sqrt_mixed_number_simplification_l2667_266717

theorem sqrt_mixed_number_simplification :
  Real.sqrt (8 + 9/16) = Real.sqrt 137 / 4 := by sorry

end sqrt_mixed_number_simplification_l2667_266717
