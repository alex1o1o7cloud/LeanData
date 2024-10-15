import Mathlib

namespace NUMINAMATH_CALUDE_hotel_revenue_calculation_l3031_303188

/-- The total revenue of a hotel for one night, given the number of single and double rooms booked and their respective prices. -/
def hotel_revenue (total_rooms single_price double_price double_rooms : ℕ) : ℕ :=
  let single_rooms := total_rooms - double_rooms
  single_rooms * single_price + double_rooms * double_price

/-- Theorem stating that under the given conditions, the hotel's revenue for one night is $14,000. -/
theorem hotel_revenue_calculation :
  hotel_revenue 260 35 60 196 = 14000 := by
  sorry

end NUMINAMATH_CALUDE_hotel_revenue_calculation_l3031_303188


namespace NUMINAMATH_CALUDE_line_segment_param_product_l3031_303153

/-- Given a line segment connecting (1, -3) and (6, 9), parameterized by x = at + b and y = ct + d
    where 0 ≤ t ≤ 1 and t = 0 corresponds to (1, -3), prove that (a+b) × (c+d) = 54. -/
theorem line_segment_param_product (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t → t ≤ 1 → 
    ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (1 = b ∧ -3 = d) →
  (6 = a + b ∧ 9 = c + d) →
  (a + b) * (c + d) = 54 := by
sorry

end NUMINAMATH_CALUDE_line_segment_param_product_l3031_303153


namespace NUMINAMATH_CALUDE_bah_equivalent_to_yahs_l3031_303148

/-- Conversion rate between bahs and rahs -/
def bah_to_rah : ℚ := 18 / 10

/-- Conversion rate between rahs and yahs -/
def rah_to_yah : ℚ := 10 / 6

/-- The number of yahs to convert -/
def yahs_to_convert : ℕ := 1500

theorem bah_equivalent_to_yahs : 
  ∃ (n : ℕ), n * bah_to_rah * rah_to_yah = yahs_to_convert ∧ n = 500 := by
  sorry

end NUMINAMATH_CALUDE_bah_equivalent_to_yahs_l3031_303148


namespace NUMINAMATH_CALUDE_unique_assignment_l3031_303137

/- Define the girls and colors as enums -/
inductive Girl : Type
  | Katya | Olya | Liza | Rita

inductive Color : Type
  | Pink | Green | Yellow | Blue

/- Define the assignment of colors to girls -/
def assignment : Girl → Color
  | Girl.Katya => Color.Green
  | Girl.Olya => Color.Blue
  | Girl.Liza => Color.Pink
  | Girl.Rita => Color.Yellow

/- Define the circular arrangement of girls -/
def nextGirl : Girl → Girl
  | Girl.Katya => Girl.Olya
  | Girl.Olya => Girl.Liza
  | Girl.Liza => Girl.Rita
  | Girl.Rita => Girl.Katya

/- Define the conditions -/
def conditions (a : Girl → Color) : Prop :=
  (a Girl.Katya ≠ Color.Pink ∧ a Girl.Katya ≠ Color.Blue) ∧
  (∃ g : Girl, a g = Color.Green ∧ 
    ((nextGirl g = Girl.Liza ∧ a (nextGirl (nextGirl g)) = Color.Yellow) ∨
     (nextGirl (nextGirl g) = Girl.Liza ∧ a (nextGirl g) = Color.Yellow))) ∧
  (a Girl.Rita ≠ Color.Green ∧ a Girl.Rita ≠ Color.Blue) ∧
  (∃ g : Girl, nextGirl g = Girl.Olya ∧ nextGirl (nextGirl g) = Girl.Rita ∧ 
    (a g = Color.Pink ∨ a (nextGirl (nextGirl (nextGirl g))) = Color.Pink))

/- Theorem statement -/
theorem unique_assignment : 
  ∀ a : Girl → Color, conditions a → a = assignment :=
sorry

end NUMINAMATH_CALUDE_unique_assignment_l3031_303137


namespace NUMINAMATH_CALUDE_y_coordinate_of_P_l3031_303126

/-- A line through the origin equidistant from two points -/
structure EquidistantLine where
  slope : ℝ
  P : ℝ × ℝ
  Q : ℝ × ℝ
  origin_line : slope * P.1 = P.2 ∧ slope * Q.1 = Q.2
  equidistant : (P.1 - 0)^2 + (P.2 - 0)^2 = (Q.1 - 0)^2 + (Q.2 - 0)^2

/-- Theorem: Given the conditions, the y-coordinate of P is 3.2 -/
theorem y_coordinate_of_P (L : EquidistantLine)
  (h_slope : L.slope = 0.8)
  (h_x_coord : L.P.1 = 4) :
  L.P.2 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_y_coordinate_of_P_l3031_303126


namespace NUMINAMATH_CALUDE_mona_unique_players_l3031_303185

/-- The number of unique players Mona grouped with in a video game --/
def unique_players (groups : ℕ) (players_per_group : ℕ) (groups_with_two_repeats : ℕ) (groups_with_one_repeat : ℕ) : ℕ :=
  groups * players_per_group - (2 * groups_with_two_repeats + groups_with_one_repeat)

/-- Theorem stating the number of unique players Mona grouped with --/
theorem mona_unique_players :
  unique_players 25 4 8 5 = 79 := by
  sorry

end NUMINAMATH_CALUDE_mona_unique_players_l3031_303185


namespace NUMINAMATH_CALUDE_value_of_a_l3031_303156

theorem value_of_a (a : ℝ) (h : a + a/4 = 5/2) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3031_303156


namespace NUMINAMATH_CALUDE_correct_possible_values_l3031_303129

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The set of possible values for 'a' -/
def PossibleValues : Set ℝ := {1/3, 3, -6}

/-- Function to count the number of intersection points between three lines -/
def countIntersections (l1 l2 l3 : Line) : ℕ := sorry

/-- Theorem stating that the set of possible values of 'a' is correct -/
theorem correct_possible_values :
  ∀ a : ℝ,
  (∃ l3 : Line,
    l3.a = a ∧ l3.b = 3 ∧ l3.c = -5 ∧
    countIntersections ⟨1, 1, 1⟩ ⟨2, -1, 8⟩ l3 ≤ 2) ↔
  a ∈ PossibleValues :=
sorry

end NUMINAMATH_CALUDE_correct_possible_values_l3031_303129


namespace NUMINAMATH_CALUDE_intersection_symmetry_implies_k_minus_m_eq_four_l3031_303169

/-- The line equation y = kx + 1 -/
def line_equation (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

/-- The circle equation x² + y² + kx + my - 4 = 0 -/
def circle_equation (k m : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 + k*x + m*y - 4 = 0

/-- The symmetry line equation x + y - 1 = 0 -/
def symmetry_line (x y : ℝ) : Prop :=
  x + y - 1 = 0

/-- Two points (x₁, y₁) and (x₂, y₂) are symmetric with respect to the line x + y - 1 = 0 -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (x₁ + x₂) / 2 + (y₁ + y₂) / 2 - 1 = 0

theorem intersection_symmetry_implies_k_minus_m_eq_four (k m : ℝ) :
  (∃ x₁ y₁ x₂ y₂ : ℝ,
    line_equation k x₁ y₁ ∧
    line_equation k x₂ y₂ ∧
    circle_equation k m x₁ y₁ ∧
    circle_equation k m x₂ y₂ ∧
    symmetric_points x₁ y₁ x₂ y₂) →
  k - m = 4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_symmetry_implies_k_minus_m_eq_four_l3031_303169


namespace NUMINAMATH_CALUDE_alex_shirt_count_l3031_303175

/-- Given that:
  - Alex has some new shirts
  - Joe has 3 more new shirts than Alex
  - Ben has a certain number of new shirts more than Joe
  - Ben has 15 new shirts
Prove that Alex has 12 new shirts. -/
theorem alex_shirt_count :
  ∀ (alex_shirts joe_shirts ben_shirts : ℕ),
  joe_shirts = alex_shirts + 3 →
  ben_shirts > joe_shirts →
  ben_shirts = 15 →
  alex_shirts = 12 := by
sorry

end NUMINAMATH_CALUDE_alex_shirt_count_l3031_303175


namespace NUMINAMATH_CALUDE_triangle_side_lengths_l3031_303179

theorem triangle_side_lengths :
  ∀ x y z : ℕ,
    x ≥ y ∧ y ≥ z →
    x + y + z = 240 →
    3 * x - 2 * (y + z) = 5 * z + 10 →
    ((x = 113 ∧ y = 112 ∧ z = 15) ∨
     (x = 114 ∧ y = 110 ∧ z = 16) ∨
     (x = 115 ∧ y = 108 ∧ z = 17) ∨
     (x = 116 ∧ y = 106 ∧ z = 18) ∨
     (x = 117 ∧ y = 104 ∧ z = 19) ∨
     (x = 118 ∧ y = 102 ∧ z = 20) ∨
     (x = 119 ∧ y = 100 ∧ z = 21)) :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_side_lengths_l3031_303179


namespace NUMINAMATH_CALUDE_b_share_is_2200_l3031_303118

/- Define the investments and a's share -/
def investment_a : ℕ := 7000
def investment_b : ℕ := 11000
def investment_c : ℕ := 18000
def share_a : ℕ := 1400

/- Define the function to calculate b's share -/
def calculate_b_share (inv_a inv_b inv_c share_a : ℕ) : ℕ :=
  let total_ratio := inv_a + inv_b + inv_c
  let total_profit := share_a * total_ratio / inv_a
  inv_b * total_profit / total_ratio

/- Theorem statement -/
theorem b_share_is_2200 : 
  calculate_b_share investment_a investment_b investment_c share_a = 2200 := by
  sorry


end NUMINAMATH_CALUDE_b_share_is_2200_l3031_303118


namespace NUMINAMATH_CALUDE_mountain_height_theorem_l3031_303102

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the problem setup -/
structure MountainProblem where
  A : Point3D
  C : Point3D
  P : Point3D
  F : Point3D
  AC_distance : ℝ
  AP_distance : ℝ
  C_elevation : ℝ
  AC_angle : ℝ
  AP_angle : ℝ
  magnetic_declination : ℝ
  latitude : ℝ

/-- The main theorem to prove -/
theorem mountain_height_theorem (problem : MountainProblem) :
  problem.AC_distance = 2200 →
  problem.AP_distance = 400 →
  problem.C_elevation = 550 →
  problem.AC_angle = 71 →
  problem.AP_angle = 64 →
  problem.magnetic_declination = 2 →
  problem.latitude = 49 →
  ∃ (height : ℝ), abs (height - 420) < 1 ∧ height = problem.A.z :=
sorry


end NUMINAMATH_CALUDE_mountain_height_theorem_l3031_303102


namespace NUMINAMATH_CALUDE_det_sin_matrix_zero_l3031_303198

theorem det_sin_matrix_zero :
  let A : Matrix (Fin 3) (Fin 3) ℝ := λ i j =>
    match i, j with
    | 0, 0 => Real.sin 1
    | 0, 1 => Real.sin 2
    | 0, 2 => Real.sin 3
    | 1, 0 => Real.sin 4
    | 1, 1 => Real.sin 5
    | 1, 2 => Real.sin 6
    | 2, 0 => Real.sin 7
    | 2, 1 => Real.sin 8
    | 2, 2 => Real.sin 9
  Matrix.det A = 0 :=
by sorry

end NUMINAMATH_CALUDE_det_sin_matrix_zero_l3031_303198


namespace NUMINAMATH_CALUDE_factor_expression_l3031_303116

theorem factor_expression (x : ℝ) :
  (10 * x^3 + 50 * x^2 - 5) - (-5 * x^3 + 15 * x^2 - 5) = 5 * x^2 * (3 * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3031_303116


namespace NUMINAMATH_CALUDE_det_of_matrix_l3031_303100

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, -4; 2, 3]

theorem det_of_matrix : Matrix.det matrix = 23 := by sorry

end NUMINAMATH_CALUDE_det_of_matrix_l3031_303100


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3031_303120

/-- The function f(x) = a^(2-x) + 2 passes through the point (2, 3) for all a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(2 - x) + 2
  f 2 = 3 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3031_303120


namespace NUMINAMATH_CALUDE_xiaoman_dumpling_probability_l3031_303150

theorem xiaoman_dumpling_probability :
  let total_dumplings : ℕ := 10
  let egg_dumplings : ℕ := 3
  let probability : ℚ := egg_dumplings / total_dumplings
  probability = 3 / 10 := by
sorry

end NUMINAMATH_CALUDE_xiaoman_dumpling_probability_l3031_303150


namespace NUMINAMATH_CALUDE_cricket_run_rate_l3031_303173

/-- Calculates the required run rate for the remaining overs in a cricket game. -/
def required_run_rate (total_overs : ℕ) (first_overs : ℕ) (first_run_rate : ℚ) (target : ℕ) : ℚ :=
  let remaining_overs := total_overs - first_overs
  let runs_scored := first_run_rate * first_overs
  let runs_needed := target - runs_scored
  runs_needed / remaining_overs

/-- Theorem stating the required run rate for the given cricket game scenario. -/
theorem cricket_run_rate : required_run_rate 50 10 (34/10) 282 = 62/10 := by
  sorry

end NUMINAMATH_CALUDE_cricket_run_rate_l3031_303173


namespace NUMINAMATH_CALUDE_garden_area_l3031_303158

-- Define the garden structure
structure Garden where
  side_length : ℝ
  perimeter : ℝ
  area : ℝ

-- Define the conditions
def garden_conditions (g : Garden) : Prop :=
  g.perimeter = 4 * g.side_length ∧
  g.area = g.side_length * g.side_length ∧
  1500 = 30 * g.side_length ∧
  1500 = 15 * g.perimeter

-- Theorem statement
theorem garden_area (g : Garden) (h : garden_conditions g) : g.area = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l3031_303158


namespace NUMINAMATH_CALUDE_radius_of_larger_circle_l3031_303122

/-- Given a configuration of four circles of radius 2 that are externally tangent to two others
    and internally tangent to a larger circle, the radius of the larger circle is 2√3 + 2. -/
theorem radius_of_larger_circle (r : ℝ) (h1 : r > 0) :
  let small_radius : ℝ := 2
  let diagonal : ℝ := 4 * Real.sqrt 2
  let large_radius : ℝ := r
  (small_radius > 0) →
  (diagonal = 4 * Real.sqrt 2) →
  (large_radius = 2 * Real.sqrt 3 + 2) :=
by
  sorry

#check radius_of_larger_circle

end NUMINAMATH_CALUDE_radius_of_larger_circle_l3031_303122


namespace NUMINAMATH_CALUDE_tetrahedron_volume_from_pentagon_tetrahedron_volume_proof_l3031_303117

/-- The volume of a tetrahedron formed from a regular pentagon -/
theorem tetrahedron_volume_from_pentagon (side_length : ℝ) 
  (h_side : side_length = 1) : ℝ :=
let diagonal_length := (1 + Real.sqrt 5) / 2
let base_area := Real.sqrt 3 / 4 * side_length ^ 2
let height := Real.sqrt ((5 + 2 * Real.sqrt 5) / 4)
let volume := (1 / 3) * base_area * height
(1 + Real.sqrt 5) / 24

/-- The theorem statement -/
theorem tetrahedron_volume_proof : 
  ∃ (v : ℝ), tetrahedron_volume_from_pentagon 1 rfl = v ∧ v = (1 + Real.sqrt 5) / 24 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_from_pentagon_tetrahedron_volume_proof_l3031_303117


namespace NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_eq_five_halves_l3031_303178

theorem sqrt_a_div_sqrt_b_eq_five_halves (a b : ℝ) 
  (h : (1/3)^2 + (1/4)^2 = ((1/5)^2 + (1/6)^2) * (25*a)/(61*b)) : 
  Real.sqrt a / Real.sqrt b = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_a_div_sqrt_b_eq_five_halves_l3031_303178


namespace NUMINAMATH_CALUDE_candy_problem_l3031_303160

/-- The number of candy pieces remaining in a bowl after some are taken. -/
def remaining_candy (initial : ℕ) (taken_by_talitha : ℕ) (taken_by_solomon : ℕ) : ℕ :=
  initial - (taken_by_talitha + taken_by_solomon)

/-- Theorem stating that given the initial amount and amounts taken by Talitha and Solomon,
    the remaining candy pieces are 88. -/
theorem candy_problem :
  remaining_candy 349 108 153 = 88 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l3031_303160


namespace NUMINAMATH_CALUDE_solution_to_equation_l3031_303146

theorem solution_to_equation :
  ∃ x y : ℝ, 3 * x^2 - 12 * y^2 + 6 * x = 0 ∧ x = 0 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_solution_to_equation_l3031_303146


namespace NUMINAMATH_CALUDE_pencil_difference_l3031_303189

/-- The cost of pencils Jamar bought -/
def jamar_cost : ℚ := 325 / 100

/-- The cost of pencils Sharona bought -/
def sharona_cost : ℚ := 425 / 100

/-- The minimum number of pencils Jamar bought -/
def jamar_min_pencils : ℕ := 15

/-- The cost difference between Sharona's and Jamar's purchases -/
def cost_difference : ℚ := sharona_cost - jamar_cost

/-- The theorem stating the difference in the number of pencils bought -/
theorem pencil_difference : ∃ (jamar_pencils sharona_pencils : ℕ) (price_per_pencil : ℚ), 
  jamar_pencils ≥ jamar_min_pencils ∧
  price_per_pencil > 1 / 100 ∧
  jamar_cost = jamar_pencils * price_per_pencil ∧
  sharona_cost = sharona_pencils * price_per_pencil ∧
  sharona_pencils - jamar_pencils = 5 := by
  sorry

end NUMINAMATH_CALUDE_pencil_difference_l3031_303189


namespace NUMINAMATH_CALUDE_percentage_without_full_time_jobs_l3031_303154

theorem percentage_without_full_time_jobs :
  let total_parents : ℝ := 100
  let mothers : ℝ := 0.6 * total_parents
  let fathers : ℝ := 0.4 * total_parents
  let mothers_with_jobs : ℝ := (7/8) * mothers
  let fathers_with_jobs : ℝ := (3/4) * fathers
  let parents_with_jobs : ℝ := mothers_with_jobs + fathers_with_jobs
  let parents_without_jobs : ℝ := total_parents - parents_with_jobs
  (parents_without_jobs / total_parents) * 100 = 18 :=
by sorry

end NUMINAMATH_CALUDE_percentage_without_full_time_jobs_l3031_303154


namespace NUMINAMATH_CALUDE_john_uber_profit_l3031_303115

/-- Calculates the profit from driving Uber given the income, initial car cost, and trade-in value. -/
def uberProfit (income : ℕ) (carCost : ℕ) (tradeInValue : ℕ) : ℕ :=
  income - (carCost - tradeInValue)

/-- Proves that John's profit from driving Uber is $18,000 given the specified conditions. -/
theorem john_uber_profit :
  let income : ℕ := 30000
  let carCost : ℕ := 18000
  let tradeInValue : ℕ := 6000
  uberProfit income carCost tradeInValue = 18000 := by
  sorry

#eval uberProfit 30000 18000 6000

end NUMINAMATH_CALUDE_john_uber_profit_l3031_303115


namespace NUMINAMATH_CALUDE_remaining_boys_average_weight_l3031_303170

/-- The average weight of the remaining 8 boys given the following conditions:
  - There are 20 boys with an average weight of 50.25 kg
  - There are 8 remaining boys
  - The average weight of all 28 boys is 48.792857142857144 kg
-/
theorem remaining_boys_average_weight :
  let num_group1 : ℕ := 20
  let avg_group1 : ℝ := 50.25
  let num_group2 : ℕ := 8
  let total_num : ℕ := num_group1 + num_group2
  let total_avg : ℝ := 48.792857142857144
  
  ((num_group1 : ℝ) * avg_group1 + (num_group2 : ℝ) * avg_group2) / (total_num : ℝ) = total_avg →
  avg_group2 = 45.15
  := by sorry

end NUMINAMATH_CALUDE_remaining_boys_average_weight_l3031_303170


namespace NUMINAMATH_CALUDE_total_skips_theorem_l3031_303140

/-- Represents the number of skips a person can do with one rock -/
structure SkipAbility :=
  (skips : ℕ)

/-- Represents the number of rocks a person skipped -/
structure RocksSkipped :=
  (rocks : ℕ)

/-- Calculates the total skips for a person -/
def totalSkips (ability : SkipAbility) (skipped : RocksSkipped) : ℕ :=
  ability.skips * skipped.rocks

theorem total_skips_theorem 
  (bob_ability : SkipAbility)
  (jim_ability : SkipAbility)
  (sally_ability : SkipAbility)
  (bob_skipped : RocksSkipped)
  (jim_skipped : RocksSkipped)
  (sally_skipped : RocksSkipped)
  (h1 : bob_ability.skips = 12)
  (h2 : jim_ability.skips = 15)
  (h3 : sally_ability.skips = 18)
  (h4 : bob_skipped.rocks = 10)
  (h5 : jim_skipped.rocks = 8)
  (h6 : sally_skipped.rocks = 12) :
  totalSkips bob_ability bob_skipped + 
  totalSkips jim_ability jim_skipped + 
  totalSkips sally_ability sally_skipped = 456 := by
  sorry

#check total_skips_theorem

end NUMINAMATH_CALUDE_total_skips_theorem_l3031_303140


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3031_303133

variable (θ : Real)
variable (α : Real)

/-- Given tan θ = 2, prove the following statements -/
theorem trigonometric_identities (h : Real.tan θ = 2) :
  ((Real.sin α + Real.sqrt 2 * Real.cos α) / (Real.sin α - Real.sqrt 2 * Real.cos α) = 3 + 2 * Real.sqrt 2) ∧
  (Real.sin θ ^ 2 + Real.sin θ * Real.cos θ - 2 * Real.cos θ ^ 2 = 4 / 5) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3031_303133


namespace NUMINAMATH_CALUDE_no_prime_solution_l3031_303104

theorem no_prime_solution : ¬∃ (p q : Nat), Prime p ∧ Prime q ∧ p > 5 ∧ q > 5 ∧ (p * q ∣ (5^p - 2^p) * (5^q - 2^q)) := by
  sorry

end NUMINAMATH_CALUDE_no_prime_solution_l3031_303104


namespace NUMINAMATH_CALUDE_work_completion_time_l3031_303181

/-- Given a work that can be completed by person a in 15 days and by person b in 30 days,
    prove that a and b together can complete the work in 10 days. -/
theorem work_completion_time (work : ℝ) (a b : ℝ) 
    (ha : a * 15 = work) 
    (hb : b * 30 = work) : 
    (a + b) * 10 = work := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l3031_303181


namespace NUMINAMATH_CALUDE_duck_weight_calculation_l3031_303112

theorem duck_weight_calculation (num_ducks : ℕ) (cost_per_duck : ℚ) (selling_price_per_pound : ℚ) (profit : ℚ) : 
  num_ducks = 30 →
  cost_per_duck = 10 →
  selling_price_per_pound = 5 →
  profit = 300 →
  (profit + num_ducks * cost_per_duck) / (selling_price_per_pound * num_ducks) = 4 := by
sorry

end NUMINAMATH_CALUDE_duck_weight_calculation_l3031_303112


namespace NUMINAMATH_CALUDE_square_diagonal_and_inscribed_circle_area_l3031_303168

/-- Given a square with side length 40√3 cm, this theorem proves the length of its diagonal
    and the area of its inscribed circle. -/
theorem square_diagonal_and_inscribed_circle_area 
  (side_length : ℝ) 
  (h_side : side_length = 40 * Real.sqrt 3) :
  ∃ (diagonal_length : ℝ) (inscribed_circle_area : ℝ),
    diagonal_length = 40 * Real.sqrt 6 ∧
    inscribed_circle_area = 1200 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_square_diagonal_and_inscribed_circle_area_l3031_303168


namespace NUMINAMATH_CALUDE_smallest_book_count_l3031_303138

theorem smallest_book_count (physics chemistry biology : ℕ) : 
  physics = 3 * (chemistry / 2) →  -- ratio of physics to chemistry is 3:2
  4 * biology = 3 * chemistry →    -- ratio of chemistry to biology is 4:3
  physics + chemistry + biology > 0 →  -- total number of books is more than 0
  ∀ n : ℕ, n > 0 → 
    (∃ p c b : ℕ, p = 3 * (c / 2) ∧ 4 * b = 3 * c ∧ p + c + b = n) →
    n ≥ 15 :=
by sorry

end NUMINAMATH_CALUDE_smallest_book_count_l3031_303138


namespace NUMINAMATH_CALUDE_least_product_of_two_primes_above_30_l3031_303135

theorem least_product_of_two_primes_above_30 (p q : ℕ) : 
  Prime p → Prime q → p ≠ q → p > 30 → q > 30 → 
  ∀ r s : ℕ, Prime r → Prime s → r ≠ s → r > 30 → s > 30 → 
  p * q ≤ r * s := by
  sorry

end NUMINAMATH_CALUDE_least_product_of_two_primes_above_30_l3031_303135


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l3031_303128

/-- A function satisfying the given functional equation -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) * f (x - y) = (f x + f y)^2 - 6 * x^2 * f y + 2 * x^2 * y^2

theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ f = fun x ↦ x^2 :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l3031_303128


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3031_303187

theorem complex_equation_solution (a : ℝ) : 
  Complex.abs ((a + Complex.I) / Complex.I) = 2 → a = Real.sqrt 3 ∨ a = -Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3031_303187


namespace NUMINAMATH_CALUDE_min_value_theorem_l3031_303166

theorem min_value_theorem (x : ℝ) (h1 : x > 0) (h2 : Real.log x + 1 ≤ x) :
  (x^2 - Real.log x + x) / x ≥ 2 ∧
  (∃ y > 0, (y^2 - Real.log y + y) / y = 2) :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3031_303166


namespace NUMINAMATH_CALUDE_multiple_properties_l3031_303149

/-- Given that c is a multiple of 4 and d is a multiple of 8, prove the following statements -/
theorem multiple_properties (c d : ℤ) 
  (hc : ∃ k : ℤ, c = 4 * k) 
  (hd : ∃ m : ℤ, d = 8 * m) : 
  (∃ n : ℤ, d = 4 * n) ∧ 
  (∃ p : ℤ, c - d = 4 * p) ∧ 
  (∃ q : ℤ, c - d = 2 * q) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l3031_303149


namespace NUMINAMATH_CALUDE_units_digit_of_product_over_1000_l3031_303139

theorem units_digit_of_product_over_1000 : 
  (20 * 21 * 22 * 23 * 24 * 25) / 1000 ≡ 2 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_product_over_1000_l3031_303139


namespace NUMINAMATH_CALUDE_living_room_area_l3031_303106

/-- Given a rectangular carpet covering 60% of a room's floor area,
    if the carpet measures 4 feet by 9 feet,
    then the total floor area of the room is 60 square feet. -/
theorem living_room_area
  (carpet_length : ℝ)
  (carpet_width : ℝ)
  (carpet_coverage : ℝ)
  (h1 : carpet_length = 4)
  (h2 : carpet_width = 9)
  (h3 : carpet_coverage = 0.6)
  : carpet_length * carpet_width / carpet_coverage = 60 := by
  sorry

end NUMINAMATH_CALUDE_living_room_area_l3031_303106


namespace NUMINAMATH_CALUDE_q_investment_correct_l3031_303143

/-- Represents the investment of two people in a business -/
structure Business where
  p_investment : ℕ
  q_investment : ℕ
  profit_ratio : Rat

/-- The business scenario with given conditions -/
def given_business : Business where
  p_investment := 40000
  q_investment := 60000
  profit_ratio := 2 / 3

/-- Theorem stating that q's investment is correct given the conditions -/
theorem q_investment_correct (b : Business) : 
  b.p_investment = 40000 ∧ 
  b.profit_ratio = 2 / 3 → 
  b.q_investment = 60000 := by
  sorry

#check q_investment_correct given_business

end NUMINAMATH_CALUDE_q_investment_correct_l3031_303143


namespace NUMINAMATH_CALUDE_min_value_of_a_l3031_303199

theorem min_value_of_a (a b c : ℝ) (ha : a > 0) (hroots : ∃ x y : ℝ, 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 2 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) (hineq : ∀ c' : ℝ, c' ≥ 1 → 25 * a + 10 * b + 4 * c' ≥ 4) : a ≥ 16/25 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_a_l3031_303199


namespace NUMINAMATH_CALUDE_square_areas_square_areas_concrete_l3031_303190

/-- Given a square with area 100, prove the areas of the inscribed square and right triangle --/
theorem square_areas (S : Real) (h1 : S^2 = 100) :
  let small_square_area := S^2 / 4
  let right_triangle_area := S^2 / 16
  (small_square_area = 50) ∧ (right_triangle_area = 12.5) := by
  sorry

/-- Alternative formulation using concrete numbers --/
theorem square_areas_concrete :
  let large_square_area := 100
  let small_square_area := large_square_area / 2
  let right_triangle_area := large_square_area / 8
  (small_square_area = 50) ∧ (right_triangle_area = 12.5) := by
  sorry

end NUMINAMATH_CALUDE_square_areas_square_areas_concrete_l3031_303190


namespace NUMINAMATH_CALUDE_complex_modulus_example_l3031_303131

theorem complex_modulus_example : Complex.abs (3 - 10*I) = Real.sqrt 109 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_example_l3031_303131


namespace NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l3031_303164

theorem complex_arithmetic_evaluation : 2 - 2 * (2 - 2 * (2 - 2 * (4 - 2))) = -10 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_evaluation_l3031_303164


namespace NUMINAMATH_CALUDE_base_eight_addition_l3031_303132

/-- Converts a number from base b to base 10 -/
def toBase10 (digits : List Nat) (b : Nat) : Nat :=
  digits.foldr (fun d acc => d + b * acc) 0

/-- Adds two numbers in base b -/
def addInBase (n1 n2 : List Nat) (b : Nat) : List Nat :=
  sorry

theorem base_eight_addition : ∃ b : Nat, 
  b > 1 ∧ 
  addInBase [4, 5, 2] [3, 1, 6] b = [7, 7, 0] ∧
  b = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_addition_l3031_303132


namespace NUMINAMATH_CALUDE_unique_m_value_l3031_303145

/-- Given a set A and a real number m, proves that m = 3 is the only valid solution -/
theorem unique_m_value (m : ℝ) : 
  let A : Set ℝ := {0, m, m^2 - 3*m + 2}
  2 ∈ A → m = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_m_value_l3031_303145


namespace NUMINAMATH_CALUDE_complement_A_inter_B_l3031_303197

-- Define the sets
def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x * (x - 3) ≥ 0}
def B : Set ℝ := {x | x ≤ 2}

-- State the theorem
theorem complement_A_inter_B :
  (Set.compl A) ∩ B = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_A_inter_B_l3031_303197


namespace NUMINAMATH_CALUDE_theater_seat_count_l3031_303141

/-- The number of seats in a theater -/
def theater_seats (people_watching : ℕ) (empty_seats : ℕ) : ℕ :=
  people_watching + empty_seats

/-- Theorem: The theater has 750 seats -/
theorem theater_seat_count : theater_seats 532 218 = 750 := by
  sorry

end NUMINAMATH_CALUDE_theater_seat_count_l3031_303141


namespace NUMINAMATH_CALUDE_percentage_of_360_equals_93_6_l3031_303119

theorem percentage_of_360_equals_93_6 (total : ℝ) (part : ℝ) (percentage : ℝ) : 
  total = 360 → part = 93.6 → percentage = (part / total) * 100 → percentage = 26 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_360_equals_93_6_l3031_303119


namespace NUMINAMATH_CALUDE_percentage_of_indian_men_l3031_303180

theorem percentage_of_indian_men (total_men : ℕ) (total_women : ℕ) (total_children : ℕ)
  (percentage_indian_women : ℚ) (percentage_indian_children : ℚ)
  (percentage_not_indian : ℚ) :
  total_men = 500 →
  total_women = 300 →
  total_children = 500 →
  percentage_indian_women = 60 / 100 →
  percentage_indian_children = 70 / 100 →
  percentage_not_indian = 55.38461538461539 / 100 →
  (total_men * (10 / 100) + total_women * percentage_indian_women + total_children * percentage_indian_children : ℚ) =
  (total_men + total_women + total_children : ℕ) * (1 - percentage_not_indian) :=
by sorry

end NUMINAMATH_CALUDE_percentage_of_indian_men_l3031_303180


namespace NUMINAMATH_CALUDE_cakes_after_school_l3031_303163

theorem cakes_after_school (croissants_per_person breakfast_people pizzas_per_person bedtime_people total_food : ℕ) 
  (h1 : croissants_per_person = 7)
  (h2 : breakfast_people = 2)
  (h3 : pizzas_per_person = 30)
  (h4 : bedtime_people = 2)
  (h5 : total_food = 110) :
  ∃ (cakes_per_person : ℕ), 
    croissants_per_person * breakfast_people + cakes_per_person * 2 + pizzas_per_person * bedtime_people = total_food ∧ 
    cakes_per_person = 18 := by
  sorry

end NUMINAMATH_CALUDE_cakes_after_school_l3031_303163


namespace NUMINAMATH_CALUDE_log_30_8_l3031_303123

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- Define the variables a and b
variable (a b : ℝ)

-- Define the conditions
axiom lg_5 : lg 5 = a
axiom lg_3 : lg 3 = b

-- State the theorem
theorem log_30_8 : (Real.log 8) / (Real.log 30) = 3 * (1 - a) / (b + 1) :=
sorry

end NUMINAMATH_CALUDE_log_30_8_l3031_303123


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l3031_303125

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 18 →
  a * b + c + d = 91 →
  a * d + b * c = 187 →
  c * d = 105 →
  a^2 + b^2 + c^2 + d^2 ≤ 107 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l3031_303125


namespace NUMINAMATH_CALUDE_quasi_pythagorean_prime_divisor_l3031_303196

theorem quasi_pythagorean_prime_divisor (a b c : ℕ+) :
  c^2 = a^2 + b^2 + a*b → ∃ p : ℕ, p.Prime ∧ p > 5 ∧ p ∣ c := by
  sorry

end NUMINAMATH_CALUDE_quasi_pythagorean_prime_divisor_l3031_303196


namespace NUMINAMATH_CALUDE_tan_period_l3031_303127

theorem tan_period (x : ℝ) : 
  let f : ℝ → ℝ := fun x => Real.tan (3 * x / 4)
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ p = 4 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_period_l3031_303127


namespace NUMINAMATH_CALUDE_constant_sum_of_roots_l3031_303157

theorem constant_sum_of_roots (b x : ℝ) (h : (6 / b) < x ∧ x < (10 / b)) :
  Real.sqrt (x^2 - 2*x + 1) + Real.sqrt (x^2 - 6*x + 9) = 2 := by
  sorry

end NUMINAMATH_CALUDE_constant_sum_of_roots_l3031_303157


namespace NUMINAMATH_CALUDE_candy_problem_l3031_303142

theorem candy_problem :
  ∀ (N : ℕ) (S : ℕ),
    N > 0 →
    (∀ i : Fin N, ∃ (a : ℕ), a > 1 ∧ a = S - (N - 1) * a - 7) →
    S = 21 := by
  sorry

end NUMINAMATH_CALUDE_candy_problem_l3031_303142


namespace NUMINAMATH_CALUDE_morgan_change_calculation_l3031_303183

/-- Calculates the change Morgan receives after buying lunch items and paying with a $50 bill. -/
theorem morgan_change_calculation (hamburger onion_rings smoothie side_salad chocolate_cake : ℚ)
  (h1 : hamburger = 5.75)
  (h2 : onion_rings = 2.50)
  (h3 : smoothie = 3.25)
  (h4 : side_salad = 3.75)
  (h5 : chocolate_cake = 4.20) :
  50 - (hamburger + onion_rings + smoothie + side_salad + chocolate_cake) = 30.55 := by
  sorry

#eval 50 - (5.75 + 2.50 + 3.25 + 3.75 + 4.20)

end NUMINAMATH_CALUDE_morgan_change_calculation_l3031_303183


namespace NUMINAMATH_CALUDE_quartic_root_sum_l3031_303155

theorem quartic_root_sum (p q : ℝ) : 
  (Complex.I + 2 : ℂ) ^ 4 + p * (Complex.I + 2 : ℂ) ^ 2 + q * (Complex.I + 2 : ℂ) + 1 = 0 →
  p + q = 10 := by
sorry

end NUMINAMATH_CALUDE_quartic_root_sum_l3031_303155


namespace NUMINAMATH_CALUDE_phase_shift_sine_specific_phase_shift_l3031_303136

/-- The phase shift of a sine function of the form y = A sin(Bx + C) is -C/B -/
theorem phase_shift_sine (A B C : ℝ) (h : B ≠ 0) :
  let f := λ x : ℝ => A * Real.sin (B * x + C)
  let phase_shift := -C / B
  ∀ x : ℝ, f (x + phase_shift) = A * Real.sin (B * x) := by
  sorry

/-- The phase shift of y = 3 sin(3x + π/4) is -π/12 -/
theorem specific_phase_shift :
  let f := λ x : ℝ => 3 * Real.sin (3 * x + π/4)
  let phase_shift := -π/12
  ∀ x : ℝ, f (x + phase_shift) = 3 * Real.sin (3 * x) := by
  sorry

end NUMINAMATH_CALUDE_phase_shift_sine_specific_phase_shift_l3031_303136


namespace NUMINAMATH_CALUDE_course_total_hours_l3031_303114

/-- Calculates the total hours spent on a course given the course duration and weekly schedule. -/
def total_course_hours (weeks : ℕ) (class_hours_1 class_hours_2 class_hours_3 homework_hours : ℕ) : ℕ :=
  weeks * (class_hours_1 + class_hours_2 + class_hours_3 + homework_hours)

/-- Proves that a 24-week course with the given weekly schedule results in 336 total hours. -/
theorem course_total_hours :
  total_course_hours 24 3 3 4 4 = 336 := by
  sorry

end NUMINAMATH_CALUDE_course_total_hours_l3031_303114


namespace NUMINAMATH_CALUDE_second_expression_proof_l3031_303159

theorem second_expression_proof (a x : ℝ) : 
  ((2 * a + 16 + x) / 2 = 84) → (a = 32) → (x = 88) := by
  sorry

end NUMINAMATH_CALUDE_second_expression_proof_l3031_303159


namespace NUMINAMATH_CALUDE_center_locus_is_conic_l3031_303186

/-- A point in a 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A conic section in a 2D plane -/
structure Conic where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- The center of a conic section -/
def center (c : Conic) : Point2D :=
  { x := 0, y := 0 }  -- Placeholder definition

/-- Checks if a point lies on a conic -/
def lies_on (p : Point2D) (c : Conic) : Prop :=
  c.a * p.x^2 + c.b * p.x * p.y + c.c * p.y^2 + c.d * p.x + c.e * p.y + c.f = 0

/-- The set of all conics passing through four given points -/
def conics_through_four_points (A B C D : Point2D) : Set Conic :=
  { c | lies_on A c ∧ lies_on B c ∧ lies_on C c ∧ lies_on D c }

/-- The locus of centers of conics passing through four points -/
def center_locus (A B C D : Point2D) : Set Point2D :=
  { p | ∃ c ∈ conics_through_four_points A B C D, center c = p }

/-- Theorem: The locus of centers of conics passing through four points is a conic -/
theorem center_locus_is_conic (A B C D : Point2D) :
  ∃ Γ : Conic, ∀ p ∈ center_locus A B C D, lies_on p Γ :=
sorry

end NUMINAMATH_CALUDE_center_locus_is_conic_l3031_303186


namespace NUMINAMATH_CALUDE_square_of_difference_l3031_303162

theorem square_of_difference (y : ℝ) (h : 4 * y^2 - 36 ≥ 0) :
  (10 - Real.sqrt (4 * y^2 - 36))^2 = 4 * y^2 + 64 - 20 * Real.sqrt (4 * y^2 - 36) := by
  sorry

end NUMINAMATH_CALUDE_square_of_difference_l3031_303162


namespace NUMINAMATH_CALUDE_emily_egg_collection_l3031_303144

/-- The number of baskets Emily used -/
def num_baskets : ℕ := 303

/-- The number of eggs in each basket -/
def eggs_per_basket : ℕ := 28

/-- The total number of eggs Emily collected -/
def total_eggs : ℕ := num_baskets * eggs_per_basket

theorem emily_egg_collection : total_eggs = 8484 := by
  sorry

end NUMINAMATH_CALUDE_emily_egg_collection_l3031_303144


namespace NUMINAMATH_CALUDE_unique_pair_l3031_303130

-- Define the properties of N and X
def is_valid_pair (N : ℕ) (X : ℚ) : Prop :=
  -- N is a two-digit natural number
  10 ≤ N ∧ N < 100 ∧
  -- X is a two-digit decimal number
  3 ≤ X ∧ X < 10 ∧
  -- N becomes 56.7 smaller when a decimal point is inserted between its digits
  (N : ℚ) = (N / 10 : ℚ) + 56.7 ∧
  -- X becomes twice as close to N after this change
  (N : ℚ) - X = 2 * ((N : ℚ) - (N / 10 : ℚ))

-- Theorem statement
theorem unique_pair : ∃! (N : ℕ) (X : ℚ), is_valid_pair N X ∧ N = 63 ∧ X = 3.6 := by
  sorry

end NUMINAMATH_CALUDE_unique_pair_l3031_303130


namespace NUMINAMATH_CALUDE_sum_of_max_and_min_is_4022_l3031_303121

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the interval
def I : Set ℝ := Set.Icc (-2011) 2011

-- State the theorem
theorem sum_of_max_and_min_is_4022 
  (h1 : ∀ x ∈ I, ∀ y ∈ I, f (x + y) = f x + f y - 2011)
  (h2 : ∀ x > 0, x ∈ I → f x > 2011)
  : (⨆ x ∈ I, f x) + (⨅ x ∈ I, f x) = 4022 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_and_min_is_4022_l3031_303121


namespace NUMINAMATH_CALUDE_complex_cube_sum_magnitude_l3031_303177

theorem complex_cube_sum_magnitude (z₁ z₂ : ℂ) 
  (h1 : Complex.abs (z₁ + z₂) = 20) 
  (h2 : Complex.abs (z₁^2 + z₂^2) = 16) : 
  Complex.abs (z₁^3 + z₂^3) = 3520 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_sum_magnitude_l3031_303177


namespace NUMINAMATH_CALUDE_profit_share_difference_l3031_303151

def investment_A : ℕ := 8000
def investment_B : ℕ := 10000
def investment_C : ℕ := 12000
def profit_share_B : ℕ := 2500

theorem profit_share_difference :
  let ratio_A : ℕ := investment_A / 2000
  let ratio_B : ℕ := investment_B / 2000
  let ratio_C : ℕ := investment_C / 2000
  let part_value : ℕ := profit_share_B / ratio_B
  let profit_A : ℕ := ratio_A * part_value
  let profit_C : ℕ := ratio_C * part_value
  profit_C - profit_A = 1000 := by sorry

end NUMINAMATH_CALUDE_profit_share_difference_l3031_303151


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l3031_303108

/-- The perimeter of a rectangular field with length 7/5 of its width and width 70 meters is 336 meters. -/
theorem rectangular_field_perimeter : 
  ∀ (width length perimeter : ℝ),
  width = 70 →
  length = (7 / 5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 336 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l3031_303108


namespace NUMINAMATH_CALUDE_smallest_prime_six_less_than_square_l3031_303192

theorem smallest_prime_six_less_than_square : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (Nat.Prime n) ∧ 
    (∃ (m : ℕ), n = m^2 - 6) ∧
    (∀ (k : ℕ), k > 0 → Nat.Prime k → (∃ (j : ℕ), k = j^2 - 6) → k ≥ n) ∧
    n = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_six_less_than_square_l3031_303192


namespace NUMINAMATH_CALUDE_no_negative_roots_l3031_303101

theorem no_negative_roots :
  ∀ x : ℝ, x < 0 → x^4 - 5*x^3 - 4*x^2 - 7*x + 4 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_negative_roots_l3031_303101


namespace NUMINAMATH_CALUDE_sine_inequality_l3031_303176

theorem sine_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) := by
  sorry

end NUMINAMATH_CALUDE_sine_inequality_l3031_303176


namespace NUMINAMATH_CALUDE_mitch_boat_financing_l3031_303174

/-- The amount Mitch has saved in dollars -/
def total_savings : ℕ := 20000

/-- The cost of a new boat per foot in dollars -/
def boat_cost_per_foot : ℕ := 1500

/-- The maximum length of boat Mitch can buy in feet -/
def max_boat_length : ℕ := 12

/-- The amount Mitch needs to keep for license and registration in dollars -/
def license_registration_cost : ℕ := 500

/-- The ratio of docking fees to license and registration cost -/
def docking_fee_ratio : ℕ := 3

theorem mitch_boat_financing :
  license_registration_cost * (docking_fee_ratio + 1) = 
    total_savings - (boat_cost_per_foot * max_boat_length) :=
by sorry

end NUMINAMATH_CALUDE_mitch_boat_financing_l3031_303174


namespace NUMINAMATH_CALUDE_combined_machine_time_l3031_303167

theorem combined_machine_time (t1 t2 : ℝ) (h1 : t1 = 20) (h2 : t2 = 30) :
  1 / (1 / t1 + 1 / t2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_combined_machine_time_l3031_303167


namespace NUMINAMATH_CALUDE_students_between_50_and_90_count_l3031_303103

/-- Represents the distribution of student scores -/
structure ScoreDistribution where
  total_students : ℕ
  mean : ℝ
  std_dev : ℝ
  above_90 : ℕ

/-- Calculates the number of students scoring between 50 and 90 -/
def students_between_50_and_90 (d : ScoreDistribution) : ℕ :=
  d.total_students - 2 * d.above_90

/-- Theorem stating the number of students scoring between 50 and 90 -/
theorem students_between_50_and_90_count
  (d : ScoreDistribution)
  (h1 : d.total_students = 10000)
  (h2 : d.mean = 70)
  (h3 : d.std_dev = 10)
  (h4 : d.above_90 = 230) :
  students_between_50_and_90 d = 9540 := by
  sorry

#check students_between_50_and_90_count

end NUMINAMATH_CALUDE_students_between_50_and_90_count_l3031_303103


namespace NUMINAMATH_CALUDE_gray_area_calculation_l3031_303113

theorem gray_area_calculation (black_area : ℝ) (width1 height1 width2 height2 : ℝ) :
  black_area = 37 ∧ 
  width1 = 8 ∧ height1 = 10 ∧ 
  width2 = 12 ∧ height2 = 9 →
  width2 * height2 - (width1 * height1 - black_area) = 65 :=
by
  sorry

end NUMINAMATH_CALUDE_gray_area_calculation_l3031_303113


namespace NUMINAMATH_CALUDE_cheese_block_volume_l3031_303105

/-- Given a normal block of cheese with volume 3 cubic feet, 
    a large block with twice the width, twice the depth, and three times the length 
    of the normal block will have a volume of 36 cubic feet. -/
theorem cheese_block_volume : 
  ∀ (w d l : ℝ), 
    w * d * l = 3 → 
    (2 * w) * (2 * d) * (3 * l) = 36 := by
  sorry

end NUMINAMATH_CALUDE_cheese_block_volume_l3031_303105


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3031_303171

/-- An isosceles triangle with two sides of length 9 and one side of length 4 has a perimeter of 22. -/
theorem isosceles_triangle_perimeter : ℝ → ℝ → ℝ → Prop :=
  fun a b c => 
    (a = 9 ∧ b = 9 ∧ c = 4) →  -- Two sides are 9, one side is 4
    (a + b > c ∧ b + c > a ∧ a + c > b) →  -- Triangle inequality
    (a = b) →  -- Isosceles condition
    (a + b + c = 22)  -- Perimeter is 22

-- The proof is omitted
theorem isosceles_triangle_perimeter_proof : isosceles_triangle_perimeter 9 9 4 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_isosceles_triangle_perimeter_proof_l3031_303171


namespace NUMINAMATH_CALUDE_sqrt_equality_l3031_303194

theorem sqrt_equality (a b x : ℝ) (h1 : a < b) (h2 : -b ≤ x) (h3 : x ≤ -a) :
  Real.sqrt (-(x+a)^3*(x+b)) = -(x+a) * Real.sqrt (-(x+a)*(x+b)) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equality_l3031_303194


namespace NUMINAMATH_CALUDE_quadratic_always_positive_l3031_303110

theorem quadratic_always_positive (b c : ℤ) 
  (h : ∀ x : ℤ, (x^2 : ℤ) + b*x + c > 0) : 
  b^2 - 4*c ≤ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_always_positive_l3031_303110


namespace NUMINAMATH_CALUDE_hexagonal_prism_diagonals_l3031_303161

/-- A regular hexagonal prism --/
structure RegularHexagonalPrism where
  /-- Number of sides in the base --/
  n : ℕ
  /-- Assertion that the base has 6 sides --/
  base_is_hexagon : n = 6

/-- The number of diagonals in a regular hexagonal prism --/
def num_diagonals (prism : RegularHexagonalPrism) : ℕ := prism.n * (prism.n - 3)

/-- Theorem: The number of diagonals in a regular hexagonal prism is 18 --/
theorem hexagonal_prism_diagonals (prism : RegularHexagonalPrism) : 
  num_diagonals prism = 18 := by
  sorry

#check hexagonal_prism_diagonals

end NUMINAMATH_CALUDE_hexagonal_prism_diagonals_l3031_303161


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3031_303124

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ) (m n p : ℕ) 
  (h_arith : ArithmeticSequence a)
  (h_distinct : m ≠ n ∧ n ≠ p ∧ m ≠ p)
  (h_positive : 0 < m ∧ 0 < n ∧ 0 < p) :
  m * (a p - a n) + n * (a m - a p) + p * (a n - a m) = 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3031_303124


namespace NUMINAMATH_CALUDE_division_simplification_l3031_303109

theorem division_simplification : 180 / (12 + 13 * 3) = 60 / 17 := by sorry

end NUMINAMATH_CALUDE_division_simplification_l3031_303109


namespace NUMINAMATH_CALUDE_exponential_inequality_l3031_303172

theorem exponential_inequality (x y a b : ℝ) 
  (hxy : x > y ∧ y > 1) 
  (hab : 0 < a ∧ a < b ∧ b < 1) : 
  a^x < b^y := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l3031_303172


namespace NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_l_l3031_303111

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 4

-- Define the line l
def l (x y : ℝ) : Prop := 3*x + 2*y + 1 = 0

-- Define the result line
def result_line (x y : ℝ) : Prop := 2*x - 3*y + 3 = 0

-- Theorem statement
theorem line_through_circle_center_perpendicular_to_l :
  ∃ (x₀ y₀ : ℝ), 
    (∀ x y, C x y ↔ (x - x₀)^2 + (y - y₀)^2 = 4) ∧ 
    (result_line x₀ y₀) ∧
    (∀ x₁ y₁ x₂ y₂, l x₁ y₁ ∧ l x₂ y₂ ∧ x₁ ≠ x₂ → 
      (y₂ - y₁) * (x₀ - x₁) = -(x₂ - x₁) * (y₀ - y₁)) :=
sorry

end NUMINAMATH_CALUDE_line_through_circle_center_perpendicular_to_l_l3031_303111


namespace NUMINAMATH_CALUDE_customers_who_tried_sample_l3031_303195

/-- Given a store that puts out product samples, this theorem calculates
    the number of customers who tried a sample based on the given conditions. -/
theorem customers_who_tried_sample
  (samples_per_box : ℕ)
  (boxes_opened : ℕ)
  (samples_left : ℕ)
  (h1 : samples_per_box = 20)
  (h2 : boxes_opened = 12)
  (h3 : samples_left = 5) :
  samples_per_box * boxes_opened - samples_left = 235 :=
by sorry

end NUMINAMATH_CALUDE_customers_who_tried_sample_l3031_303195


namespace NUMINAMATH_CALUDE_president_and_committee_selection_l3031_303165

/-- The number of ways to choose a president and a 3-person committee from a group of 10 people,
    where the order of committee selection doesn't matter and the president cannot be on the committee. -/
def select_president_and_committee (total_people : ℕ) (committee_size : ℕ) : ℕ :=
  total_people * (Nat.choose (total_people - 1) committee_size)

/-- Theorem stating that the number of ways to choose a president and a 3-person committee
    from a group of 10 people, where the order of committee selection doesn't matter and
    the president cannot be on the committee, is equal to 840. -/
theorem president_and_committee_selection :
  select_president_and_committee 10 3 = 840 := by
  sorry


end NUMINAMATH_CALUDE_president_and_committee_selection_l3031_303165


namespace NUMINAMATH_CALUDE_comic_books_sale_proof_l3031_303134

/-- The number of comic books sold by Scott and Sam -/
def comic_books_sold (initial_total remaining : ℕ) : ℕ :=
  initial_total - remaining

theorem comic_books_sale_proof :
  comic_books_sold 90 25 = 65 := by
  sorry

end NUMINAMATH_CALUDE_comic_books_sale_proof_l3031_303134


namespace NUMINAMATH_CALUDE_prime_divisibility_l3031_303147

theorem prime_divisibility (p q : ℕ) (n : ℕ) 
  (h_p_prime : Prime p) 
  (h_q_prime : Prime q) 
  (h_distinct : p ≠ q) 
  (h_pq_div : (p * q) ∣ (n^(p*q) + 1)) 
  (h_p3q3_div : (p^3 * q^3) ∣ (n^(p*q) + 1)) :
  p^2 ∣ (n + 1) ∨ q^2 ∣ (n + 1) := by
sorry

end NUMINAMATH_CALUDE_prime_divisibility_l3031_303147


namespace NUMINAMATH_CALUDE_quadratic_shift_l3031_303182

/-- Represents a quadratic function of the form y = -(x+a)^2 + b -/
def QuadraticFunction (a b : ℝ) := λ x : ℝ => -(x + a)^2 + b

/-- Represents a horizontal shift of a function -/
def HorizontalShift (f : ℝ → ℝ) (shift : ℝ) := λ x : ℝ => f (x - shift)

/-- Theorem: Shifting the graph of y = -(x+2)^2 + 1 by 1 unit to the right 
    results in the function y = -(x+1)^2 + 1 -/
theorem quadratic_shift :
  HorizontalShift (QuadraticFunction 2 1) 1 = QuadraticFunction 1 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_shift_l3031_303182


namespace NUMINAMATH_CALUDE_circle_line_tangency_l3031_303193

/-- The circle equation -/
def circle_equation (x y k : ℝ) : Prop :=
  x^2 + y^2 - 2*k*x - 2*y = 0

/-- The line equation -/
def line_equation (x y k : ℝ) : Prop :=
  x + y = 2*k

/-- The tangency condition -/
def are_tangent (k : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_equation x y k ∧ line_equation x y k

/-- The main theorem -/
theorem circle_line_tangency (k : ℝ) :
  are_tangent k → k = -1 := by sorry

end NUMINAMATH_CALUDE_circle_line_tangency_l3031_303193


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_seven_sqrt_three_over_three_l3031_303152

theorem sqrt_sum_equals_seven_sqrt_three_over_three :
  Real.sqrt 12 + Real.sqrt (1/3) = 7 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_seven_sqrt_three_over_three_l3031_303152


namespace NUMINAMATH_CALUDE_brians_trip_distance_l3031_303184

/-- Calculates the distance traveled given car efficiency and gas used -/
def distance_traveled (efficiency : ℝ) (gas_used : ℝ) : ℝ :=
  efficiency * gas_used

/-- Proves that Brian's car travels 60 miles given the conditions -/
theorem brians_trip_distance :
  let efficiency : ℝ := 20
  let gas_used : ℝ := 3
  distance_traveled efficiency gas_used = 60 := by
  sorry

end NUMINAMATH_CALUDE_brians_trip_distance_l3031_303184


namespace NUMINAMATH_CALUDE_service_provider_selection_l3031_303191

theorem service_provider_selection (n : ℕ) (k : ℕ) (h1 : n = 25) (h2 : k = 4) :
  (n - 0) * (n - 1) * (n - 2) * (n - 3) = 303600 :=
by sorry

end NUMINAMATH_CALUDE_service_provider_selection_l3031_303191


namespace NUMINAMATH_CALUDE_power_five_mod_eighteen_l3031_303107

theorem power_five_mod_eighteen : 5^100 % 18 = 13 := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_eighteen_l3031_303107
