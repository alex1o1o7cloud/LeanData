import Mathlib

namespace yoz_perpendicular_x_xoz_perpendicular_y_xoy_perpendicular_z_l1801_180183

-- Define the three-dimensional Cartesian coordinate system
structure CartesianCoordinate3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the coordinate planes
def yoz_plane : Set CartesianCoordinate3D := {p | p.x = 0}
def xoz_plane : Set CartesianCoordinate3D := {p | p.y = 0}
def xoy_plane : Set CartesianCoordinate3D := {p | p.z = 0}

-- Define the axes
def x_axis : Set CartesianCoordinate3D := {p | p.y = 0 ∧ p.z = 0}
def y_axis : Set CartesianCoordinate3D := {p | p.x = 0 ∧ p.z = 0}
def z_axis : Set CartesianCoordinate3D := {p | p.x = 0 ∧ p.y = 0}

-- Define perpendicularity between a plane and an axis
def perpendicular (plane : Set CartesianCoordinate3D) (axis : Set CartesianCoordinate3D) : Prop :=
  ∀ p ∈ plane, ∀ q ∈ axis, (p.x - q.x) * q.x + (p.y - q.y) * q.y + (p.z - q.z) * q.z = 0

-- Theorem statements
theorem yoz_perpendicular_x : perpendicular yoz_plane x_axis := sorry

theorem xoz_perpendicular_y : perpendicular xoz_plane y_axis := sorry

theorem xoy_perpendicular_z : perpendicular xoy_plane z_axis := sorry

end yoz_perpendicular_x_xoz_perpendicular_y_xoy_perpendicular_z_l1801_180183


namespace factorization_equality_l1801_180187

theorem factorization_equality (a b : ℝ) : a^2 - 4*a*b + 4*b^2 = (a - 2*b)^2 := by
  sorry

end factorization_equality_l1801_180187


namespace only_five_students_l1801_180135

/-- Represents the number of students -/
def n : ℕ := sorry

/-- Represents the total number of problems solved -/
def S : ℕ := sorry

/-- Represents the number of problems solved by one student -/
def a : ℕ := sorry

/-- The condition that each student solved more than one-fifth of the problems solved by others -/
axiom condition1 : a > (S - a) / 5

/-- The condition that each student solved less than one-third of the problems solved by others -/
axiom condition2 : a < (S - a) / 3

/-- The total number of problems is the sum of problems solved by all students -/
axiom total_problems : S = n * a

/-- The theorem stating that the only possible number of students is 5 -/
theorem only_five_students : n = 5 := by sorry

end only_five_students_l1801_180135


namespace power_division_rule_l1801_180178

theorem power_division_rule (a : ℝ) : a^3 / a^2 = a := by sorry

end power_division_rule_l1801_180178


namespace eggs_per_hen_l1801_180199

/-- Given 303.0 eggs collected from 28.0 hens, prove that the number of eggs
    laid by each hen, when rounded to the nearest whole number, is 11. -/
theorem eggs_per_hen (total_eggs : ℝ) (num_hens : ℝ) 
    (h1 : total_eggs = 303.0) (h2 : num_hens = 28.0) :
  round (total_eggs / num_hens) = 11 := by
  sorry

end eggs_per_hen_l1801_180199


namespace tree_growth_rate_consistency_l1801_180136

theorem tree_growth_rate_consistency :
  ∃ (a b : ℝ), 
    (a + b) / 2 = 0.15 ∧ 
    (1 + a) * (1 + b) = 0.9 := by
  sorry

end tree_growth_rate_consistency_l1801_180136


namespace geometric_sequence_sum_l1801_180105

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- Given a geometric sequence a_n where a_3 + a_5 = 20 and a_4 = 8, prove that a_2 + a_6 = 34 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geo : IsGeometricSequence a)
    (h_sum : a 3 + a 5 = 20) (h_fourth : a 4 = 8) : a 2 + a 6 = 34 := by
  sorry

end geometric_sequence_sum_l1801_180105


namespace min_value_theorem_l1801_180171

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∃ (min : ℝ), min = 2 + 2 * Real.sqrt 2 ∧ ∀ z, z = (2 / x) + (x / y) → z ≥ min :=
by sorry

end min_value_theorem_l1801_180171


namespace parallel_vectors_m_value_l1801_180118

/-- Two vectors are parallel if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  ∀ m : ℝ,
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (-2, m)
  parallel a b → m = -4 := by
sorry

end parallel_vectors_m_value_l1801_180118


namespace next_red_probability_l1801_180108

/-- Represents the count of balls of each color -/
structure BallCount where
  red : ℕ
  blue : ℕ
  yellow : ℕ

/-- Represents the result of pulling out balls -/
structure PullResult where
  pulled : ℕ
  redBlueDifference : ℤ

/-- Calculates the probability of pulling a red ball next -/
def probabilityNextRed (initial : BallCount) (result : PullResult) : ℚ :=
  9/26

theorem next_red_probability 
  (initial : BallCount)
  (result : PullResult)
  (h1 : initial.red = 50)
  (h2 : initial.blue = 50)
  (h3 : initial.yellow = 30)
  (h4 : result.pulled = 65)
  (h5 : result.redBlueDifference = 5) :
  probabilityNextRed initial result = 9/26 := by
  sorry

end next_red_probability_l1801_180108


namespace triangle_area_arithmetic_progression_l1801_180195

/-- The area of a triangle with base 2a - d and height 2a + d is 2a^2 - d^2/2 -/
theorem triangle_area_arithmetic_progression (a d : ℝ) (h_a : a > 0) :
  let base := 2 * a - d
  let height := 2 * a + d
  (1 / 2 : ℝ) * base * height = 2 * a^2 - d^2 / 2 :=
by sorry

end triangle_area_arithmetic_progression_l1801_180195


namespace problem_statement_l1801_180113

theorem problem_statement (a b : ℝ) (h : a + b = 1) :
  (a^3 + b^3 ≥ 1/4) ∧
  (∃ x : ℝ, |x - a| + |x - b| ≤ 5 → 0 ≤ 2*a + 3*b ∧ 2*a + 3*b ≤ 5) :=
by sorry

end problem_statement_l1801_180113


namespace reflection_line_property_l1801_180175

/-- A line that reflects a point (x₁, y₁) to (x₂, y₂) -/
structure ReflectionLine where
  m : ℝ
  b : ℝ
  x₁ : ℝ
  y₁ : ℝ
  x₂ : ℝ
  y₂ : ℝ
  reflects : ((x₂ - x₁) * m + y₁ + y₂) / 2 = m * ((x₂ + x₁) / 2) + b

/-- The theorem stating that for a line y = mx + b that reflects (1, -4) to (7, 2), 3m + 2b = 3 -/
theorem reflection_line_property (line : ReflectionLine) 
    (h1 : line.x₁ = 1) (h2 : line.y₁ = -4) (h3 : line.x₂ = 7) (h4 : line.y₂ = 2) : 
    3 * line.m + 2 * line.b = 3 := by
  sorry

end reflection_line_property_l1801_180175


namespace specific_plot_fencing_cost_l1801_180127

/-- Represents a rectangular plot with given dimensions and fencing cost -/
structure RectangularPlot where
  length : ℝ
  breadth : ℝ
  fencingCostPerMeter : ℝ

/-- Calculates the total cost of fencing a rectangular plot -/
def totalFencingCost (plot : RectangularPlot) : ℝ :=
  2 * (plot.length + plot.breadth) * plot.fencingCostPerMeter

/-- Theorem stating the total fencing cost for a specific rectangular plot -/
theorem specific_plot_fencing_cost :
  let plot : RectangularPlot := {
    length := 65,
    breadth := 35,
    fencingCostPerMeter := 26.5
  }
  totalFencingCost plot = 5300 := by sorry

end specific_plot_fencing_cost_l1801_180127


namespace organization_size_after_five_years_l1801_180140

def organization_size (n : ℕ) : ℕ :=
  match n with
  | 0 => 20
  | k + 1 => 4 * organization_size k - 21

theorem organization_size_after_five_years :
  organization_size 5 = 13343 := by
  sorry

end organization_size_after_five_years_l1801_180140


namespace divisible_by_nineteen_l1801_180115

theorem divisible_by_nineteen (n : ℕ) :
  ∃ k : ℤ, (5 : ℤ)^(2*n+1) * 2^(n+2) + 3^(n+2) * 2^(2*n+1) = 19 * k := by
  sorry

end divisible_by_nineteen_l1801_180115


namespace total_tomatoes_l1801_180129

def number_of_rows : ℕ := 30
def plants_per_row : ℕ := 10
def tomatoes_per_plant : ℕ := 20

theorem total_tomatoes : 
  number_of_rows * plants_per_row * tomatoes_per_plant = 6000 := by
  sorry

end total_tomatoes_l1801_180129


namespace number_of_broadcasting_methods_l1801_180103

/-- Represents the number of commercial advertisements -/
def num_commercial_ads : ℕ := 4

/-- Represents the number of public service advertisements -/
def num_public_service_ads : ℕ := 2

/-- Represents the total number of advertisements -/
def total_ads : ℕ := num_commercial_ads + num_public_service_ads

/-- Represents the fact that public service ads must be at the beginning and end -/
def public_service_ads_fixed : Prop := True

theorem number_of_broadcasting_methods : 
  (num_commercial_ads = 4 ∧ 
   num_public_service_ads = 2 ∧ 
   total_ads = 6 ∧ 
   public_service_ads_fixed) → 
  (Nat.factorial num_commercial_ads = 24) := by
  sorry

end number_of_broadcasting_methods_l1801_180103


namespace count_integer_pairs_l1801_180176

theorem count_integer_pairs : 
  let w_count := (Finset.range 450).filter (fun w => w % 23 = 5) |>.card
  let n_count := (Finset.range 450).filter (fun n => n % 17 = 7) |>.card
  w_count * n_count = 540 := by
sorry

end count_integer_pairs_l1801_180176


namespace circle_sector_radius_l1801_180174

theorem circle_sector_radius 
  (area : ℝ) 
  (arc_length : ℝ) 
  (h1 : area = 8.75) 
  (h2 : arc_length = 3.5) : 
  ∃ (radius : ℝ), radius = 5 ∧ area = (1/2) * radius * arc_length :=
by
  sorry

end circle_sector_radius_l1801_180174


namespace eighteen_wheel_truck_toll_l1801_180111

/-- Calculates the number of axles for a truck given the total number of wheels,
    the number of wheels on the front axle, and the number of wheels on other axles. -/
def calculateAxles (totalWheels frontAxleWheels otherAxleWheels : ℕ) : ℕ :=
  1 + (totalWheels - frontAxleWheels) / otherAxleWheels

/-- Calculates the toll for a truck based on the number of axles. -/
def calculateToll (axles : ℕ) : ℚ :=
  1.5 + 1.5 * (axles - 2 : ℚ)

/-- Theorem stating that the toll for an 18-wheel truck with 2 wheels on the front axle
    and 4 wheels on each other axle is $6.00. -/
theorem eighteen_wheel_truck_toll :
  let axles := calculateAxles 18 2 4
  calculateToll axles = 6 := by
  sorry

#eval calculateAxles 18 2 4
#eval calculateToll (calculateAxles 18 2 4)

end eighteen_wheel_truck_toll_l1801_180111


namespace parallel_vectors_x_value_l1801_180131

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (1, 2) (x, 4) → x = 2 := by
  sorry

end parallel_vectors_x_value_l1801_180131


namespace crosswalk_parallelogram_l1801_180191

/-- A parallelogram with given dimensions -/
structure Parallelogram where
  side1 : ℝ
  side2 : ℝ
  height1 : ℝ
  height2 : ℝ

/-- The theorem about the parallelogram representing the crosswalk -/
theorem crosswalk_parallelogram (p : Parallelogram) 
  (h1 : p.side1 = 18)
  (h2 : p.height1 = 60)
  (h3 : p.side2 = 60) :
  p.height2 = 18 := by
  sorry

#check crosswalk_parallelogram

end crosswalk_parallelogram_l1801_180191


namespace largest_solution_quadratic_equation_l1801_180180

theorem largest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 5 * (9 * x^2 + 9 * x + 11) - x * (10 * x - 50)
  ∃ x : ℝ, f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x
  ↔ 
  ∃ x : ℝ, x = (-19 + Real.sqrt 53) / 14 ∧ f x = 0 ∧ ∀ y : ℝ, f y = 0 → y ≤ x :=
by sorry

end largest_solution_quadratic_equation_l1801_180180


namespace prime_pair_sum_is_106_l1801_180120

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def prime_pair_sum : ℕ → Prop
| S => ∃ (primes : Finset ℕ), 
    (∀ p ∈ primes, is_prime p ∧ is_prime (p + 2) ∧ p * (p + 2) ≤ 2007) ∧
    (∀ p : ℕ, is_prime p → is_prime (p + 2) → p * (p + 2) ≤ 2007 → p ∈ primes) ∧
    (Finset.sum primes id = S)

theorem prime_pair_sum_is_106 : prime_pair_sum 106 := by sorry

end prime_pair_sum_is_106_l1801_180120


namespace quadratic_always_positive_l1801_180173

theorem quadratic_always_positive (m : ℝ) (h : m > 3) :
  ∀ x : ℝ, m * x^2 - (m + 3) * x + m > 0 := by
  sorry

end quadratic_always_positive_l1801_180173


namespace empty_seats_arrangements_l1801_180163

/-- The number of chairs in a row -/
def num_chairs : ℕ := 8

/-- The number of students taking seats -/
def num_students : ℕ := 4

/-- The number of empty seats -/
def num_empty_seats : ℕ := num_chairs - num_students

/-- Calculates the number of ways to arrange all empty seats adjacent to each other -/
def adjacent_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial (n - k + 1)

/-- Calculates the number of ways to arrange all empty seats not adjacent to each other -/
def non_adjacent_arrangements (n : ℕ) (k : ℕ) : ℕ :=
  Nat.factorial k * Nat.choose (n - k + 1) k

theorem empty_seats_arrangements :
  adjacent_arrangements num_chairs num_empty_seats = 120 ∧
  non_adjacent_arrangements num_chairs num_empty_seats = 120 := by
  sorry

end empty_seats_arrangements_l1801_180163


namespace triangle_perimeter_l1801_180160

/-- A line passing through the origin -/
structure OriginLine where
  slope : ℝ

/-- The intersection point of two lines -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the problem setup -/
def TriangleSetup (l : OriginLine) : Prop :=
  ∃ (p q : Point),
    -- The line intersects x = 1
    p.x = 1 ∧ p.y = -l.slope
    -- The line intersects y = 1 + (√2/2)x
    ∧ q.x = 1 ∧ q.y = 1 + (Real.sqrt 2 / 2)
    -- The three lines form an equilateral triangle
    ∧ (p.x - 0)^2 + (p.y - 0)^2 = (q.x - p.x)^2 + (q.y - p.y)^2
    ∧ (q.x - 0)^2 + (q.y - 0)^2 = (q.x - p.x)^2 + (q.y - p.y)^2

/-- The main theorem -/
theorem triangle_perimeter (l : OriginLine) :
  TriangleSetup l → (3 : ℝ) + 3 * Real.sqrt 2 = 
    let p := Point.mk 1 (-l.slope)
    let q := Point.mk 1 (1 + Real.sqrt 2 / 2)
    3 * Real.sqrt ((q.x - p.x)^2 + (q.y - p.y)^2) :=
by
  sorry


end triangle_perimeter_l1801_180160


namespace card_z_value_l1801_180141

/-- Given four cards W, X, Y, Z with specific tagging rules, prove that Z is tagged with 400. -/
theorem card_z_value : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun w x y z =>
    w = 200 ∧
    x = w / 2 ∧
    y = w + x ∧
    w + x + y + z = 1000 →
    z = 400

/-- Proof of the card_z_value theorem -/
lemma prove_card_z_value : card_z_value 200 100 300 400 := by
  sorry

end card_z_value_l1801_180141


namespace equal_radii_radii_formula_l1801_180158

/-- Two non-intersecting circles with their tangent circles -/
structure TwoCirclesWithTangents where
  /-- Center of the first circle -/
  O₁ : ℝ × ℝ
  /-- Center of the second circle -/
  O₂ : ℝ × ℝ
  /-- Radius of the first circle -/
  R₁ : ℝ
  /-- Radius of the second circle -/
  R₂ : ℝ
  /-- Distance between centers of the two circles -/
  d : ℝ
  /-- The circles are non-intersecting -/
  non_intersecting : d > R₁ + R₂
  /-- Radius of circle K₁ -/
  r₁ : ℝ
  /-- Radius of circle K₂ -/
  r₂ : ℝ
  /-- K₁ is tangent to the first circle and two rays from A₁ -/
  K₁_tangent : r₁ = (2 * R₁ * R₂) / d
  /-- K₂ is tangent to the second circle and two rays from A₂ -/
  K₂_tangent : r₂ = (2 * R₁ * R₂) / d

/-- Theorem: The radii of K₁ and K₂ are equal -/
theorem equal_radii (c : TwoCirclesWithTangents) : c.r₁ = c.r₂ := by
  sorry

/-- Theorem: The radii of K₁ and K₂ can be expressed as (2 * R₁ * R₂) / d -/
theorem radii_formula (c : TwoCirclesWithTangents) : c.r₁ = (2 * c.R₁ * c.R₂) / c.d ∧ c.r₂ = (2 * c.R₁ * c.R₂) / c.d := by
  sorry

end equal_radii_radii_formula_l1801_180158


namespace abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l1801_180198

theorem abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two :
  (∀ x : ℝ, x < -2 → |x| > 2) ∧
  (∃ x : ℝ, |x| > 2 ∧ ¬(x < -2)) :=
sorry

end abs_gt_two_necessary_not_sufficient_for_x_lt_neg_two_l1801_180198


namespace obtuse_angle_measure_l1801_180167

theorem obtuse_angle_measure (α β : Real) (p : Real) :
  (∃ (x y : Real), x^2 + p*(x+1) + 1 = 0 ∧ y^2 + p*(y+1) + 1 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  (α > 0 ∧ β > 0 ∧ α + β < Real.pi) →
  ∃ (γ : Real), γ = Real.pi - α - β ∧ γ = 3*Real.pi/4 :=
by sorry

end obtuse_angle_measure_l1801_180167


namespace product_from_lcm_and_gcd_l1801_180150

theorem product_from_lcm_and_gcd (a b : ℕ+) 
  (h1 : Nat.lcm a b = 60) 
  (h2 : Nat.gcd a b = 12) : 
  a * b = 720 := by
  sorry

end product_from_lcm_and_gcd_l1801_180150


namespace new_distance_between_cars_l1801_180133

/-- Calculates the new distance between cars in a convoy after speed reduction -/
theorem new_distance_between_cars 
  (initial_speed : ℝ) 
  (initial_distance : ℝ) 
  (reduced_speed : ℝ) 
  (h1 : initial_speed = 80) 
  (h2 : initial_distance = 10) 
  (h3 : reduced_speed = 60) : 
  (reduced_speed * (initial_distance / 1000) / initial_speed) * 1000 = 7.5 := by
  sorry

end new_distance_between_cars_l1801_180133


namespace fraction_denominator_l1801_180192

theorem fraction_denominator (n : ℕ) (d : ℕ) (h1 : n = 325) (h2 : (n : ℚ) / d = 1 / 8) 
  (h3 : ∃ (seq : ℕ → ℕ), (∀ k, seq k < 10) ∧ 
    (∀ k, ((n : ℚ) / d - (n : ℚ) / d).floor + (seq k) / 10^(k+1) = ((n : ℚ) / d * 10^(k+1)).floor / 10^(k+1)) ∧ 
    seq 80 = 5) : 
  d = 8 := by sorry

end fraction_denominator_l1801_180192


namespace sum_of_absolute_coefficients_l1801_180182

theorem sum_of_absolute_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) :
  (∀ x : ℝ, (1 - x)^9 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7 + a₈*x^8 + a₉*x^9) →
  |a₀| + |a₁| + |a₂| + |a₃| + |a₄| + |a₅| + |a₆| + |a₇| + |a₈| + |a₉| = 512 :=
by
  sorry

end sum_of_absolute_coefficients_l1801_180182


namespace shell_ratio_l1801_180125

/-- The number of shells each person has -/
structure ShellCounts where
  david : ℕ
  mia : ℕ
  ava : ℕ
  alice : ℕ

/-- The conditions of the problem -/
def shell_problem (sc : ShellCounts) : Prop :=
  sc.david = 15 ∧
  sc.mia = 4 * sc.david ∧
  sc.ava = sc.mia + 20 ∧
  sc.david + sc.mia + sc.ava + sc.alice = 195

/-- The theorem to prove -/
theorem shell_ratio (sc : ShellCounts) 
  (h : shell_problem sc) : sc.alice * 2 = sc.ava := by
  sorry

end shell_ratio_l1801_180125


namespace plain_cookie_price_l1801_180128

theorem plain_cookie_price 
  (chocolate_chip_price : ℝ) 
  (total_boxes : ℝ) 
  (total_value : ℝ) 
  (plain_boxes : ℝ) 
  (h1 : chocolate_chip_price = 1.25)
  (h2 : total_boxes = 1585)
  (h3 : total_value = 1586.25)
  (h4 : plain_boxes = 793.125) : 
  (total_value - (total_boxes - plain_boxes) * chocolate_chip_price) / plain_boxes = 0.75 := by
  sorry

end plain_cookie_price_l1801_180128


namespace percent_asian_in_west_1990_l1801_180172

/-- Represents the population of Asians in millions for each region in the U.S. in 1990 -/
structure AsianPopulation where
  northeast : ℕ
  midwest : ℕ
  south : ℕ
  west : ℕ

/-- Calculates the percentage of Asians living in the West to the nearest percent -/
def percentInWest (pop : AsianPopulation) : ℕ :=
  let total := pop.northeast + pop.midwest + pop.south + pop.west
  let westPercentage := (pop.west * 100) / total
  -- Round to nearest percent
  if westPercentage % 10 ≥ 5 then
    (westPercentage / 10 + 1) * 10
  else
    (westPercentage / 10) * 10

/-- The given population data for Asians in 1990 -/
def population1990 : AsianPopulation :=
  { northeast := 2
  , midwest := 2
  , south := 2
  , west := 6 }

/-- Theorem stating that the percentage of Asians living in the West in 1990 is 50% -/
theorem percent_asian_in_west_1990 : percentInWest population1990 = 50 := by
  sorry


end percent_asian_in_west_1990_l1801_180172


namespace min_m_and_x_range_l1801_180152

theorem min_m_and_x_range (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a^2 + 3*b^2 = 3) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m) ∧
            (∀ m' : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → Real.sqrt 5 * a + b ≤ m') → m ≤ m') ∧
            m = 4) ∧
  (∀ x : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a^2 + 3*b^2 = 3 → 2 * |x - 1| + |x| ≥ Real.sqrt 5 * a + b) →
            (x ≤ -2/3 ∨ x ≥ 2)) :=
by sorry

end min_m_and_x_range_l1801_180152


namespace sand_container_problem_l1801_180154

theorem sand_container_problem (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (a * Real.exp (-8 * b) = a / 2) →
  (∃ t : ℝ, a * Real.exp (-b * t) = a / 8 ∧ t > 0) →
  (∃ t : ℝ, a * Real.exp (-b * t) = a / 8 ∧ t = 24) :=
by sorry

end sand_container_problem_l1801_180154


namespace sin_alpha_value_l1801_180190

theorem sin_alpha_value (α : Real) (h : Real.sin (Real.pi - α) = -1/3) :
  Real.sin α = -1/3 := by
  sorry

end sin_alpha_value_l1801_180190


namespace max_value_of_S_fourth_power_l1801_180184

theorem max_value_of_S_fourth_power :
  let S (x : ℝ) := |Real.sqrt (x^2 + 4*x + 5) - Real.sqrt (x^2 + 2*x + 5)|
  ∀ x : ℝ, (S x)^4 ≤ 4 ∧ ∃ y : ℝ, (S y)^4 = 4 := by
  sorry

end max_value_of_S_fourth_power_l1801_180184


namespace first_group_size_l1801_180188

/-- Represents the work done by a group of workers -/
structure Work where
  workers : ℕ
  days : ℕ
  hectares : ℕ

/-- The work principle: workers * days is proportional to hectares for any two Work instances -/
axiom work_principle {w1 w2 : Work} : 
  w1.workers * w1.days * w2.hectares = w2.workers * w2.days * w1.hectares

/-- The first group's work -/
def first_group : Work := { workers := 0, days := 24, hectares := 80 }

/-- The second group's work -/
def second_group : Work := { workers := 36, days := 30, hectares := 400 }

/-- The theorem to prove -/
theorem first_group_size : first_group.workers = 9 := by
  sorry

end first_group_size_l1801_180188


namespace square_difference_formula_l1801_180102

theorem square_difference_formula (x y : ℚ) 
  (h1 : x + y = 8/15)
  (h2 : x - y = 2/15) : 
  x^2 - y^2 = 16/225 := by
  sorry

end square_difference_formula_l1801_180102


namespace rectangle_area_change_l1801_180197

/-- Given a rectangle with area 540 square centimeters, if its length is decreased by 20%
    and its width is increased by 15%, then its new area is 496.8 square centimeters. -/
theorem rectangle_area_change (l w : ℝ) (h1 : l * w = 540) : 
  (0.8 * l) * (1.15 * w) = 496.8 := by
  sorry

end rectangle_area_change_l1801_180197


namespace quadratic_trinomial_form_is_quadratic_trinomial_l1801_180139

-- Define variables
variable (x y : ℝ)

-- Define A, B, and C
def A : ℝ := x^2 * y + 2
def B : ℝ := 3 * x^2 * y + x
def C : ℝ := 4 * x^2 * y - x * y

-- Theorem statement
theorem quadratic_trinomial_form :
  A x y + B x y - C x y = 2 + x + x * y :=
by sorry

-- Theorem to classify the result as a quadratic trinomial
theorem is_quadratic_trinomial :
  ∃ (a b c : ℝ), A x y + B x y - C x y = a + b * x + c * x * y :=
by sorry

end quadratic_trinomial_form_is_quadratic_trinomial_l1801_180139


namespace sum_in_first_quadrant_l1801_180121

/-- Given complex numbers z₁ and z₂, prove that their sum is in the first quadrant -/
theorem sum_in_first_quadrant (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 1 + 2*I) (h₂ : z₂ = 1 - I) : 
  let z := z₁ + z₂
  (z.re > 0 ∧ z.im > 0) := by sorry

end sum_in_first_quadrant_l1801_180121


namespace smallest_composite_no_small_factors_l1801_180138

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

def has_no_small_prime_factors (n : ℕ) : Prop := ∀ p, p < 20 → p.Prime → ¬(p ∣ n)

theorem smallest_composite_no_small_factors :
  (is_composite 529) ∧
  (has_no_small_prime_factors 529) ∧
  (∀ m : ℕ, m < 529 → ¬(is_composite m ∧ has_no_small_prime_factors m)) :=
sorry

end smallest_composite_no_small_factors_l1801_180138


namespace arithmetic_mean_of_fractions_l1801_180179

theorem arithmetic_mean_of_fractions : 
  (3/4 : ℚ) + (5/8 : ℚ) / 2 = 11/16 := by sorry

end arithmetic_mean_of_fractions_l1801_180179


namespace geometric_mean_of_one_and_four_l1801_180144

theorem geometric_mean_of_one_and_four :
  ∀ x : ℝ, x^2 = 1 * 4 → x = 2 ∨ x = -2 := by
  sorry

end geometric_mean_of_one_and_four_l1801_180144


namespace M_intersect_N_eq_open_interval_l1801_180157

-- Define the sets M and N
def M : Set ℝ := {y : ℝ | ∃ x : ℝ, x > 0 ∧ y = 2^x}
def N : Set ℝ := {x : ℝ | 2*x - x^2 > 0}

-- State the theorem
theorem M_intersect_N_eq_open_interval : M ∩ N = Set.Ioo 1 2 := by sorry

end M_intersect_N_eq_open_interval_l1801_180157


namespace triangle_to_pentagon_area_ratio_l1801_180142

/-- The ratio of the area of an isosceles right triangle to the area of a pentagon formed by the triangle and a rectangle -/
theorem triangle_to_pentagon_area_ratio :
  let triangle_leg : ℝ := 2
  let triangle_hypotenuse : ℝ := triangle_leg * Real.sqrt 2
  let rectangle_width : ℝ := triangle_hypotenuse
  let rectangle_height : ℝ := 2 * triangle_leg
  let triangle_area : ℝ := (1 / 2) * triangle_leg ^ 2
  let rectangle_area : ℝ := rectangle_width * rectangle_height
  let pentagon_area : ℝ := triangle_area + rectangle_area
  triangle_area / pentagon_area = 2 / (2 + 8 * Real.sqrt 2) :=
by sorry

end triangle_to_pentagon_area_ratio_l1801_180142


namespace seaweed_distribution_l1801_180151

theorem seaweed_distribution (total_seaweed : ℝ) (fire_percentage : ℝ) (human_percentage : ℝ) :
  total_seaweed = 400 ∧ 
  fire_percentage = 0.5 ∧ 
  human_percentage = 0.25 →
  (total_seaweed * (1 - fire_percentage) * (1 - human_percentage)) = 150 := by
  sorry

end seaweed_distribution_l1801_180151


namespace acid_solution_volume_l1801_180153

/-- Given a solution with 1.6 litres of pure acid and a concentration of 20%,
    prove that the total volume of the solution is 8 litres. -/
theorem acid_solution_volume (pure_acid : ℝ) (concentration : ℝ) (total_volume : ℝ) 
    (h1 : pure_acid = 1.6)
    (h2 : concentration = 0.2)
    (h3 : pure_acid = concentration * total_volume) : 
  total_volume = 8 := by
sorry

end acid_solution_volume_l1801_180153


namespace function_equality_l1801_180124

theorem function_equality : ∀ x : ℝ, x = 3 * x^3 := by sorry

end function_equality_l1801_180124


namespace range_of_k_l1801_180106

-- Define the condition function
def condition (x : ℝ) : Prop := 3 / (x + 1) < 1

-- Define the sufficient condition
def sufficient_condition (k : ℝ) : Prop := ∀ x, x > k → condition x

-- Define the not necessary condition
def not_necessary_condition (k : ℝ) : Prop := ∃ x, condition x ∧ x ≤ k

-- State the theorem
theorem range_of_k :
  ∀ k, (sufficient_condition k ∧ not_necessary_condition k) ↔ k ∈ Set.Ici 2 :=
sorry

end range_of_k_l1801_180106


namespace smallest_number_l1801_180130

theorem smallest_number (a b c d : ℝ) (h1 : a = 0) (h2 : b = -Real.rpow 8 (1/3)) (h3 : c = 2) (h4 : d = -1.7) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
sorry

end smallest_number_l1801_180130


namespace eighth_term_value_arithmetic_sequence_eighth_term_l1801_180109

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  -- The sum of the first six terms
  sum_first_six : ℚ
  -- The seventh term
  seventh_term : ℚ
  -- Property: The sum of the first six terms is 21
  sum_property : sum_first_six = 21
  -- Property: The seventh term is 8
  seventh_property : seventh_term = 8

/-- Theorem: The eighth term of the arithmetic sequence is 65/7 -/
theorem eighth_term_value (seq : ArithmeticSequence) : ℚ :=
  65 / 7

/-- The main theorem: Given the conditions, the eighth term is 65/7 -/
theorem arithmetic_sequence_eighth_term (seq : ArithmeticSequence) :
  eighth_term_value seq = 65 / 7 := by
  sorry


end eighth_term_value_arithmetic_sequence_eighth_term_l1801_180109


namespace arithmetic_sequence_with_geometric_subsequence_l1801_180143

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (x y z : ℝ) : Prop :=
  y / x = z / y

theorem arithmetic_sequence_with_geometric_subsequence
  (a : ℕ → ℝ)
  (h_arith : is_arithmetic_sequence a)
  (h_first : a 1 = 1)
  (h_geom : is_geometric_sequence (a 1) (a 3) (a 9)) :
  (∀ n : ℕ, a n = 1) ∨ (∀ n : ℕ, a n = n) :=
sorry

end arithmetic_sequence_with_geometric_subsequence_l1801_180143


namespace rhombus_area_l1801_180164

/-- The area of a rhombus with side length 3 cm and an interior angle of 45 degrees is 9 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 3) (h2 : θ = π/4) :
  s * s * Real.sin θ = 9 := by
  sorry

end rhombus_area_l1801_180164


namespace only_f3_is_quadratic_l1801_180117

-- Define the concept of a quadratic function
def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

-- Define the given functions
def f1 (x : ℝ) : ℝ := 3 * x
def f2 (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c
def f3 (x : ℝ) : ℝ := (x - 1)^2
def f4 (x : ℝ) : ℝ := 2

-- State the theorem
theorem only_f3_is_quadratic :
  (¬ is_quadratic f1) ∧
  (¬ ∀ a b c, is_quadratic (f2 a b c)) ∧
  is_quadratic f3 ∧
  (¬ is_quadratic f4) :=
sorry

end only_f3_is_quadratic_l1801_180117


namespace sqrt_expression_equality_l1801_180149

theorem sqrt_expression_equality : 
  (Real.sqrt 3 + 2)^2 - (2 * Real.sqrt 3 + 3 * Real.sqrt 2) * (3 * Real.sqrt 2 - 2 * Real.sqrt 3) = 1 + 4 * Real.sqrt 3 := by
  sorry

end sqrt_expression_equality_l1801_180149


namespace product_simplification_l1801_180112

theorem product_simplification (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  ((2*x + 2*y + 2*z)⁻¹) * (x⁻¹ + y⁻¹ + z⁻¹) * ((x*y + y*z + x*z)⁻¹) * 
  (2*(x*y)⁻¹ + 2*(y*z)⁻¹ + 2*(x*z)⁻¹) = (x^2 * y^2 * z^2)⁻¹ :=
by sorry

end product_simplification_l1801_180112


namespace rosalina_gifts_l1801_180119

theorem rosalina_gifts (emilio jorge pedro : ℕ) 
  (h1 : emilio = 11) 
  (h2 : jorge = 6) 
  (h3 : pedro = 4) : 
  emilio + jorge + pedro = 21 := by
  sorry

end rosalina_gifts_l1801_180119


namespace solutions_are_correct_l1801_180100

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * x^2 - 5 * x - 2 = 0
def equation2 (x : ℝ) : Prop := x^2 - 1 = 2 * (x + 1)
def equation3 (x : ℝ) : Prop := 4 * x^2 + 4 * x + 1 = 3 * (3 - x)^2
def equation4 (x : ℝ) : Prop := (2 * x + 8) * (x - 2) = x^2 + 2 * x - 17

-- Theorem stating the solutions are correct
theorem solutions_are_correct :
  (equation1 (-1/3) ∧ equation1 2) ∧
  (equation2 (-1) ∧ equation2 3) ∧
  (equation3 (-11 + 7 * Real.sqrt 3) ∧ equation3 (-11 - 7 * Real.sqrt 3)) ∧
  (equation4 (-1)) := by
  sorry

#check solutions_are_correct

end solutions_are_correct_l1801_180100


namespace intersection_in_fourth_quadrant_l1801_180170

theorem intersection_in_fourth_quadrant (k : ℝ) :
  let line1 : ℝ → ℝ := λ x => -2 * x + 3 * k + 14
  let line2 : ℝ → ℝ := λ y => (3 * k + 2 + 4 * y) / 1
  let x := k + 6
  let y := k + 2
  (∀ x', line1 x' = line2 x') →
  (x > 0 ∧ y < 0) →
  -6 < k ∧ k < -2 :=
by sorry

end intersection_in_fourth_quadrant_l1801_180170


namespace catch_difference_l1801_180123

theorem catch_difference (joe_catches derek_catches tammy_catches : ℕ) : 
  joe_catches = 23 →
  derek_catches = 2 * joe_catches - 4 →
  tammy_catches = 30 →
  tammy_catches > derek_catches / 3 →
  tammy_catches - derek_catches / 3 = 16 := by
sorry

end catch_difference_l1801_180123


namespace simplify_exponents_l1801_180137

theorem simplify_exponents (t : ℝ) (h : t ≠ 0) : (t^5 * t^3) / t^2 = t^6 := by
  sorry

end simplify_exponents_l1801_180137


namespace triangle_side_length_l1801_180126

theorem triangle_side_length (a b c : ℝ) (h1 : a + b + c = 55) (h2 : b = 20) (h3 : c = 30) : a = 5 := by
  sorry

end triangle_side_length_l1801_180126


namespace train_speed_l1801_180194

/-- Calculates the speed of a train crossing a bridge -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) 
  (h1 : train_length = 300)
  (h2 : bridge_length = 300)
  (h3 : crossing_time = 45) :
  (train_length + bridge_length) / crossing_time = 13.33333333333333 := by
sorry

end train_speed_l1801_180194


namespace florist_roses_sold_l1801_180166

/-- Represents the number of roses sold by a florist -/
def roses_sold (initial : ℕ) (picked : ℕ) (final : ℕ) : ℕ :=
  initial + picked - final

/-- Theorem stating that the florist sold 16 roses -/
theorem florist_roses_sold :
  roses_sold 37 19 40 = 16 := by
  sorry

end florist_roses_sold_l1801_180166


namespace fathers_age_l1801_180165

/-- Proves that the father's age is 64 years given the problem conditions -/
theorem fathers_age (son_age : ℕ) : 
  (4 * son_age = 4 * son_age) →  -- Father is four times as old as his son
  ((son_age - 10) + (4 * son_age - 10) = 60) →  -- Sum of ages 10 years ago was 60
  (4 * son_age = 64) :=  -- Father's present age is 64
by
  sorry

#check fathers_age

end fathers_age_l1801_180165


namespace triangle_side_length_l1801_180132

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  b + c = 2 * Real.sqrt 3 →
  A = π / 3 →
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  a = Real.sqrt 6 := by sorry

end triangle_side_length_l1801_180132


namespace correct_arrival_times_l1801_180146

/-- Represents the train journey with given parameters -/
structure TrainJourney where
  totalDistance : ℝ
  uphillDistance1 : ℝ
  flatDistance : ℝ
  uphillDistance2 : ℝ
  speedDifference : ℝ
  stationDistances : List ℝ
  stopTime : ℝ
  departureTime : ℝ
  arrivalTime : ℝ

/-- Calculate arrival times at intermediate stations -/
def calculateArrivalTimes (journey : TrainJourney) : List ℝ :=
  sorry

/-- Main theorem: Arrival times at stations are correct -/
theorem correct_arrival_times (journey : TrainJourney)
  (h1 : journey.totalDistance = 185)
  (h2 : journey.uphillDistance1 = 40)
  (h3 : journey.flatDistance = 105)
  (h4 : journey.uphillDistance2 = 40)
  (h5 : journey.speedDifference = 10)
  (h6 : journey.stationDistances = [20, 70, 100, 161])
  (h7 : journey.stopTime = 3/60)
  (h8 : journey.departureTime = 8)
  (h9 : journey.arrivalTime = 10 + 22/60) :
  calculateArrivalTimes journey = [8 + 15/60, 8 + 53/60, 9 + 21/60, 10 + 34/60] :=
sorry

end correct_arrival_times_l1801_180146


namespace technician_round_trip_l1801_180189

/-- Represents the percentage of a round-trip journey completed -/
def round_trip_percentage (outbound_percent : ℝ) (return_percent : ℝ) : ℝ :=
  (outbound_percent + return_percent * outbound_percent) * 50

/-- Theorem stating that completing the outbound journey and 10% of the return journey
    results in 55% of the round-trip being completed -/
theorem technician_round_trip :
  round_trip_percentage 100 10 = 55 := by
  sorry

end technician_round_trip_l1801_180189


namespace work_completion_time_l1801_180168

/-- Given:
  * Mahesh can complete the entire work in 45 days
  * Mahesh works for 20 days
  * Rajesh finishes the remaining work in 30 days
  Prove that Y will take 54 days to complete the work -/
theorem work_completion_time (mahesh_full_time rajesh_completion_time mahesh_work_time : ℕ)
  (h1 : mahesh_full_time = 45)
  (h2 : mahesh_work_time = 20)
  (h3 : rajesh_completion_time = 30) :
  54 = (mahesh_full_time * rajesh_completion_time) / (rajesh_completion_time - mahesh_work_time) :=
by sorry

end work_completion_time_l1801_180168


namespace binary_op_property_l1801_180181

-- Define a binary operation on a type S
def binary_op (S : Type) := S → S → S

-- State the theorem
theorem binary_op_property {S : Type} (op : binary_op S) 
  (h : ∀ (a b : S), op (op a b) a = b) :
  ∀ (a b : S), op a (op b a) = b := by
  sorry

end binary_op_property_l1801_180181


namespace unique_solution_system_l1801_180193

theorem unique_solution_system (a b c : ℝ) : 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 →
  a^4 + b^2 * c^2 = 16 * a ∧
  b^4 + c^2 * a^2 = 16 * b ∧
  c^4 + a^2 * b^2 = 16 * c →
  a = 2 ∧ b = 2 ∧ c = 2 := by
sorry

end unique_solution_system_l1801_180193


namespace sequence_gcd_property_l1801_180104

theorem sequence_gcd_property (a : ℕ → ℕ) :
  (∀ i j : ℕ, i ≠ j → Nat.gcd (a i) (a j) = Nat.gcd i j) →
  ∀ i : ℕ, a i = i := by
sorry

end sequence_gcd_property_l1801_180104


namespace two_b_values_for_two_integer_solutions_l1801_180101

theorem two_b_values_for_two_integer_solutions : 
  ∃! (s : Finset ℤ), 
    (∀ b ∈ s, ∃! (t : Finset ℤ), (∀ x ∈ t, x^2 + b*x + 5 ≤ 0) ∧ t.card = 2) ∧ 
    s.card = 2 := by
  sorry

end two_b_values_for_two_integer_solutions_l1801_180101


namespace compound_proposition_truth_l1801_180161

theorem compound_proposition_truth (p q : Prop) 
  (hp : p) (hq : ¬q) : p ∧ ¬q := by
  sorry

end compound_proposition_truth_l1801_180161


namespace go_stones_perimeter_l1801_180134

/-- Calculates the number of stones on the perimeter of a rectangle made of Go stones -/
def perimeter_stones (width : ℕ) (height : ℕ) : ℕ :=
  2 * (width + height) - 4

theorem go_stones_perimeter :
  let width : ℕ := 4
  let height : ℕ := 8
  perimeter_stones width height = 20 := by sorry

end go_stones_perimeter_l1801_180134


namespace melissa_games_l1801_180196

/-- The number of games Melissa played -/
def number_of_games (total_points : ℕ) (points_per_game : ℕ) : ℕ :=
  total_points / points_per_game

/-- Proof that Melissa played 3 games -/
theorem melissa_games : number_of_games 21 7 = 3 := by
  sorry

end melissa_games_l1801_180196


namespace no_function_satisfies_equation_l1801_180162

open Real

theorem no_function_satisfies_equation :
  ¬∃ f : ℝ → ℝ, (∀ x : ℝ, x > 0 → f x > 0) ∧
    (∀ x y : ℝ, x > 0 → y > 0 → f (x + y) = f x + f y + 1 / 2012) := by
  sorry

end no_function_satisfies_equation_l1801_180162


namespace sqrt_x_minus_one_real_l1801_180185

theorem sqrt_x_minus_one_real (x : ℝ) (h : x = 2) : ∃ y : ℝ, y ^ 2 = x - 1 := by
  sorry

end sqrt_x_minus_one_real_l1801_180185


namespace result_calculation_l1801_180114

/-- Definition of x as the solution to x = 2 + (√3 / (2 + (√3 / (2 + ...)))) -/
noncomputable def x : ℝ := 2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / (2 + Real.sqrt 3 / 2))))))

/-- Theorem stating the result of the calculation -/
theorem result_calculation : 1 / ((x + 2) * (x - 3)) = (5 + Real.sqrt 3) / -22 := by
  sorry

end result_calculation_l1801_180114


namespace union_of_A_and_B_l1801_180148

def A : Set ℕ := {1, 2, 4}
def B : Set ℕ := {2, 6}

theorem union_of_A_and_B :
  A ∪ B = {1, 2, 4, 6} := by sorry

end union_of_A_and_B_l1801_180148


namespace prob_not_all_same_l1801_180147

/-- The number of sides on each die -/
def numSides : ℕ := 6

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The probability that not all dice show the same number when rolling 
    five fair 6-sided dice -/
theorem prob_not_all_same : 
  (1 - (numSides : ℚ) / (numSides ^ numDice)) = 1295 / 1296 := by
  sorry

end prob_not_all_same_l1801_180147


namespace button_difference_l1801_180159

theorem button_difference (sue_buttons kendra_buttons mari_buttons : ℕ) : 
  sue_buttons = 6 →
  sue_buttons = kendra_buttons / 2 →
  mari_buttons = 64 →
  mari_buttons - 5 * kendra_buttons = 4 := by
  sorry

end button_difference_l1801_180159


namespace sum_reciprocals_of_factors_12_l1801_180156

/-- The set of natural-number factors of 12 -/
def factors_of_12 : Finset ℕ := {1, 2, 3, 4, 6, 12}

/-- The sum of the reciprocals of the natural-number factors of 12 -/
def sum_reciprocals : ℚ := (factors_of_12.sum fun n => (1 : ℚ) / n)

/-- Theorem: The sum of the reciprocals of the natural-number factors of 12 is equal to 7/3 -/
theorem sum_reciprocals_of_factors_12 : sum_reciprocals = 7 / 3 := by
  sorry

end sum_reciprocals_of_factors_12_l1801_180156


namespace geometric_sequence_problem_l1801_180116

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

/-- The problem statement -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geom : GeometricSequence a)
    (h_a3 : a 3 = 6)
    (h_sum : a 3 + a 5 + a 7 = 78) :
  a 5 = 18 := by
  sorry

end geometric_sequence_problem_l1801_180116


namespace multiples_of_15_between_16_and_181_l1801_180122

theorem multiples_of_15_between_16_and_181 : 
  (Finset.filter (fun n => n % 15 = 0 ∧ 16 < n ∧ n < 181) (Finset.range 181)).card = 11 := by
  sorry

end multiples_of_15_between_16_and_181_l1801_180122


namespace inequality_range_l1801_180186

theorem inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ≥ -1 → Real.log (x + 2) + a * (x^2 + x) ≥ 0) ↔ 0 ≤ a ∧ a ≤ 1 := by
  sorry

end inequality_range_l1801_180186


namespace arcsin_negative_one_l1801_180155

theorem arcsin_negative_one : Real.arcsin (-1) = -π / 2 := by
  sorry

end arcsin_negative_one_l1801_180155


namespace square_field_area_l1801_180110

/-- The area of a square field with side length 8 meters is 64 square meters. -/
theorem square_field_area : 
  ∀ (side_length area : ℝ), 
  side_length = 8 → 
  area = side_length ^ 2 → 
  area = 64 := by sorry

end square_field_area_l1801_180110


namespace divisibility_implies_equality_l1801_180145

theorem divisibility_implies_equality (a b n : ℕ) 
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) : 
  a = b^n := by
sorry

end divisibility_implies_equality_l1801_180145


namespace michelle_crayon_boxes_l1801_180177

/-- Given Michelle has 35 crayons in total and each box holds 5 crayons, 
    prove that the number of boxes Michelle has is 7. -/
theorem michelle_crayon_boxes 
  (total_crayons : ℕ) 
  (crayons_per_box : ℕ) 
  (h1 : total_crayons = 35)
  (h2 : crayons_per_box = 5) : 
  total_crayons / crayons_per_box = 7 := by
  sorry

#check michelle_crayon_boxes

end michelle_crayon_boxes_l1801_180177


namespace parallelogram_fourth_vertex_l1801_180107

/-- A parallelogram in 2D space --/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ

/-- Check if a given point is a valid fourth vertex of a parallelogram --/
def isValidFourthVertex (p : Parallelogram) (point : ℝ × ℝ) : Prop :=
  point = (11, 4) ∨ point = (-1, 12) ∨ point = (3, -12)

/-- The main theorem --/
theorem parallelogram_fourth_vertex 
  (p : Parallelogram) 
  (h1 : p.v1 = (1, 0)) 
  (h2 : p.v2 = (5, 8)) 
  (h3 : p.v3 = (7, -4)) : 
  isValidFourthVertex p p.v4 := by
  sorry

end parallelogram_fourth_vertex_l1801_180107


namespace rectangle_area_change_l1801_180169

/-- Given a rectangle with area 300 square meters, if its length is doubled and
    its width is tripled, the area of the new rectangle will be 1800 square meters. -/
theorem rectangle_area_change (length width : ℝ) 
    (h_area : length * width = 300) : 
    (2 * length) * (3 * width) = 1800 := by
  sorry

end rectangle_area_change_l1801_180169
