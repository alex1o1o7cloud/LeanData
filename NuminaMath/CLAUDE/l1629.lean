import Mathlib

namespace pastor_prayer_ratio_l1629_162934

/-- Represents the number of prayers for a pastor on a given day --/
structure DailyPrayers where
  weekday : ℕ
  sunday : ℕ

/-- Represents the total prayers for a pastor in a week --/
def WeeklyPrayers (d : DailyPrayers) : ℕ := 6 * d.weekday + d.sunday

/-- Pastor Paul's prayer schedule --/
def paul : DailyPrayers :=
  { weekday := 20
    sunday := 40 }

/-- Pastor Bruce's prayer schedule --/
def bruce : DailyPrayers :=
  { weekday := paul.weekday / 2
    sunday := WeeklyPrayers paul - WeeklyPrayers { weekday := paul.weekday / 2, sunday := 0 } - 20 }

theorem pastor_prayer_ratio :
  bruce.sunday / paul.sunday = 2 := by sorry

end pastor_prayer_ratio_l1629_162934


namespace average_geometric_sequence_l1629_162963

theorem average_geometric_sequence (y : ℝ) : 
  let sequence := [0, 3*y, 9*y, 27*y, 81*y]
  (sequence.sum / sequence.length : ℝ) = 24*y := by
  sorry

end average_geometric_sequence_l1629_162963


namespace sqrt_three_properties_l1629_162935

theorem sqrt_three_properties : ∃ x : ℝ, Irrational x ∧ 0 < x ∧ x < 3 :=
  by
  use Real.sqrt 3
  sorry

end sqrt_three_properties_l1629_162935


namespace distinct_integers_product_sum_l1629_162902

theorem distinct_integers_product_sum (p q r s t : ℤ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ 
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ 
  r ≠ s ∧ r ≠ t ∧ 
  s ≠ t → 
  (8 - p) * (8 - q) * (8 - r) * (8 - s) * (8 - t) = 80 →
  p + q + r + s + t = 36 := by
sorry

end distinct_integers_product_sum_l1629_162902


namespace cubic_identity_for_fifty_l1629_162907

theorem cubic_identity_for_fifty : 50^3 + 3*(50^2) + 3*50 + 1 = 261051 := by
  sorry

end cubic_identity_for_fifty_l1629_162907


namespace parabola_vertex_l1629_162951

/-- The vertex of a parabola defined by y = -(x+1)^2 is the point (-1, 0) -/
theorem parabola_vertex (x y : ℝ) : 
  y = -(x + 1)^2 → (∃ (a : ℝ), y = a * (x + 1)^2 ∧ a = -1) → 
  (∃ (h k : ℝ), y = -(x - h)^2 + k ∧ h = -1 ∧ k = 0) :=
by sorry

end parabola_vertex_l1629_162951


namespace component_scrap_probability_l1629_162960

/-- The probability of a component passing the first inspection -/
def prob_pass_first : ℝ := 0.8

/-- The probability of a component passing the second inspection -/
def prob_pass_second : ℝ := 0.9

/-- The probability of a component being scrapped -/
def prob_scrapped : ℝ := (1 - prob_pass_first) * (1 - prob_pass_second)

theorem component_scrap_probability : prob_scrapped = 0.02 := by
  sorry

end component_scrap_probability_l1629_162960


namespace total_cars_is_seventeen_l1629_162953

/-- The number of cars Tommy has -/
def tommy_cars : ℕ := 3

/-- The number of cars Jessie has -/
def jessie_cars : ℕ := 3

/-- The number of additional cars Jessie's older brother has compared to Tommy and Jessie combined -/
def brother_additional_cars : ℕ := 5

/-- The total number of cars for all three of them -/
def total_cars : ℕ := tommy_cars + jessie_cars + (tommy_cars + jessie_cars + brother_additional_cars)

theorem total_cars_is_seventeen : total_cars = 17 := by
  sorry

end total_cars_is_seventeen_l1629_162953


namespace num_factors_41040_eq_80_l1629_162910

/-- The number of positive factors of 41040 -/
def num_factors_41040 : ℕ :=
  (Finset.filter (· ∣ 41040) (Finset.range 41041)).card

/-- Theorem stating that the number of positive factors of 41040 is 80 -/
theorem num_factors_41040_eq_80 : num_factors_41040 = 80 := by
  sorry

end num_factors_41040_eq_80_l1629_162910


namespace smallest_common_multiple_of_6_and_5_l1629_162941

theorem smallest_common_multiple_of_6_and_5 : ∃ n : ℕ, 
  n > 0 ∧ 
  6 ∣ n ∧ 
  5 ∣ n ∧ 
  ∀ m : ℕ, m > 0 → 6 ∣ m → 5 ∣ m → n ≤ m :=
by
  use 30
  sorry

end smallest_common_multiple_of_6_and_5_l1629_162941


namespace star_value_l1629_162971

/-- The operation * for non-zero integers -/
def star (a b : ℤ) : ℚ := (a : ℚ)⁻¹ + (b : ℚ)⁻¹

/-- Theorem: If a + b = 10 and ab = 24, then a * b = 5/12 -/
theorem star_value (a b : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) 
  (sum_eq : a + b = 10) (prod_eq : a * b = 24) : 
  star a b = 5 / 12 := by
  sorry

end star_value_l1629_162971


namespace tommys_tomato_profit_l1629_162938

/-- Represents the problem of calculating Tommy's profit from selling tomatoes -/
theorem tommys_tomato_profit :
  let crate_capacity : ℕ := 20 -- kg
  let num_crates : ℕ := 3
  let crates_cost : ℕ := 330 -- $
  let selling_price : ℕ := 6 -- $ per kg
  let rotten_tomatoes : ℕ := 3 -- kg
  
  let total_capacity : ℕ := crate_capacity * num_crates
  let sellable_tomatoes : ℕ := total_capacity - rotten_tomatoes
  let revenue : ℕ := sellable_tomatoes * selling_price
  let profit : ℤ := revenue - crates_cost

  profit = 12 := by
  sorry

/- Note: We use ℕ (natural numbers) for non-negative integers and ℤ (integers) for the final profit calculation to allow for the possibility of negative profit. -/

end tommys_tomato_profit_l1629_162938


namespace arccos_gt_twice_arcsin_l1629_162999

theorem arccos_gt_twice_arcsin (x : ℝ) : 
  -1 ≤ x ∧ x ≤ 1 → (Real.arccos x > 2 * Real.arcsin x ↔ -1 < x ∧ x ≤ (1 - Real.sqrt 3) / 2) :=
by sorry

end arccos_gt_twice_arcsin_l1629_162999


namespace number_above_345_l1629_162986

/-- Represents the triangular array structure -/
structure TriangularArray where
  /-- Returns the number of elements in the k-th row -/
  elementsInRow : ℕ → ℕ
  /-- Returns the sum of elements up to and including the k-th row -/
  sumUpToRow : ℕ → ℕ
  /-- First row has one element -/
  first_row_one : elementsInRow 1 = 1
  /-- Each row has three more elements than the previous -/
  row_increment : ∀ k, elementsInRow (k + 1) = elementsInRow k + 3
  /-- Sum formula for elements up to k-th row -/
  sum_formula : ∀ k, sumUpToRow k = k * (3 * k - 1) / 2

theorem number_above_345 (arr : TriangularArray) :
  ∃ (row : ℕ) (pos : ℕ),
    arr.sumUpToRow (row - 1) < 345 ∧
    345 ≤ arr.sumUpToRow row ∧
    pos = 345 - arr.sumUpToRow (row - 1) ∧
    arr.sumUpToRow (row - 2) + pos = 308 :=
  sorry

end number_above_345_l1629_162986


namespace factorization_of_quadratic_l1629_162930

theorem factorization_of_quadratic (a : ℝ) : a^2 + 5*a = a*(a+5) := by
  sorry

end factorization_of_quadratic_l1629_162930


namespace sine_inequality_l1629_162913

theorem sine_inequality (x : Real) (h : 0 < x ∧ x < Real.pi / 4) :
  Real.sin (Real.sin x) < Real.sin x ∧ Real.sin x < Real.sin (Real.tan x) :=
by sorry

end sine_inequality_l1629_162913


namespace table_covering_l1629_162926

/-- A tile type used to cover the table -/
inductive Tile
  | Square  -- 2×2 square tile
  | LShaped -- L-shaped tile with 5 cells

/-- Represents a covering of the table -/
def Covering (m n : ℕ) := List (ℕ × ℕ × Tile)

/-- Checks if a covering is valid for the given table dimensions -/
def IsValidCovering (m n : ℕ) (c : Covering m n) : Prop := sorry

/-- The main theorem stating the condition for possible covering -/
theorem table_covering (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  (∃ c : Covering m n, IsValidCovering m n c) ↔ (6 ∣ m ∨ 6 ∣ n) :=
sorry

end table_covering_l1629_162926


namespace quadratic_equation_solution_l1629_162972

theorem quadratic_equation_solution : 
  ∃ x₁ x₂ : ℝ, x₁ = 2 ∧ x₂ = (1/2 : ℝ) ∧ 
  (∀ x : ℝ, 2 * x^2 - 5 * x + 2 = 0 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end quadratic_equation_solution_l1629_162972


namespace salary_restoration_l1629_162917

theorem salary_restoration (original_salary : ℝ) (original_salary_positive : original_salary > 0) :
  let reduced_salary := 0.8 * original_salary
  let restored_salary := reduced_salary * 1.25
  restored_salary = original_salary := by
sorry

end salary_restoration_l1629_162917


namespace cosine_equation_solutions_l1629_162936

theorem cosine_equation_solutions :
  ∃! (n : ℕ), ∃ (S : Finset ℝ),
    (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi) ∧
    (∀ x ∈ S, 3 * (Real.cos x)^3 - 7 * (Real.cos x)^2 + 3 * Real.cos x = 0) ∧
    Finset.card S = n ∧
    n = 4 :=
by sorry

end cosine_equation_solutions_l1629_162936


namespace potato_distribution_l1629_162906

theorem potato_distribution (total_potatoes : ℕ) (num_people : ℕ) (potatoes_per_person : ℕ) :
  total_potatoes = 24 →
  num_people = 3 →
  total_potatoes = num_people * potatoes_per_person →
  potatoes_per_person = 8 := by
  sorry

end potato_distribution_l1629_162906


namespace total_clothing_cost_l1629_162903

def shorts_cost : ℚ := 14.28
def jacket_cost : ℚ := 4.74

theorem total_clothing_cost : shorts_cost + jacket_cost = 19.02 := by
  sorry

end total_clothing_cost_l1629_162903


namespace orange_ring_weight_l1629_162912

/-- The weight of the orange ring in an experiment, given the weights of other rings and the total weight -/
theorem orange_ring_weight 
  (total_weight : Float) 
  (purple_weight : Float) 
  (white_weight : Float) 
  (h1 : total_weight = 0.8333333333) 
  (h2 : purple_weight = 0.3333333333333333) 
  (h3 : white_weight = 0.4166666666666667) : 
  total_weight - purple_weight - white_weight = 0.0833333333 := by
  sorry

end orange_ring_weight_l1629_162912


namespace min_value_expression_min_value_attainable_l1629_162977

theorem min_value_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  4 * a^4 + 8 * b^4 + 16 * c^4 + 1 / (a * b * c) ≥ 10 := by
  sorry

theorem min_value_attainable :
  ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧
  4 * a^4 + 8 * b^4 + 16 * c^4 + 1 / (a * b * c) = 10 := by
  sorry

end min_value_expression_min_value_attainable_l1629_162977


namespace problem_statement_l1629_162997

theorem problem_statement : (-3)^7 / 3^5 + 5^5 - 8^2 = 3052 := by
  sorry

end problem_statement_l1629_162997


namespace fixed_points_subset_stable_points_l1629_162919

-- Define the function type
def RealFunction := ℝ → ℝ

-- Define the set of fixed points
def FixedPoints (f : RealFunction) : Set ℝ :=
  {x : ℝ | f x = x}

-- Define the set of stable points
def StablePoints (f : RealFunction) : Set ℝ :=
  {x : ℝ | f (f x) = x}

-- Theorem statement
theorem fixed_points_subset_stable_points (f : RealFunction) :
  FixedPoints f ⊆ StablePoints f := by
  sorry


end fixed_points_subset_stable_points_l1629_162919


namespace division_remainder_problem_l1629_162983

theorem division_remainder_problem (dividend : Nat) (divisor : Nat) : 
  Prime dividend → Prime divisor → dividend = divisor * 7 + 1054 := by
  sorry

end division_remainder_problem_l1629_162983


namespace coprime_count_2016_l1629_162968

theorem coprime_count_2016 : Nat.totient 2016 = 576 := by
  sorry

end coprime_count_2016_l1629_162968


namespace ln_cube_inequality_l1629_162911

theorem ln_cube_inequality (a b : ℝ) : 
  (∃ a b, a^3 < b^3 ∧ ¬(Real.log a < Real.log b)) ∧ 
  (∀ a b, Real.log a < Real.log b → a^3 < b^3) :=
sorry

end ln_cube_inequality_l1629_162911


namespace division_problem_l1629_162985

theorem division_problem (n : ℕ) : 
  n / 18 = 11 ∧ n % 18 = 1 → n = 199 := by
  sorry

end division_problem_l1629_162985


namespace rhinoceros_grazing_area_l1629_162904

theorem rhinoceros_grazing_area 
  (initial_population : ℕ) 
  (watering_area : ℕ) 
  (population_increase_rate : ℚ) 
  (total_preserve_area : ℕ) 
  (h1 : initial_population = 8000)
  (h2 : watering_area = 10000)
  (h3 : population_increase_rate = 1/10)
  (h4 : total_preserve_area = 890000) :
  let final_population := initial_population + initial_population * population_increase_rate
  let grazing_area := total_preserve_area - watering_area
  grazing_area / final_population = 100 := by
sorry

end rhinoceros_grazing_area_l1629_162904


namespace max_gcd_consecutive_terms_l1629_162993

def a (n : ℕ) : ℕ := 2 * Nat.factorial n + n

theorem max_gcd_consecutive_terms (n : ℕ) : Nat.gcd (a n) (a (n + 1)) ≤ 1 := by
  sorry

end max_gcd_consecutive_terms_l1629_162993


namespace total_viewing_time_l1629_162939

/-- The viewing times for the original animal types -/
def original_times : List Nat := [4, 6, 7, 5, 9]

/-- The viewing times for the new animal types -/
def new_times : List Nat := [3, 7, 8, 10]

/-- The total number of animal types -/
def total_types : Nat := original_times.length + new_times.length

theorem total_viewing_time :
  (List.sum original_times) + (List.sum new_times) = 59 := by
  sorry

end total_viewing_time_l1629_162939


namespace big_sixteen_game_count_l1629_162952

/-- Represents a basketball league with the given structure -/
structure BasketballLeague where
  totalTeams : Nat
  divisionsCount : Nat
  intraGameCount : Nat
  interGameCount : Nat

/-- Calculates the total number of scheduled games in the league -/
def totalGames (league : BasketballLeague) : Nat :=
  let teamsPerDivision := league.totalTeams / league.divisionsCount
  let intraGamesPerDivision := teamsPerDivision * (teamsPerDivision - 1) / 2 * league.intraGameCount
  let totalIntraGames := intraGamesPerDivision * league.divisionsCount
  let totalInterGames := league.totalTeams * teamsPerDivision * league.interGameCount / 2
  totalIntraGames + totalInterGames

/-- Theorem stating that the Big Sixteen Basketball League schedules 296 games -/
theorem big_sixteen_game_count :
  let bigSixteen : BasketballLeague := {
    totalTeams := 16
    divisionsCount := 2
    intraGameCount := 3
    interGameCount := 2
  }
  totalGames bigSixteen = 296 := by
  sorry

end big_sixteen_game_count_l1629_162952


namespace shell_collection_ratio_l1629_162954

theorem shell_collection_ratio :
  ∀ (laurie_shells ben_shells alan_shells : ℕ),
    laurie_shells = 36 →
    ben_shells = laurie_shells / 3 →
    alan_shells = 48 →
    alan_shells / ben_shells = 4 :=
by
  sorry

end shell_collection_ratio_l1629_162954


namespace parallel_line_through_point_l1629_162925

/-- A line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

/-- Check if a point lies on a line -/
def pointOnLine (x y : ℝ) (l : Line) : Prop :=
  y = l.slope * x + l.intercept

/-- The given line y = -2x + 1 -/
def givenLine : Line :=
  { slope := -2, intercept := 1 }

theorem parallel_line_through_point :
  ∃ (l : Line), parallel l givenLine ∧ pointOnLine (-1) 2 l ∧ l.slope * x + l.intercept = -2 * x :=
sorry

end parallel_line_through_point_l1629_162925


namespace units_digit_of_n_l1629_162937

def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_of_n (m n : ℕ) (h1 : m * n = 31^4) (h2 : units_digit m = 6) : 
  units_digit n = 7 := by sorry

end units_digit_of_n_l1629_162937


namespace surface_area_of_specific_cut_tetrahedron_l1629_162946

/-- Represents a right prism with equilateral triangular bases -/
structure RightPrism where
  height : ℝ
  base_side_length : ℝ

/-- Represents the tetrahedron formed by cutting the prism -/
structure CutTetrahedron where
  prism : RightPrism

/-- Calculate the surface area of the cut tetrahedron -/
noncomputable def surface_area (tetra : CutTetrahedron) : ℝ :=
  sorry

/-- Theorem statement for the surface area of the specific cut tetrahedron -/
theorem surface_area_of_specific_cut_tetrahedron :
  let prism := RightPrism.mk 20 10
  let tetra := CutTetrahedron.mk prism
  surface_area tetra = 50 + (25 * Real.sqrt 3) / 4 + (5 * Real.sqrt 118.75) / 2 :=
by sorry

end surface_area_of_specific_cut_tetrahedron_l1629_162946


namespace point_plane_configuration_exists_l1629_162923

-- Define a type for points in space
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a type for planes in space
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define a function to check if a point lies on a plane
def pointOnPlane (p : Point) (pl : Plane) : Prop :=
  pl.a * p.x + pl.b * p.y + pl.c * p.z + pl.d = 0

-- Define a function to check if a set of points is collinear
def collinear (points : Set Point) : Prop :=
  ∃ (a b c : ℝ), ∀ p ∈ points, a * p.x + b * p.y + c * p.z = 0

-- State the theorem
theorem point_plane_configuration_exists :
  ∃ (points : Set Point) (planes : Set Plane),
    -- There are several points and planes
    (points.Nonempty ∧ planes.Nonempty) ∧
    -- Through any two points, exactly two planes pass
    (∀ p q : Point, p ∈ points → q ∈ points → p ≠ q →
      ∃! (pl1 pl2 : Plane), pl1 ∈ planes ∧ pl2 ∈ planes ∧ pl1 ≠ pl2 ∧
        pointOnPlane p pl1 ∧ pointOnPlane q pl1 ∧
        pointOnPlane p pl2 ∧ pointOnPlane q pl2) ∧
    -- Each plane contains at least four points
    (∀ pl : Plane, pl ∈ planes →
      ∃ (p1 p2 p3 p4 : Point), p1 ∈ points ∧ p2 ∈ points ∧ p3 ∈ points ∧ p4 ∈ points ∧
        p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
        pointOnPlane p1 pl ∧ pointOnPlane p2 pl ∧ pointOnPlane p3 pl ∧ pointOnPlane p4 pl) ∧
    -- Not all points lie on a single line
    ¬collinear points :=
by
  sorry

end point_plane_configuration_exists_l1629_162923


namespace naoh_equals_agoh_l1629_162929

/-- Represents the number of moles of a substance -/
def Moles : Type := ℝ

/-- The reaction between AgNO3 and NaOH to form AgOH -/
structure Reaction where
  agno3_initial : Moles
  agoh_formed : Moles
  naoh_combined : Moles

/-- The conditions of the reaction -/
class ReactionConditions (r : Reaction) where
  agno3_agoh_equal : r.agno3_initial = r.agoh_formed
  one_to_one_ratio : r.agoh_formed = r.naoh_combined

/-- Theorem stating that the number of moles of NaOH combined equals the number of moles of AgOH formed -/
theorem naoh_equals_agoh (r : Reaction) [ReactionConditions r] : r.naoh_combined = r.agoh_formed := by
  sorry

end naoh_equals_agoh_l1629_162929


namespace smallest_angle_in_characteristic_triangle_l1629_162994

/-- A characteristic triangle is a triangle where one interior angle is twice another. -/
structure CharacteristicTriangle where
  α : ℝ  -- The larger angle (characteristic angle)
  β : ℝ  -- The smaller angle
  γ : ℝ  -- The third angle
  angle_sum : α + β + γ = 180
  characteristic : α = 2 * β

/-- The smallest angle in a characteristic triangle with characteristic angle 100° is 30°. -/
theorem smallest_angle_in_characteristic_triangle :
  ∀ (t : CharacteristicTriangle), t.α = 100 → min t.α (min t.β t.γ) = 30 := by
  sorry

end smallest_angle_in_characteristic_triangle_l1629_162994


namespace order_of_x_l1629_162915

theorem order_of_x (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)
  (eq1 : x₁ + x₂ + x₃ = a₁)
  (eq2 : x₂ + x₃ + x₁ = a₂)
  (eq3 : x₃ + x₄ + x₅ = a₃)
  (eq4 : x₄ + x₅ + x₁ = a₄)
  (eq5 : x₅ + x₁ + x₂ = a₅)
  (h : a₁ > a₂ ∧ a₂ > a₃ ∧ a₃ > a₄ ∧ a₄ > a₅) :
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ := by
  sorry

end order_of_x_l1629_162915


namespace total_crayons_for_six_children_l1629_162940

/-- Calculates the total number of crayons given the number of children and crayons per child -/
def total_crayons (num_children : ℕ) (crayons_per_child : ℕ) : ℕ :=
  num_children * crayons_per_child

/-- Theorem: Given 6 children with 3 crayons each, the total number of crayons is 18 -/
theorem total_crayons_for_six_children :
  total_crayons 6 3 = 18 := by
  sorry

end total_crayons_for_six_children_l1629_162940


namespace payment_calculation_l1629_162962

/-- Calculates the payment per safely delivered bowl -/
def payment_per_bowl (total_bowls : ℕ) (fee : ℚ) (cost_per_damaged : ℚ) 
  (lost_bowls : ℕ) (broken_bowls : ℕ) (total_payment : ℚ) : ℚ :=
  let safely_delivered := total_bowls - lost_bowls - broken_bowls
  (total_payment - fee) / safely_delivered

theorem payment_calculation : 
  let result := payment_per_bowl 638 100 4 12 15 1825
  ∃ (ε : ℚ), ε > 0 ∧ ε < (1/100) ∧ |result - (282/100)| < ε := by
  sorry

end payment_calculation_l1629_162962


namespace complement_of_A_in_U_l1629_162957

def U : Set Int := {-2, -1, 0, 1, 2}

def A : Set Int := {x | x ∈ U ∧ x^2 + x - 2 < 0}

theorem complement_of_A_in_U :
  {x | x ∈ U ∧ x ∉ A} = {-2, 1, 2} := by sorry

end complement_of_A_in_U_l1629_162957


namespace infinite_solutions_l1629_162975

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The first equation: 3x - 4y = 10 -/
def equation1 (p : Point) : Prop := 3 * p.x - 4 * p.y = 10

/-- The second equation: 9x - 12y = 30 -/
def equation2 (p : Point) : Prop := 9 * p.x - 12 * p.y = 30

/-- A solution satisfies both equations -/
def is_solution (p : Point) : Prop := equation1 p ∧ equation2 p

/-- The set of all solutions -/
def solution_set : Set Point := {p | is_solution p}

/-- The theorem stating that there are infinitely many solutions -/
theorem infinite_solutions : Set.Infinite solution_set := by sorry

end infinite_solutions_l1629_162975


namespace solution_set_f_geq_neg_two_max_a_for_f_leq_x_minus_a_l1629_162921

-- Define the function f
def f (x : ℝ) : ℝ := |x + 2| - 2 * |x - 1|

-- Theorem for the solution set of f(x) ≥ -2
theorem solution_set_f_geq_neg_two :
  {x : ℝ | f x ≥ -2} = {x : ℝ | -2/3 ≤ x ∧ x ≤ 6} := by sorry

-- Theorem for the maximum value of a
theorem max_a_for_f_leq_x_minus_a :
  ∀ a : ℝ, (∀ x : ℝ, f x ≤ x - a) ↔ a ≤ -2 := by sorry

end solution_set_f_geq_neg_two_max_a_for_f_leq_x_minus_a_l1629_162921


namespace student_addition_mistake_l1629_162916

theorem student_addition_mistake (a b : ℤ) :
  (a + 10 * b = 7182) ∧ (a + b = 3132) → (a = 2682 ∧ b = 450) := by
  sorry

end student_addition_mistake_l1629_162916


namespace exam_time_ratio_l1629_162927

theorem exam_time_ratio :
  let total_time : ℕ := 3 * 60  -- 3 hours in minutes
  let time_type_a : ℕ := 120    -- Time spent on type A problems in minutes
  let time_type_b : ℕ := total_time - time_type_a  -- Time spent on type B problems
  (time_type_a : ℚ) / time_type_b = 2 / 1 :=
by sorry

end exam_time_ratio_l1629_162927


namespace females_with_advanced_degrees_l1629_162995

theorem females_with_advanced_degrees 
  (total_employees : ℕ)
  (female_employees : ℕ)
  (advanced_degree_employees : ℕ)
  (males_with_college_only : ℕ)
  (h1 : total_employees = 148)
  (h2 : female_employees = 92)
  (h3 : advanced_degree_employees = 78)
  (h4 : males_with_college_only = 31) :
  total_employees - female_employees - males_with_college_only - 
  (advanced_degree_employees - (total_employees - female_employees - males_with_college_only)) = 53 := by
  sorry

end females_with_advanced_degrees_l1629_162995


namespace stating_dual_polyhedra_equal_spheres_l1629_162982

/-- Represents a regular polyhedron with its associated sphere radii -/
structure RegularPolyhedron where
  R : ℝ  -- radius of circumscribed sphere
  r : ℝ  -- radius of inscribed sphere
  p : ℝ  -- radius of half-inscribed sphere

/-- Represents a pair of dual regular polyhedra -/
structure DualPolyhedraPair where
  T1 : RegularPolyhedron
  T2 : RegularPolyhedron

/-- 
Theorem stating that for dual regular polyhedra with equal inscribed spheres,
their circumscribed spheres are also equal.
-/
theorem dual_polyhedra_equal_spheres (pair : DualPolyhedraPair) :
  pair.T1.r = pair.T2.r → pair.T1.R = pair.T2.R := by
  sorry


end stating_dual_polyhedra_equal_spheres_l1629_162982


namespace vector_sum_proof_l1629_162992

def v1 : Fin 3 → ℤ := ![- 7, 3, 5]
def v2 : Fin 3 → ℤ := ![4, - 1, - 6]
def v3 : Fin 3 → ℤ := ![1, 8, 2]

theorem vector_sum_proof :
  (v1 + v2 + v3) = ![- 2, 10, 1] := by sorry

end vector_sum_proof_l1629_162992


namespace fraction_equality_implies_equality_l1629_162996

theorem fraction_equality_implies_equality (a b : ℝ) : 
  a / (-5 : ℝ) = b / (-5 : ℝ) → a = b := by
  sorry

end fraction_equality_implies_equality_l1629_162996


namespace georges_initial_money_l1629_162909

theorem georges_initial_money (shirt_cost sock_cost money_left : ℕ) :
  shirt_cost = 24 →
  sock_cost = 11 →
  money_left = 65 →
  shirt_cost + sock_cost + money_left = 100 :=
by sorry

end georges_initial_money_l1629_162909


namespace weight_difference_after_one_year_l1629_162958

/-- Calculates the final weight of the labrador puppy after one year -/
def labrador_final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 1.1
  let weight2 := weight1 * 1.2
  let weight3 := weight2 * 1.25
  weight3 + 5

/-- Calculates the final weight of the dachshund puppy after one year -/
def dachshund_final_weight (initial_weight : ℝ) : ℝ :=
  let weight1 := initial_weight * 1.05
  let weight2 := weight1 * 1.15
  let weight3 := weight2 - 1
  let weight4 := weight3 * 1.2
  weight4 + 3

/-- The difference in weight between the labrador and dachshund puppies after one year -/
theorem weight_difference_after_one_year :
  labrador_final_weight 40 - dachshund_final_weight 12 = 51.812 := by
  sorry

end weight_difference_after_one_year_l1629_162958


namespace triangle_side_length_l1629_162931

theorem triangle_side_length (a b c : ℝ) (A : ℝ) :
  a = 4 → b = 2 → Real.cos A = 1/4 → c^2 = a^2 + b^2 - 2*a*b*(Real.cos A) → c = 4 := by
  sorry

end triangle_side_length_l1629_162931


namespace price_difference_l1629_162945

def coupon_A (P : ℝ) : ℝ := 0.20 * P
def coupon_B : ℝ := 40
def coupon_C (P : ℝ) : ℝ := 0.30 * (P - 150)

def valid_price (P : ℝ) : Prop :=
  P > 150 ∧ coupon_A P ≥ max coupon_B (coupon_C P)

theorem price_difference : 
  ∃ (x y : ℝ), valid_price x ∧ valid_price y ∧
  (∀ P, valid_price P → x ≤ P ∧ P ≤ y) ∧
  y - x = 250 :=
sorry

end price_difference_l1629_162945


namespace parabola_vertex_on_x_axis_l1629_162944

/-- A parabola with equation y = x^2 - 6x + c has its vertex on the x-axis if and only if c = 9 -/
theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ x : ℝ, x^2 - 6*x + c = 0 ∧ ∀ y : ℝ, y^2 - 6*y + c ≥ x^2 - 6*x + c) ↔ c = 9 := by
  sorry

end parabola_vertex_on_x_axis_l1629_162944


namespace square_implies_four_right_angles_but_not_conversely_l1629_162948

-- Define a quadrilateral
structure Quadrilateral :=
  (A B C D : Point)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  -- A square has four equal sides and four right angles
  sorry

-- Define a quadrilateral with four right angles
def has_four_right_angles (q : Quadrilateral) : Prop :=
  -- A quadrilateral has four right angles
  sorry

-- Theorem statement
theorem square_implies_four_right_angles_but_not_conversely :
  (∀ q : Quadrilateral, is_square q → has_four_right_angles q) ∧
  (∃ q : Quadrilateral, has_four_right_angles q ∧ ¬is_square q) :=
sorry

end square_implies_four_right_angles_but_not_conversely_l1629_162948


namespace matrix_determinant_l1629_162980

def matrix : Matrix (Fin 2) (Fin 2) ℤ := !![5, -3; 4, 7]

theorem matrix_determinant :
  Matrix.det matrix = 47 := by sorry

end matrix_determinant_l1629_162980


namespace sum_remainder_theorem_l1629_162969

/-- Calculates the sum S as defined in the problem -/
def calculate_sum : ℚ := sorry

/-- Finds the closest natural number to a given rational number -/
def closest_natural (q : ℚ) : ℕ := sorry

/-- The main theorem stating that the remainder when the closest natural number
    to the sum S is divided by 5 is equal to 4 -/
theorem sum_remainder_theorem : 
  (closest_natural calculate_sum) % 5 = 4 := by sorry

end sum_remainder_theorem_l1629_162969


namespace square_roots_problem_l1629_162965

theorem square_roots_problem (x a b : ℝ) (hx : x > 0) 
  (h_roots : x = a^2 ∧ x = (a + b)^2) (h_sum : 2*a + b = 0) :
  (a = -2 → b = 4 ∧ x = 4) ∧
  (b = 6 → a = -3 ∧ x = 9) ∧
  (a^2*x + (a + b)^2*x = 8 → x = 2) := by
sorry

end square_roots_problem_l1629_162965


namespace negation_equivalence_l1629_162987

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ = x₀ - 1) ↔ 
  (∀ x : ℝ, x > 0 → Real.log x ≠ x - 1) :=
by sorry

end negation_equivalence_l1629_162987


namespace ninth_term_of_arithmetic_sequence_l1629_162979

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
structure ArithmeticSequence (α : Type*) [Add α] where
  a : ℕ → α  -- The sequence
  d : α      -- The common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 3rd term is 3/8 and the 15th term is 7/9, 
    the 9th term is equal to 83/144. -/
theorem ninth_term_of_arithmetic_sequence 
  (seq : ArithmeticSequence ℚ) 
  (h3 : seq.a 3 = 3/8) 
  (h15 : seq.a 15 = 7/9) : 
  seq.a 9 = 83/144 := by
sorry

end ninth_term_of_arithmetic_sequence_l1629_162979


namespace walkway_problem_l1629_162949

/-- Represents the scenario of a person walking on a moving walkway -/
structure WalkwayScenario where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time taken to walk when the walkway is stopped -/
def time_when_stopped (scenario : WalkwayScenario) : ℝ :=
  -- The actual calculation is not provided here
  sorry

/-- Theorem stating the correct time when the walkway is stopped -/
theorem walkway_problem (scenario : WalkwayScenario) 
  (h1 : scenario.length = 80)
  (h2 : scenario.time_with = 40)
  (h3 : scenario.time_against = 120) :
  time_when_stopped scenario = 60 := by
  sorry

end walkway_problem_l1629_162949


namespace salem_poem_words_per_line_l1629_162998

/-- A poem with a given structure and word count -/
structure Poem where
  stanzas : ℕ
  lines_per_stanza : ℕ
  total_words : ℕ

/-- Calculate the number of words per line in a poem -/
def words_per_line (p : Poem) : ℕ :=
  p.total_words / (p.stanzas * p.lines_per_stanza)

/-- Theorem: Given a poem with 20 stanzas, 10 lines per stanza, and 1600 total words,
    the number of words per line is 8 -/
theorem salem_poem_words_per_line :
  let p : Poem := { stanzas := 20, lines_per_stanza := 10, total_words := 1600 }
  words_per_line p = 8 := by
  sorry

#check salem_poem_words_per_line

end salem_poem_words_per_line_l1629_162998


namespace triangle_trig_ratio_l1629_162989

theorem triangle_trig_ratio (A B C : ℝ) (h : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (ratio : Real.sin A / Real.sin B = 2/3 ∧ Real.sin B / Real.sin C = 3/4) : 
  Real.cos C = -1/4 := by
  sorry

end triangle_trig_ratio_l1629_162989


namespace number_of_students_l1629_162964

theorem number_of_students (total_books : ℕ) : 
  (∃ (x : ℕ), 3 * x + 20 = total_books ∧ 4 * x = total_books + 25) → 
  (∃ (x : ℕ), x = 45 ∧ 3 * x + 20 = total_books ∧ 4 * x = total_books + 25) :=
by sorry

end number_of_students_l1629_162964


namespace sum_15_terms_l1629_162966

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- Sum of the first 5 terms -/
  sum5 : ℝ
  /-- Sum of the first 10 terms -/
  sum10 : ℝ
  /-- The sequence is arithmetic -/
  is_arithmetic : True
  /-- The sum of the first 5 terms is 10 -/
  sum5_eq_10 : sum5 = 10
  /-- The sum of the first 10 terms is 50 -/
  sum10_eq_50 : sum10 = 50

/-- Theorem: The sum of the first 15 terms is 120 -/
theorem sum_15_terms (seq : ArithmeticSequence) : ∃ (sum15 : ℝ), sum15 = 120 := by
  sorry

end sum_15_terms_l1629_162966


namespace inequality_solution_set_l1629_162914

theorem inequality_solution_set (x : ℝ) :
  (∀ x, -x^2 - 3*x + 4 > 0 ↔ -4 < x ∧ x < 1) :=
by sorry

end inequality_solution_set_l1629_162914


namespace prob_select_boy_is_correct_l1629_162932

/-- Represents the number of boys in the calligraphy group -/
def calligraphy_boys : ℕ := 6

/-- Represents the number of girls in the calligraphy group -/
def calligraphy_girls : ℕ := 4

/-- Represents the number of boys in the original art group -/
def art_boys : ℕ := 5

/-- Represents the number of girls in the original art group -/
def art_girls : ℕ := 5

/-- Represents the number of people selected from the calligraphy group -/
def selected_from_calligraphy : ℕ := 2

/-- Calculates the probability of selecting a boy from the new art group -/
def prob_select_boy : ℚ := 31/60

theorem prob_select_boy_is_correct :
  prob_select_boy = 31/60 := by sorry

end prob_select_boy_is_correct_l1629_162932


namespace theta_range_l1629_162947

theorem theta_range (θ : Real) (h1 : θ ∈ Set.Icc 0 (2 * Real.pi)) 
  (h2 : Real.cos θ ^ 5 - Real.sin θ ^ 5 < 7 * (Real.sin θ ^ 3 - Real.cos θ ^ 3)) :
  θ ∈ Set.Ioo (Real.pi / 4) (5 * Real.pi / 4) := by
  sorry

end theta_range_l1629_162947


namespace square_sum_equals_six_l1629_162981

theorem square_sum_equals_six (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -1) : 
  x^2 + y^2 = 6 := by
sorry

end square_sum_equals_six_l1629_162981


namespace right_triangle_identification_l1629_162990

def is_right_triangle (a b c : ℕ) : Prop := a * a + b * b = c * c

theorem right_triangle_identification :
  (¬ is_right_triangle 2 3 4) ∧
  (is_right_triangle 3 4 5) ∧
  (¬ is_right_triangle 4 5 6) ∧
  (¬ is_right_triangle 5 6 7) :=
sorry

end right_triangle_identification_l1629_162990


namespace find_B_l1629_162928

-- Define the polynomial g(x)
def g (A B C D x : ℝ) : ℝ := A * x^3 + B * x^2 + C * x + D

-- State the theorem
theorem find_B :
  ∀ (A B C D : ℝ),
  (∀ x : ℝ, g A B C D x = 0 ↔ x = -2 ∨ x = 1 ∨ x = 2) →
  g A B C D 0 = -8 →
  B = -2 :=
by
  sorry

end find_B_l1629_162928


namespace import_value_calculation_l1629_162955

theorem import_value_calculation (export_value import_value : ℝ) : 
  export_value = 8.07 ∧ 
  export_value = 1.5 * import_value + 1.11 → 
sorry

end import_value_calculation_l1629_162955


namespace scientific_notation_110000_l1629_162956

theorem scientific_notation_110000 : 
  110000 = 1.1 * (10 : ℝ) ^ 5 := by sorry

end scientific_notation_110000_l1629_162956


namespace probability_is_three_fifths_l1629_162967

/-- The number of red balls in the box -/
def num_red : ℕ := 2

/-- The number of black balls in the box -/
def num_black : ℕ := 3

/-- The total number of balls in the box -/
def total_balls : ℕ := num_red + num_black

/-- The number of ways to choose 2 balls from the box -/
def total_combinations : ℕ := (total_balls * (total_balls - 1)) / 2

/-- The number of ways to choose 1 red ball and 1 black ball -/
def different_color_combinations : ℕ := num_red * num_black

/-- The probability of drawing two balls with different colors -/
def probability_different_colors : ℚ := different_color_combinations / total_combinations

theorem probability_is_three_fifths :
  probability_different_colors = 3 / 5 := by
  sorry

end probability_is_three_fifths_l1629_162967


namespace sqrt_81_div_3_l1629_162970

theorem sqrt_81_div_3 : Real.sqrt 81 / 3 = 3 := by
  sorry

end sqrt_81_div_3_l1629_162970


namespace no_nontrivial_solutions_l1629_162905

theorem no_nontrivial_solutions (x y z t : ℤ) :
  x^2 = 2*y^2 ∧ x^4 + 3*y^4 + 27*z^4 = 9*t^4 → x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0 := by
  sorry

end no_nontrivial_solutions_l1629_162905


namespace good_sets_exist_l1629_162933

-- Define a "good" subset of natural numbers
def is_good (A : Set ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (∃! p : ℕ, Prime p ∧ p ∣ n ∧ (n - p) ∈ A)

-- Define the set of perfect squares
def perfect_squares : Set ℕ := {n : ℕ | ∃ k : ℕ, n = k^2}

-- Define the set of prime numbers
def prime_set : Set ℕ := {p : ℕ | Prime p}

theorem good_sets_exist :
  (is_good perfect_squares) ∧ 
  (is_good prime_set) ∧ 
  (Set.Infinite prime_set) ∧ 
  (perfect_squares ∩ prime_set = ∅) := by
  sorry

end good_sets_exist_l1629_162933


namespace remainder_theorem_l1629_162976

theorem remainder_theorem (n : ℤ) (h : ∃ k : ℤ, n = 100 * k - 1) :
  (n^2 + 2*n + 3 + n^3) % 100 = 1 := by
sorry

end remainder_theorem_l1629_162976


namespace square_root_squared_l1629_162978

theorem square_root_squared (x : ℝ) : (Real.sqrt x)^2 = 49 → x = 49 := by
  sorry

end square_root_squared_l1629_162978


namespace cubic_polynomial_evaluation_l1629_162908

theorem cubic_polynomial_evaluation : 7^3 - 3 * 7^2 + 3 * 7 - 1 = 216 := by
  sorry

end cubic_polynomial_evaluation_l1629_162908


namespace equal_tabletops_and_legs_l1629_162918

/-- Represents the amount of wood used for tabletops -/
def wood_for_tabletops : ℝ := 3

/-- Represents the amount of wood used for legs -/
def wood_for_legs : ℝ := 5 - wood_for_tabletops

/-- Represents the number of tabletops that can be made from 1 cubic meter of wood -/
def tabletops_per_cubic_meter : ℝ := 50

/-- Represents the number of legs that can be made from 1 cubic meter of wood -/
def legs_per_cubic_meter : ℝ := 300

/-- Represents the number of legs per table -/
def legs_per_table : ℝ := 4

theorem equal_tabletops_and_legs :
  wood_for_tabletops * tabletops_per_cubic_meter = 
  wood_for_legs * legs_per_cubic_meter / legs_per_table := by
  sorry

end equal_tabletops_and_legs_l1629_162918


namespace race_finish_orders_l1629_162943

/-- The number of possible finish orders for a race with 4 participants and no ties -/
def finish_orders : ℕ := 24

/-- The number of participants in the race -/
def num_participants : ℕ := 4

/-- Theorem: The number of possible finish orders for a race with 4 participants and no ties is 24 -/
theorem race_finish_orders : 
  finish_orders = Nat.factorial num_participants :=
sorry

end race_finish_orders_l1629_162943


namespace sum_even_integers_l1629_162959

theorem sum_even_integers (x y : ℕ) : 
  (x = (40 + 60) * ((60 - 40) / 2 + 1) / 2) →  -- Sum formula for arithmetic sequence
  (y = (60 - 40) / 2 + 1) →                    -- Number of terms in arithmetic sequence
  (x + y = 561) → 
  (x = 550) := by
sorry

end sum_even_integers_l1629_162959


namespace books_to_decorations_ratio_l1629_162950

theorem books_to_decorations_ratio 
  (total_books : ℕ) 
  (books_per_shelf : ℕ) 
  (decorations_per_shelf : ℕ) 
  (initial_shelves : ℕ) 
  (h1 : total_books = 42)
  (h2 : books_per_shelf = 2)
  (h3 : decorations_per_shelf = 1)
  (h4 : initial_shelves = 3) :
  (total_books : ℚ) / ((total_books / (books_per_shelf * initial_shelves)) * decorations_per_shelf) = 6 / 1 := by
sorry

end books_to_decorations_ratio_l1629_162950


namespace complex_power_4_l1629_162920

theorem complex_power_4 : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
  Complex.ofReal (-40.5) + Complex.I * Complex.ofReal (40.5 * Real.sqrt 3) := by
sorry

end complex_power_4_l1629_162920


namespace three_digit_rotations_divisibility_l1629_162991

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  hundreds : Nat
  tens : Nat
  ones : Nat
  h_hundreds : hundreds ≥ 1 ∧ hundreds ≤ 9
  h_tens : tens ≥ 0 ∧ tens ≤ 9
  h_ones : ones ≥ 0 ∧ ones ≤ 9

/-- Converts a ThreeDigitNumber to its numeric value -/
def ThreeDigitNumber.toNat (n : ThreeDigitNumber) : Nat :=
  100 * n.hundreds + 10 * n.tens + n.ones

/-- Rotates the digits of a ThreeDigitNumber once to the left -/
def ThreeDigitNumber.rotateLeft (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.tens
  tens := n.ones
  ones := n.hundreds
  h_hundreds := by sorry
  h_tens := by sorry
  h_ones := by sorry

/-- Rotates the digits of a ThreeDigitNumber twice to the left -/
def ThreeDigitNumber.rotateLeftTwice (n : ThreeDigitNumber) : ThreeDigitNumber where
  hundreds := n.ones
  tens := n.hundreds
  ones := n.tens
  h_hundreds := by sorry
  h_tens := by sorry
  h_ones := by sorry

theorem three_digit_rotations_divisibility (n : ThreeDigitNumber) :
  27 ∣ n.toNat → 27 ∣ (n.rotateLeft).toNat ∧ 27 ∣ (n.rotateLeftTwice).toNat := by
  sorry

end three_digit_rotations_divisibility_l1629_162991


namespace intersection_of_M_and_N_l1629_162984

-- Define the sets M and N
def M : Set ℝ := {x | -1 < x ∧ x < 1}
def N : Set ℝ := {x | x / (x - 1) ≤ 0}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {x : ℝ | 0 ≤ x ∧ x < 1} := by sorry

end intersection_of_M_and_N_l1629_162984


namespace right_triangular_pyramid_property_l1629_162900

/-- A right-angled triangular pyramid -/
structure RightTriangularPyramid where
  /-- Area of the first right-angle face -/
  S₁ : ℝ
  /-- Area of the second right-angle face -/
  S₂ : ℝ
  /-- Area of the third right-angle face -/
  S₃ : ℝ
  /-- Area of the oblique face -/
  S : ℝ
  /-- All areas are positive -/
  S₁_pos : S₁ > 0
  S₂_pos : S₂ > 0
  S₃_pos : S₃ > 0
  S_pos : S > 0
  /-- Lateral edges are perpendicular to each other -/
  lateral_edges_perpendicular : True

/-- The property of a right-angled triangular pyramid -/
theorem right_triangular_pyramid_property (p : RightTriangularPyramid) :
  p.S₁^2 + p.S₂^2 + p.S₃^2 = p.S^2 := by
  sorry

end right_triangular_pyramid_property_l1629_162900


namespace expression_simplification_l1629_162973

theorem expression_simplification (y : ℝ) :
  y * (4 * y^2 - 3) - 6 * (y^2 - 3 * y + 8) = 4 * y^3 - 6 * y^2 + 15 * y - 48 := by
  sorry

end expression_simplification_l1629_162973


namespace first_car_speed_l1629_162961

/-- Proves that the speed of the first car is 54 miles per hour given the conditions of the problem -/
theorem first_car_speed (total_distance : ℝ) (second_car_speed : ℝ) (time_difference : ℝ) (total_time : ℝ)
  (h1 : total_distance = 80)
  (h2 : second_car_speed = 60)
  (h3 : time_difference = 1/6)
  (h4 : total_time = 1.5) :
  ∃ (first_car_speed : ℝ), first_car_speed = 54 ∧
    second_car_speed * total_time = first_car_speed * (total_time + time_difference) := by
  sorry


end first_car_speed_l1629_162961


namespace water_bottle_cost_l1629_162942

theorem water_bottle_cost (cola_price : ℝ) (juice_price : ℝ) (water_price : ℝ)
  (cola_sold : ℕ) (juice_sold : ℕ) (water_sold : ℕ) (total_revenue : ℝ)
  (h1 : cola_price = 3)
  (h2 : juice_price = 1.5)
  (h3 : cola_sold = 15)
  (h4 : juice_sold = 12)
  (h5 : water_sold = 25)
  (h6 : total_revenue = 88)
  (h7 : total_revenue = cola_price * cola_sold + juice_price * juice_sold + water_price * water_sold) :
  water_price = 1 := by
sorry

end water_bottle_cost_l1629_162942


namespace sin_square_equation_solution_l1629_162988

theorem sin_square_equation_solution (x : ℝ) :
  (Real.sin (3 * x))^2 + (Real.sin (4 * x))^2 = (Real.sin (5 * x))^2 + (Real.sin (6 * x))^2 →
  (∃ l : ℤ, x = l * π / 2) ∨ (∃ n : ℤ, x = n * π / 9) :=
by sorry

end sin_square_equation_solution_l1629_162988


namespace new_figure_length_is_32_l1629_162924

/-- Represents the dimensions of the original polygon --/
structure PolygonDimensions where
  vertical_side : ℝ
  top_first_horizontal : ℝ
  top_second_horizontal : ℝ
  remaining_horizontal : ℝ
  last_vertical_drop : ℝ

/-- Calculates the total length of segments in the new figure after removing four sides --/
def newFigureLength (d : PolygonDimensions) : ℝ :=
  d.vertical_side + (d.top_first_horizontal + d.top_second_horizontal + d.remaining_horizontal) +
  (d.vertical_side - d.last_vertical_drop) + d.last_vertical_drop

/-- Theorem stating that for the given dimensions, the new figure length is 32 units --/
theorem new_figure_length_is_32 (d : PolygonDimensions)
    (h1 : d.vertical_side = 10)
    (h2 : d.top_first_horizontal = 3)
    (h3 : d.top_second_horizontal = 4)
    (h4 : d.remaining_horizontal = 5)
    (h5 : d.last_vertical_drop = 2) :
    newFigureLength d = 32 := by
  sorry

end new_figure_length_is_32_l1629_162924


namespace complex_coordinate_l1629_162922

theorem complex_coordinate (i : ℂ) (z : ℂ) : 
  i * i = -1 → z = (2 + 4*i) / i → (z.re = 4 ∧ z.im = -2) :=
by sorry

end complex_coordinate_l1629_162922


namespace opposite_of_negative_fraction_l1629_162974

theorem opposite_of_negative_fraction (n : ℕ) (hn : n ≠ 0) :
  (-(1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

#check opposite_of_negative_fraction

end opposite_of_negative_fraction_l1629_162974


namespace obtuse_angle_equation_l1629_162901

theorem obtuse_angle_equation (α : Real) : 
  α > π / 2 ∧ α < π →
  Real.sin α * (1 + Real.sqrt 3 * Real.tan (10 * π / 180)) = 1 →
  α = 140 * π / 180 := by
sorry

end obtuse_angle_equation_l1629_162901
