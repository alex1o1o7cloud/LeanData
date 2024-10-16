import Mathlib

namespace NUMINAMATH_CALUDE_lisas_teaspoons_l1529_152910

theorem lisas_teaspoons (num_children : ℕ) (baby_spoons_per_child : ℕ) (decorative_spoons : ℕ) 
  (large_spoons : ℕ) (total_spoons : ℕ) :
  num_children = 4 →
  baby_spoons_per_child = 3 →
  decorative_spoons = 2 →
  large_spoons = 10 →
  total_spoons = 39 →
  total_spoons - (num_children * baby_spoons_per_child + decorative_spoons + large_spoons) = 15 := by
  sorry

end NUMINAMATH_CALUDE_lisas_teaspoons_l1529_152910


namespace NUMINAMATH_CALUDE_jamie_remaining_capacity_l1529_152931

/-- The maximum amount of liquid Jamie can consume before needing the bathroom -/
def max_liquid : ℕ := 32

/-- The amount of liquid Jamie has already consumed -/
def consumed_liquid : ℕ := 24

/-- The amount of additional liquid Jamie can consume -/
def remaining_capacity : ℕ := max_liquid - consumed_liquid

theorem jamie_remaining_capacity :
  remaining_capacity = 8 := by
  sorry

end NUMINAMATH_CALUDE_jamie_remaining_capacity_l1529_152931


namespace NUMINAMATH_CALUDE_sugar_percentage_in_kola_solution_l1529_152942

/-- Calculates the percentage of sugar in a kola solution after adding ingredients -/
theorem sugar_percentage_in_kola_solution
  (initial_volume : ℝ)
  (initial_water_percent : ℝ)
  (initial_kola_percent : ℝ)
  (added_sugar : ℝ)
  (added_water : ℝ)
  (added_kola : ℝ)
  (h1 : initial_volume = 340)
  (h2 : initial_water_percent = 88)
  (h3 : initial_kola_percent = 5)
  (h4 : added_sugar = 3.2)
  (h5 : added_water = 10)
  (h6 : added_kola = 6.8) :
  let initial_sugar_percent := 100 - initial_water_percent - initial_kola_percent
  let initial_sugar_volume := initial_sugar_percent / 100 * initial_volume
  let final_sugar_volume := initial_sugar_volume + added_sugar
  let final_volume := initial_volume + added_sugar + added_water + added_kola
  let final_sugar_percent := final_sugar_volume / final_volume * 100
  final_sugar_percent = 7.5 := by
sorry

end NUMINAMATH_CALUDE_sugar_percentage_in_kola_solution_l1529_152942


namespace NUMINAMATH_CALUDE_cubic_inequality_l1529_152950

theorem cubic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^3 * b + b^3 * c + c^3 * a - a^2 * b * c - b^2 * c * a - c^2 * a * b ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_inequality_l1529_152950


namespace NUMINAMATH_CALUDE_line_intersects_circle_l1529_152909

/-- 
Given a > 0, prove that the line x + a²y - a = 0 intersects 
the circle (x - a)² + (y - 1/a)² = 1
-/
theorem line_intersects_circle (a : ℝ) (h : a > 0) : 
  ∃ (x y : ℝ), (x + a^2 * y - a = 0) ∧ ((x - a)^2 + (y - 1/a)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l1529_152909


namespace NUMINAMATH_CALUDE_negative_comparison_l1529_152912

theorem negative_comparison : -2023 > -2024 := by
  sorry

end NUMINAMATH_CALUDE_negative_comparison_l1529_152912


namespace NUMINAMATH_CALUDE_alvin_marbles_lost_l1529_152970

/-- Proves that Alvin lost 18 marbles in the first game -/
theorem alvin_marbles_lost (initial_marbles : ℕ) (won_marbles : ℕ) (final_marbles : ℕ) 
  (h1 : initial_marbles = 57)
  (h2 : won_marbles = 25)
  (h3 : final_marbles = 64) :
  initial_marbles - (final_marbles - won_marbles) = 18 := by
  sorry

#check alvin_marbles_lost

end NUMINAMATH_CALUDE_alvin_marbles_lost_l1529_152970


namespace NUMINAMATH_CALUDE_number_puzzle_l1529_152981

theorem number_puzzle : ∃ x : ℝ, x = 280 ∧ (x / 5 + 4 = x / 4 - 10) := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l1529_152981


namespace NUMINAMATH_CALUDE_range_of_y_minus_x_l1529_152929

-- Define the triangle ABC and points D and P
variable (A B C D P : ℝ × ℝ)

-- Define vectors
def vec (X Y : ℝ × ℝ) : ℝ × ℝ := (Y.1 - X.1, Y.2 - X.2)

-- Conditions
variable (h1 : vec D C = (2 * (vec A D).1, 2 * (vec A D).2))
variable (h2 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (t * B.1 + (1 - t) * D.1, t * B.2 + (1 - t) * D.2))
variable (h3 : ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ vec A P = (x * (vec A B).1 + y * (vec A C).1, x * (vec A B).2 + y * (vec A C).2))

-- Theorem statement
theorem range_of_y_minus_x :
  ∃ S : Set ℝ, S = {z | ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 
    vec A P = (x * (vec A B).1 + y * (vec A C).1, x * (vec A B).2 + y * (vec A C).2) ∧
    z = y - x} ∧
  S = {z | -1 < z ∧ z < 1/3} :=
sorry

end NUMINAMATH_CALUDE_range_of_y_minus_x_l1529_152929


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l1529_152908

theorem sum_of_x_and_y_is_two (x y : ℝ) (h : x^2 + y^2 = 12*x - 8*y - 56) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_is_two_l1529_152908


namespace NUMINAMATH_CALUDE_min_distance_circle_to_line_l1529_152916

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + (y - 3)^2 = 2

-- Define point M
def point_M : ℝ × ℝ := (-2, 1)

-- Define trajectory E
def trajectory_E (x y : ℝ) : Prop := 4*x + 2*y - 3 = 0

-- Theorem statement
theorem min_distance_circle_to_line :
  ∃ (min_dist : ℝ),
    min_dist = (11 * Real.sqrt 5) / 10 - Real.sqrt 2 ∧
    ∀ (a b : ℝ × ℝ),
      circle_C a.1 a.2 →
      trajectory_E b.1 b.2 →
      Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) ≥ min_dist :=
sorry

end NUMINAMATH_CALUDE_min_distance_circle_to_line_l1529_152916


namespace NUMINAMATH_CALUDE_ratio_of_multiples_l1529_152941

theorem ratio_of_multiples (w x y z : ℝ) 
  (hx : x = 4 * y) 
  (hy : y = 3 * z) 
  (hz : z = 5 * w) : 
  (x * z) / (y * w) = 20 := by
sorry

end NUMINAMATH_CALUDE_ratio_of_multiples_l1529_152941


namespace NUMINAMATH_CALUDE_smaller_two_digit_factor_l1529_152961

theorem smaller_two_digit_factor (a b : ℕ) : 
  10 ≤ a ∧ a < 100 ∧ 
  10 ≤ b ∧ b < 100 ∧ 
  a * b = 3774 → 
  min a b = 51 := by
sorry

end NUMINAMATH_CALUDE_smaller_two_digit_factor_l1529_152961


namespace NUMINAMATH_CALUDE_tan_theta_plus_pi_fourth_l1529_152985

theorem tan_theta_plus_pi_fourth (θ : Real) 
  (h : (Real.cos (2 * θ) + 1) / (1 + 2 * Real.sin (2 * θ)) = -2/3) : 
  Real.tan (θ + π/4) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_plus_pi_fourth_l1529_152985


namespace NUMINAMATH_CALUDE_function_value_negation_l1529_152926

/-- Given a function f(x) = a * sin(πx + α) + b * cos(πx + β) where f(2002) = 3,
    prove that f(2003) = -f(2002). -/
theorem function_value_negation (a b α β : ℝ) :
  let f : ℝ → ℝ := λ x => a * Real.sin (π * x + α) + b * Real.cos (π * x + β)
  f 2002 = 3 → f 2003 = -f 2002 := by
  sorry

end NUMINAMATH_CALUDE_function_value_negation_l1529_152926


namespace NUMINAMATH_CALUDE_max_n_A_theorem_l1529_152953

/-- A set of four distinct positive integers -/
structure FourSet where
  a₁ : ℕ+
  a₂ : ℕ+
  a₃ : ℕ+
  a₄ : ℕ+
  distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄

/-- Sum of elements in a FourSet -/
def S_A (A : FourSet) : ℕ+ :=
  A.a₁ + A.a₂ + A.a₃ + A.a₄

/-- Number of pairs (i, j) with 1 ≤ i < j ≤ 4 such that (aᵢ + a_j) divides S_A -/
def n_A (A : FourSet) : ℕ :=
  let pairs := [(A.a₁, A.a₂), (A.a₁, A.a₃), (A.a₁, A.a₄), (A.a₂, A.a₃), (A.a₂, A.a₄), (A.a₃, A.a₄)]
  (pairs.filter (fun (x, y) => (S_A A).val % (x + y).val = 0)).length

/-- Theorem stating the maximum value of n_A and the form of A when this maximum is achieved -/
theorem max_n_A_theorem (A : FourSet) :
  n_A A ≤ 4 ∧
  (n_A A = 4 →
    (∃ c : ℕ+, A.a₁ = c ∧ A.a₂ = 5 * c ∧ A.a₃ = 7 * c ∧ A.a₄ = 11 * c) ∨
    (∃ c : ℕ+, A.a₁ = c ∧ A.a₂ = 11 * c ∧ A.a₃ = 19 * c ∧ A.a₄ = 29 * c)) := by
  sorry

end NUMINAMATH_CALUDE_max_n_A_theorem_l1529_152953


namespace NUMINAMATH_CALUDE_complex_fraction_equals_25_l1529_152968

theorem complex_fraction_equals_25 :
  ((5 / 2) / (1 / 2) * (5 / 2)) / ((5 / 2) * (1 / 2) / (5 / 2)) = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equals_25_l1529_152968


namespace NUMINAMATH_CALUDE_complex_sum_power_l1529_152955

theorem complex_sum_power (i : ℂ) : i * i = -1 → (1 - i)^2016 + (1 + i)^2016 = 2^1009 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_power_l1529_152955


namespace NUMINAMATH_CALUDE_expression_factorization_l1529_152993

theorem expression_factorization (y : ℝ) :
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 9 * y^4 + 9) = 3 * (4 * y^6 + 15 * y^4 - 6) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l1529_152993


namespace NUMINAMATH_CALUDE_f_properties_triangle_property_l1529_152933

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2 - 1

theorem f_properties :
  ∃ (max_value : ℝ) (max_set : Set ℝ),
    (∀ x, f x ≤ max_value) ∧
    (∀ x, x ∈ max_set ↔ f x = max_value) ∧
    max_value = 2 ∧
    max_set = {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6} := by sorry

theorem triangle_property :
  ∀ (A B C : ℝ) (a b c : ℝ),
    a = 1 → b = Real.sqrt 3 → f A = 2 →
    (A + B + C = Real.pi) →
    (Real.sin A / a = Real.sin B / b) →
    (Real.sin B / b = Real.sin C / c) →
    (C = Real.pi / 6 ∨ C = Real.pi / 2) := by sorry

end NUMINAMATH_CALUDE_f_properties_triangle_property_l1529_152933


namespace NUMINAMATH_CALUDE_cost_per_candy_bar_l1529_152903

-- Define the given conditions
def boxes_sold : ℕ := 5
def candy_bars_per_box : ℕ := 10
def selling_price_per_bar : ℚ := 3/2  -- $1.50 as a rational number
def total_profit : ℚ := 25

-- Define the theorem
theorem cost_per_candy_bar :
  let total_bars := boxes_sold * candy_bars_per_box
  let total_revenue := total_bars * selling_price_per_bar
  let total_cost := total_revenue - total_profit
  total_cost / total_bars = 1 := by sorry

end NUMINAMATH_CALUDE_cost_per_candy_bar_l1529_152903


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l1529_152900

/-- Two points are symmetric with respect to the origin if their coordinates are negations of each other -/
def symmetric_wrt_origin (A B : ℝ × ℝ) : Prop :=
  A.1 = -B.1 ∧ A.2 = -B.2

theorem symmetric_points_difference (a b : ℝ) :
  let A : ℝ × ℝ := (-2, b)
  let B : ℝ × ℝ := (a, 3)
  symmetric_wrt_origin A B → a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l1529_152900


namespace NUMINAMATH_CALUDE_segment_length_l1529_152972

/-- Given a line segment CD with points R and S on it, prove that CD has length 273.6 -/
theorem segment_length (C D R S : ℝ) : 
  (R > (C + D) / 2) →  -- R is on the same side of the midpoint as S
  (S > (C + D) / 2) →  -- S is on the same side of the midpoint as R
  (R - C) / (D - R) = 3 / 5 →  -- R divides CD in ratio 3:5
  (S - C) / (D - S) = 4 / 7 →  -- S divides CD in ratio 4:7
  S - R = 3 →  -- RS = 3
  D - C = 273.6 :=  -- CD = 273.6
by sorry

end NUMINAMATH_CALUDE_segment_length_l1529_152972


namespace NUMINAMATH_CALUDE_evening_campers_count_l1529_152952

def morning_campers : ℕ := 36
def afternoon_campers : ℕ := 13
def total_campers : ℕ := 98

theorem evening_campers_count : 
  total_campers - (morning_campers + afternoon_campers) = 49 := by
  sorry

end NUMINAMATH_CALUDE_evening_campers_count_l1529_152952


namespace NUMINAMATH_CALUDE_church_attendance_l1529_152978

/-- Proves the number of female adults in a church given the number of children, male adults, and total people. -/
theorem church_attendance (children : ℕ) (male_adults : ℕ) (total_people : ℕ) 
  (h1 : children = 80)
  (h2 : male_adults = 60)
  (h3 : total_people = 200) :
  total_people - (children + male_adults) = 60 := by
  sorry

#check church_attendance

end NUMINAMATH_CALUDE_church_attendance_l1529_152978


namespace NUMINAMATH_CALUDE_intersection_complement_equals_set_l1529_152958

-- Define the sets M and N
def M : Set ℝ := {x | ∃ y, y = Real.log (x^2 - 1) / Real.log 10}
def N : Set ℝ := {x | 0 < x ∧ x < 2}

-- State the theorem
theorem intersection_complement_equals_set :
  N ∩ (Mᶜ) = {x | 0 < x ∧ x ≤ 1} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equals_set_l1529_152958


namespace NUMINAMATH_CALUDE_min_price_reduction_l1529_152984

theorem min_price_reduction (price_2004 : ℝ) (h1 : price_2004 > 0) : 
  let price_2005 := price_2004 * (1 - 0.15)
  let min_reduction := (price_2005 - price_2004 * 0.75) / price_2005 * 100
  ∀ ε > 0, ∃ δ > 0, 
    abs (min_reduction - 11.8) < δ ∧ 
    price_2004 * (1 - 0.15) * (1 - (min_reduction + ε) / 100) < price_2004 * 0.75 ∧
    price_2004 * (1 - 0.15) * (1 - (min_reduction - ε) / 100) > price_2004 * 0.75 :=
by sorry

end NUMINAMATH_CALUDE_min_price_reduction_l1529_152984


namespace NUMINAMATH_CALUDE_smallest_positive_shift_l1529_152913

-- Define a function f with period 20
def f : ℝ → ℝ := sorry

-- Define the periodicity property
axiom f_periodic : ∀ x : ℝ, f (x - 20) = f x

-- Define the property for the scaled and shifted function
def scaled_shifted_property (a : ℝ) : Prop :=
  ∀ x : ℝ, f ((x - a) / 4) = f (x / 4)

-- Theorem statement
theorem smallest_positive_shift :
  ∃ a : ℝ, a > 0 ∧ scaled_shifted_property a ∧
  ∀ b : ℝ, b > 0 ∧ scaled_shifted_property b → a ≤ b :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_shift_l1529_152913


namespace NUMINAMATH_CALUDE_senate_democrats_count_l1529_152901

/-- Given the conditions of the House of Representatives and Senate composition,
    prove that the number of Democrats in the Senate is 55. -/
theorem senate_democrats_count : 
  ∀ (house_total house_dem house_rep senate_total senate_dem senate_rep : ℕ),
  house_total = 434 →
  house_total = house_dem + house_rep →
  house_rep = house_dem + 30 →
  senate_total = 100 →
  senate_total = senate_dem + senate_rep →
  5 * senate_rep = 4 * senate_dem →
  senate_dem = 55 := by
  sorry

end NUMINAMATH_CALUDE_senate_democrats_count_l1529_152901


namespace NUMINAMATH_CALUDE_circle_radius_proof_l1529_152957

theorem circle_radius_proof (r₁ r₂ : ℝ) : 
  r₂ = 2 →                             -- The smaller circle has a radius of 2 cm
  (π * r₁^2) = 4 * (π * r₂^2) →        -- The area of one circle is four times the area of the other
  r₁ = 4 :=                            -- The radius of the larger circle is 4 cm
by sorry

end NUMINAMATH_CALUDE_circle_radius_proof_l1529_152957


namespace NUMINAMATH_CALUDE_tetrahedron_circumsphere_area_l1529_152960

/-- The surface area of a circumscribed sphere of a regular tetrahedron with side length 2 -/
theorem tetrahedron_circumsphere_area : 
  let side_length : ℝ := 2
  let circumradius : ℝ := side_length * Real.sqrt 3 / 3
  let sphere_area : ℝ := 4 * Real.pi * circumradius^2
  sphere_area = 16 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_circumsphere_area_l1529_152960


namespace NUMINAMATH_CALUDE_gcd_of_three_numbers_l1529_152983

theorem gcd_of_three_numbers : Nat.gcd 279 (Nat.gcd 372 465) = 93 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_three_numbers_l1529_152983


namespace NUMINAMATH_CALUDE_organizationalStructureIsCorrect_l1529_152917

-- Define the types of diagrams
inductive Diagram
  | Flowchart
  | ProcessFlow
  | KnowledgeStructure
  | OrganizationalStructure

-- Define the properties a diagram should have
structure DiagramProperties where
  reflectsRelationships : Bool
  showsVerticalHorizontal : Bool
  reflectsOrganizationalStructure : Bool
  interpretsOrganizationalFunctions : Bool

-- Define a function to check if a diagram has the required properties
def hasRequiredProperties (d : Diagram) : DiagramProperties :=
  match d with
  | Diagram.OrganizationalStructure => {
      reflectsRelationships := true,
      showsVerticalHorizontal := true,
      reflectsOrganizationalStructure := true,
      interpretsOrganizationalFunctions := true
    }
  | _ => {
      reflectsRelationships := false,
      showsVerticalHorizontal := false,
      reflectsOrganizationalStructure := false,
      interpretsOrganizationalFunctions := false
    }

-- Theorem: The Organizational Structure Diagram is the correct choice for describing factory composition
theorem organizationalStructureIsCorrect :
  ∀ (d : Diagram),
    (hasRequiredProperties d).reflectsRelationships ∧
    (hasRequiredProperties d).showsVerticalHorizontal ∧
    (hasRequiredProperties d).reflectsOrganizationalStructure ∧
    (hasRequiredProperties d).interpretsOrganizationalFunctions
    →
    d = Diagram.OrganizationalStructure :=
  sorry

end NUMINAMATH_CALUDE_organizationalStructureIsCorrect_l1529_152917


namespace NUMINAMATH_CALUDE_bill_donuts_l1529_152998

theorem bill_donuts (total : ℕ) (secretary_takes : ℕ) (final : ℕ) : 
  total = 50 →
  secretary_takes = 4 →
  final = 22 →
  final * 2 = total - secretary_takes - (total - secretary_takes - final * 2) :=
by sorry

end NUMINAMATH_CALUDE_bill_donuts_l1529_152998


namespace NUMINAMATH_CALUDE_cloth_sale_loss_per_metre_l1529_152956

/-- Calculates the loss per metre for a cloth sale -/
theorem cloth_sale_loss_per_metre
  (total_metres : ℕ)
  (total_selling_price : ℕ)
  (cost_price_per_metre : ℕ)
  (h1 : total_metres = 600)
  (h2 : total_selling_price = 18000)
  (h3 : cost_price_per_metre = 35) :
  (cost_price_per_metre * total_metres - total_selling_price) / total_metres = 5 := by
  sorry

#check cloth_sale_loss_per_metre

end NUMINAMATH_CALUDE_cloth_sale_loss_per_metre_l1529_152956


namespace NUMINAMATH_CALUDE_kelly_apples_l1529_152927

/-- Given Kelly's initial apples and the number of apples she needs to pick,
    calculate the total number of apples she will have. -/
def total_apples (initial : ℕ) (to_pick : ℕ) : ℕ :=
  initial + to_pick

/-- Theorem stating that Kelly will have 105 apples altogether -/
theorem kelly_apples :
  total_apples 56 49 = 105 := by
  sorry

end NUMINAMATH_CALUDE_kelly_apples_l1529_152927


namespace NUMINAMATH_CALUDE_max_product_constrained_max_product_achieved_l1529_152904

theorem max_product_constrained (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 20 → x * y ≤ 25 := by
  sorry

theorem max_product_achieved (x y : ℝ) : 
  x > 0 → y > 0 → x + 4 * y = 20 → x = 10 ∧ y = 2.5 → x * y = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_product_constrained_max_product_achieved_l1529_152904


namespace NUMINAMATH_CALUDE_first_player_always_wins_l1529_152994

/-- Represents a point on a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a color of a dot --/
inductive Color
  | Red
  | Blue

/-- Represents a dot on the plane --/
structure Dot where
  point : Point
  color : Color

/-- Represents the game state --/
structure GameState where
  dots : List Dot

/-- Represents a player's strategy --/
def Strategy := GameState → Point

/-- Checks if three points form an equilateral triangle --/
def isEquilateralTriangle (p1 p2 p3 : Point) : Prop := sorry

/-- The main theorem stating that the first player can always win --/
theorem first_player_always_wins :
  ∀ (second_player_strategy : Strategy),
  ∃ (first_player_strategy : Strategy) (n : ℕ),
  ∀ (game : GameState),
  ∃ (p1 p2 p3 : Point),
  (p1 ∈ game.dots.map Dot.point) ∧
  (p2 ∈ game.dots.map Dot.point) ∧
  (p3 ∈ game.dots.map Dot.point) ∧
  isEquilateralTriangle p1 p2 p3 :=
sorry

end NUMINAMATH_CALUDE_first_player_always_wins_l1529_152994


namespace NUMINAMATH_CALUDE_inequality_condition_l1529_152905

theorem inequality_condition (a b c : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ Real.sqrt (a^2 + b^2) < c :=
sorry

end NUMINAMATH_CALUDE_inequality_condition_l1529_152905


namespace NUMINAMATH_CALUDE_largest_number_problem_l1529_152940

theorem largest_number_problem (a b c : ℝ) 
  (h_order : a < b ∧ b < c)
  (h_sum : a + b + c = 67)
  (h_diff_large : c - b = 7)
  (h_diff_small : b - a = 5) :
  c = 86 / 3 := by
sorry

end NUMINAMATH_CALUDE_largest_number_problem_l1529_152940


namespace NUMINAMATH_CALUDE_function_roots_l1529_152935

def has_at_least_roots (f : ℝ → ℝ) (n : ℕ) (a b : ℝ) : Prop :=
  ∃ (S : Finset ℝ), S.card ≥ n ∧ (∀ x ∈ S, a ≤ x ∧ x ≤ b ∧ f x = 0)

theorem function_roots (g : ℝ → ℝ) 
  (h1 : ∀ x, g (3 + x) = g (3 - x))
  (h2 : ∀ x, g (8 + x) = g (8 - x))
  (h3 : g 0 = 0) :
  has_at_least_roots g 501 (-2000) 2000 := by
  sorry

end NUMINAMATH_CALUDE_function_roots_l1529_152935


namespace NUMINAMATH_CALUDE_compare_negative_numbers_l1529_152976

theorem compare_negative_numbers : -4 < -2.1 := by
  sorry

end NUMINAMATH_CALUDE_compare_negative_numbers_l1529_152976


namespace NUMINAMATH_CALUDE_line_problem_l1529_152991

-- Define the lines and point
def l1 (m : ℝ) : ℝ → ℝ → Prop := λ x y => (m - 2) * x + m * y - 8 = 0
def l2 (m : ℝ) : ℝ → ℝ → Prop := λ x y => m * x + y - 3 = 0
def P (m : ℝ) : ℝ × ℝ := (1, 2 * m)

-- Define perpendicularity of lines
def perpendicular (f g : ℝ → ℝ → Prop) : Prop := sorry

-- Define a line passing through a point
def passes_through (f : ℝ → ℝ → Prop) (p : ℝ × ℝ) : Prop := sorry

-- Define the sum of intercepts
def sum_of_intercepts (f : ℝ → ℝ → Prop) : ℝ := sorry

-- Theorem statement
theorem line_problem (m : ℝ) :
  (perpendicular (l1 m) (l2 m) → m = 1 ∨ m = 0) ∧
  (l2 m (P m).1 (P m).2 →
    ∃ l : ℝ → ℝ → Prop,
      passes_through l (P m) ∧
      sum_of_intercepts l = 0 →
      ∀ x y, l x y ↔ x - y + 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_line_problem_l1529_152991


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1529_152921

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  (2 * x - 1 > 5) ∧ (-x < -6)

-- Define the solution set
def solution_set : Set ℝ :=
  {x | x > 6}

-- Theorem stating that the solution set is correct
theorem inequality_system_solution :
  ∀ x : ℝ, inequality_system x ↔ x ∈ solution_set :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1529_152921


namespace NUMINAMATH_CALUDE_yellow_marbles_count_l1529_152988

theorem yellow_marbles_count (total : ℕ) (red : ℕ) (blue : ℕ) (yellow : ℕ) : 
  total = 85 →
  blue = 3 * red →
  red = 14 →
  total = red + blue + yellow →
  yellow = 29 := by
sorry

end NUMINAMATH_CALUDE_yellow_marbles_count_l1529_152988


namespace NUMINAMATH_CALUDE_sqrt_of_repeating_ones_100_l1529_152973

theorem sqrt_of_repeating_ones_100 :
  let x := (10^100 - 1) / (9 * 10^100)
  0.10049987498 < Real.sqrt x ∧ Real.sqrt x < 0.10049987499 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_of_repeating_ones_100_l1529_152973


namespace NUMINAMATH_CALUDE_complex_sum_l1529_152930

def alphabet_value (n : ℕ) : ℤ :=
  match n % 26 with
  | 0  => 3
  | 1  => 2
  | 2  => 3
  | 3  => 2
  | 4  => 1
  | 5  => 0
  | 6  => -1
  | 7  => -2
  | 8  => -3
  | 9  => -2
  | 10 => -1
  | 11 => 0
  | 12 => 1
  | 13 => 2
  | 14 => 3
  | 15 => 2
  | 16 => 1
  | 17 => 0
  | 18 => -1
  | 19 => -2
  | 20 => -3
  | 21 => -2
  | 22 => -1
  | 23 => 0
  | 24 => 1
  | 25 => 2
  | _  => 3

theorem complex_sum : 
  alphabet_value 3 + alphabet_value 15 + alphabet_value 13 + 
  alphabet_value 16 + alphabet_value 12 + alphabet_value 5 + 
  alphabet_value 24 = 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_l1529_152930


namespace NUMINAMATH_CALUDE_upper_pyramid_volume_l1529_152922

/-- The volume of the upper smaller pyramid formed by cutting a right square pyramid -/
theorem upper_pyramid_volume (base_edge slant_edge cut_height : ℝ) 
  (h_base : base_edge = 12 * Real.sqrt 2)
  (h_slant : slant_edge = 15)
  (h_cut : cut_height = 5) : 
  ∃ (volume : ℝ), volume = (1/6) * ((12 * Real.sqrt 2 * (Real.sqrt 153 - 5)) / Real.sqrt 153)^2 * (Real.sqrt 153 - 5) := by
  sorry

end NUMINAMATH_CALUDE_upper_pyramid_volume_l1529_152922


namespace NUMINAMATH_CALUDE_board_cutting_theorem_l1529_152920

def is_valid_board_size (n : ℕ) : Prop :=
  ∃ m : ℕ, n * n = 5 * m ∧ n > 5

theorem board_cutting_theorem (n : ℕ) :
  (∃ m : ℕ, m > 0 ∧ n * n = m + 4 * m) ↔ is_valid_board_size n :=
sorry

end NUMINAMATH_CALUDE_board_cutting_theorem_l1529_152920


namespace NUMINAMATH_CALUDE_complex_inequality_nonexistence_l1529_152980

theorem complex_inequality_nonexistence : 
  ∀ (a b c : ℂ) (h : ℕ), a ≠ 0 → b ≠ 0 → c ≠ 0 → 
  ∃ (k l m : ℤ), (abs k + abs l + abs m ≥ 1996) ∧ 
  (Complex.abs (1 + k * a + l * b + m * c) ≤ 1 / h) := by
  sorry

end NUMINAMATH_CALUDE_complex_inequality_nonexistence_l1529_152980


namespace NUMINAMATH_CALUDE_work_days_calculation_l1529_152907

theorem work_days_calculation (days_a days_b : ℕ) (wage_c : ℕ) (total_earning : ℕ) :
  days_a = 6 →
  days_b = 9 →
  wage_c = 95 →
  total_earning = 1406 →
  ∃ (days_c : ℕ),
    (3 * wage_c * days_a + 4 * wage_c * days_b + 5 * wage_c * days_c = 5 * total_earning) ∧
    days_c = 4 :=
by sorry

end NUMINAMATH_CALUDE_work_days_calculation_l1529_152907


namespace NUMINAMATH_CALUDE_min_x_plus_y_l1529_152963

def is_median (x : ℝ) : Prop := 
  x ≥ 2 ∧ x ≤ 4

def average_condition (x y : ℝ) : Prop :=
  (-1 + 5 + (-1/x) + y) / 4 = 3

theorem min_x_plus_y (x y : ℝ) 
  (h1 : is_median x) 
  (h2 : average_condition x y) : 
  x + y ≥ 21/2 := by
  sorry

end NUMINAMATH_CALUDE_min_x_plus_y_l1529_152963


namespace NUMINAMATH_CALUDE_dot_product_zero_nonzero_vectors_l1529_152944

theorem dot_product_zero_nonzero_vectors :
  ∃ (a b : ℝ × ℝ), a ≠ (0, 0) ∧ b ≠ (0, 0) ∧ a.1 * b.1 + a.2 * b.2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_zero_nonzero_vectors_l1529_152944


namespace NUMINAMATH_CALUDE_proper_subset_of_A_l1529_152945

def A : Set ℝ := {x | x^2 < 5*x}

theorem proper_subset_of_A : Set.Subset (Set.Ioo 1 5) A ∧ (Set.Ioo 1 5) ≠ A := by
  sorry

end NUMINAMATH_CALUDE_proper_subset_of_A_l1529_152945


namespace NUMINAMATH_CALUDE_distance_AE_BF_is_19_2_l1529_152967

/-- A rectangular parallelepiped with given dimensions and midpoints -/
structure Parallelepiped where
  -- Edge lengths
  ab : ℝ
  ad : ℝ
  aa1 : ℝ
  -- Ensure it's a rectangular parallelepiped
  is_rectangular : True
  -- Ensure E is midpoint of A₁B₁
  e_is_midpoint_a1b1 : True
  -- Ensure F is midpoint of B₁C₁
  f_is_midpoint_b1c1 : True

/-- The distance between lines AE and BF in the parallelepiped -/
def distance_AE_BF (p : Parallelepiped) : ℝ := sorry

/-- Theorem: The distance between AE and BF is 19.2 -/
theorem distance_AE_BF_is_19_2 (p : Parallelepiped) 
  (h1 : p.ab = 30) (h2 : p.ad = 32) (h3 : p.aa1 = 20) : 
  distance_AE_BF p = 19.2 := by sorry

end NUMINAMATH_CALUDE_distance_AE_BF_is_19_2_l1529_152967


namespace NUMINAMATH_CALUDE_discount_savings_l1529_152989

theorem discount_savings (original_price discounted_price : ℝ) 
  (h1 : discounted_price = original_price * 0.8)
  (h2 : discounted_price = 48)
  (h3 : original_price > 0) :
  (original_price - discounted_price) / 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_discount_savings_l1529_152989


namespace NUMINAMATH_CALUDE_solve_installment_problem_l1529_152914

def installment_problem (cash_price : ℕ) (down_payment : ℕ) (first_four_payment : ℕ) (next_four_payment : ℕ) (total_months : ℕ) (installment_markup : ℕ) : Prop :=
  let total_installment_price := cash_price + installment_markup
  let paid_so_far := down_payment + 4 * first_four_payment + 4 * next_four_payment
  let remaining_amount := total_installment_price - paid_so_far
  let last_four_months := total_months - 8
  remaining_amount / last_four_months = 30

theorem solve_installment_problem :
  installment_problem 450 100 40 35 12 70 :=
sorry

end NUMINAMATH_CALUDE_solve_installment_problem_l1529_152914


namespace NUMINAMATH_CALUDE_product_of_two_numbers_l1529_152995

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x * y = 15 * (x - y)) 
  (h2 : x + y = 8 * (x - y)) : 
  x * y = 100 / 7 := by
sorry

end NUMINAMATH_CALUDE_product_of_two_numbers_l1529_152995


namespace NUMINAMATH_CALUDE_logarithm_equality_l1529_152924

-- Define the logarithm base 10 function
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the logarithm base 5 function
noncomputable def log5 (x : ℝ) : ℝ := Real.log x / Real.log 5

-- State the theorem
theorem logarithm_equality : lg 2 + lg 5 + 2 * log5 10 - log5 20 = 2 := by
  sorry

end NUMINAMATH_CALUDE_logarithm_equality_l1529_152924


namespace NUMINAMATH_CALUDE_rose_cost_is_six_l1529_152999

/-- The cost of each rose when buying in bulk -/
def rose_cost (dozen : ℕ) (discount_percent : ℚ) (total_paid : ℚ) : ℚ :=
  total_paid / (discount_percent / 100) / (dozen * 12)

/-- Theorem: The cost of each rose is $6 -/
theorem rose_cost_is_six :
  rose_cost 5 80 288 = 6 := by
  sorry

end NUMINAMATH_CALUDE_rose_cost_is_six_l1529_152999


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1529_152966

theorem min_value_sum_of_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_9 : a + b + c + d + e + f = 9) : 
  2/a + 4/b + 6/c + 16/d + 20/e + 30/f ≥ 2053.78 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l1529_152966


namespace NUMINAMATH_CALUDE_kimikos_age_l1529_152902

theorem kimikos_age (kimiko omi arlette : ℝ) 
  (h1 : omi = 2 * kimiko)
  (h2 : arlette = 3/4 * kimiko)
  (h3 : (kimiko + omi + arlette) / 3 = 35) :
  kimiko = 28 := by
  sorry

end NUMINAMATH_CALUDE_kimikos_age_l1529_152902


namespace NUMINAMATH_CALUDE_james_writes_to_fourteen_people_l1529_152936

/-- Represents James' writing habits and calculates the number of people he writes to daily --/
def james_writing (pages_per_hour : ℕ) (pages_per_person_per_day : ℕ) (hours_per_week : ℕ) : ℕ :=
  (pages_per_hour * hours_per_week) / pages_per_person_per_day

/-- Theorem stating that James writes to 14 people daily --/
theorem james_writes_to_fourteen_people :
  james_writing 10 5 7 = 14 := by
  sorry

end NUMINAMATH_CALUDE_james_writes_to_fourteen_people_l1529_152936


namespace NUMINAMATH_CALUDE_book_cost_price_l1529_152934

theorem book_cost_price (cost : ℝ) : cost = 300 :=
  by
  have h1 : 1.12 * cost + 18 = 1.18 * cost := by sorry
  sorry

end NUMINAMATH_CALUDE_book_cost_price_l1529_152934


namespace NUMINAMATH_CALUDE_teacher_selection_problem_l1529_152997

/-- The number of ways to select k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

theorem teacher_selection_problem (male female total selected : ℕ) 
  (h1 : male = 3)
  (h2 : female = 6)
  (h3 : total = male + female)
  (h4 : selected = 5) :
  choose total selected - choose female selected = 120 := by sorry

end NUMINAMATH_CALUDE_teacher_selection_problem_l1529_152997


namespace NUMINAMATH_CALUDE_store_buying_combinations_l1529_152979

/-- The number of students --/
def num_students : ℕ := 4

/-- The number of item choices for each student --/
def num_choices : ℕ := 2

/-- The total number of possible buying combinations --/
def total_combinations : ℕ := num_choices ^ num_students

/-- The number of valid buying combinations --/
def valid_combinations : ℕ := total_combinations - 1

theorem store_buying_combinations :
  valid_combinations = 15 := by sorry

end NUMINAMATH_CALUDE_store_buying_combinations_l1529_152979


namespace NUMINAMATH_CALUDE_inequality_range_l1529_152932

theorem inequality_range (m : ℝ) : 
  (∀ x : ℝ, (m^2 - 2*m - 3)*x^2 - (m - 3)*x - 1 < 0) ↔ 
  m > -1/5 ∧ m ≤ 3 := by sorry

end NUMINAMATH_CALUDE_inequality_range_l1529_152932


namespace NUMINAMATH_CALUDE_robins_pieces_l1529_152951

theorem robins_pieces (gum_packages : ℕ) (candy_packages : ℕ) (pieces_per_package : ℕ) : 
  gum_packages = 28 → candy_packages = 14 → pieces_per_package = 6 →
  gum_packages * pieces_per_package + candy_packages * pieces_per_package = 252 := by
sorry

end NUMINAMATH_CALUDE_robins_pieces_l1529_152951


namespace NUMINAMATH_CALUDE_orange_savings_percentage_l1529_152925

-- Define the given conditions
def family_size : ℕ := 4
def orange_cost : ℚ := 3/2  -- $1.5 as a rational number
def planned_spending : ℚ := 15

-- Define the theorem
theorem orange_savings_percentage :
  let saved_amount := family_size * orange_cost
  let savings_ratio := saved_amount / planned_spending
  savings_ratio * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_orange_savings_percentage_l1529_152925


namespace NUMINAMATH_CALUDE_value_of_S_l1529_152987

theorem value_of_S : ∀ S : ℕ, 
  S = 6 * 10000 + 5 * 1000 + 4 * 10 + 3 * 1 → S = 65043 := by
  sorry

end NUMINAMATH_CALUDE_value_of_S_l1529_152987


namespace NUMINAMATH_CALUDE_circle_radius_sqrt_29_l1529_152923

/-- Given a circle with center on the x-axis that passes through points (2,2) and (-1,5),
    prove that its radius is √29 -/
theorem circle_radius_sqrt_29 :
  ∃ (x : ℝ), 
    (x - 2)^2 + 2^2 = (x + 1)^2 + 5^2 →
    Real.sqrt ((x - 2)^2 + 2^2) = Real.sqrt 29 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_sqrt_29_l1529_152923


namespace NUMINAMATH_CALUDE_quadratic_expression_value_l1529_152915

theorem quadratic_expression_value (x y : ℝ) 
  (eq1 : 4 * x + y = 17) 
  (eq2 : x + 4 * y = 23) : 
  17 * x^2 + 34 * x * y + 17 * y^2 = 818 := by
sorry

end NUMINAMATH_CALUDE_quadratic_expression_value_l1529_152915


namespace NUMINAMATH_CALUDE_company_workforce_l1529_152971

theorem company_workforce (initial_employees : ℕ) : 
  (initial_employees * 6 / 10 : ℚ) = (initial_employees + 20) * 11 / 20 →
  initial_employees + 20 = 240 := by
  sorry

end NUMINAMATH_CALUDE_company_workforce_l1529_152971


namespace NUMINAMATH_CALUDE_f_strictly_increasing_iff_a_in_range_l1529_152986

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a * x^2 + 1 else (a^2 - 1) * Real.exp (a * x)

-- State the theorem
theorem f_strictly_increasing_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, (deriv (f a)) x > 0) ↔ (1 < a ∧ a ≤ Real.sqrt 2) :=
sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_iff_a_in_range_l1529_152986


namespace NUMINAMATH_CALUDE_bisecting_circle_relation_l1529_152959

/-- A circle that always bisects another circle -/
structure BisectingCircle where
  a : ℝ
  b : ℝ
  eq_bisecting : ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = b^2 + 1
  eq_bisected : ∀ (x y : ℝ), (x + 1)^2 + (y + 1)^2 = 4
  bisects : ∀ (x y : ℝ), (x - a)^2 + (y - b)^2 = b^2 + 1 → 
    ∃ (t : ℝ), (x + 1)^2 + (y + 1)^2 = 4 ∧ 
    ((1 - t) * x + t * (-1))^2 + ((1 - t) * y + t * (-1))^2 = 1

/-- The relationship between a and b in a bisecting circle -/
theorem bisecting_circle_relation (c : BisectingCircle) : 
  c.a^2 + 2*c.a + 2*c.b + 5 = 0 := by
  sorry

end NUMINAMATH_CALUDE_bisecting_circle_relation_l1529_152959


namespace NUMINAMATH_CALUDE_simplify_expression_l1529_152965

-- Define the trigonometric identity
axiom trig_identity (θ : Real) : Real.sin θ ^ 2 + Real.cos θ ^ 2 = 1

-- Define the theorem
theorem simplify_expression : 
  2 - Real.sin (21 * π / 180) ^ 2 - Real.cos (21 * π / 180) ^ 2 
  + Real.sin (17 * π / 180) ^ 4 + Real.sin (17 * π / 180) ^ 2 * Real.cos (17 * π / 180) ^ 2 
  + Real.cos (17 * π / 180) ^ 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1529_152965


namespace NUMINAMATH_CALUDE_sum_base6_to_55_l1529_152977

/-- Converts a number from base 6 to base 10 -/
def base6ToBase10 (n : ℕ) : ℕ := sorry

/-- Sums numbers from 1 to n in base 6 -/
def sumBase6 (n : ℕ) : ℕ := sorry

theorem sum_base6_to_55 : base6ToBase10 (sumBase6 55) = 630 := by sorry

end NUMINAMATH_CALUDE_sum_base6_to_55_l1529_152977


namespace NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l1529_152928

theorem sqrt_seven_to_sixth : (Real.sqrt 7) ^ 6 = 343 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_to_sixth_l1529_152928


namespace NUMINAMATH_CALUDE_find_m_l1529_152919

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A (m : ℕ) : Set ℕ := {x ∈ U | x^2 - 5*x + m = 0}

-- Define the complement of A in U
def C_UA (m : ℕ) : Set ℕ := U \ A m

-- Theorem statement
theorem find_m : ∃ m : ℕ, C_UA m = {1, 4} := by
  sorry

end NUMINAMATH_CALUDE_find_m_l1529_152919


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1529_152969

theorem min_value_of_expression (x y : ℝ) : (x^3*y - 1)^2 + (x + y)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1529_152969


namespace NUMINAMATH_CALUDE_janes_breakfast_l1529_152939

theorem janes_breakfast (b m : ℕ) : 
  b + m = 7 →
  (90 * b + 40 * m) % 100 = 0 →
  b = 4 :=
by sorry

end NUMINAMATH_CALUDE_janes_breakfast_l1529_152939


namespace NUMINAMATH_CALUDE_max_chord_length_l1529_152982

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 8*y

-- Define a point on the parabola
structure Point := (x : ℝ) (y : ℝ)

-- Define a chord on the parabola
structure Chord := (A : Point) (B : Point)

-- Define the condition for the midpoint of the chord
def midpointCondition (c : Chord) : Prop := (c.A.y + c.B.y) / 2 = 4

-- Define the length of the chord
def chordLength (c : Chord) : ℝ := abs (c.A.x - c.B.x)

-- Theorem statement
theorem max_chord_length :
  ∀ c : Chord,
  parabola c.A.x c.A.y →
  parabola c.B.x c.B.y →
  midpointCondition c →
  ∃ maxLength : ℝ, maxLength = 12 ∧ ∀ otherChord : Chord,
    parabola otherChord.A.x otherChord.A.y →
    parabola otherChord.B.x otherChord.B.y →
    midpointCondition otherChord →
    chordLength otherChord ≤ maxLength :=
sorry

end NUMINAMATH_CALUDE_max_chord_length_l1529_152982


namespace NUMINAMATH_CALUDE_january_oil_bill_l1529_152962

theorem january_oil_bill (january february : ℝ) 
  (h1 : february / january = 5 / 4)
  (h2 : (february + 30) / january = 3 / 2) : 
  january = 120 := by
  sorry

end NUMINAMATH_CALUDE_january_oil_bill_l1529_152962


namespace NUMINAMATH_CALUDE_fraction_product_squared_l1529_152996

theorem fraction_product_squared :
  (8 / 9 : ℚ)^2 * (1 / 3 : ℚ)^2 = 64 / 729 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_squared_l1529_152996


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l1529_152975

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 2 = 1

-- Define the asymptote equations
def asymptote1 (x y : ℝ) : Prop := y = (Real.sqrt 2 / 2) * x
def asymptote2 (x y : ℝ) : Prop := y = -(Real.sqrt 2 / 2) * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ (x y : ℝ), hyperbola x y →
  (asymptote1 x y ∨ asymptote2 x y) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l1529_152975


namespace NUMINAMATH_CALUDE_geometric_progression_proof_l1529_152937

/-- Given a geometric progression with b₃ = -1 and b₆ = 27/8,
    prove that the first term b₁ = -4/9 and the common ratio q = -3/2 -/
theorem geometric_progression_proof (b : ℕ → ℚ) :
  b 3 = -1 ∧ b 6 = 27/8 →
  (∃ q : ℚ, ∀ n : ℕ, b (n + 1) = b n * q) →
  b 1 = -4/9 ∧ (∀ n : ℕ, b (n + 1) = b n * (-3/2)) := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_proof_l1529_152937


namespace NUMINAMATH_CALUDE_survey_ratings_l1529_152947

theorem survey_ratings (total : ℕ) (excellent_percent : ℚ) (satisfactory_remaining_percent : ℚ) (needs_improvement : ℕ) :
  total = 120 →
  excellent_percent = 15 / 100 →
  satisfactory_remaining_percent = 80 / 100 →
  needs_improvement = 6 →
  ∃ (very_satisfactory_percent : ℚ),
    very_satisfactory_percent = 16 / 100 ∧
    excellent_percent + very_satisfactory_percent + 
    (satisfactory_remaining_percent * (1 - excellent_percent - needs_improvement / total)) +
    (needs_improvement / total) = 1 :=
by sorry

end NUMINAMATH_CALUDE_survey_ratings_l1529_152947


namespace NUMINAMATH_CALUDE_sum_of_first_60_digits_l1529_152964

/-- The decimal representation of 1/9999 -/
def decimal_rep : ℚ := 1 / 9999

/-- The sequence of digits in the decimal representation of 1/9999 -/
def digit_sequence : ℕ → ℕ
  | n => match n % 4 with
         | 0 => 0
         | 1 => 0
         | 2 => 0
         | 3 => 1
         | _ => 0  -- This case is technically unreachable

/-- The sum of the first n digits in the sequence -/
def digit_sum (n : ℕ) : ℕ := (List.range n).map digit_sequence |>.sum

theorem sum_of_first_60_digits :
  digit_sum 60 = 15 :=
sorry

end NUMINAMATH_CALUDE_sum_of_first_60_digits_l1529_152964


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l1529_152943

def f (a x : ℝ) : ℝ := |x + 1| + |x - a|

theorem solution_set_implies_a_value (a : ℝ) (h1 : a > 0) :
  (∀ x : ℝ, f a x ≥ 5 ↔ x ≤ -2 ∨ x > 3) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l1529_152943


namespace NUMINAMATH_CALUDE_leila_spending_difference_l1529_152906

theorem leila_spending_difference : 
  ∀ (total_money sweater_cost jewelry_cost remaining : ℕ),
  sweater_cost = 40 →
  4 * sweater_cost = total_money →
  remaining = 20 →
  total_money = sweater_cost + jewelry_cost + remaining →
  jewelry_cost - sweater_cost = 60 := by
sorry

end NUMINAMATH_CALUDE_leila_spending_difference_l1529_152906


namespace NUMINAMATH_CALUDE_swimmer_speed_l1529_152938

/-- The speed of a swimmer in still water, given downstream and upstream swim data -/
theorem swimmer_speed (downstream_distance upstream_distance : ℝ) 
  (time : ℝ) (h1 : downstream_distance = 30) (h2 : upstream_distance = 20) 
  (h3 : time = 5) : ∃ (v_man v_stream : ℝ),
  downstream_distance / time = v_man + v_stream ∧
  upstream_distance / time = v_man - v_stream ∧
  v_man = 5 := by
  sorry

end NUMINAMATH_CALUDE_swimmer_speed_l1529_152938


namespace NUMINAMATH_CALUDE_red_non_honda_percentage_l1529_152948

/-- Calculates the percentage of red non-Honda cars in Chennai --/
theorem red_non_honda_percentage
  (total_cars : ℕ) 
  (honda_cars : ℕ) 
  (red_honda_ratio : ℚ) 
  (total_red_ratio : ℚ) 
  (h1 : total_cars = 900)
  (h2 : honda_cars = 500)
  (h3 : red_honda_ratio = 90 / 100)
  (h4 : total_red_ratio = 60 / 100) :
  (total_red_ratio * total_cars - red_honda_ratio * honda_cars) / (total_cars - honda_cars) = 9 / 40 :=
by
  sorry

#eval (9 : ℚ) / 40 -- This should evaluate to 0.225 or 22.5%

end NUMINAMATH_CALUDE_red_non_honda_percentage_l1529_152948


namespace NUMINAMATH_CALUDE_stratified_sampling_middle_schools_l1529_152911

theorem stratified_sampling_middle_schools 
  (total_schools : ℕ) 
  (middle_schools : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_schools = 700) 
  (h2 : middle_schools = 200) 
  (h3 : sample_size = 70) :
  (sample_size : ℚ) * (middle_schools : ℚ) / (total_schools : ℚ) = 20 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_middle_schools_l1529_152911


namespace NUMINAMATH_CALUDE_range_of_a_l1529_152992

theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) →
  (∃ x : ℝ, x^2 + 2*a*x + (2 - a) = 0) →
  a ≤ -2 ∨ a = 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1529_152992


namespace NUMINAMATH_CALUDE_ad_broadcast_solution_l1529_152954

/-- Represents the number of ads remaining after the k-th broadcasting -/
def remaining_ads (m : ℕ) (k : ℕ) : ℚ :=
  if k = 0 then m
  else (7/8 : ℚ) * remaining_ads m (k-1) - (7/8 : ℚ) * k

/-- The total number of ads broadcast up to and including the k-th insert -/
def ads_broadcast (m : ℕ) (k : ℕ) : ℚ :=
  m - remaining_ads m k

theorem ad_broadcast_solution (n : ℕ) (m : ℕ) (h1 : n > 1) 
  (h2 : ads_broadcast m n = m) 
  (h3 : ∀ k < n, ads_broadcast m k < m) :
  n = 7 ∧ m = 49 := by
  sorry


end NUMINAMATH_CALUDE_ad_broadcast_solution_l1529_152954


namespace NUMINAMATH_CALUDE_max_rooks_is_400_l1529_152974

/-- Represents a rectangular hole on a chessboard -/
structure Hole :=
  (x : Nat) (y : Nat) (width : Nat) (height : Nat)

/-- Represents a 300x300 chessboard with a hole -/
structure Board :=
  (hole : Hole)
  (is_valid : hole.x + hole.width < 300 ∧ hole.y + hole.height < 300)

/-- The maximum number of non-attacking rooks on a 300x300 board with a hole -/
def max_rooks (b : Board) : Nat :=
  sorry

/-- Theorem: The maximum number of non-attacking rooks is 400 for any valid hole -/
theorem max_rooks_is_400 (b : Board) : max_rooks b = 400 :=
  sorry

end NUMINAMATH_CALUDE_max_rooks_is_400_l1529_152974


namespace NUMINAMATH_CALUDE_polynomial_derivative_value_l1529_152918

theorem polynomial_derivative_value (a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x, (3*x - 1)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5) →
  a₁ + 2*a₂ + 3*a₃ + 4*a₄ + 5*a₅ = 240 := by
sorry

end NUMINAMATH_CALUDE_polynomial_derivative_value_l1529_152918


namespace NUMINAMATH_CALUDE_election_winner_percentage_l1529_152949

theorem election_winner_percentage (total_votes : ℕ) (winner_votes : ℕ) (margin : ℕ) : 
  winner_votes = 490 →
  margin = 280 →
  total_votes = winner_votes + (winner_votes - margin) →
  (winner_votes : ℚ) / (total_votes : ℚ) = 7/10 :=
by
  sorry

end NUMINAMATH_CALUDE_election_winner_percentage_l1529_152949


namespace NUMINAMATH_CALUDE_motel_weekly_charge_l1529_152990

/-- The weekly charge for Casey's motel stay --/
def weekly_charge : ℕ → Prop :=
  fun w => 
    let months : ℕ := 3
    let weeks_per_month : ℕ := 4
    let monthly_rate : ℕ := 1000
    let savings : ℕ := 360
    let total_weeks : ℕ := months * weeks_per_month
    let total_monthly_cost : ℕ := months * monthly_rate
    (total_weeks * w = total_monthly_cost + savings) ∧ (w = 280)

/-- Proof that the weekly charge is $280 --/
theorem motel_weekly_charge : weekly_charge 280 := by
  sorry

end NUMINAMATH_CALUDE_motel_weekly_charge_l1529_152990


namespace NUMINAMATH_CALUDE_number_of_divisors_210_l1529_152946

theorem number_of_divisors_210 : Nat.card (Nat.divisors 210) = 16 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_210_l1529_152946
