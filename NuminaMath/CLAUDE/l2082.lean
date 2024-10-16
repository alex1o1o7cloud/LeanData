import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_common_ratios_l2082_208211

/-- Given two nonconstant geometric sequences with terms k, a₁, a₂ and k, b₁, b₂ respectively,
    with different common ratios p and r, if a₂-b₂=5(a₁-b₁), then p + r = 5. -/
theorem sum_of_common_ratios (k p r : ℝ) (h_p_neq_r : p ≠ r) (h_p_neq_1 : p ≠ 1) (h_r_neq_1 : r ≠ 1)
    (h_eq : k * p^2 - k * r^2 = 5 * (k * p - k * r)) :
  p + r = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_common_ratios_l2082_208211


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2082_208270

-- Define set A
def A : Set ℝ := {x | |x - 2| ≤ 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ A, y = -2 * x + 2}

-- Theorem statement
theorem intersection_of_A_and_B : A ∩ B = {x | 0 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2082_208270


namespace NUMINAMATH_CALUDE_fraction_simplification_l2082_208248

theorem fraction_simplification :
  (4 : ℝ) / (Real.sqrt 108 + 2 * Real.sqrt 12 + 2 * Real.sqrt 27) = Real.sqrt 3 / 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l2082_208248


namespace NUMINAMATH_CALUDE_gnome_escape_strategy_exists_l2082_208225

-- Define the color type
inductive Color
| Red | Orange | Yellow | Green | Blue | Indigo | Violet

-- Define the type for a gnome's view (5 colors they can see)
def GnomeView := Fin 5 → Color

-- Define the type for a strategy (takes a view and returns a guess)
def Strategy := GnomeView → Color

-- Define the type for a hat distribution (6 hats on gnomes, 1 hidden)
def HatDistribution := Fin 7 → Color

-- Function to count correct guesses given a strategy and hat distribution
def countCorrectGuesses (s : Strategy) (d : HatDistribution) : Nat :=
  sorry

-- The main theorem
theorem gnome_escape_strategy_exists :
  ∃ (s : Strategy), ∀ (d : HatDistribution),
    countCorrectGuesses s d ≥ 3 := by
  sorry

end NUMINAMATH_CALUDE_gnome_escape_strategy_exists_l2082_208225


namespace NUMINAMATH_CALUDE_sprinter_probabilities_l2082_208221

/-- Probabilities of three independent events -/
def prob_A : ℚ := 2/5
def prob_B : ℚ := 3/4
def prob_C : ℚ := 1/3

/-- Probability of all three events occurring -/
def prob_all_three : ℚ := prob_A * prob_B * prob_C

/-- Probability of exactly two events occurring -/
def prob_two : ℚ := 
  prob_A * prob_B * (1 - prob_C) + 
  prob_A * (1 - prob_B) * prob_C + 
  (1 - prob_A) * prob_B * prob_C

/-- Probability of at least one event occurring -/
def prob_at_least_one : ℚ := 1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

theorem sprinter_probabilities :
  prob_all_three = 1/10 ∧ 
  prob_two = 23/60 ∧ 
  prob_at_least_one = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_sprinter_probabilities_l2082_208221


namespace NUMINAMATH_CALUDE_kingdom_guards_bound_l2082_208209

/-- Represents a road between two castles with a number of guards -/
structure Road where
  castle1 : Nat
  castle2 : Nat
  guards : Nat

/-- Kingdom Wierdo with its castles and roads -/
structure Kingdom where
  N : Nat
  roads : List Road

/-- Check if the kingdom satisfies the guard policy -/
def satisfiesPolicy (k : Kingdom) : Prop :=
  (∀ r ∈ k.roads, r.guards ≤ 4) ∧
  (∀ a b c : Nat, a < k.N → b < k.N → c < k.N →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = b) ∨ (r.castle1 = b ∧ r.castle2 = a)) →
    (∃ r ∈ k.roads, (r.castle1 = b ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = b)) →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = a)) →
    ∀ r ∈ k.roads, ((r.castle1 = a ∧ r.castle2 = b) ∨ (r.castle1 = b ∧ r.castle2 = a) ∨
                    (r.castle1 = b ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = b) ∨
                    (r.castle1 = a ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = a)) →
      r.guards ≤ 3) ∧
  (∀ a b c d : Nat, a < k.N → b < k.N → c < k.N → d < k.N →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = b) ∨ (r.castle1 = b ∧ r.castle2 = a)) →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = a)) →
    (∃ r ∈ k.roads, (r.castle1 = a ∧ r.castle2 = d) ∨ (r.castle1 = d ∧ r.castle2 = a)) →
    (∃ r ∈ k.roads, (r.castle1 = b ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = b)) →
    (∃ r ∈ k.roads, (r.castle1 = b ∧ r.castle2 = d) ∨ (r.castle1 = d ∧ r.castle2 = b)) →
    (∃ r ∈ k.roads, (r.castle1 = c ∧ r.castle2 = d) ∨ (r.castle1 = d ∧ r.castle2 = c)) →
    ¬(∀ r ∈ k.roads, ((r.castle1 = a ∧ r.castle2 = b) ∨ (r.castle1 = b ∧ r.castle2 = a) ∨
                      (r.castle1 = a ∧ r.castle2 = c) ∨ (r.castle1 = c ∧ r.castle2 = a) ∨
                      (r.castle1 = a ∧ r.castle2 = d) ∨ (r.castle1 = d ∧ r.castle2 = a)) →
      r.guards = 3))

theorem kingdom_guards_bound (k : Kingdom) (h : satisfiesPolicy k) :
  (k.roads.map (·.guards)).sum ≤ k.N ^ 2 :=
sorry

end NUMINAMATH_CALUDE_kingdom_guards_bound_l2082_208209


namespace NUMINAMATH_CALUDE_paper_I_maximum_mark_l2082_208278

/-- The maximum mark for paper I -/
def maximum_mark : ℕ := 186

/-- The passing percentage as a rational number -/
def passing_percentage : ℚ := 35 / 100

/-- The marks scored by the candidate -/
def scored_marks : ℕ := 42

/-- The marks by which the candidate failed -/
def failing_margin : ℕ := 23

/-- Theorem stating the maximum mark for paper I -/
theorem paper_I_maximum_mark :
  (↑maximum_mark * passing_percentage).floor = scored_marks + failing_margin :=
sorry

end NUMINAMATH_CALUDE_paper_I_maximum_mark_l2082_208278


namespace NUMINAMATH_CALUDE_tangent_plane_and_normal_line_at_point_A_l2082_208256

-- Define the elliptic paraboloid
def elliptic_paraboloid (x y z : ℝ) : Prop := z = 2 * x^2 + y^2

-- Define the point A
def point_A : ℝ × ℝ × ℝ := (1, -1, 3)

-- Define the tangent plane equation
def tangent_plane (x y z : ℝ) : Prop := 4 * x - 2 * y - z - 3 = 0

-- Define the normal line equations
def normal_line (x y z : ℝ) : Prop :=
  (x - 1) / 4 = (y + 1) / (-2) ∧ (y + 1) / (-2) = (z - 3) / (-1)

-- Theorem statement
theorem tangent_plane_and_normal_line_at_point_A :
  ∀ x y z : ℝ,
  elliptic_paraboloid x y z →
  (x, y, z) = point_A →
  tangent_plane x y z ∧ normal_line x y z :=
sorry

end NUMINAMATH_CALUDE_tangent_plane_and_normal_line_at_point_A_l2082_208256


namespace NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2082_208269

theorem gcd_from_lcm_and_ratio (X Y : ℕ+) :
  Nat.lcm X Y = 180 →
  (X : ℚ) / Y = 2 / 5 →
  Nat.gcd X Y = 18 := by
sorry

end NUMINAMATH_CALUDE_gcd_from_lcm_and_ratio_l2082_208269


namespace NUMINAMATH_CALUDE_nth_k_gonal_number_l2082_208266

/-- The nth k-gonal number -/
def N (n k : ℕ) : ℚ :=
  ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n

/-- Theorem stating the properties of the nth k-gonal number -/
theorem nth_k_gonal_number (k : ℕ) (h : k ≥ 3) :
  ∀ n : ℕ, N n k = ((k - 2) / 2 : ℚ) * n^2 + ((4 - k) / 2 : ℚ) * n ∧
  N 10 24 = 1000 := by sorry

end NUMINAMATH_CALUDE_nth_k_gonal_number_l2082_208266


namespace NUMINAMATH_CALUDE_yellow_face_probability_l2082_208277

/-- The probability of rolling a yellow face on a 12-sided die with 4 yellow faces is 1/3 -/
theorem yellow_face_probability (total_faces : ℕ) (yellow_faces : ℕ) 
  (h1 : total_faces = 12) (h2 : yellow_faces = 4) : 
  (yellow_faces : ℚ) / total_faces = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_face_probability_l2082_208277


namespace NUMINAMATH_CALUDE_tammy_orange_picking_l2082_208222

/-- Proves that given the conditions of Tammy's orange selling business, 
    she picks 12 oranges from each tree each day. -/
theorem tammy_orange_picking :
  let num_trees : ℕ := 10
  let oranges_per_pack : ℕ := 6
  let price_per_pack : ℕ := 2
  let total_earnings : ℕ := 840
  let num_weeks : ℕ := 3
  let days_per_week : ℕ := 7

  (num_trees > 0) →
  (oranges_per_pack > 0) →
  (price_per_pack > 0) →
  (total_earnings > 0) →
  (num_weeks > 0) →
  (days_per_week > 0) →

  (total_earnings / price_per_pack * oranges_per_pack) / (num_weeks * days_per_week) / num_trees = 12 :=
by
  sorry


end NUMINAMATH_CALUDE_tammy_orange_picking_l2082_208222


namespace NUMINAMATH_CALUDE_range_of_f_l2082_208206

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 2^x - 5 else 3 * Real.sin x

-- State the theorem about the range of f
theorem range_of_f :
  Set.range f = Set.Ioc (-5) 3 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l2082_208206


namespace NUMINAMATH_CALUDE_team_formation_with_girls_l2082_208298

-- Define the total number of people
def total_people : Nat := 10

-- Define the number of boys
def num_boys : Nat := 5

-- Define the number of girls
def num_girls : Nat := 5

-- Define the team size
def team_size : Nat := 3

-- Theorem statement
theorem team_formation_with_girls (total_people num_boys num_girls team_size : Nat) 
  (h1 : total_people = num_boys + num_girls)
  (h2 : num_boys = 5)
  (h3 : num_girls = 5)
  (h4 : team_size = 3) :
  (Nat.choose total_people team_size) - (Nat.choose num_boys team_size) = 110 := by
  sorry

end NUMINAMATH_CALUDE_team_formation_with_girls_l2082_208298


namespace NUMINAMATH_CALUDE_elena_car_rental_cost_l2082_208238

/-- Calculates the total cost of a car rental given the daily rate, mileage rate, number of days, and miles driven. -/
def car_rental_cost (daily_rate : ℚ) (mileage_rate : ℚ) (days : ℕ) (miles : ℕ) : ℚ :=
  daily_rate * days + mileage_rate * miles

/-- Proves that Elena's car rental cost is $215 given the specified conditions. -/
theorem elena_car_rental_cost :
  car_rental_cost 30 0.25 3 500 = 215 := by
  sorry

end NUMINAMATH_CALUDE_elena_car_rental_cost_l2082_208238


namespace NUMINAMATH_CALUDE_probability_at_least_3_of_6_l2082_208284

def probability_at_least_k_successes (n k : ℕ) (p : ℚ) : ℚ :=
  Finset.sum (Finset.range (n - k + 1))
    (λ i => Nat.choose n (k + i) * p ^ (k + i) * (1 - p) ^ (n - k - i))

theorem probability_at_least_3_of_6 :
  probability_at_least_k_successes 6 3 (2/3) = 656/729 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_3_of_6_l2082_208284


namespace NUMINAMATH_CALUDE_air_conditioner_sales_l2082_208267

theorem air_conditioner_sales (ac_ratio : ℕ) (ref_ratio : ℕ) (difference : ℕ) : 
  ac_ratio = 5 ∧ ref_ratio = 3 ∧ difference = 54 →
  ac_ratio * (difference / (ac_ratio - ref_ratio)) = 135 :=
by sorry

end NUMINAMATH_CALUDE_air_conditioner_sales_l2082_208267


namespace NUMINAMATH_CALUDE_solve_equation_for_k_l2082_208285

theorem solve_equation_for_k (k : ℝ) (h1 : k ≠ 0) :
  (∀ x : ℝ, (x^2 - k) * (x + k + 1) = x^3 + k * (x^2 - x - 4)) →
  k = -3 :=
by sorry

end NUMINAMATH_CALUDE_solve_equation_for_k_l2082_208285


namespace NUMINAMATH_CALUDE_horner_v4_for_f_at_neg_two_l2082_208292

/-- Horner's method for polynomial evaluation -/
def horner_step (a : ℝ) (x : ℝ) (v : ℝ) : ℝ := v * x + a

/-- The polynomial f(x) = 3x^6 + 5x^5 + 6x^4 + 20x^3 - 8x^2 + 35x + 12 -/
def f (x : ℝ) : ℝ := 3*x^6 + 5*x^5 + 6*x^4 + 20*x^3 - 8*x^2 + 35*x + 12

/-- Theorem: v_4 in Horner's method for f(x) when x = -2 is -16 -/
theorem horner_v4_for_f_at_neg_two :
  let x : ℝ := -2
  let v0 : ℝ := 3
  let v1 : ℝ := horner_step 5 x v0
  let v2 : ℝ := horner_step 6 x v1
  let v3 : ℝ := horner_step 20 x v2
  let v4 : ℝ := horner_step (-8) x v3
  v4 = -16 := by sorry

end NUMINAMATH_CALUDE_horner_v4_for_f_at_neg_two_l2082_208292


namespace NUMINAMATH_CALUDE_abc_product_l2082_208291

theorem abc_product (a b c : ℕ) : 
  a * b * c + a * b + b * c + a * c + a + b + c = 164 → a * b * c = 80 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l2082_208291


namespace NUMINAMATH_CALUDE_river_crossing_possible_l2082_208228

/-- Represents the state of the river crossing -/
structure RiverState where
  left_soldiers : Nat
  left_robbers : Nat
  right_soldiers : Nat
  right_robbers : Nat

/-- Represents a boat trip -/
inductive BoatTrip
  | SoldierSoldier
  | SoldierRobber
  | RobberRobber
  | Soldier
  | Robber

/-- Checks if a state is safe (soldiers not outnumbered by robbers) -/
def is_safe_state (state : RiverState) : Prop :=
  (state.left_soldiers ≥ state.left_robbers || state.left_soldiers = 0) &&
  (state.right_soldiers ≥ state.right_robbers || state.right_soldiers = 0)

/-- Applies a boat trip to a state -/
def apply_trip (state : RiverState) (trip : BoatTrip) (direction : Bool) : RiverState :=
  sorry

/-- Checks if the final state is reached -/
def is_final_state (state : RiverState) : Prop :=
  state.left_soldiers = 0 && state.left_robbers = 0 &&
  state.right_soldiers = 3 && state.right_robbers = 3

/-- Theorem: There exists a sequence of boat trips that safely transports everyone across -/
theorem river_crossing_possible : ∃ (trips : List (BoatTrip × Bool)),
  let final_state := trips.foldl (λ s (trip, dir) => apply_trip s trip dir)
    (RiverState.mk 3 3 0 0)
  is_final_state final_state ∧
  ∀ (intermediate_state : RiverState),
    intermediate_state ∈ trips.scanl (λ s (trip, dir) => apply_trip s trip dir)
      (RiverState.mk 3 3 0 0) →
    is_safe_state intermediate_state :=
  sorry

end NUMINAMATH_CALUDE_river_crossing_possible_l2082_208228


namespace NUMINAMATH_CALUDE_distance_between_rectangle_vertices_l2082_208242

/-- Given an acute-angled triangle ABC with AB = √3, AC = 1, and angle BAC = 60°,
    and equal rectangles AMNB and APQC built outward on sides AB and AC respectively,
    the distance between vertices N and Q is 2√(2 + √3). -/
theorem distance_between_rectangle_vertices (A B C M N P Q : ℝ × ℝ) :
  let AB := Real.sqrt 3
  let AC := 1
  let angle_BAC := 60 * π / 180
  -- Triangle ABC properties
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 →
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 →
  (C.1 - A.1) * (B.1 - A.1) + (C.2 - A.2) * (B.2 - A.2) = AB * AC * Real.cos angle_BAC →
  -- Rectangle properties
  (M.1 - A.1)^2 + (M.2 - A.2)^2 = (P.1 - A.1)^2 + (P.2 - A.2)^2 →
  (N.1 - B.1)^2 + (N.2 - B.2)^2 = (Q.1 - C.1)^2 + (Q.2 - C.2)^2 →
  (M.1 - A.1) * (B.1 - A.1) + (M.2 - A.2) * (B.2 - A.2) = 0 →
  (P.1 - A.1) * (C.1 - A.1) + (P.2 - A.2) * (C.2 - A.2) = 0 →
  -- Conclusion
  (N.1 - Q.1)^2 + (N.2 - Q.2)^2 = 4 * (2 + Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_distance_between_rectangle_vertices_l2082_208242


namespace NUMINAMATH_CALUDE_triangular_prism_properties_l2082_208259

/-- Represents a triangular prism -/
structure TriangularPrism where
  AB : ℝ
  AC : ℝ
  AA₁ : ℝ
  angleCAB : ℝ

/-- The volume of a triangular prism -/
def volume (p : TriangularPrism) : ℝ := sorry

/-- The surface area of a triangular prism -/
def surfaceArea (p : TriangularPrism) : ℝ := sorry

theorem triangular_prism_properties (p : TriangularPrism)
    (h1 : p.AB = 1)
    (h2 : p.AC = 1)
    (h3 : p.AA₁ = Real.sqrt 2)
    (h4 : p.angleCAB = 2 * π / 3) : -- 120° in radians
  volume p = Real.sqrt 6 / 4 ∧
  surfaceArea p = 2 * Real.sqrt 2 + Real.sqrt 6 + Real.sqrt 3 / 2 := by
  sorry

#check triangular_prism_properties

end NUMINAMATH_CALUDE_triangular_prism_properties_l2082_208259


namespace NUMINAMATH_CALUDE_circle_area_through_point_l2082_208247

/-- The area of a circle with center R(5, -2) passing through the point S(-4, 7) is 162π. -/
theorem circle_area_through_point : 
  let R : ℝ × ℝ := (5, -2)
  let S : ℝ × ℝ := (-4, 7)
  let radius := Real.sqrt ((R.1 - S.1)^2 + (R.2 - S.2)^2)
  π * radius^2 = 162 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_point_l2082_208247


namespace NUMINAMATH_CALUDE_dividend_calculation_l2082_208204

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 36)
  (h2 : quotient = 19)
  (h3 : remainder = 5) :
  divisor * quotient + remainder = 689 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l2082_208204


namespace NUMINAMATH_CALUDE_multiply_three_neg_two_l2082_208262

theorem multiply_three_neg_two : 3 * (-2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_multiply_three_neg_two_l2082_208262


namespace NUMINAMATH_CALUDE_street_lights_on_triangular_playground_l2082_208212

theorem street_lights_on_triangular_playground (side_length : ℝ) (interval : ℝ) :
  side_length = 10 ∧ interval = 3 →
  (3 * side_length) / interval = 10 := by
sorry

end NUMINAMATH_CALUDE_street_lights_on_triangular_playground_l2082_208212


namespace NUMINAMATH_CALUDE_unique_solution_l2082_208290

/-- The set A of solutions to the quadratic equation (a^2 - 1)x^2 + (a + 1)x + 1 = 0 -/
def A (a : ℝ) : Set ℝ :=
  {x | (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0}

/-- The set A contains exactly one element if and only if a = 1 or a = 5/3 -/
theorem unique_solution (a : ℝ) : (∃! x, x ∈ A a) ↔ (a = 1 ∨ a = 5/3) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2082_208290


namespace NUMINAMATH_CALUDE_perfect_squares_digits_parity_l2082_208263

/-- A natural number is a perfect square if it is equal to the square of some natural number. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- The units digit of a natural number. -/
def units_digit (n : ℕ) : ℕ := n % 10

/-- The tens digit of a natural number. -/
def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem perfect_squares_digits_parity (a b : ℕ) (x y : ℕ) :
  is_perfect_square a →
  is_perfect_square b →
  units_digit a = 1 →
  tens_digit a = x →
  units_digit b = 6 →
  tens_digit b = y →
  Even x ∧ Odd y :=
sorry

end NUMINAMATH_CALUDE_perfect_squares_digits_parity_l2082_208263


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l2082_208299

/-- An isosceles trapezoid with specific measurements -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculate the area of the isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 40,
    diagonal_length := 50,
    longer_base := 60
  }
  trapezoid_area t = 1336 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l2082_208299


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2082_208249

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => 3 * x^2 - 4 * x - 7
  {x : ℝ | f x < 0} = {x : ℝ | -1 < x ∧ x < 7/3} := by
sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2082_208249


namespace NUMINAMATH_CALUDE_omega_range_l2082_208271

theorem omega_range (ω : ℝ) (h_pos : ω > 0) :
  (∃ a b : ℝ, π ≤ a ∧ a < b ∧ b ≤ 2*π ∧ Real.sin (ω*a) + Real.sin (ω*b) = 2) →
  (ω ∈ Set.Ioo (1/4 : ℝ) (1/2) ∪ Set.Ioi (5/4 : ℝ)) :=
by sorry

end NUMINAMATH_CALUDE_omega_range_l2082_208271


namespace NUMINAMATH_CALUDE_odd_function_inequality_l2082_208231

-- Define the properties of the function f
def IsOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def HasPositiveProduct (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ ≠ x₂ → (x₁ - x₂) * (f x₁ - f x₂) > 0

-- State the theorem
theorem odd_function_inequality (f : ℝ → ℝ) 
  (h_odd : IsOddFunction f) (h_pos : HasPositiveProduct f) : 
  f 4 < f (-6) := by
  sorry

end NUMINAMATH_CALUDE_odd_function_inequality_l2082_208231


namespace NUMINAMATH_CALUDE_exactly_one_zero_iff_m_eq_zero_or_nine_l2082_208205

/-- A quadratic function of the form y = mx² - 6x + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 6 * x + 1

/-- The discriminant of the quadratic function f -/
def discriminant (m : ℝ) : ℝ := (-6)^2 - 4 * m * 1

/-- The function f has exactly one zero -/
def has_exactly_one_zero (m : ℝ) : Prop :=
  (m = 0 ∧ ∃! x, f m x = 0) ∨
  (m ≠ 0 ∧ discriminant m = 0)

theorem exactly_one_zero_iff_m_eq_zero_or_nine (m : ℝ) :
  has_exactly_one_zero m ↔ m = 0 ∨ m = 9 := by
  sorry

end NUMINAMATH_CALUDE_exactly_one_zero_iff_m_eq_zero_or_nine_l2082_208205


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2082_208200

theorem imaginary_part_of_z (z : ℂ) : z = (1 - Complex.I) / Complex.I → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2082_208200


namespace NUMINAMATH_CALUDE_flowchart_properties_l2082_208274

/-- A flowchart is a type of diagram that represents a process or algorithm. -/
def Flowchart : Type := sorry

/-- A block in a flowchart represents a step or decision in the process. -/
def Block : Type := sorry

/-- The start block of a flowchart. -/
def start_block : Block := sorry

/-- The end block of a flowchart. -/
def end_block : Block := sorry

/-- An input block in a flowchart. -/
def input_block : Block := sorry

/-- An output block in a flowchart. -/
def output_block : Block := sorry

/-- A decision block in a flowchart. -/
def decision_block : Block := sorry

/-- A function that checks if a flowchart has both start and end blocks. -/
def has_start_and_end (f : Flowchart) : Prop := sorry

/-- A function that checks if input blocks are only after the start block. -/
def input_after_start (f : Flowchart) : Prop := sorry

/-- A function that checks if output blocks are only before the end block. -/
def output_before_end (f : Flowchart) : Prop := sorry

/-- A function that checks if decision blocks are the only ones with multiple exit points. -/
def decision_multiple_exits (f : Flowchart) : Prop := sorry

/-- A function that checks if the way conditions are described in decision blocks is unique. -/
def unique_decision_conditions (f : Flowchart) : Prop := sorry

theorem flowchart_properties (f : Flowchart) :
  (has_start_and_end f ∧ 
   input_after_start f ∧ 
   output_before_end f ∧ 
   decision_multiple_exits f) ∧
  ¬(unique_decision_conditions f) := by sorry

end NUMINAMATH_CALUDE_flowchart_properties_l2082_208274


namespace NUMINAMATH_CALUDE_equality_of_positive_reals_l2082_208250

theorem equality_of_positive_reals (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d) / (b + 2*c + d) + (b^2 - c*a) / (c + 2*d + a) + 
  (c^2 - d*b) / (d + 2*a + b) + (d^2 - a*c) / (a + 2*b + c) = 0 →
  a = b ∧ b = c ∧ c = d := by
sorry

end NUMINAMATH_CALUDE_equality_of_positive_reals_l2082_208250


namespace NUMINAMATH_CALUDE_game_winner_l2082_208201

/-- Represents the state of the game with three balls -/
structure GameState where
  n : ℕ -- number of empty holes between one outer ball and the middle ball
  k : ℕ -- number of empty holes between the other outer ball and the middle ball

/-- Determines if a player can make a move in the given game state -/
def canMove (state : GameState) : Prop :=
  state.n > 0 ∨ state.k > 0

/-- Determines if the first player wins in the given game state -/
def firstPlayerWins (state : GameState) : Prop :=
  (state.n + state.k) % 2 = 1

theorem game_winner (state : GameState) :
  canMove state → (firstPlayerWins state ↔ ¬firstPlayerWins { n := state.k, k := state.n - 1 }) ∧
                  (¬firstPlayerWins state ↔ ¬firstPlayerWins { n := state.n - 1, k := state.k }) :=
sorry

end NUMINAMATH_CALUDE_game_winner_l2082_208201


namespace NUMINAMATH_CALUDE_minimal_coloring_exists_l2082_208254

/-- Define the function f for a given set M and subset A -/
def f (M : Finset ℕ) (A : Finset ℕ) : Finset ℕ :=
  M.filter (fun x => (A.filter (fun a => x % a = 0)).card % 2 = 1)

/-- The main theorem -/
theorem minimal_coloring_exists :
  ∀ (M : Finset ℕ), M.card = 2017 →
  ∃ (c : Finset ℕ → Bool),
    ∀ (A : Finset ℕ), A ⊆ M →
      A ≠ f M A → c A ≠ c (f M A) :=
by sorry

end NUMINAMATH_CALUDE_minimal_coloring_exists_l2082_208254


namespace NUMINAMATH_CALUDE_shaded_area_is_108pi_l2082_208289

/-- Represents a point on a line -/
structure Point :=
  (x : ℝ)

/-- Represents a semicircle -/
structure Semicircle :=
  (center : Point)
  (radius : ℝ)

/-- The configuration of points and semicircles -/
structure Configuration :=
  (A B C D E F : Point)
  (AF AB BC CD DE EF : Semicircle)

/-- The conditions of the problem -/
def problem_conditions (config : Configuration) : Prop :=
  let {A, B, C, D, E, F, AF, AB, BC, CD, DE, EF} := config
  (B.x - A.x = 6) ∧ 
  (C.x - B.x = 6) ∧ 
  (D.x - C.x = 6) ∧ 
  (E.x - D.x = 6) ∧ 
  (F.x - E.x = 6) ∧
  (AF.radius = 15) ∧
  (AB.radius = 3) ∧
  (BC.radius = 3) ∧
  (CD.radius = 3) ∧
  (DE.radius = 3) ∧
  (EF.radius = 3)

/-- The area of the shaded region -/
def shaded_area (config : Configuration) : ℝ :=
  sorry  -- Actual calculation would go here

/-- The theorem stating that the shaded area is 108π -/
theorem shaded_area_is_108pi (config : Configuration) 
  (h : problem_conditions config) : shaded_area config = 108 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_108pi_l2082_208289


namespace NUMINAMATH_CALUDE_vertex_to_center_equals_side_length_l2082_208276

/-- A regular hexagon with side length 16 units -/
structure RegularHexagon :=
  (side_length : ℝ)
  (is_regular : Bool)
  (side_length_eq_16 : side_length = 16)

/-- The length of a segment from a vertex to the center of a regular hexagon -/
def vertex_to_center_length (h : RegularHexagon) : ℝ := sorry

/-- Theorem: The length of a segment from a vertex to the center of a regular hexagon
    with side length 16 units is equal to 16 units -/
theorem vertex_to_center_equals_side_length (h : RegularHexagon) :
  vertex_to_center_length h = h.side_length :=
sorry

end NUMINAMATH_CALUDE_vertex_to_center_equals_side_length_l2082_208276


namespace NUMINAMATH_CALUDE_josh_candy_purchase_l2082_208296

/-- Given an initial amount of money and the cost of a purchase, 
    calculate the remaining change. -/
def calculate_change (initial_amount cost : ℚ) : ℚ :=
  initial_amount - cost

/-- Prove that given an initial amount of $1.80 and a purchase of $0.45, 
    the remaining change is $1.35. -/
theorem josh_candy_purchase : 
  calculate_change (180/100) (45/100) = 135/100 := by
  sorry

end NUMINAMATH_CALUDE_josh_candy_purchase_l2082_208296


namespace NUMINAMATH_CALUDE_tank_depth_l2082_208239

/-- Proves that a tank with given dimensions and plastering cost has a depth of 6 meters -/
theorem tank_depth (length width : ℝ) (cost_per_sqm total_cost : ℝ) : 
  length = 25 → 
  width = 12 → 
  cost_per_sqm = 0.75 → 
  total_cost = 558 → 
  ∃ d : ℝ, d = 6 ∧ cost_per_sqm * (2 * (length * d) + 2 * (width * d) + (length * width)) = total_cost :=
by
  sorry

#check tank_depth

end NUMINAMATH_CALUDE_tank_depth_l2082_208239


namespace NUMINAMATH_CALUDE_max_area_triangle_OPQ_l2082_208214

/-- Parabola in Cartesian coordinates -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a parabola -/
structure PointOnParabola (c : Parabola) where
  x : ℝ
  y : ℝ
  h : x^2 = 2 * c.p * y

/-- Line intersecting a parabola -/
structure IntersectingLine (c : Parabola) where
  k : ℝ
  b : ℝ
  h : ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ x₁^2 = 2 * c.p * (k * x₁ + b) ∧ x₂^2 = 2 * c.p * (k * x₂ + b)

/-- Theorem: Maximum area of triangle OPQ -/
theorem max_area_triangle_OPQ (c : Parabola) (a : PointOnParabola c) (l : IntersectingLine c) :
  a.x^2 + a.y^2 = (3/2)^2 →
  (∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧
    x₁^2 = 2 * c.p * y₁ ∧ x₂^2 = 2 * c.p * y₂ ∧
    y₁ = l.k * x₁ + l.b ∧ y₂ = l.k * x₂ + l.b ∧
    (y₁ + y₂) / 2 = 1) →
  (∃ (area : ℝ), area ≤ 2 ∧
    ∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ →
      x₁^2 = 2 * c.p * y₁ → x₂^2 = 2 * c.p * y₂ →
      y₁ = l.k * x₁ + l.b → y₂ = l.k * x₂ + l.b →
      (y₁ + y₂) / 2 = 1 →
      area ≥ abs (x₁ * y₂ - x₂ * y₁) / 2) :=
sorry

end NUMINAMATH_CALUDE_max_area_triangle_OPQ_l2082_208214


namespace NUMINAMATH_CALUDE_measure_one_kg_grain_l2082_208234

/-- Represents a balance scale --/
structure BalanceScale where
  isInaccurate : Bool

/-- Represents a weight --/
structure Weight where
  mass : ℝ
  isAccurate : Bool

/-- Represents a bag of grain --/
structure GrainBag where
  mass : ℝ

/-- Function to measure a specific mass of grain --/
def measureGrain (scale : BalanceScale) (reference : Weight) (bag : GrainBag) (targetMass : ℝ) : Prop :=
  scale.isInaccurate ∧ reference.isAccurate ∧ reference.mass = targetMass

/-- Theorem stating that it's possible to measure 1 kg of grain using inaccurate scales and an accurate 1 kg weight --/
theorem measure_one_kg_grain 
  (scale : BalanceScale) 
  (reference : Weight) 
  (bag : GrainBag) : 
  measureGrain scale reference bag 1 → 
  ∃ (measuredGrain : GrainBag), measuredGrain.mass = 1 :=
sorry

end NUMINAMATH_CALUDE_measure_one_kg_grain_l2082_208234


namespace NUMINAMATH_CALUDE_luke_coin_count_l2082_208224

theorem luke_coin_count (quarter_piles dime_piles coins_per_pile : ℕ) : 
  quarter_piles = 5 → dime_piles = 5 → coins_per_pile = 3 →
  quarter_piles * coins_per_pile + dime_piles * coins_per_pile = 30 := by
  sorry

end NUMINAMATH_CALUDE_luke_coin_count_l2082_208224


namespace NUMINAMATH_CALUDE_w_plus_reciprocal_w_traces_ellipse_l2082_208215

theorem w_plus_reciprocal_w_traces_ellipse :
  ∀ (w : ℂ) (x y : ℝ),
  (Complex.abs w = 3) →
  (w + w⁻¹ = x + y * Complex.I) →
  ∃ (a b : ℝ), (x^2 / a^2 + y^2 / b^2 = 1 ∧ a ≠ b ∧ a > 0 ∧ b > 0) := by
  sorry

end NUMINAMATH_CALUDE_w_plus_reciprocal_w_traces_ellipse_l2082_208215


namespace NUMINAMATH_CALUDE_trigonometric_simplification_l2082_208243

theorem trigonometric_simplification :
  let numerator := Real.sin (15 * π / 180) + Real.sin (25 * π / 180) + Real.sin (35 * π / 180) + 
                   Real.sin (45 * π / 180) + Real.sin (55 * π / 180) + Real.sin (65 * π / 180) + 
                   Real.sin (75 * π / 180) + Real.sin (85 * π / 180)
  let denominator := Real.cos (10 * π / 180) * Real.cos (20 * π / 180) * Real.cos (30 * π / 180)
  numerator / denominator = (16 * Real.sin (50 * π / 180) * Real.cos (20 * π / 180)) / Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_simplification_l2082_208243


namespace NUMINAMATH_CALUDE_original_number_l2082_208236

theorem original_number (x : ℝ) : ((x - 8 + 7) / 5) * 4 = 16 → x = 21 := by
  sorry

end NUMINAMATH_CALUDE_original_number_l2082_208236


namespace NUMINAMATH_CALUDE_total_coins_remain_odd_cannot_achieve_equal_coins_l2082_208279

/-- Represents the state of Petya's coins -/
structure CoinState where
  two_kopeck : ℕ
  ten_kopeck : ℕ

/-- The initial state of Petya's coins -/
def initial_state : CoinState := { two_kopeck := 1, ten_kopeck := 0 }

/-- Represents a coin insertion operation -/
inductive InsertionOperation
  | insert_two_kopeck
  | insert_ten_kopeck

/-- Applies an insertion operation to a coin state -/
def apply_insertion (state : CoinState) (op : InsertionOperation) : CoinState :=
  match op with
  | InsertionOperation.insert_two_kopeck => 
      { two_kopeck := state.two_kopeck - 1, ten_kopeck := state.ten_kopeck + 5 }
  | InsertionOperation.insert_ten_kopeck => 
      { two_kopeck := state.two_kopeck + 5, ten_kopeck := state.ten_kopeck - 1 }

/-- The total number of coins in a given state -/
def total_coins (state : CoinState) : ℕ := state.two_kopeck + state.ten_kopeck

/-- Theorem stating that the total number of coins remains odd after any sequence of insertions -/
theorem total_coins_remain_odd (ops : List InsertionOperation) : 
  Odd (total_coins (ops.foldl apply_insertion initial_state)) := by
  sorry

/-- Theorem stating that Petya cannot achieve an equal number of two-kopeck and ten-kopeck coins -/
theorem cannot_achieve_equal_coins (ops : List InsertionOperation) : 
  let final_state := ops.foldl apply_insertion initial_state
  ¬(final_state.two_kopeck = final_state.ten_kopeck) := by
  sorry

end NUMINAMATH_CALUDE_total_coins_remain_odd_cannot_achieve_equal_coins_l2082_208279


namespace NUMINAMATH_CALUDE_problem_statement_l2082_208281

theorem problem_statement (a b c : ℚ) 
  (h : (3*a - 2*b + c - 4)^2 + (a + 2*b - 3*c + 6)^2 + (2*a - b + 2*c - 2)^2 ≤ 0) : 
  2*a + b - 4*c = -4 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2082_208281


namespace NUMINAMATH_CALUDE_f_not_monotonic_iff_a_in_range_l2082_208237

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (1-a)*x^2 - a*(a+2)*x

def is_not_monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x y z, a < x ∧ x < y ∧ y < z ∧ z < b ∧ 
  ((f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z))

theorem f_not_monotonic_iff_a_in_range (a : ℝ) :
  is_not_monotonic (f a) (-1) 1 ↔ 
  (a > -5 ∧ a < -1/2) ∨ (a > -1/2 ∧ a < 1) :=
sorry

end NUMINAMATH_CALUDE_f_not_monotonic_iff_a_in_range_l2082_208237


namespace NUMINAMATH_CALUDE_specific_figure_perimeter_l2082_208223

/-- Calculates the perimeter of a figure composed of a central square and four smaller squares attached to its sides. -/
def figure_perimeter (central_side_length : ℝ) (small_side_length : ℝ) : ℝ :=
  4 * central_side_length + 4 * (3 * small_side_length)

/-- Theorem stating that the perimeter of the specific figure is 140 -/
theorem specific_figure_perimeter :
  figure_perimeter 20 5 = 140 := by
  sorry

#eval figure_perimeter 20 5

end NUMINAMATH_CALUDE_specific_figure_perimeter_l2082_208223


namespace NUMINAMATH_CALUDE_hyperbola_equation_l2082_208245

/-- Given a hyperbola with the standard equation (x²/a² - y²/b² = 1),
    one focus at (-2, 0), and the angle between asymptotes is 60°,
    prove that its equation is either x² - y²/3 = 1 or x²/3 - y² = 1 -/
theorem hyperbola_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (a^2 + b^2 = 4) →
  (b / a = Real.sqrt 3 ∨ b / a = Real.sqrt 3 / 3) →
  ((∀ x y : ℝ, x^2 - y^2 / 3 = 1) ∨ (∀ x y : ℝ, x^2 / 3 - y^2 = 1)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l2082_208245


namespace NUMINAMATH_CALUDE_min_chess_pieces_chess_pieces_solution_l2082_208219

theorem min_chess_pieces (n : ℕ) : 
  (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) → n ≥ 103 :=
by sorry

theorem chess_pieces_solution : 
  ∃ (n : ℕ), (n % 3 = 1) ∧ (n % 5 = 3) ∧ (n % 7 = 5) ∧ n = 103 :=
by sorry

end NUMINAMATH_CALUDE_min_chess_pieces_chess_pieces_solution_l2082_208219


namespace NUMINAMATH_CALUDE_game_score_total_l2082_208246

theorem game_score_total (dad_score : ℕ) (olaf_score : ℕ) : 
  dad_score = 7 → 
  olaf_score = 3 * dad_score → 
  olaf_score + dad_score = 28 := by
sorry

end NUMINAMATH_CALUDE_game_score_total_l2082_208246


namespace NUMINAMATH_CALUDE_edge_pairs_determine_plane_l2082_208283

/-- A regular octahedron -/
structure RegularOctahedron where
  /-- The number of edges in a regular octahedron -/
  num_edges : ℕ
  /-- The number of edges that intersect with any given edge -/
  num_intersecting_edges : ℕ
  /-- Property: A regular octahedron has 12 edges -/
  edge_count : num_edges = 12
  /-- Property: Each edge intersects with 8 other edges -/
  intersecting_edge_count : num_intersecting_edges = 8

/-- The number of unordered pairs of edges that determine a plane in a regular octahedron -/
def num_edge_pairs_determine_plane (o : RegularOctahedron) : ℕ :=
  (o.num_edges * o.num_intersecting_edges) / 2

/-- Theorem: The number of unordered pairs of edges that determine a plane in a regular octahedron is 48 -/
theorem edge_pairs_determine_plane (o : RegularOctahedron) :
  num_edge_pairs_determine_plane o = 48 := by
  sorry

end NUMINAMATH_CALUDE_edge_pairs_determine_plane_l2082_208283


namespace NUMINAMATH_CALUDE_division_problem_l2082_208217

theorem division_problem (n : ℚ) : n / 4 = 12 → n / 3 = 16 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l2082_208217


namespace NUMINAMATH_CALUDE_max_cone_section_area_l2082_208220

/-- The maximum area of a cone section passing through the vertex, given the cone's height and volume --/
theorem max_cone_section_area (h : ℝ) (v : ℝ) : h = 1 → v = π → 
  ∃ (max_area : ℝ), max_area = 2 ∧ 
  ∀ (section_area : ℝ), section_area ≤ max_area := by
  sorry


end NUMINAMATH_CALUDE_max_cone_section_area_l2082_208220


namespace NUMINAMATH_CALUDE_floor_sum_example_l2082_208232

theorem floor_sum_example : ⌊(23.7 : ℝ)⌋ + ⌊(-23.7 : ℝ)⌋ = -1 := by
  sorry

end NUMINAMATH_CALUDE_floor_sum_example_l2082_208232


namespace NUMINAMATH_CALUDE_sheela_income_proof_l2082_208286

/-- Sheela's monthly income in Rupees -/
def monthly_income : ℝ := 22666.67

/-- The amount Sheela deposited in the bank in Rupees -/
def deposit : ℝ := 3400

/-- The percentage of monthly income that was deposited -/
def deposit_percentage : ℝ := 0.15

theorem sheela_income_proof :
  deposit = deposit_percentage * monthly_income :=
by sorry

end NUMINAMATH_CALUDE_sheela_income_proof_l2082_208286


namespace NUMINAMATH_CALUDE_initial_rulers_count_l2082_208235

/-- The number of rulers initially in the drawer -/
def initial_rulers : ℕ := sorry

/-- The number of crayons initially in the drawer -/
def initial_crayons : ℕ := 34

/-- The number of rulers taken out of the drawer -/
def rulers_taken : ℕ := 11

/-- The number of rulers remaining in the drawer after removal -/
def rulers_remaining : ℕ := 3

theorem initial_rulers_count : initial_rulers = 14 := by sorry

end NUMINAMATH_CALUDE_initial_rulers_count_l2082_208235


namespace NUMINAMATH_CALUDE_ab_value_l2082_208213

theorem ab_value (a b : ℝ) (h : 48 * (a * b) = (a * b) * 65) : a * b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l2082_208213


namespace NUMINAMATH_CALUDE_problem_1_l2082_208233

theorem problem_1 : (-1)^2020 * (2020 - Real.pi)^0 - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2082_208233


namespace NUMINAMATH_CALUDE_roma_winning_strategy_l2082_208207

/-- The game state representing the positions of chips on a board -/
structure GameState where
  k : ℕ  -- number of cells
  n : ℕ  -- number of chips
  positions : List ℕ  -- positions of chips

/-- The rating of a chip at a given position -/
def chipRating (pos : ℕ) : ℕ := 2^pos

/-- The total rating of all chips in the game state -/
def totalRating (state : GameState) : ℕ :=
  state.positions.map chipRating |>.sum

/-- Roma's strategy to maintain or reduce the total rating -/
def romaStrategy (state : GameState) : GameState :=
  sorry

theorem roma_winning_strategy (k n : ℕ) (h : n < 2^(k-3)) :
  ∀ (state : GameState), state.k = k → state.n = n →
    ∀ (finalState : GameState), finalState = (romaStrategy state) →
      ∀ (pos : ℕ), pos ∈ finalState.positions → pos < k - 1 := by
  sorry

end NUMINAMATH_CALUDE_roma_winning_strategy_l2082_208207


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l2082_208255

/-- Calculates the total number of students in three grades given stratified sampling information -/
def totalStudents (sampleSize : ℕ) (firstGradeSample : ℕ) (thirdGradeSample : ℕ) (secondGradeTotal : ℕ) : ℕ :=
  let secondGradeSample := sampleSize - firstGradeSample - thirdGradeSample
  sampleSize * (secondGradeTotal / secondGradeSample)

/-- The total number of students in three grades is 900 given the stratified sampling information -/
theorem stratified_sampling_theorem (sampleSize : ℕ) (firstGradeSample : ℕ) (thirdGradeSample : ℕ) (secondGradeTotal : ℕ)
  (h1 : sampleSize = 45)
  (h2 : firstGradeSample = 20)
  (h3 : thirdGradeSample = 10)
  (h4 : secondGradeTotal = 300) :
  totalStudents sampleSize firstGradeSample thirdGradeSample secondGradeTotal = 900 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_theorem_l2082_208255


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_negative_four_range_of_a_for_inequality_l2082_208261

-- Define the function f
def f (x a : ℝ) : ℝ := |2*x + a| + |x - 2|

-- Theorem for part (1)
theorem solution_set_when_a_is_negative_four :
  {x : ℝ | f x (-4) ≥ 6} = {x : ℝ | x ≤ 0 ∨ x ≥ 4} := by sorry

-- Theorem for part (2)
theorem range_of_a_for_inequality :
  {a : ℝ | ∀ x, f x a ≥ 3*a^2 - |2 - x|} = {a : ℝ | -1 ≤ a ∧ a ≤ 4/3} := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_negative_four_range_of_a_for_inequality_l2082_208261


namespace NUMINAMATH_CALUDE_sine_graph_horizontal_compression_l2082_208244

/-- Given a function f(x) = 2sin(x + π/3), if we shorten the horizontal coordinates
    of its graph to 1/2 of the original while keeping the vertical coordinates unchanged,
    the resulting function is g(x) = 2sin(2x + π/3) -/
theorem sine_graph_horizontal_compression (x : ℝ) :
  let f : ℝ → ℝ := λ x ↦ 2 * Real.sin (x + π/3)
  let g : ℝ → ℝ := λ x ↦ 2 * Real.sin (2*x + π/3)
  let h : ℝ → ℝ := λ x ↦ f (x/2)
  h = g :=
by sorry

end NUMINAMATH_CALUDE_sine_graph_horizontal_compression_l2082_208244


namespace NUMINAMATH_CALUDE_product_of_numbers_l2082_208272

theorem product_of_numbers (a b : ℝ) (h1 : a + b = 2) (h2 : a^3 + b^3 = 16) : a * b = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l2082_208272


namespace NUMINAMATH_CALUDE_solution_in_second_quadrant_l2082_208240

theorem solution_in_second_quadrant :
  ∃ (x y : ℝ), 
    (y = 2*x + 2) ∧ 
    (y = -x + 1) ∧ 
    (x < 0) ∧ 
    (y > 0) := by
  sorry

end NUMINAMATH_CALUDE_solution_in_second_quadrant_l2082_208240


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2082_208258

theorem inequality_system_solution_set :
  let S := {x : ℝ | 3 * x + 2 ≥ 1 ∧ (5 - x) / 2 < 0}
  S = {x : ℝ | -1/3 ≤ x ∧ x < 5} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2082_208258


namespace NUMINAMATH_CALUDE_subset_implies_a_range_l2082_208257

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 - 2)^2 ≤ 5/4}
def B (a : ℝ) : Set (ℝ × ℝ) := {p | |p.1 - 1| + 2*|p.2 - 2| ≤ a}

-- State the theorem
theorem subset_implies_a_range (a : ℝ) : A ⊆ B a → a ≥ 5/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_range_l2082_208257


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2082_208253

/-- Given an arithmetic sequence {aₙ}, where Sₙ is the sum of the first n terms,
    prove that S₈ = 80 when a₃ = 20 - a₆ -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) →  -- arithmetic sequence condition
  (∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) →  -- sum formula
  a 3 = 20 - a 6 →
  S 8 = 80 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l2082_208253


namespace NUMINAMATH_CALUDE_volume_of_cut_cube_piece_l2082_208280

theorem volume_of_cut_cube_piece (cube_edge : ℝ) (piece_base_side : ℝ) (piece_height : ℝ) : 
  cube_edge = 1 →
  piece_base_side = 1/3 →
  piece_height = 1/3 →
  (1/3) * (piece_base_side^2) * piece_height = 1/81 :=
by sorry

end NUMINAMATH_CALUDE_volume_of_cut_cube_piece_l2082_208280


namespace NUMINAMATH_CALUDE_integral_x4_over_2minusx2_32_l2082_208260

theorem integral_x4_over_2minusx2_32 :
  ∫ x in (0:ℝ)..1, x^4 / (2 - x^2)^(3/2) = 5/2 - 3*π/4 := by
  sorry

end NUMINAMATH_CALUDE_integral_x4_over_2minusx2_32_l2082_208260


namespace NUMINAMATH_CALUDE_postcards_cost_l2082_208297

/-- Represents a country --/
inductive Country
| Italy
| Germany
| Canada
| Japan

/-- Represents a decade --/
inductive Decade
| Fifties
| Sixties
| Seventies
| Eighties
| Nineties

/-- Price of a postcard in cents for a given country --/
def price (c : Country) : ℕ :=
  match c with
  | Country.Italy => 8
  | Country.Germany => 8
  | Country.Canada => 5
  | Country.Japan => 7

/-- Number of postcards for a given country and decade --/
def quantity (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Italy, Decade.Fifties => 5
  | Country.Italy, Decade.Sixties => 12
  | Country.Italy, Decade.Seventies => 11
  | Country.Italy, Decade.Eighties => 10
  | Country.Italy, Decade.Nineties => 6
  | Country.Germany, Decade.Fifties => 9
  | Country.Germany, Decade.Sixties => 5
  | Country.Germany, Decade.Seventies => 13
  | Country.Germany, Decade.Eighties => 15
  | Country.Germany, Decade.Nineties => 7
  | Country.Canada, Decade.Fifties => 3
  | Country.Canada, Decade.Sixties => 7
  | Country.Canada, Decade.Seventies => 6
  | Country.Canada, Decade.Eighties => 10
  | Country.Canada, Decade.Nineties => 11
  | Country.Japan, Decade.Fifties => 6
  | Country.Japan, Decade.Sixties => 8
  | Country.Japan, Decade.Seventies => 9
  | Country.Japan, Decade.Eighties => 5
  | Country.Japan, Decade.Nineties => 9

/-- Total cost of postcards for a given country and set of decades --/
def totalCost (c : Country) (decades : List Decade) : ℕ :=
  (decades.map (quantity c)).sum * price c

/-- Theorem: The total cost of postcards from Canada and Japan issued in the '60s, '70s, and '80s is 269 cents --/
theorem postcards_cost :
  totalCost Country.Canada [Decade.Sixties, Decade.Seventies, Decade.Eighties] +
  totalCost Country.Japan [Decade.Sixties, Decade.Seventies, Decade.Eighties] = 269 := by
  sorry

end NUMINAMATH_CALUDE_postcards_cost_l2082_208297


namespace NUMINAMATH_CALUDE_count_valid_house_numbers_l2082_208287

/-- A two-digit prime number less than 50 -/
def TwoDigitPrime : Type := { n : ℕ // n ≥ 10 ∧ n < 50 ∧ Nat.Prime n }

/-- The set of all two-digit primes less than 50 -/
def TwoDigitPrimes : Finset TwoDigitPrime := sorry

/-- A four-digit house number ABCD where AB and CD are distinct two-digit primes less than 50 -/
structure HouseNumber where
  ab : TwoDigitPrime
  cd : TwoDigitPrime
  distinct : ab ≠ cd

/-- The set of all valid house numbers -/
def ValidHouseNumbers : Finset HouseNumber := sorry

theorem count_valid_house_numbers : Finset.card ValidHouseNumbers = 110 := by sorry

end NUMINAMATH_CALUDE_count_valid_house_numbers_l2082_208287


namespace NUMINAMATH_CALUDE_man_speed_is_4_l2082_208288

/-- Represents the speed of water in a stream. -/
def stream_speed : ℝ := sorry

/-- Represents the speed of a man swimming in still water. -/
def man_speed : ℝ := sorry

/-- The distance traveled downstream. -/
def downstream_distance : ℝ := 30

/-- The distance traveled upstream. -/
def upstream_distance : ℝ := 18

/-- The time taken for both downstream and upstream swims. -/
def swim_time : ℝ := 6

/-- Theorem stating that the man's speed in still water is 4 km/h. -/
theorem man_speed_is_4 : 
  downstream_distance = (man_speed + stream_speed) * swim_time ∧ 
  upstream_distance = (man_speed - stream_speed) * swim_time → 
  man_speed = 4 := by sorry

end NUMINAMATH_CALUDE_man_speed_is_4_l2082_208288


namespace NUMINAMATH_CALUDE_pictures_hung_vertically_l2082_208252

/-- Given a total of 30 pictures, with half hung horizontally and 5 hung haphazardly,
    prove that 10 pictures are hung vertically. -/
theorem pictures_hung_vertically (total : ℕ) (horizontal : ℕ) (haphazard : ℕ) :
  total = 30 →
  horizontal = total / 2 →
  haphazard = 5 →
  total - horizontal - haphazard = 10 := by
sorry

end NUMINAMATH_CALUDE_pictures_hung_vertically_l2082_208252


namespace NUMINAMATH_CALUDE_farm_field_area_l2082_208275

/-- Represents the farm field ploughing scenario -/
structure FarmField where
  planned_daily_rate : ℝ
  actual_daily_rate : ℝ
  extra_days : ℕ
  remaining_area : ℝ

/-- Calculates the total area of the farm field -/
def total_area (f : FarmField) : ℝ :=
  sorry

/-- Theorem stating that the total area of the farm field is 312 hectares -/
theorem farm_field_area (f : FarmField) 
  (h1 : f.planned_daily_rate = 260)
  (h2 : f.actual_daily_rate = 85)
  (h3 : f.extra_days = 2)
  (h4 : f.remaining_area = 40) :
  total_area f = 312 :=
sorry

end NUMINAMATH_CALUDE_farm_field_area_l2082_208275


namespace NUMINAMATH_CALUDE_camp_attendance_l2082_208294

theorem camp_attendance (total_lawrence : ℕ) (stayed_home : ℕ) (went_to_camp : ℕ)
  (h1 : total_lawrence = 1538832)
  (h2 : stayed_home = 644997)
  (h3 : went_to_camp = 893835)
  (h4 : total_lawrence = stayed_home + went_to_camp) :
  0 = went_to_camp - (total_lawrence - stayed_home) :=
by sorry

end NUMINAMATH_CALUDE_camp_attendance_l2082_208294


namespace NUMINAMATH_CALUDE_rectangle_perimeter_area_sum_l2082_208241

-- Define the coordinates of the rectangle
def vertex1 : ℤ × ℤ := (1, 2)
def vertex2 : ℤ × ℤ := (1, 6)
def vertex3 : ℤ × ℤ := (7, 6)
def vertex4 : ℤ × ℤ := (7, 2)

-- Define the function to calculate the perimeter and area sum
def perimeterAreaSum (v1 v2 v3 v4 : ℤ × ℤ) : ℤ :=
  let width := (v3.1 - v1.1).natAbs
  let height := (v2.2 - v1.2).natAbs
  2 * (width + height) + width * height

-- Theorem statement
theorem rectangle_perimeter_area_sum :
  perimeterAreaSum vertex1 vertex2 vertex3 vertex4 = 44 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_perimeter_area_sum_l2082_208241


namespace NUMINAMATH_CALUDE_expression_calculation_l2082_208210

theorem expression_calculation : 1453 - 250 * 2 + 130 / 5 = 979 := by
  sorry

end NUMINAMATH_CALUDE_expression_calculation_l2082_208210


namespace NUMINAMATH_CALUDE_weightlifting_winner_l2082_208226

theorem weightlifting_winner (A B C : ℕ) 
  (sum_AB : A + B = 220)
  (sum_AC : A + C = 240)
  (sum_BC : B + C = 250) :
  max A (max B C) = 135 := by
sorry

end NUMINAMATH_CALUDE_weightlifting_winner_l2082_208226


namespace NUMINAMATH_CALUDE_symmetric_graph_phi_l2082_208293

/-- Given a function f and a real number φ, proves that if the graph of y = f(x + φ) 
    is symmetric about x = 0 and |φ| ≤ π/2, then φ = π/6 -/
theorem symmetric_graph_phi (f : ℝ → ℝ) (φ : ℝ) : 
  (∀ x : ℝ, f x = 2 * Real.sin (x + π/3)) →
  (∀ x : ℝ, f (x + φ) = f (-x + φ)) →
  |φ| ≤ π/2 →
  φ = π/6 := by
  sorry

#check symmetric_graph_phi

end NUMINAMATH_CALUDE_symmetric_graph_phi_l2082_208293


namespace NUMINAMATH_CALUDE_power_function_symmetry_l2082_208216

/-- A function f is a power function if it can be written as f(x) = kx^n for some constant k and real number n. -/
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ (k n : ℝ), ∀ x, f x = k * x ^ n

/-- A function f is symmetric about the y-axis if f(x) = f(-x) for all x in the domain of f. -/
def isSymmetricAboutYAxis (f : ℝ → ℝ) : Prop :=
  ∀ x, f x = f (-x)

/-- The main theorem stating the properties of the function f(x) = (2m^2 - m)x^(2m+3) -/
theorem power_function_symmetry (m : ℝ) :
  let f : ℝ → ℝ := fun x ↦ (2 * m^2 - m) * x^(2*m + 3)
  isPowerFunction f ∧ isSymmetricAboutYAxis f →
  (m = -1/2) ∧
  (∀ a : ℝ, 3/2 < a ∧ a < 2 ↔ (a - 1)^m < (2*a - 3)^m) :=
by sorry

end NUMINAMATH_CALUDE_power_function_symmetry_l2082_208216


namespace NUMINAMATH_CALUDE_car_distance_calculation_l2082_208229

/-- Proves that a car traveling at 260 km/h for 2 2/5 hours covers a distance of 624 km -/
theorem car_distance_calculation (speed : ℝ) (time : ℝ) (distance : ℝ) : 
  speed = 260 → time = 2 + 2/5 → distance = speed * time → distance = 624 := by
  sorry

end NUMINAMATH_CALUDE_car_distance_calculation_l2082_208229


namespace NUMINAMATH_CALUDE_min_value_expression_l2082_208264

theorem min_value_expression (x₁ x₂ : ℝ) (h1 : x₁ + x₂ = 16) (h2 : x₁ > x₂) :
  (x₁^2 + x₂^2) / (x₁ - x₂) ≥ 16 ∧ ∃ x₁ x₂, x₁ + x₂ = 16 ∧ x₁ > x₂ ∧ (x₁^2 + x₂^2) / (x₁ - x₂) = 16 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2082_208264


namespace NUMINAMATH_CALUDE_strawberry_distribution_l2082_208208

/-- Represents the distribution of strawberries in buckets -/
structure StrawberryDistribution where
  buckets : Fin 5 → ℕ

/-- The initial distribution of strawberries -/
def initial_distribution : StrawberryDistribution :=
  { buckets := λ _ => 60 }

/-- Removes a specified number of strawberries from each bucket -/
def remove_from_each (d : StrawberryDistribution) (amount : ℕ) : StrawberryDistribution :=
  { buckets := λ i => d.buckets i - amount }

/-- Adds strawberries to specific buckets -/
def add_to_buckets (d : StrawberryDistribution) (additions : Fin 5 → ℕ) : StrawberryDistribution :=
  { buckets := λ i => d.buckets i + additions i }

/-- The final distribution of strawberries after all adjustments -/
def final_distribution : StrawberryDistribution :=
  add_to_buckets
    (remove_from_each initial_distribution 20)
    (λ i => match i with
      | 0 => 15
      | 1 => 15
      | 2 => 25
      | _ => 0)

/-- Theorem stating the final distribution of strawberries -/
theorem strawberry_distribution :
  final_distribution.buckets = λ i => match i with
    | 0 => 55
    | 1 => 55
    | 2 => 65
    | _ => 40 := by sorry

end NUMINAMATH_CALUDE_strawberry_distribution_l2082_208208


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l2082_208251

theorem quadratic_roots_sum (x : ℝ) (h : x^2 - 9*x + 20 = 0) : 
  ∃ (y : ℝ), y ≠ x ∧ y^2 - 9*y + 20 = 0 ∧ x + y = 9 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l2082_208251


namespace NUMINAMATH_CALUDE_unbroken_seashells_l2082_208295

theorem unbroken_seashells (total : ℕ) (broken : ℕ) (unbroken : ℕ) 
  (h1 : total = 7)
  (h2 : broken = 4)
  (h3 : unbroken = total - broken) :
  unbroken = 3 := by
  sorry

end NUMINAMATH_CALUDE_unbroken_seashells_l2082_208295


namespace NUMINAMATH_CALUDE_evaluate_expression_l2082_208230

theorem evaluate_expression : 150 * (150 - 4) - 2 * (150 * 150 - 4) = -23092 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2082_208230


namespace NUMINAMATH_CALUDE_subtract_and_add_l2082_208203

theorem subtract_and_add : (3005 - 3000) + 10 = 15 := by
  sorry

end NUMINAMATH_CALUDE_subtract_and_add_l2082_208203


namespace NUMINAMATH_CALUDE_chocolate_milk_probability_l2082_208273

theorem chocolate_milk_probability : 
  let n : ℕ := 7  -- number of days
  let k : ℕ := 5  -- number of successful days
  let p : ℚ := 3/4  -- probability of success on each day
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_chocolate_milk_probability_l2082_208273


namespace NUMINAMATH_CALUDE_magician_earnings_l2082_208202

/-- Calculates the money earned by a magician selling card decks --/
def money_earned (price_per_deck : ℕ) (starting_decks : ℕ) (ending_decks : ℕ) : ℕ :=
  (starting_decks - ending_decks) * price_per_deck

/-- Proves that the magician earned 4 dollars --/
theorem magician_earnings : money_earned 2 5 3 = 4 := by
  sorry

end NUMINAMATH_CALUDE_magician_earnings_l2082_208202


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l2082_208282

theorem third_root_of_cubic (c d : ℚ) :
  (∃ x : ℚ, c * x^3 + (c - 3*d) * x^2 + (2*d + 4*c) * x + (12 - 2*c) = 0) ∧
  (c * 1^3 + (c - 3*d) * 1^2 + (2*d + 4*c) * 1 + (12 - 2*c) = 0) ∧
  (c * (-3)^3 + (c - 3*d) * (-3)^2 + (2*d + 4*c) * (-3) + (12 - 2*c) = 0) →
  c * 4^3 + (c - 3*d) * 4^2 + (2*d + 4*c) * 4 + (12 - 2*c) = 0 :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l2082_208282


namespace NUMINAMATH_CALUDE_sum_and_count_equals_851_l2082_208218

/-- Sum of integers from a to b, inclusive -/
def sumIntegers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

/-- Count of even integers from a to b, inclusive -/
def countEvenIntegers (a b : ℕ) : ℕ := (b - a) / 2 + 1

/-- The sum of integers from 30 to 50 (inclusive) plus the count of even integers
    in the same range equals 851 -/
theorem sum_and_count_equals_851 : sumIntegers 30 50 + countEvenIntegers 30 50 = 851 := by
  sorry

end NUMINAMATH_CALUDE_sum_and_count_equals_851_l2082_208218


namespace NUMINAMATH_CALUDE_dot_product_of_vectors_l2082_208268

theorem dot_product_of_vectors (a b : ℝ × ℝ) :
  a = (2, 1) →
  ‖b‖ = Real.sqrt 3 →
  ‖a + b‖ = 4 →
  a • b = 4 := by sorry

end NUMINAMATH_CALUDE_dot_product_of_vectors_l2082_208268


namespace NUMINAMATH_CALUDE_expression_simplification_l2082_208227

theorem expression_simplification :
  Real.sqrt (1 + 3) * Real.sqrt (4 + Real.sqrt (1 + 3 + 5 + 7 + 9)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2082_208227


namespace NUMINAMATH_CALUDE_probability_one_head_one_tail_l2082_208265

/-- Represents the outcome of a single coin toss -/
inductive CoinOutcome
| Heads
| Tails

/-- Represents the outcome of tossing two coins simultaneously -/
def TwoCoinsOutcome := CoinOutcome × CoinOutcome

/-- The set of all possible outcomes when tossing two coins -/
def allOutcomes : Finset TwoCoinsOutcome := sorry

/-- The set of favorable outcomes (one head and one tail) -/
def favorableOutcomes : Finset TwoCoinsOutcome := sorry

/-- Probability of an event in a finite sample space -/
def probability (event : Finset TwoCoinsOutcome) : ℚ :=
  event.card / allOutcomes.card

theorem probability_one_head_one_tail :
  probability favorableOutcomes = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_one_head_one_tail_l2082_208265
