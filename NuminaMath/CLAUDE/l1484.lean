import Mathlib

namespace min_decimal_digits_fraction_l1484_148485

theorem min_decimal_digits_fraction (n : ℕ) (d : ℕ) (h : n = 987654321 ∧ d = 2^30 * 5^5) :
  (∃ k : ℕ, k = 30 ∧ 
    ∀ m : ℕ, m < k → ∃ r : ℚ, r ≠ 0 ∧ (n : ℚ) / d * 10^m - ((n : ℚ) / d * 10^m).floor ≠ 0) ∧
    (∃ q : ℚ, (n : ℚ) / d = q ∧ (q * 10^30).floor / 10^30 = q) :=
sorry

end min_decimal_digits_fraction_l1484_148485


namespace todds_initial_gum_l1484_148478

theorem todds_initial_gum (initial : ℕ) : 
  (∃ (after_steve after_emily : ℕ),
    after_steve = initial + 16 ∧
    after_emily = after_steve - 12 ∧
    after_emily = 54) →
  initial = 50 := by
sorry

end todds_initial_gum_l1484_148478


namespace zCoordinate_when_x_is_seven_l1484_148400

/-- A line passing through two points in 3D space -/
structure Line3D where
  point1 : ℝ × ℝ × ℝ
  point2 : ℝ × ℝ × ℝ

/-- Calculate the z-coordinate for a given x-coordinate on the line -/
def zCoordinate (l : Line3D) (x : ℝ) : ℝ :=
  sorry

/-- Theorem stating that for the given line, when x = 7, z = -4 -/
theorem zCoordinate_when_x_is_seven :
  let l : Line3D := { point1 := (1, 3, 2), point2 := (4, 4, -1) }
  zCoordinate l 7 = -4 := by
  sorry

end zCoordinate_when_x_is_seven_l1484_148400


namespace triangle_existence_l1484_148462

theorem triangle_existence (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^4 + b^4 + c^4 + 4*a^2*b^2*c^2 = 2*(a^2*b^2 + a^2*c^2 + b^2*c^2)) :
  ∃ (α β γ : ℝ), α + β + γ = π ∧ Real.sin α = a ∧ Real.sin β = b ∧ Real.sin γ = c := by
sorry

end triangle_existence_l1484_148462


namespace saree_price_theorem_l1484_148477

/-- The original price of sarees before discounts -/
def original_price : ℝ := 550

/-- The first discount rate -/
def discount1 : ℝ := 0.18

/-- The second discount rate -/
def discount2 : ℝ := 0.12

/-- The final sale price after both discounts -/
def final_price : ℝ := 396.88

/-- Theorem stating that the original price of sarees is approximately 550,
    given the final price after two successive discounts -/
theorem saree_price_theorem :
  ∃ ε > 0, abs (original_price - (final_price / ((1 - discount1) * (1 - discount2)))) < ε :=
sorry

end saree_price_theorem_l1484_148477


namespace negation_of_universal_proposition_l1484_148427

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x ≥ 0 → x^3 + x ≥ 0) ↔ (∃ x : ℝ, x < 0 ∧ x^3 + x < 0) := by sorry

end negation_of_universal_proposition_l1484_148427


namespace volume_between_concentric_spheres_l1484_148453

theorem volume_between_concentric_spheres :
  let r₁ : ℝ := 4
  let r₂ : ℝ := 8
  let V₁ := (4 / 3) * π * r₁^3
  let V₂ := (4 / 3) * π * r₂^3
  V₂ - V₁ = (1792 / 3) * π := by sorry

end volume_between_concentric_spheres_l1484_148453


namespace solve_for_q_l1484_148463

theorem solve_for_q : ∀ (k h q : ℚ),
  (3/4 : ℚ) = k/48 ∧ 
  (3/4 : ℚ) = (h+k)/60 ∧ 
  (3/4 : ℚ) = (q-h)/80 →
  q = 69 := by sorry

end solve_for_q_l1484_148463


namespace man_rowing_downstream_speed_l1484_148421

/-- The speed of a man rowing downstream, given his speed in still water and the speed of the stream. -/
def speed_downstream (speed_still_water : ℝ) (speed_stream : ℝ) : ℝ :=
  speed_still_water + speed_stream

/-- Theorem: The speed of the man rowing downstream is 18 kmph. -/
theorem man_rowing_downstream_speed :
  let speed_still_water : ℝ := 12
  let speed_stream : ℝ := 6
  speed_downstream speed_still_water speed_stream = 18 := by
  sorry

end man_rowing_downstream_speed_l1484_148421


namespace only_14_and_28_satisfy_l1484_148488

def satisfies_condition (n : ℕ) : Prop :=
  ∃ (x y : ℕ), n = 10 * x + y ∧ y ≤ 9 ∧ n = 14 * x

theorem only_14_and_28_satisfy :
  ∀ n : ℕ, satisfies_condition n ↔ n = 14 ∨ n = 28 := by
  sorry

end only_14_and_28_satisfy_l1484_148488


namespace colin_speed_proof_l1484_148450

theorem colin_speed_proof (bruce_speed tony_speed brandon_speed colin_speed : ℝ) : 
  bruce_speed = 1 →
  tony_speed = 2 * bruce_speed →
  brandon_speed = (1 / 3) * tony_speed →
  colin_speed = 4 →
  ∃ (multiple : ℝ), colin_speed = multiple * brandon_speed ∧ colin_speed = 4 := by
sorry

end colin_speed_proof_l1484_148450


namespace vector_magnitude_condition_l1484_148471

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_condition (a b : V) :
  ¬(∀ a b : V, ‖a‖ = ‖b‖ → ‖a + b‖ = ‖a - b‖) ∧
  ¬(∀ a b : V, ‖a + b‖ = ‖a - b‖ → ‖a‖ = ‖b‖) :=
by sorry

end vector_magnitude_condition_l1484_148471


namespace square_difference_635_615_l1484_148418

theorem square_difference_635_615 : 635^2 - 615^2 = 25000 := by sorry

end square_difference_635_615_l1484_148418


namespace percentage_calculation_l1484_148479

theorem percentage_calculation (x : ℝ) (h : 0.4 * x = 160) : 0.5 * x = 200 := by
  sorry

end percentage_calculation_l1484_148479


namespace cube_with_holes_surface_area_l1484_148458

/-- Represents a cube with square holes cut through each face. -/
structure CubeWithHoles where
  edge_length : ℝ
  hole_side_length : ℝ

/-- Calculates the total surface area of a cube with square holes, including inside surfaces. -/
def total_surface_area (cube : CubeWithHoles) : ℝ :=
  let original_surface_area := 6 * cube.edge_length ^ 2
  let hole_area := 6 * cube.hole_side_length ^ 2
  let new_exposed_area := 6 * 4 * cube.hole_side_length ^ 2
  original_surface_area - hole_area + new_exposed_area

/-- Theorem stating that a cube with edge length 4 and hole side length 2 has a total surface area of 168. -/
theorem cube_with_holes_surface_area :
  let cube := CubeWithHoles.mk 4 2
  total_surface_area cube = 168 := by
  sorry

end cube_with_holes_surface_area_l1484_148458


namespace min_adults_in_park_l1484_148492

theorem min_adults_in_park (total_people : ℕ) (total_amount : ℚ) 
  (adult_price youth_price child_price : ℚ) :
  total_people = 100 →
  total_amount = 100 →
  adult_price = 3 →
  youth_price = 2 →
  child_price = (3 : ℚ) / 10 →
  ∃ (adults youths children : ℕ),
    adults + youths + children = total_people ∧
    adult_price * adults + youth_price * youths + child_price * children = total_amount ∧
    adults = 2 ∧
    ∀ (a y c : ℕ),
      a + y + c = total_people →
      adult_price * a + youth_price * y + child_price * c = total_amount →
      a ≥ 2 := by
  sorry

end min_adults_in_park_l1484_148492


namespace max_a_value_l1484_148466

/-- The function f(x) = x^2 + 2ax - 1 -/
def f (a : ℝ) (x : ℝ) := x^2 + 2*a*x - 1

/-- The theorem stating the maximum value of a -/
theorem max_a_value :
  ∀ a : ℝ,
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Ici 1 → x₂ ∈ Set.Ici 1 → x₁ < x₂ →
    x₂ * (f a x₁) - x₁ * (f a x₂) < a * (x₁ - x₂)) ↔
  a ≤ 2 :=
sorry

end max_a_value_l1484_148466


namespace circle_with_parallel_tangents_l1484_148410

-- Define the type for points in 2D space
def Point := ℝ × ℝ

-- Define three non-collinear points
variable (A B C : Point)

-- Define the property of non-collinearity
def NonCollinear (A B C : Point) : Prop :=
  let (x₁, y₁) := A
  let (x₂, y₂) := B
  let (x₃, y₃) := C
  (x₂ - x₁) * (y₃ - y₁) ≠ (y₂ - y₁) * (x₃ - x₁)

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a tangent line to a circle
def IsTangent (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

-- Define parallel lines
def Parallel (l₁ l₂ : Point → Prop) : Prop :=
  ∀ (p q : Point), l₁ p ∧ l₂ q → ∃ (k : ℝ), k ≠ 0 ∧ 
    (p.1 - q.1) * k = (p.2 - q.2)

-- Theorem statement
theorem circle_with_parallel_tangents 
  (h : NonCollinear A B C) : 
  ∃ (c : Circle), c.center = C ∧ 
    ∃ (t₁ t₂ : Point → Prop), 
      IsTangent A c ∧ IsTangent B c ∧ 
      Parallel t₁ t₂ :=
sorry

end circle_with_parallel_tangents_l1484_148410


namespace afternoon_snowfall_l1484_148416

theorem afternoon_snowfall (total : ℝ) (morning : ℝ) (afternoon : ℝ)
  (h1 : total = 0.625)
  (h2 : morning = 0.125)
  (h3 : total = morning + afternoon) :
  afternoon = 0.500 := by
sorry

end afternoon_snowfall_l1484_148416


namespace colorNGon_correct_l1484_148401

/-- The number of ways to color exactly k vertices of an n-gon in red,
    such that no two consecutive vertices are red. -/
def colorNGon (n k : ℕ) : ℕ :=
  Nat.choose (n - k - 1) (k - 1) + Nat.choose (n - k) k

/-- Theorem stating that the number of ways to color exactly k vertices of an n-gon in red,
    such that no two consecutive vertices are red, is equal to ⁽ⁿ⁻ᵏ⁻¹ᵏ⁻¹⁾ + ⁽ⁿ⁻ᵏᵏ⁾. -/
theorem colorNGon_correct (n k : ℕ) (h1 : n > k) (h2 : k > 0) :
  colorNGon n k = Nat.choose (n - k - 1) (k - 1) + Nat.choose (n - k) k := by
  sorry

#eval colorNGon 5 2  -- Example usage

end colorNGon_correct_l1484_148401


namespace two_pencils_length_l1484_148431

def pencil_length : ℕ := 12

theorem two_pencils_length : pencil_length + pencil_length = 24 := by
  sorry

end two_pencils_length_l1484_148431


namespace greater_number_sum_difference_l1484_148489

theorem greater_number_sum_difference (x y : ℝ) 
  (sum_eq : x + y = 22) 
  (diff_eq : x - y = 4) : 
  max x y = 13 := by
sorry

end greater_number_sum_difference_l1484_148489


namespace shaanxi_temp_difference_l1484_148419

/-- The temperature difference between two regions -/
def temperature_difference (temp1 : ℝ) (temp2 : ℝ) : ℝ :=
  temp1 - temp2

/-- The highest temperature in Shaanxi South -/
def shaanxi_south_temp : ℝ := 6

/-- The highest temperature in Shaanxi North -/
def shaanxi_north_temp : ℝ := -3

/-- Theorem: The temperature difference between Shaanxi South and Shaanxi North is 9°C -/
theorem shaanxi_temp_difference :
  temperature_difference shaanxi_south_temp shaanxi_north_temp = 9 := by
  sorry

end shaanxi_temp_difference_l1484_148419


namespace real_solutions_quadratic_l1484_148482

theorem real_solutions_quadratic (x y : ℝ) : 
  (3 * y^2 + 6 * x * y + 2 * x + 4 = 0) → 
  (∃ y : ℝ, 3 * y^2 + 6 * x * y + 2 * x + 4 = 0) ↔ 
  (x ≤ -2/3 ∨ x ≥ 4) :=
by sorry

end real_solutions_quadratic_l1484_148482


namespace total_tanks_needed_l1484_148440

/-- Calculates the minimum number of tanks needed to fill all balloons --/
def minTanksNeeded (smallBalloons mediumBalloons largeBalloons : Nat)
  (smallCapacity mediumCapacity largeCapacity : Nat)
  (heliumTankCapacity hydrogenTankCapacity mixtureTankCapacity : Nat) : Nat :=
  let heliumNeeded := smallBalloons * smallCapacity
  let hydrogenNeeded := mediumBalloons * mediumCapacity
  let mixtureNeeded := largeBalloons * largeCapacity
  let heliumTanks := (heliumNeeded + heliumTankCapacity - 1) / heliumTankCapacity
  let hydrogenTanks := (hydrogenNeeded + hydrogenTankCapacity - 1) / hydrogenTankCapacity
  let mixtureTanks := (mixtureNeeded + mixtureTankCapacity - 1) / mixtureTankCapacity
  heliumTanks + hydrogenTanks + mixtureTanks

theorem total_tanks_needed :
  minTanksNeeded 5000 5000 5000 20 30 50 1000 1200 1500 = 392 := by
  sorry

#eval minTanksNeeded 5000 5000 5000 20 30 50 1000 1200 1500

end total_tanks_needed_l1484_148440


namespace interest_rate_equality_l1484_148420

theorem interest_rate_equality (I : ℝ) (r : ℝ) : 
  I = 1000 * 0.12 * 2 → 
  I = 200 * r * 12 → 
  r = 0.1 := by
  sorry

end interest_rate_equality_l1484_148420


namespace unique_polynomial_reconstruction_l1484_148413

/-- A polynomial with non-negative integer coefficients -/
def NonNegIntPolynomial (P : ℕ → ℕ) : Prop :=
  ∃ n : ℕ, ∀ k > n, P k = 0

/-- The polynomial is non-constant -/
def NonConstant (P : ℕ → ℕ) : Prop :=
  ∃ k : ℕ, P k ≠ P 0

theorem unique_polynomial_reconstruction
  (P : ℕ → ℕ)
  (h_non_neg : NonNegIntPolynomial P)
  (h_non_const : NonConstant P) :
  ∀ Q : ℕ → ℕ,
    NonNegIntPolynomial Q →
    NonConstant Q →
    P 2 = Q 2 →
    P (P 2) = Q (Q 2) →
    ∀ x, P x = Q x :=
sorry

end unique_polynomial_reconstruction_l1484_148413


namespace hockey_cards_count_l1484_148437

theorem hockey_cards_count (hockey : ℕ) (football : ℕ) (baseball : ℕ) : 
  baseball = football - 50 →
  football = 4 * hockey →
  hockey + football + baseball = 1750 →
  hockey = 200 := by
sorry

end hockey_cards_count_l1484_148437


namespace function_value_implies_b_equals_negative_one_l1484_148498

noncomputable def f (b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then 2 * x - b else 2^x

theorem function_value_implies_b_equals_negative_one (b : ℝ) :
  (f b (f b (1/2)) = 4) → b = -1 := by
  sorry

end function_value_implies_b_equals_negative_one_l1484_148498


namespace two_sin_45_equals_sqrt_2_l1484_148438

theorem two_sin_45_equals_sqrt_2 (α : Real) (h : α = Real.pi / 4) : 
  2 * Real.sin α = Real.sqrt 2 := by
  sorry

end two_sin_45_equals_sqrt_2_l1484_148438


namespace drunkard_theorem_l1484_148452

structure PubSystem where
  states : Finset (Fin 4)
  transition : Fin 4 → Fin 4 → ℚ
  start_state : Fin 4

def drunkard_walk (ps : PubSystem) (n : ℕ) : Fin 4 → ℚ :=
  sorry

theorem drunkard_theorem (ps : PubSystem) :
  (ps.states = {0, 1, 2, 3}) →
  (ps.transition 0 1 = 1/3) →
  (ps.transition 0 2 = 1/3) →
  (ps.transition 0 3 = 0) →
  (ps.transition 1 0 = 1/2) →
  (ps.transition 1 2 = 1/3) →
  (ps.transition 1 3 = 1/2) →
  (ps.transition 2 0 = 1/2) →
  (ps.transition 2 1 = 1/3) →
  (ps.transition 2 3 = 1/2) →
  (ps.transition 3 1 = 1/3) →
  (ps.transition 3 2 = 1/3) →
  (ps.transition 3 0 = 0) →
  (ps.start_state = 0) →
  ((drunkard_walk ps 5) 2 = 55/162) ∧
  (∀ n > 5, (drunkard_walk ps n) 1 > (drunkard_walk ps n) 0 ∧
            (drunkard_walk ps n) 2 > (drunkard_walk ps n) 0 ∧
            (drunkard_walk ps n) 1 > (drunkard_walk ps n) 3 ∧
            (drunkard_walk ps n) 2 > (drunkard_walk ps n) 3) :=
by sorry

end drunkard_theorem_l1484_148452


namespace box_volume_increase_l1484_148451

/-- Given a rectangular box with length l, width w, and height h, 
    if the volume is 3456, surface area is 1368, and sum of edges is 192,
    prove that increasing each dimension by 2 results in a volume of 5024 -/
theorem box_volume_increase (l w h : ℝ) 
  (hv : l * w * h = 3456)
  (hs : 2 * (l * w + w * h + h * l) = 1368)
  (he : 4 * (l + w + h) = 192) :
  (l + 2) * (w + 2) * (h + 2) = 5024 := by
  sorry

end box_volume_increase_l1484_148451


namespace M_intersect_N_equals_M_l1484_148404

def M : Set ℝ := {x | x^2 - 3*x + 2 = 0}
def N : Set ℝ := {x | x*(x-1)*(x-2) = 0}

theorem M_intersect_N_equals_M : M ∩ N = M := by
  sorry

end M_intersect_N_equals_M_l1484_148404


namespace different_color_chips_probability_l1484_148434

/-- The probability of drawing two chips of different colors from a bag with replacement -/
theorem different_color_chips_probability
  (total_chips : ℕ)
  (blue_chips : ℕ)
  (yellow_chips : ℕ)
  (h_total : total_chips = blue_chips + yellow_chips)
  (h_blue : blue_chips = 5)
  (h_yellow : yellow_chips = 3) :
  (blue_chips : ℚ) / total_chips * (yellow_chips : ℚ) / total_chips +
  (yellow_chips : ℚ) / total_chips * (blue_chips : ℚ) / total_chips =
  15 / 32 :=
sorry

end different_color_chips_probability_l1484_148434


namespace sqrt_of_nine_l1484_148403

-- Define the square root function
def sqrt (x : ℝ) : Set ℝ := {y : ℝ | y^2 = x}

-- Theorem statement
theorem sqrt_of_nine : sqrt 9 = {3, -3} := by
  sorry

end sqrt_of_nine_l1484_148403


namespace polynomial_division_remainder_l1484_148445

/-- The remainder when x^3 + 3x^2 is divided by x^2 - 7x + 2 is 68x - 20 -/
theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  x^3 + 3*x^2 = (x^2 - 7*x + 2) * q + (68*x - 20) := by sorry

end polynomial_division_remainder_l1484_148445


namespace rectangles_with_equal_areas_have_reciprocal_proportions_l1484_148405

theorem rectangles_with_equal_areas_have_reciprocal_proportions 
  (a b c d : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : d > 0) 
  (h5 : a * b = c * d) : 
  a / c = d / b := by
sorry


end rectangles_with_equal_areas_have_reciprocal_proportions_l1484_148405


namespace complex_fraction_simplification_l1484_148487

theorem complex_fraction_simplification (z : ℂ) :
  z = 1 - I → 2 / z = 1 + I := by
  sorry

end complex_fraction_simplification_l1484_148487


namespace parabola_ratio_l1484_148483

-- Define the parabola R
def Parabola (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = a * p.1^2}

-- Define the vertex and focus of a parabola
def vertex (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry
def focus (P : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- Define the locus of midpoints
def midpointLocus (R : Set (ℝ × ℝ)) (W : ℝ × ℝ) : Set (ℝ × ℝ) := sorry

-- Main theorem
theorem parabola_ratio 
  (R : Set (ℝ × ℝ)) 
  (h_R : ∃ a : ℝ, R = Parabola a) 
  (W₁ : ℝ × ℝ) 
  (G₁ : ℝ × ℝ) 
  (h_W₁ : W₁ = vertex R) 
  (h_G₁ : G₁ = focus R) 
  (S : Set (ℝ × ℝ)) 
  (h_S : S = midpointLocus R W₁) 
  (W₂ : ℝ × ℝ) 
  (G₂ : ℝ × ℝ) 
  (h_W₂ : W₂ = vertex S) 
  (h_G₂ : G₂ = focus S) : 
  ‖G₁ - G₂‖ / ‖W₁ - W₂‖ = 1/4 := by sorry


end parabola_ratio_l1484_148483


namespace gcd_digits_bound_l1484_148496

theorem gcd_digits_bound (a b : ℕ) (ha : 10^6 ≤ a ∧ a < 10^7) (hb : 10^6 ≤ b ∧ b < 10^7) 
  (hlcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : 
  Nat.gcd a b < 10^4 := by
sorry

end gcd_digits_bound_l1484_148496


namespace pascal_21st_number_23_row_l1484_148402

/-- The binomial coefficient -/
def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- The number of elements in the nth row of Pascal's triangle -/
def pascal_row_length (n : ℕ) : ℕ := n + 1

theorem pascal_21st_number_23_row : 
  let row := 22
  let position := 21
  pascal_row_length row = 23 → binomial row (row + 1 - position) = 231 := by
  sorry

end pascal_21st_number_23_row_l1484_148402


namespace number_of_groups_l1484_148411

theorem number_of_groups (max min interval : ℝ) (h1 : max = 140) (h2 : min = 51) (h3 : interval = 10) :
  ⌈(max - min) / interval⌉ = 9 := by
  sorry

end number_of_groups_l1484_148411


namespace divided_proportion_problem_l1484_148494

theorem divided_proportion_problem (total : ℚ) (a b c : ℚ) (h1 : total = 782) 
  (h2 : a = 1/2) (h3 : b = 1/3) (h4 : c = 3/4) : 
  (total * a) / (a + b + c) = 247 := by
  sorry

end divided_proportion_problem_l1484_148494


namespace marble_capacity_l1484_148432

theorem marble_capacity (v₁ v₂ : ℝ) (m₁ : ℕ) (h₁ : v₁ > 0) (h₂ : v₂ > 0) :
  v₁ = 36 → m₁ = 180 → v₂ = 108 →
  (v₂ / v₁ * m₁ : ℝ) = 540 := by sorry

end marble_capacity_l1484_148432


namespace quadratic_rewrite_sum_l1484_148460

/-- Given a quadratic polynomial 6x^2 + 36x + 150, prove that when rewritten in the form a(x+b)^2+c, 
    where a, b, and c are constants, a + b + c = 105 -/
theorem quadratic_rewrite_sum (x : ℝ) : 
  ∃ (a b c : ℝ), (∀ x, 6*x^2 + 36*x + 150 = a*(x+b)^2 + c) ∧ (a + b + c = 105) := by
sorry

end quadratic_rewrite_sum_l1484_148460


namespace point_set_characterization_l1484_148470

theorem point_set_characterization (x y : ℝ) : 
  (∀ t : ℝ, -1 ≤ t ∧ t ≤ 1 → t^2 + y*t + x ≥ 0) ↔ 
  (y ∈ Set.Icc (-2) 2 → x ≥ y^2/4) ∧ 
  (y < -2 → x ≥ -y - 1) ∧ 
  (y > 2 → x ≥ y - 1) := by
sorry

end point_set_characterization_l1484_148470


namespace tetrahedron_sphere_radius_relation_l1484_148497

/-- Given a tetrahedron with congruent triangular faces, an inscribed sphere,
    and a triangular face with known angles and circumradius,
    prove the relationship between the inscribed sphere radius and the face properties. -/
theorem tetrahedron_sphere_radius_relation
  (r : ℝ) -- radius of inscribed sphere
  (R : ℝ) -- radius of circumscribed circle of a face
  (α β γ : ℝ) -- angles of a triangular face
  (h_positive_r : r > 0)
  (h_positive_R : R > 0)
  (h_angle_sum : α + β + γ = π)
  (h_positive_angles : 0 < α ∧ 0 < β ∧ 0 < γ)
  (h_congruent_faces : True) -- placeholder for the condition of congruent faces
  : r = R * Real.sqrt (Real.cos α * Real.cos β * Real.cos γ) :=
by sorry

end tetrahedron_sphere_radius_relation_l1484_148497


namespace coloring_exists_l1484_148409

/-- A coloring of numbers from 1 to 2n -/
def Coloring (n : ℕ) := Fin (2*n) → Fin n

/-- Predicate to check if a coloring is valid -/
def ValidColoring (n : ℕ) (c : Coloring n) : Prop :=
  (∀ color : Fin n, ∃! (a b : Fin (2*n)), c a = color ∧ c b = color ∧ a ≠ b) ∧
  (∀ diff : Fin n, ∃! (a b : Fin (2*n)), c a = c b ∧ a ≠ b ∧ a.val - b.val = diff.val + 1)

/-- The sequence of n for which the coloring is possible -/
def ColoringSequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 3 * ColoringSequence n + 1

theorem coloring_exists (m : ℕ) : ∃ c : Coloring (ColoringSequence m), ValidColoring (ColoringSequence m) c := by
  sorry

end coloring_exists_l1484_148409


namespace line_length_after_erasing_l1484_148423

/-- Given a line of 1.5 meters with 15.25 centimeters erased, the resulting length is 134.75 centimeters. -/
theorem line_length_after_erasing (original_length : Real) (erased_length : Real) :
  original_length = 1.5 ∧ erased_length = 15.25 →
  original_length * 100 - erased_length = 134.75 :=
by sorry

end line_length_after_erasing_l1484_148423


namespace point_in_second_quadrant_l1484_148448

theorem point_in_second_quadrant :
  let x : ℝ := Real.sin (2014 * π / 180)
  let y : ℝ := Real.tan (2014 * π / 180)
  x < 0 ∧ y > 0 := by
  sorry

end point_in_second_quadrant_l1484_148448


namespace equal_length_different_turns_l1484_148417

/-- Represents a point in the triangular grid -/
structure Point where
  x : ℤ
  y : ℤ

/-- Represents a direction in the triangular grid -/
inductive Direction
  | Up
  | UpRight
  | DownRight
  | Down
  | DownLeft
  | UpLeft

/-- Represents a route in the triangular grid -/
structure Route where
  start : Point
  steps : List Direction
  leftTurns : ℕ

/-- Calculates the length of a route -/
def routeLength (r : Route) : ℕ := r.steps.length

/-- Theorem: There exist two routes in a triangular grid with different numbers of left turns but equal length -/
theorem equal_length_different_turns :
  ∃ (start finish : Point) (route1 route2 : Route),
    route1.start = start ∧
    route2.start = start ∧
    (routeLength route1 = routeLength route2) ∧
    route1.leftTurns = 4 ∧
    route2.leftTurns = 1 :=
  sorry

end equal_length_different_turns_l1484_148417


namespace intersection_of_M_and_N_l1484_148467

-- Define the sets M and N
def M : Set ℝ := {x | Real.sqrt x < 4}
def N : Set ℝ := {x | 3 * x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 16} := by sorry

end intersection_of_M_and_N_l1484_148467


namespace cube_root_four_solution_l1484_148429

theorem cube_root_four_solution (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h1 : a ^ b = b ^ a) (h2 : b = 4 * a) : a = (4 : ℝ) ^ (1/3) :=
by
  sorry

end cube_root_four_solution_l1484_148429


namespace sum_of_abs_roots_of_P_l1484_148435

/-- The polynomial P(x) = x^3 - 6x^2 + 5x + 12 -/
def P (x : ℝ) : ℝ := x^3 - 6*x^2 + 5*x + 12

/-- Theorem: The sum of the absolute values of the roots of P(x) is 8 -/
theorem sum_of_abs_roots_of_P :
  ∃ (x₁ x₂ x₃ : ℝ),
    (P x₁ = 0) ∧ (P x₂ = 0) ∧ (P x₃ = 0) ∧
    (∀ x, P x = 0 → x = x₁ ∨ x = x₂ ∨ x = x₃) ∧
    |x₁| + |x₂| + |x₃| = 8 :=
sorry

end sum_of_abs_roots_of_P_l1484_148435


namespace supermarket_difference_l1484_148493

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 41

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 60

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

/-- There are more supermarkets in the US than in Canada -/
axiom more_in_us : us_supermarkets > canada_supermarkets

theorem supermarket_difference : us_supermarkets - canada_supermarkets = 22 := by
  sorry

end supermarket_difference_l1484_148493


namespace statement_IV_must_be_false_l1484_148414

/-- Represents a two-digit number -/
structure TwoDigitNumber where
  value : Nat
  is_two_digit : 10 ≤ value ∧ value ≤ 99

/-- Represents the four statements about the number -/
structure Statements (n : TwoDigitNumber) where
  I : Bool
  II : Bool
  III : Bool
  IV : Bool
  I_def : I ↔ n.value = 12
  II_def : II ↔ n.value % 10 ≠ 2
  III_def : III ↔ n.value = 35
  IV_def : IV ↔ n.value % 10 ≠ 5
  three_true : I + II + III + IV = 3

theorem statement_IV_must_be_false (n : TwoDigitNumber) (s : Statements n) :
  s.IV = false :=
sorry

end statement_IV_must_be_false_l1484_148414


namespace min_value_quadratic_l1484_148430

theorem min_value_quadratic (x y : ℝ) : 
  x^2 + 2*x*y + 2*y^2 ≥ 0 ∧ (x^2 + 2*x*y + 2*y^2 = 0 ↔ x = 0 ∧ y = 0) :=
sorry

end min_value_quadratic_l1484_148430


namespace range_of_m_l1484_148424

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |x - m| ≤ 2) →
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end range_of_m_l1484_148424


namespace equation_a_is_linear_l1484_148443

/-- Definition of a linear equation -/
def is_linear_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, f x = a * x + b

/-- The equation x = 1 -/
def equation_a (x : ℝ) : ℝ := x - 1

theorem equation_a_is_linear : is_linear_equation equation_a := by
  sorry

#check equation_a_is_linear

end equation_a_is_linear_l1484_148443


namespace square_area_from_perimeter_l1484_148428

/-- The area of a square with a perimeter of 40 meters is 100 square meters. -/
theorem square_area_from_perimeter :
  ∀ (side : ℝ), 
  (4 * side = 40) →  -- perimeter is 40 meters
  (side * side = 100) -- area is 100 square meters
:= by sorry

end square_area_from_perimeter_l1484_148428


namespace special_function_unique_l1484_148480

/-- A function satisfying the given properties -/
def special_function (g : ℝ → ℝ) : Prop :=
  g 2 = 2 ∧ ∀ x y : ℝ, g (x * y + g x) = x * g y + g x

theorem special_function_unique (g : ℝ → ℝ) (h : special_function g) :
  ∀ x : ℝ, g x = 2 * x :=
sorry

end special_function_unique_l1484_148480


namespace least_three_digit_multiple_l1484_148459

theorem least_three_digit_multiple : ∃ n : ℕ, 
  (n ≥ 100 ∧ n < 1000) ∧ 
  3 ∣ n ∧ 4 ∣ n ∧ 9 ∣ n ∧
  (∀ m : ℕ, (m ≥ 100 ∧ m < 1000) ∧ 3 ∣ m ∧ 4 ∣ m ∧ 9 ∣ m → n ≤ m) ∧
  n = 108 :=
by sorry

end least_three_digit_multiple_l1484_148459


namespace total_owed_correct_l1484_148465

/-- Calculates the total amount owed after one year given monthly charges and interest rates -/
def totalOwed (jan_charge feb_charge mar_charge apr_charge : ℝ)
              (jan_rate feb_rate mar_rate apr_rate : ℝ) : ℝ :=
  let jan_total := jan_charge * (1 + jan_rate)
  let feb_total := feb_charge * (1 + feb_rate)
  let mar_total := mar_charge * (1 + mar_rate)
  let apr_total := apr_charge * (1 + apr_rate)
  jan_total + feb_total + mar_total + apr_total

/-- The theorem stating the total amount owed after one year -/
theorem total_owed_correct :
  totalOwed 35 45 55 25 0.05 0.07 0.04 0.06 = 168.60 := by
  sorry

end total_owed_correct_l1484_148465


namespace garden_walkway_area_l1484_148472

/-- Calculates the total area of walkways in a garden with given specifications -/
def walkway_area (rows : ℕ) (columns : ℕ) (bed_length : ℕ) (bed_width : ℕ) (walkway_width : ℕ) : ℕ :=
  let total_width := columns * bed_length + (columns + 1) * walkway_width
  let total_height := rows * bed_width + (rows + 1) * walkway_width
  let total_area := total_width * total_height
  let bed_area := rows * columns * bed_length * bed_width
  total_area - bed_area

theorem garden_walkway_area :
  walkway_area 4 3 8 3 2 = 416 :=
by sorry

end garden_walkway_area_l1484_148472


namespace hot_dog_bun_distribution_l1484_148415

/-- Hot dog bun distribution problem -/
theorem hot_dog_bun_distribution
  (buns_per_package : ℕ)
  (packages_bought : ℕ)
  (num_classes : ℕ)
  (students_per_class : ℕ)
  (h1 : buns_per_package = 8)
  (h2 : packages_bought = 30)
  (h3 : num_classes = 4)
  (h4 : students_per_class = 30) :
  (buns_per_package * packages_bought) / (num_classes * students_per_class) = 2 :=
sorry

end hot_dog_bun_distribution_l1484_148415


namespace range_of_m_when_p_or_q_false_l1484_148455

theorem range_of_m_when_p_or_q_false (m : ℝ) : 
  (¬(∃ x : ℝ, m * x^2 + 1 ≤ 0) ∧ ¬(∀ x : ℝ, x^2 + m * x + 1 > 0)) → m ≥ 2 := by
  sorry

end range_of_m_when_p_or_q_false_l1484_148455


namespace Q_value_l1484_148436

-- Define the relationship between P, Q, and U
def varies_directly_inversely (P Q U : ℚ) : Prop :=
  ∃ k : ℚ, P = k * Q / U

-- Define the initial conditions
def initial_conditions (P Q U : ℚ) : Prop :=
  P = 12 ∧ Q = 1/2 ∧ U = 16/25

-- Define the final conditions
def final_conditions (P U : ℚ) : Prop :=
  P = 27 ∧ U = 9/49

-- Theorem statement
theorem Q_value :
  ∀ P Q U : ℚ,
  varies_directly_inversely P Q U →
  initial_conditions P Q U →
  final_conditions P U →
  Q = 225/696 :=
by sorry

end Q_value_l1484_148436


namespace be_length_l1484_148484

-- Define the parallelogram and points
structure Parallelogram :=
  (A B C D E F G : ℝ × ℝ)

-- Define the conditions
def is_valid_configuration (p : Parallelogram) : Prop :=
  let ⟨A, B, C, D, E, F, G⟩ := p
  -- F is on the extension of AD
  ∃ t : ℝ, t > 1 ∧ F = A + t • (D - A) ∧
  -- ABCD is a parallelogram
  B - A = C - D ∧ D - A = C - B ∧
  -- BF intersects AC at E
  ∃ s : ℝ, 0 < s ∧ s < 1 ∧ E = A + s • (C - A) ∧
  -- BF intersects DC at G
  ∃ r : ℝ, 0 < r ∧ r < 1 ∧ G = D + r • (C - D) ∧
  -- EF = 15
  ‖F - E‖ = 15 ∧
  -- GF = 20
  ‖F - G‖ = 20

-- The theorem to prove
theorem be_length (p : Parallelogram) (h : is_valid_configuration p) : 
  ‖p.B - p.E‖ = 5 * Real.sqrt 3 := by sorry

end be_length_l1484_148484


namespace sum_six_smallest_multiples_of_11_l1484_148456

theorem sum_six_smallest_multiples_of_11 : 
  (Finset.range 6).sum (fun i => 11 * (i + 1)) = 231 := by
  sorry

end sum_six_smallest_multiples_of_11_l1484_148456


namespace three_quantities_problem_l1484_148481

theorem three_quantities_problem (x y z : ℕ) : 
  y = x + 8 →
  z = 3 * x →
  x + y + z = 108 →
  (x = 20 ∧ y = 28 ∧ z = 60) := by
  sorry

end three_quantities_problem_l1484_148481


namespace blocks_eaten_l1484_148439

theorem blocks_eaten (initial_blocks remaining_blocks : ℕ) 
  (h1 : initial_blocks = 55)
  (h2 : remaining_blocks = 26) :
  initial_blocks - remaining_blocks = 29 := by
  sorry

end blocks_eaten_l1484_148439


namespace trig_identity_l1484_148447

theorem trig_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10/3 := by
  sorry

end trig_identity_l1484_148447


namespace smallest_prime_divisor_of_sum_l1484_148461

theorem smallest_prime_divisor_of_sum (p : ℕ) : 
  Prime p ∧ p ∣ (2^12 + 3^10 + 7^15) → p = 2 :=
sorry

end smallest_prime_divisor_of_sum_l1484_148461


namespace kiwi_juice_blend_percentage_l1484_148491

/-- The amount of juice (in ounces) that can be extracted from one kiwi -/
def kiwi_juice : ℚ := 6 / 4

/-- The amount of juice (in ounces) that can be extracted from one apple -/
def apple_juice : ℚ := 10 / 3

/-- The number of kiwis used in the blend -/
def kiwis_in_blend : ℕ := 5

/-- The number of apples used in the blend -/
def apples_in_blend : ℕ := 4

/-- The percentage of kiwi juice in the blend -/
def kiwi_juice_percentage : ℚ := 
  (kiwi_juice * kiwis_in_blend) / 
  (kiwi_juice * kiwis_in_blend + apple_juice * apples_in_blend) * 100

theorem kiwi_juice_blend_percentage :
  kiwi_juice_percentage = 36 := by sorry

end kiwi_juice_blend_percentage_l1484_148491


namespace arithmetic_subsequence_l1484_148449

theorem arithmetic_subsequence (a : ℕ → ℝ) (d : ℝ) (h : ∀ n : ℕ, a (n + 1) = a n + d) :
  ∃ c : ℝ, ∀ k : ℕ+, a (3 * k - 1) = c + (k - 1) * (3 * d) :=
sorry

end arithmetic_subsequence_l1484_148449


namespace grid_property_l1484_148486

/-- Represents a 3x3 grid -/
structure Grid :=
  (cells : Matrix (Fin 3) (Fin 3) ℤ)

/-- Represents an operation on the grid -/
inductive Operation
  | add_adjacent : Fin 3 → Fin 3 → Fin 3 → Fin 3 → Operation
  | subtract_adjacent : Fin 3 → Fin 3 → Fin 3 → Fin 3 → Operation

/-- Applies an operation to a grid -/
def apply_operation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- The sum of all cells in a grid -/
def grid_sum (g : Grid) : ℤ :=
  sorry

/-- The difference between shaded and non-shaded cells -/
def shaded_difference (g : Grid) (shaded : Set (Fin 3 × Fin 3)) : ℤ :=
  sorry

/-- Theorem stating the property of the grid after operations -/
theorem grid_property (initial : Grid) (final : Grid) (ops : List Operation) 
    (shaded : Set (Fin 3 × Fin 3)) :
  (∀ op ∈ ops, grid_sum (apply_operation initial op) = grid_sum initial) →
  (∀ op ∈ ops, shaded_difference (apply_operation initial op) shaded = shaded_difference initial shaded) →
  (∃ A : ℤ, final.cells 0 0 = A ∧ 4 * 2010 + A - 4 * 2010 = 5) →
  final.cells 0 0 = 5 :=
by sorry

end grid_property_l1484_148486


namespace darcy_remaining_clothes_l1484_148469

def remaining_clothes_to_fold (total_shirts : ℕ) (total_shorts : ℕ) (folded_shirts : ℕ) (folded_shorts : ℕ) : ℕ :=
  (total_shirts + total_shorts) - (folded_shirts + folded_shorts)

theorem darcy_remaining_clothes : 
  remaining_clothes_to_fold 20 8 12 5 = 11 := by
  sorry

end darcy_remaining_clothes_l1484_148469


namespace at_least_two_equations_have_solution_l1484_148499

theorem at_least_two_equations_have_solution (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  ∃ (x y : ℝ), (
    ((x - a) * (x - b) = x - c ∧ (y - b) * (y - c) = y - a) ∨
    ((x - a) * (x - b) = x - c ∧ (y - c) * (y - a) = y - b) ∨
    ((x - b) * (x - c) = x - a ∧ (y - c) * (y - a) = y - b)
  ) :=
sorry

end at_least_two_equations_have_solution_l1484_148499


namespace sum_of_roots_equation_l1484_148422

theorem sum_of_roots_equation (x : ℝ) : 
  (10 = (x^3 - 5*x^2 - 10*x) / (x + 2)) → 
  (∃ (y z : ℝ), x + y + z = 5 ∧ 
    10 = (y^3 - 5*y^2 - 10*y) / (y + 2) ∧
    10 = (z^3 - 5*z^2 - 10*z) / (z + 2)) :=
by sorry

end sum_of_roots_equation_l1484_148422


namespace sum_of_roots_l1484_148407

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 1716 := by
sorry

end sum_of_roots_l1484_148407


namespace carina_coffee_amount_l1484_148454

/-- The amount of coffee Carina has, given the following conditions:
  * She has coffee divided into 5- and 10-ounce packages
  * She has 2 more 5-ounce packages than 10-ounce packages
  * She has 5 10-ounce packages
-/
theorem carina_coffee_amount :
  let num_10oz_packages : ℕ := 5
  let num_5oz_packages : ℕ := num_10oz_packages + 2
  let total_ounces : ℕ := num_10oz_packages * 10 + num_5oz_packages * 5
  total_ounces = 85 := by sorry

end carina_coffee_amount_l1484_148454


namespace quadratic_discriminant_l1484_148495

/-- The discriminant of the quadratic equation 2x^2 + (2 + 1/2)x + 1/2 is 9/4 -/
theorem quadratic_discriminant : 
  let a : ℚ := 2
  let b : ℚ := 2 + 1/2
  let c : ℚ := 1/2
  (b^2 - 4*a*c) = 9/4 := by
  sorry

end quadratic_discriminant_l1484_148495


namespace min_value_reciprocal_sum_l1484_148406

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 5) :
  (1 / (x + 2) + 1 / (y + 2)) ≥ 4 / 9 ∧
  ∃ x₀ y₀ : ℝ, x₀ > 0 ∧ y₀ > 0 ∧ x₀ + y₀ = 5 ∧ 1 / (x₀ + 2) + 1 / (y₀ + 2) = 4 / 9 :=
by sorry


end min_value_reciprocal_sum_l1484_148406


namespace smallest_sum_squared_pythagorean_triple_l1484_148475

theorem smallest_sum_squared_pythagorean_triple (p q r : ℤ) 
  (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (h : p^2 + q^2 = r^2) : 
  ∃ (p' q' r' : ℤ), p'^2 + q'^2 = r'^2 ∧ (p' + q' + r')^2 = 4 ∧ 
  ∀ (a b c : ℤ), a^2 + b^2 = c^2 → (a + b + c)^2 ≥ 4 :=
sorry

end smallest_sum_squared_pythagorean_triple_l1484_148475


namespace trapezoid_height_l1484_148425

/-- The height of a trapezoid with specific properties -/
theorem trapezoid_height (a b : ℝ) (h_ab : a < b) : ∃ h : ℝ,
  h = a * b / (b - a) ∧
  ∃ (AB CD : ℝ) (angle_diagonals angle_sides : ℝ),
    AB = a ∧
    CD = b ∧
    angle_diagonals = 90 ∧
    angle_sides = 45 ∧
    h > 0 :=
by sorry

end trapezoid_height_l1484_148425


namespace f_properties_l1484_148474

noncomputable def f (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

noncomputable def g (x : ℝ) : ℝ := f x - abs x

theorem f_properties :
  (∀ y ∈ Set.range f, 0 < y ∧ y < 4) ∧
  (∀ x, f x + f (-x) = 4) ∧
  (∃! a b, a < b ∧ g a = 0 ∧ g b = 0 ∧ ∀ x, x ≠ a ∧ x ≠ b → g x ≠ 0) :=
sorry

end f_properties_l1484_148474


namespace complex_number_equality_l1484_148408

theorem complex_number_equality (z : ℂ) : (1 - Complex.I) * z = Complex.abs (1 + Complex.I * Real.sqrt 3) → z = 1 + Complex.I := by
  sorry

end complex_number_equality_l1484_148408


namespace greatest_distance_between_circle_centers_l1484_148441

theorem greatest_distance_between_circle_centers
  (rectangle_width : ℝ) (rectangle_height : ℝ) (circle_diameter : ℝ)
  (hw : rectangle_width = 15)
  (hh : rectangle_height = 17)
  (hd : circle_diameter = 7) :
  let inner_width := rectangle_width - circle_diameter
  let inner_height := rectangle_height - circle_diameter
  Real.sqrt (inner_width ^ 2 + inner_height ^ 2) = Real.sqrt 164 :=
by sorry

end greatest_distance_between_circle_centers_l1484_148441


namespace simplify_fraction_product_l1484_148490

theorem simplify_fraction_product : (1 / (2 + Real.sqrt 3)) * (1 / (2 - Real.sqrt 3)) = 1 := by
  sorry

end simplify_fraction_product_l1484_148490


namespace arithmetic_sequence_sum_l1484_148476

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that in an arithmetic sequence where a₂ + 2a₆ + a₁₀ = 120, a₃ + a₉ = 60. -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h_sum : a 2 + 2 * a 6 + a 10 = 120) : a 3 + a 9 = 60 := by
  sorry


end arithmetic_sequence_sum_l1484_148476


namespace max_value_a_plus_b_l1484_148464

/-- Given that -1/4 * x^2 ≤ ax + b ≤ e^x for all x ∈ ℝ, the maximum value of a + b is 2 -/
theorem max_value_a_plus_b (a b : ℝ) : 
  (∀ x : ℝ, -1/4 * x^2 ≤ a * x + b ∧ a * x + b ≤ Real.exp x) → 
  (∀ c d : ℝ, (∀ x : ℝ, -1/4 * x^2 ≤ c * x + d ∧ c * x + d ≤ Real.exp x) → a + b ≥ c + d) →
  a + b = 2 := by
  sorry

end max_value_a_plus_b_l1484_148464


namespace representation_of_2019_representation_of_any_integer_l1484_148442

theorem representation_of_2019 : ∃ (a b c : ℤ), 2019 = a^2 + b^2 - c^2 := by sorry

theorem representation_of_any_integer : ∀ (n : ℤ), ∃ (a b c d : ℤ), n = a^2 + b^2 + c^2 - d^2 := by sorry

end representation_of_2019_representation_of_any_integer_l1484_148442


namespace inscribed_tetrahedron_volume_l1484_148444

/-- A regular tetrahedron inscribed in a unit sphere -/
structure InscribedTetrahedron where
  /-- The tetrahedron is regular -/
  regular : Bool
  /-- All vertices lie on the surface of the sphere -/
  vertices_on_sphere : Bool
  /-- The sphere has radius 1 -/
  sphere_radius : ℝ
  /-- Three vertices of the base are on a great circle of the sphere -/
  base_on_great_circle : Bool

/-- The volume of the inscribed regular tetrahedron -/
def tetrahedron_volume (t : InscribedTetrahedron) : ℝ :=
  sorry

/-- Theorem stating the volume of the inscribed regular tetrahedron -/
theorem inscribed_tetrahedron_volume 
  (t : InscribedTetrahedron) 
  (h1 : t.regular = true) 
  (h2 : t.vertices_on_sphere = true) 
  (h3 : t.sphere_radius = 1) 
  (h4 : t.base_on_great_circle = true) : 
  tetrahedron_volume t = Real.sqrt 3 / 4 :=
sorry

end inscribed_tetrahedron_volume_l1484_148444


namespace stairs_climbed_together_l1484_148433

/-- The number of stairs Samir climbed -/
def samir_stairs : ℕ := 318

/-- The number of stairs Veronica climbed -/
def veronica_stairs : ℕ := samir_stairs / 2 + 18

/-- The total number of stairs Samir and Veronica climbed together -/
def total_stairs : ℕ := samir_stairs + veronica_stairs

theorem stairs_climbed_together : total_stairs = 495 := by
  sorry

end stairs_climbed_together_l1484_148433


namespace diagonals_25_sided_polygon_l1484_148446

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 25 sides has 275 diagonals -/
theorem diagonals_25_sided_polygon : num_diagonals 25 = 275 := by
  sorry

end diagonals_25_sided_polygon_l1484_148446


namespace geometric_sequence_sum_inequality_l1484_148457

/-- Given a geometric sequence with positive terms and common ratio not equal to 1,
    prove that the sum of the first and fourth terms is greater than
    the sum of the second and third terms. -/
theorem geometric_sequence_sum_inequality
  (a : ℕ → ℝ)  -- The geometric sequence
  (q : ℝ)      -- The common ratio
  (h1 : ∀ n, a n > 0)  -- All terms are positive
  (h2 : q ≠ 1)  -- Common ratio is not 1
  (h3 : ∀ n, a (n + 1) = a n * q)  -- Definition of geometric sequence
  : a 1 + a 4 > a 2 + a 3 :=
by sorry

end geometric_sequence_sum_inequality_l1484_148457


namespace quadratic_sum_l1484_148412

/-- A quadratic function with vertex at (2, 8) and passing through (0, 0) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_sum (a b c : ℝ) :
  (∀ x, f a b c x = a * x^2 + b * x + c) →
  f a b c 2 = 8 →
  (∀ x, f a b c x ≤ f a b c 2) →
  f a b c 0 = 0 →
  a + b + 2*c = 6 := by sorry

end quadratic_sum_l1484_148412


namespace decimal_2015_is_octal_3737_l1484_148468

/-- Converts a natural number from decimal to octal representation -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 8) ((m % 8) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid octal number -/
def is_valid_octal (l : List ℕ) : Prop :=
  l.all (· < 8) ∧ l ≠ []

theorem decimal_2015_is_octal_3737 :
  decimal_to_octal 2015 = [3, 7, 3, 7] ∧ is_valid_octal [3, 7, 3, 7] := by
  sorry

#eval decimal_to_octal 2015

end decimal_2015_is_octal_3737_l1484_148468


namespace remaining_popsicle_sticks_l1484_148426

theorem remaining_popsicle_sticks 
  (initial : ℝ) 
  (given_to_lisa : ℝ) 
  (given_to_peter : ℝ) 
  (given_to_you : ℝ) 
  (h1 : initial = 63.5) 
  (h2 : given_to_lisa = 18.2) 
  (h3 : given_to_peter = 21.7) 
  (h4 : given_to_you = 10.1) : 
  initial - (given_to_lisa + given_to_peter + given_to_you) = 13.5 := by
sorry

end remaining_popsicle_sticks_l1484_148426


namespace digits_of_2_pow_100_l1484_148473

theorem digits_of_2_pow_100 (N : ℕ) :
  (N = (Nat.digits 10 (2^100)).length) → 29 ≤ N ∧ N ≤ 34 := by
  sorry

end digits_of_2_pow_100_l1484_148473
