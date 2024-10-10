import Mathlib

namespace louis_fabric_purchase_l4134_413496

theorem louis_fabric_purchase (fabric_cost_per_yard : ℝ) (pattern_cost : ℝ) (thread_cost_per_spool : ℝ) (total_spent : ℝ) : 
  fabric_cost_per_yard = 24 →
  pattern_cost = 15 →
  thread_cost_per_spool = 3 →
  total_spent = 141 →
  (total_spent - pattern_cost - 2 * thread_cost_per_spool) / fabric_cost_per_yard = 5 :=
by
  sorry

end louis_fabric_purchase_l4134_413496


namespace fiona_reach_probability_l4134_413452

/-- Represents a lily pad with its number and whether it contains a predator -/
structure LilyPad :=
  (number : Nat)
  (hasPredator : Bool)

/-- Represents Fiona's possible moves -/
inductive Move
  | Hop
  | Jump

/-- Represents the frog's journey -/
def FrogJourney := List LilyPad

/-- The probability of each move -/
def moveProbability : Move → ℚ
  | Move.Hop => 1/2
  | Move.Jump => 1/2

/-- The number of pads to move for each move type -/
def moveDistance : Move → Nat
  | Move.Hop => 1
  | Move.Jump => 2

/-- The lily pads in the pond -/
def lilyPads : List LilyPad :=
  List.range 16 |> List.map (λ n => ⟨n, n ∈ [4, 7, 11]⟩)

/-- Check if a journey is safe (doesn't land on predator pads) -/
def isSafeJourney (journey : FrogJourney) : Bool :=
  journey.all (λ pad => !pad.hasPredator)

/-- Calculate the probability of a specific journey -/
def journeyProbability (journey : FrogJourney) : ℚ :=
  sorry

/-- The theorem to prove -/
theorem fiona_reach_probability :
  ∃ (safeJourneys : List FrogJourney),
    (∀ j ∈ safeJourneys, j.head? = some ⟨0, false⟩ ∧
                         j.getLast? = some ⟨14, false⟩ ∧
                         isSafeJourney j) ∧
    (safeJourneys.map journeyProbability).sum = 3/256 :=
  sorry

end fiona_reach_probability_l4134_413452


namespace calculate_expression_l4134_413437

theorem calculate_expression : 125^2 - 50 * 125 + 25^2 = 10000 := by
  sorry

end calculate_expression_l4134_413437


namespace sandwiches_sold_out_l4134_413401

theorem sandwiches_sold_out (original : ℕ) (available : ℕ) (h1 : original = 9) (h2 : available = 4) :
  original - available = 5 := by
sorry

end sandwiches_sold_out_l4134_413401


namespace smallest_angle_divisible_isosceles_l4134_413405

/-- An isosceles triangle that can be divided into two isosceles triangles -/
structure DivisibleIsoscelesTriangle where
  /-- The measure of one of the equal angles in the original isosceles triangle -/
  α : Real
  /-- The triangle is isosceles -/
  isIsosceles : α ≥ 0 ∧ α ≤ 90
  /-- The triangle can be divided into two isosceles triangles -/
  isDivisible : ∃ (β γ : Real), (β > 0 ∧ γ > 0) ∧ 
    ((β = α ∧ γ = (180 - α) / 2) ∨ (β = (180 - α) / 2 ∧ γ = (3 * α - 180) / 2))

/-- The smallest angle in a divisible isosceles triangle is 180/7 degrees -/
theorem smallest_angle_divisible_isosceles (t : DivisibleIsoscelesTriangle) :
  min t.α (180 - 2 * t.α) ≥ 180 / 7 ∧ 
  ∃ (t' : DivisibleIsoscelesTriangle), min t'.α (180 - 2 * t'.α) = 180 / 7 :=
sorry

end smallest_angle_divisible_isosceles_l4134_413405


namespace no_solution_condition_l4134_413498

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, x ≠ 3 → (x / (x - 3) + (3 * a) / (3 - x) ≠ 2 * a)) ↔ (a = 1 ∨ a = 1/2) := by
  sorry

end no_solution_condition_l4134_413498


namespace intersection_A_B_union_A_B_complement_intersection_A_B_in_U_l4134_413400

-- Define the sets A, B, and U
def A : Set ℝ := {x : ℝ | -4 ≤ x ∧ x ≤ -2}
def B : Set ℝ := {x : ℝ | x + 3 ≥ 0}
def U : Set ℝ := {x : ℝ | x ≤ -1}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {x : ℝ | -3 ≤ x ∧ x ≤ -2} := by sorry

-- Theorem for the union of A and B
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ -4} := by sorry

-- Theorem for the complement of A ∩ B in U
theorem complement_intersection_A_B_in_U : (A ∩ B)ᶜ ∩ U = {x : ℝ | x < -3 ∨ (-2 < x ∧ x ≤ -1)} := by sorry

end intersection_A_B_union_A_B_complement_intersection_A_B_in_U_l4134_413400


namespace right_triangle_inequality_l4134_413416

theorem right_triangle_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : a + b ≤ c * Real.sqrt 2 := by
  sorry

end right_triangle_inequality_l4134_413416


namespace mary_seashells_l4134_413411

theorem mary_seashells (sam_seashells : ℕ) (total_seashells : ℕ) 
  (h1 : sam_seashells = 18) 
  (h2 : total_seashells = 65) : 
  total_seashells - sam_seashells = 47 := by
  sorry

end mary_seashells_l4134_413411


namespace xw_value_l4134_413499

/-- Triangle XYZ with point W on YZ such that XW is perpendicular to YZ -/
structure TriangleXYZW where
  /-- The length of side XY -/
  XY : ℝ
  /-- The length of side XZ -/
  XZ : ℝ
  /-- The length of XW, where W is on YZ and XW ⟂ YZ -/
  XW : ℝ
  /-- The length of YW -/
  YW : ℝ
  /-- The length of ZW -/
  ZW : ℝ
  /-- XY equals 15 -/
  xy_eq : XY = 15
  /-- XZ equals 26 -/
  xz_eq : XZ = 26
  /-- YW:ZW ratio is 3:4 -/
  yw_zw_ratio : YW / ZW = 3 / 4
  /-- Pythagorean theorem for XYW -/
  pythagoras_xyw : YW ^ 2 = XY ^ 2 - XW ^ 2
  /-- Pythagorean theorem for XZW -/
  pythagoras_xzw : ZW ^ 2 = XZ ^ 2 - XW ^ 2

/-- The main theorem: If the conditions are met, then XW = 42/√7 -/
theorem xw_value (t : TriangleXYZW) : t.XW = 42 / Real.sqrt 7 := by
  sorry


end xw_value_l4134_413499


namespace hotel_room_charges_l4134_413480

theorem hotel_room_charges (P R G : ℝ) 
  (h1 : P = R * (1 - 0.5)) 
  (h2 : P = G * (1 - 0.2)) : 
  R = G * (1 + 0.6) := by
sorry

end hotel_room_charges_l4134_413480


namespace disease_mortality_percentage_l4134_413449

theorem disease_mortality_percentage (population : ℝ) 
  (h1 : population > 0) 
  (affected_percentage : ℝ) 
  (h2 : affected_percentage = 15) 
  (death_percentage : ℝ) 
  (h3 : death_percentage = 8) : 
  (affected_percentage / 100) * (death_percentage / 100) * 100 = 1.2 := by
  sorry

end disease_mortality_percentage_l4134_413449


namespace inequality_proof_l4134_413478

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_inequality : a * b * c ≤ a + b + c) : 
  a^2 + b^2 + c^2 ≥ Real.sqrt 3 * a * b * c := by
  sorry

end inequality_proof_l4134_413478


namespace triangle_abc_proof_l4134_413421

theorem triangle_abc_proof (a b c : ℝ) (A B C : ℝ) :
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  Real.sqrt 3 * a * Real.cos C - c * Real.sin A = 0 →
  b = 6 →
  1/2 * a * b * Real.sin C = 6 * Real.sqrt 3 →
  C = π/3 ∧ c = 2 * Real.sqrt 7 := by
  sorry

end triangle_abc_proof_l4134_413421


namespace small_tile_position_l4134_413434

/-- Represents a position on the 7x7 grid -/
structure Position :=
  (x : Fin 7)
  (y : Fin 7)

/-- Represents a 1x3 tile on the grid -/
structure Tile :=
  (start : Position)
  (horizontal : Bool)

/-- The configuration of the 7x7 grid -/
structure GridConfig :=
  (tiles : Finset Tile)
  (small_tile : Position)

/-- Checks if a position is at the center or adjoins a boundary -/
def is_center_or_boundary (p : Position) : Prop :=
  p.x = 0 ∨ p.x = 3 ∨ p.x = 6 ∨ p.y = 0 ∨ p.y = 3 ∨ p.y = 6

/-- Checks if the configuration is valid -/
def is_valid_config (config : GridConfig) : Prop :=
  config.tiles.card = 16 ∧
  ∀ t ∈ config.tiles, (t.horizontal → t.start.y < 6) ∧
                      (¬t.horizontal → t.start.x < 6)

theorem small_tile_position (config : GridConfig) 
  (h : is_valid_config config) :
  is_center_or_boundary config.small_tile :=
sorry

end small_tile_position_l4134_413434


namespace ellipse_angle_tangent_product_l4134_413423

/-- Given an ellipse with eccentricity e and a point P on the ellipse,
    if α is the angle PF₁F₂ and β is the angle PF₂F₁, where F₁ and F₂ are the foci,
    then tan(α/2) * tan(β/2) = (1 - e) / (1 + e) -/
theorem ellipse_angle_tangent_product (a b : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) (α β e : ℝ)
  (h_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (h_foci : F₁ ≠ F₂)
  (h_eccentricity : e = Real.sqrt (a^2 - b^2) / a)
  (h_angle_α : α = Real.arccos ((P.1 - F₁.1) * (F₂.1 - F₁.1) + (P.2 - F₁.2) * (F₂.2 - F₁.2)) /
    (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) * Real.sqrt ((F₂.1 - F₁.1)^2 + (F₂.2 - F₁.2)^2)))
  (h_angle_β : β = Real.arccos ((P.1 - F₂.1) * (F₁.1 - F₂.1) + (P.2 - F₂.2) * (F₁.2 - F₂.2)) /
    (Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) * Real.sqrt ((F₁.1 - F₂.1)^2 + (F₁.2 - F₂.2)^2)))
  : Real.tan (α/2) * Real.tan (β/2) = (1 - e) / (1 + e) := by
  sorry

end ellipse_angle_tangent_product_l4134_413423


namespace movie_ticket_final_price_l4134_413417

def movie_ticket_price (initial_price : ℝ) (year1_increase year2_decrease year3_increase year4_decrease year5_increase tax discount : ℝ) : ℝ :=
  let price1 := initial_price * (1 + year1_increase)
  let price2 := price1 * (1 - year2_decrease)
  let price3 := price2 * (1 + year3_increase)
  let price4 := price3 * (1 - year4_decrease)
  let price5 := price4 * (1 + year5_increase)
  let price_with_tax := price5 * (1 + tax)
  price_with_tax * (1 - discount)

theorem movie_ticket_final_price :
  let initial_price : ℝ := 100
  let year1_increase : ℝ := 0.12
  let year2_decrease : ℝ := 0.05
  let year3_increase : ℝ := 0.08
  let year4_decrease : ℝ := 0.04
  let year5_increase : ℝ := 0.06
  let tax : ℝ := 0.07
  let discount : ℝ := 0.10
  ∃ ε > 0, |movie_ticket_price initial_price year1_increase year2_decrease year3_increase year4_decrease year5_increase tax discount - 112.61| < ε :=
sorry

end movie_ticket_final_price_l4134_413417


namespace ny_mets_fans_l4134_413445

/-- The number of NY Mets fans in a town with specific fan ratios --/
theorem ny_mets_fans (total : ℕ) (y m r d : ℚ) : 
  total = 780 →
  y / m = 3 / 2 →
  m / r = 4 / 5 →
  r / d = 7 / (3/2) →
  y + m + r + d = total →
  ⌊m⌋ = 178 := by
  sorry

end ny_mets_fans_l4134_413445


namespace remainder_problem_l4134_413475

theorem remainder_problem (n : ℤ) : 
  (n % 4 = 3) → (n % 9 = 5) → (n % 36 = 23) := by
sorry

end remainder_problem_l4134_413475


namespace distance_between_points_l4134_413462

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (3, 6)
  let p2 : ℝ × ℝ := (-7, -2)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 41 :=
by sorry

end distance_between_points_l4134_413462


namespace modulus_of_z_equals_sqrt2_over_2_l4134_413436

/-- The modulus of the complex number z = i / (1 - i) is equal to √2/2 -/
theorem modulus_of_z_equals_sqrt2_over_2 : 
  let z : ℂ := Complex.I / (1 - Complex.I)
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end modulus_of_z_equals_sqrt2_over_2_l4134_413436


namespace simplification_proof_l4134_413481

theorem simplification_proof (a b : ℝ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : 3*a - b/3 ≠ 0) :
  (3*a - b/3)⁻¹ * ((3*a)⁻¹ - (b/3)⁻¹) = -(a*b)⁻¹ := by
  sorry

end simplification_proof_l4134_413481


namespace min_value_expression_l4134_413428

theorem min_value_expression (x y z : ℝ) 
  (hx : 0 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 2) (hz : 0 ≤ z ∧ z ≤ 2) : 
  1/((2 - x)*(2 - y)*(2 - z)) + 1/((2 + x)*(2 + y)*(2 + z)) + 1/(1 + (x+y+z)/3) ≥ 2 ∧
  (x = 0 ∧ y = 0 ∧ z = 0 → 1/((2 - x)*(2 - y)*(2 - z)) + 1/((2 + x)*(2 + y)*(2 + z)) + 1/(1 + (x+y+z)/3) = 2) :=
by sorry

end min_value_expression_l4134_413428


namespace mari_made_79_buttons_l4134_413485

/-- The number of buttons made by each person -/
structure ButtonCounts where
  kendra : ℕ
  mari : ℕ
  sue : ℕ
  jess : ℕ
  tom : ℕ

/-- The conditions of the button-making problem -/
def ButtonProblem (counts : ButtonCounts) : Prop :=
  counts.kendra = 15 ∧
  counts.mari = 5 * counts.kendra + 4 ∧
  counts.sue = 2 * counts.kendra / 3 ∧
  counts.jess = 2 * (counts.sue + counts.kendra) ∧
  counts.tom = 3 * counts.jess / 4

/-- Mari made 79 buttons -/
theorem mari_made_79_buttons (counts : ButtonCounts) 
  (h : ButtonProblem counts) : counts.mari = 79 := by
  sorry

end mari_made_79_buttons_l4134_413485


namespace inverse_relationship_scenarios_l4134_413492

/-- Represents a scenario with two variables that may have an inverse relationship -/
structure Scenario where
  x : ℝ
  y : ℝ
  k : ℝ
  h_k_nonzero : k ≠ 0

/-- Checks if a scenario satisfies the inverse relationship y = k/x -/
def has_inverse_relationship (s : Scenario) : Prop :=
  s.y = s.k / s.x

/-- Rectangle scenario with fixed area -/
def rectangle_scenario (area x y : ℝ) (h : area ≠ 0) : Scenario where
  x := x
  y := y
  k := area
  h_k_nonzero := h

/-- Village land scenario with fixed total arable land -/
def village_land_scenario (total_land n S : ℝ) (h : total_land ≠ 0) : Scenario where
  x := n
  y := S
  k := total_land
  h_k_nonzero := h

/-- Car travel scenario with fixed speed -/
def car_travel_scenario (speed s t : ℝ) (h : speed ≠ 0) : Scenario where
  x := t
  y := s
  k := speed
  h_k_nonzero := h

theorem inverse_relationship_scenarios 
  (rect : Scenario) 
  (village : Scenario) 
  (car : Scenario) : 
  has_inverse_relationship rect ∧ 
  has_inverse_relationship village ∧ 
  ¬has_inverse_relationship car := by
  sorry

end inverse_relationship_scenarios_l4134_413492


namespace total_cookies_calculation_l4134_413403

/-- The number of cookies Kristy baked -/
def total_cookies : ℕ := sorry

/-- The number of cookies Kristy ate -/
def kristy_ate : ℕ := 3

/-- The number of cookies Kristy's brother took -/
def brother_took : ℕ := 2

/-- The number of cookies the first friend took -/
def first_friend_took : ℕ := 4

/-- The number of cookies the second friend took (net) -/
def second_friend_took : ℕ := 4

/-- The number of cookies the third friend took -/
def third_friend_took : ℕ := 8

/-- The number of cookies the fourth friend took -/
def fourth_friend_took : ℕ := 3

/-- The number of cookies the fifth friend took -/
def fifth_friend_took : ℕ := 7

/-- The number of cookies left -/
def cookies_left : ℕ := 5

/-- Theorem stating that the total number of cookies is equal to the sum of all distributed cookies and the remaining cookies -/
theorem total_cookies_calculation :
  total_cookies = kristy_ate + brother_took + first_friend_took + second_friend_took +
                  third_friend_took + fourth_friend_took + fifth_friend_took + cookies_left :=
by sorry

end total_cookies_calculation_l4134_413403


namespace calligraphy_book_characters_l4134_413415

/-- The number of characters written per day in the first practice -/
def first_practice_chars_per_day : ℕ := 25

/-- The additional characters written per day in the second practice -/
def additional_chars_per_day : ℕ := 3

/-- The number of days fewer in the second practice compared to the first -/
def days_difference : ℕ := 3

/-- The total number of characters in the book -/
def total_characters : ℕ := 700

theorem calligraphy_book_characters :
  ∃ (x : ℕ), 
    x > days_difference ∧
    first_practice_chars_per_day * x = 
    (first_practice_chars_per_day + additional_chars_per_day) * (x - days_difference) ∧
    total_characters = first_practice_chars_per_day * x :=
by sorry

end calligraphy_book_characters_l4134_413415


namespace four_five_equality_and_precision_l4134_413413

/-- Represents a decimal number with its value and precision -/
structure Decimal where
  value : ℚ
  precision : ℕ

/-- 4.5 as a Decimal -/
def d1 : Decimal := { value := 4.5, precision := 1 }

/-- 4.50 as a Decimal -/
def d2 : Decimal := { value := 4.5, precision := 2 }

/-- Two Decimals are equal in magnitude if their values are equal -/
def equal_magnitude (a b : Decimal) : Prop := a.value = b.value

/-- Two Decimals differ in precision if their precisions are different -/
def differ_precision (a b : Decimal) : Prop := a.precision ≠ b.precision

/-- Theorem stating that 4.5 and 4.50 are equal in magnitude but differ in precision -/
theorem four_five_equality_and_precision : 
  equal_magnitude d1 d2 ∧ differ_precision d1 d2 := by
  sorry

end four_five_equality_and_precision_l4134_413413


namespace quadratic_two_distinct_roots_l4134_413458

theorem quadratic_two_distinct_roots (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ (a - 1) * x^2 - 2 * x + 1 = 0 ∧ (a - 1) * y^2 - 2 * y + 1 = 0) ↔
  (a < 2 ∧ a ≠ 1) :=
sorry

end quadratic_two_distinct_roots_l4134_413458


namespace cookie_fraction_with_nuts_l4134_413491

theorem cookie_fraction_with_nuts 
  (total_cookies : ℕ) 
  (nuts_per_cookie : ℕ) 
  (total_nuts : ℕ) 
  (h1 : total_cookies = 60) 
  (h2 : nuts_per_cookie = 2) 
  (h3 : total_nuts = 72) : 
  (total_nuts / nuts_per_cookie : ℚ) / total_cookies = 3/5 := by
  sorry

end cookie_fraction_with_nuts_l4134_413491


namespace h_zero_at_seven_fifths_l4134_413463

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: The value of b that satisfies h(b) = 0 is b = 7/5 -/
theorem h_zero_at_seven_fifths : 
  ∃ b : ℝ, h b = 0 ∧ b = 7/5 := by sorry

end h_zero_at_seven_fifths_l4134_413463


namespace football_game_attendance_l4134_413447

/-- Proves that the number of children attending a football game is 80, given the ticket prices, total attendance, and total revenue. -/
theorem football_game_attendance
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_attendance : ℕ)
  (total_revenue : ℕ)
  (h1 : adult_price = 60)
  (h2 : child_price = 25)
  (h3 : total_attendance = 280)
  (h4 : total_revenue = 14000)
  : ∃ (adults children : ℕ),
    adults + children = total_attendance ∧
    adult_price * adults + child_price * children = total_revenue ∧
    children = 80 := by
  sorry

end football_game_attendance_l4134_413447


namespace shaded_area_calculation_l4134_413489

theorem shaded_area_calculation (π : Real) :
  let semicircle_area := π * 2^2 / 2
  let quarter_circle_area := π * 1^2 / 4
  semicircle_area - 2 * quarter_circle_area = 3 * π / 2 := by
  sorry

end shaded_area_calculation_l4134_413489


namespace fraction_well_defined_at_negative_one_l4134_413435

theorem fraction_well_defined_at_negative_one :
  ∀ x : ℝ, x = -1 → (x^2 + 1 ≠ 0) := by sorry

end fraction_well_defined_at_negative_one_l4134_413435


namespace toy_production_lot_l4134_413441

theorem toy_production_lot (total : ℕ) 
  (h_red : total * 2 / 5 = total * 40 / 100)
  (h_small : total / 2 = total * 50 / 100)
  (h_red_small : total / 10 = total * 10 / 100)
  (h_red_large : total * 3 / 10 = 60) :
  total * 2 / 5 = 40 := by
  sorry

end toy_production_lot_l4134_413441


namespace existence_of_larger_element_l4134_413457

/-- A doubly infinite array of positive integers -/
def InfiniteArray := ℕ+ → ℕ+ → ℕ+

/-- The property that each positive integer appears exactly eight times in the array -/
def EightOccurrences (a : InfiniteArray) : Prop :=
  ∀ n : ℕ+, (∃ (s : Finset (ℕ+ × ℕ+)), s.card = 8 ∧ (∀ (p : ℕ+ × ℕ+), p ∈ s ↔ a p.1 p.2 = n))

/-- The main theorem -/
theorem existence_of_larger_element (a : InfiniteArray) (h : EightOccurrences a) :
  ∃ (m n : ℕ+), a m n > m * n := by
  sorry

end existence_of_larger_element_l4134_413457


namespace cubic_sum_equals_twenty_l4134_413439

theorem cubic_sum_equals_twenty (x y z : ℝ) (h1 : x = y + z) (h2 : x = 2) :
  x^3 + 3*y^2 + 3*z^2 + 3*x*y*z = 20 := by
sorry

end cubic_sum_equals_twenty_l4134_413439


namespace circle_max_cube_root_sum_l4134_413497

theorem circle_max_cube_root_sum (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  ∀ a b : ℝ, a^2 + b^2 = 1 → 
  Real.sqrt (|x|^3 + |y|^3) ≤ Real.sqrt (2 * Real.sqrt 2 + 1) / Real.sqrt 3 :=
by sorry

end circle_max_cube_root_sum_l4134_413497


namespace biff_ticket_cost_l4134_413426

/-- Represents the cost of Biff's bus ticket in dollars -/
def ticket_cost : ℝ := 11

/-- Represents the cost of drinks and snacks in dollars -/
def snacks_cost : ℝ := 3

/-- Represents the cost of headphones in dollars -/
def headphones_cost : ℝ := 16

/-- Represents Biff's hourly rate for online work in dollars per hour -/
def online_rate : ℝ := 12

/-- Represents the hourly cost of WiFi access in dollars per hour -/
def wifi_cost : ℝ := 2

/-- Represents the duration of the bus trip in hours -/
def trip_duration : ℝ := 3

theorem biff_ticket_cost :
  ticket_cost + snacks_cost + headphones_cost + wifi_cost * trip_duration =
  online_rate * trip_duration :=
by sorry

end biff_ticket_cost_l4134_413426


namespace min_value_theorem_l4134_413465

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 ∧ 
  ∃ y : ℝ, (y^2 + 9) / Real.sqrt (y^2 + 5) = 4 := by
sorry

end min_value_theorem_l4134_413465


namespace smallest_base_for_78_l4134_413477

theorem smallest_base_for_78 :
  ∃ (b : ℕ), b > 0 ∧ b^2 ≤ 78 ∧ 78 < b^3 ∧ ∀ (x : ℕ), x > 0 ∧ x^2 ≤ 78 ∧ 78 < x^3 → b ≤ x :=
by sorry

end smallest_base_for_78_l4134_413477


namespace problem_solution_l4134_413494

theorem problem_solution :
  (∀ x : ℝ, x^2 = 0 → x = 0) ∧
  (∀ x : ℝ, x^2 < 2*x → x > 0) ∧
  (∀ x : ℝ, x > 2 → x^2 > x) :=
by sorry

end problem_solution_l4134_413494


namespace milk_water_ratio_in_combined_mixture_l4134_413407

/-- Represents a mixture of milk and water -/
structure Mixture where
  milk : ℚ
  water : ℚ

/-- Calculates the ratio of milk to water in a mixture -/
def ratioMilkToWater (m : Mixture) : ℚ × ℚ :=
  (m.milk, m.water)

/-- Combines multiple mixtures into a single mixture -/
def combineMixtures (mixtures : List Mixture) : Mixture :=
  { milk := mixtures.map (·.milk) |>.sum,
    water := mixtures.map (·.water) |>.sum }

theorem milk_water_ratio_in_combined_mixture :
  let m1 := Mixture.mk (7 : ℚ) (2 : ℚ)
  let m2 := Mixture.mk (8 : ℚ) (1 : ℚ)
  let m3 := Mixture.mk (9 : ℚ) (3 : ℚ)
  let combined := combineMixtures [m1, m2, m3]
  ratioMilkToWater combined = (29, 7) := by
  sorry

#check milk_water_ratio_in_combined_mixture

end milk_water_ratio_in_combined_mixture_l4134_413407


namespace perpendicular_slope_to_OA_l4134_413422

/-- Given point A(3, 5) and O as the origin, prove that the slope of the line perpendicular to OA is -3/5 -/
theorem perpendicular_slope_to_OA :
  let A : ℝ × ℝ := (3, 5)
  let O : ℝ × ℝ := (0, 0)
  let slope_OA : ℝ := (A.2 - O.2) / (A.1 - O.1)
  let slope_perpendicular : ℝ := -1 / slope_OA
  slope_perpendicular = -3/5 := by sorry

end perpendicular_slope_to_OA_l4134_413422


namespace abs_neg_2023_l4134_413433

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end abs_neg_2023_l4134_413433


namespace x_minus_y_equals_eight_l4134_413431

theorem x_minus_y_equals_eight (x y : ℝ) 
  (h1 : |x| + x - y = 14) 
  (h2 : x + |y| + y = 6) : 
  x - y = 8 := by
sorry

end x_minus_y_equals_eight_l4134_413431


namespace equivalent_angle_for_negative_463_l4134_413448

-- Define the angle equivalence relation
def angle_equivalent (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

-- State the theorem
theorem equivalent_angle_for_negative_463 :
  ∀ k : ℤ, angle_equivalent (-463) (k * 360 + 257) :=
by sorry

end equivalent_angle_for_negative_463_l4134_413448


namespace tuesday_sales_l4134_413484

theorem tuesday_sales (initial_stock : ℕ) (monday_sales wednesday_sales thursday_sales friday_sales : ℕ)
  (unsold_percentage : ℚ) :
  initial_stock = 700 →
  monday_sales = 50 →
  wednesday_sales = 60 →
  thursday_sales = 48 →
  friday_sales = 40 →
  unsold_percentage = 60 / 100 →
  ∃ (tuesday_sales : ℕ),
    tuesday_sales = 82 ∧
    (initial_stock : ℚ) * (1 - unsold_percentage) =
      monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_sales :=
by sorry

end tuesday_sales_l4134_413484


namespace pen_count_l4134_413430

theorem pen_count (initial : ℕ) (received : ℕ) (given_away : ℕ) : 
  initial = 20 → received = 22 → given_away = 19 → 
  ((initial + received) * 2 - given_away) = 65 := by
sorry

end pen_count_l4134_413430


namespace jasmine_concentration_proof_l4134_413425

/-- Proves that adding 5 liters of jasmine and 15 liters of water to an 80-liter solution
    with 10% jasmine results in a new solution with 13% jasmine concentration. -/
theorem jasmine_concentration_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (added_jasmine : ℝ) (added_water : ℝ) (final_concentration : ℝ) : 
  initial_volume = 80 →
  initial_concentration = 0.10 →
  added_jasmine = 5 →
  added_water = 15 →
  final_concentration = 0.13 →
  (initial_volume * initial_concentration + added_jasmine) / 
  (initial_volume + added_jasmine + added_water) = final_concentration :=
by sorry

end jasmine_concentration_proof_l4134_413425


namespace geometric_sequence_minimum_l4134_413442

theorem geometric_sequence_minimum (b₁ b₂ b₃ : ℝ) : 
  b₁ = 1 → (∃ r : ℝ, b₂ = b₁ * r ∧ b₃ = b₂ * r) → 
  (∀ b₂' b₃' : ℝ, (∃ r' : ℝ, b₂' = b₁ * r' ∧ b₃' = b₂' * r') → 
    3 * b₂ + 4 * b₃ ≤ 3 * b₂' + 4 * b₃') →
  3 * b₂ + 4 * b₃ = -9/16 :=
by sorry

end geometric_sequence_minimum_l4134_413442


namespace car_trading_theorem_l4134_413461

/-- Represents the profit and purchase constraints for a car trading company. -/
structure CarTrading where
  profit_A_2_B_5 : ℕ  -- Profit from selling 2 A and 5 B
  profit_A_1_B_2 : ℕ  -- Profit from selling 1 A and 2 B
  price_A : ℕ         -- Purchase price of model A
  price_B : ℕ         -- Purchase price of model B
  total_budget : ℕ    -- Total budget
  total_units : ℕ     -- Total number of cars to purchase

/-- Theorem stating the profit per unit and minimum purchase of model A -/
theorem car_trading_theorem (ct : CarTrading) 
  (h1 : ct.profit_A_2_B_5 = 31000)
  (h2 : ct.profit_A_1_B_2 = 13000)
  (h3 : ct.price_A = 120000)
  (h4 : ct.price_B = 150000)
  (h5 : ct.total_budget = 3000000)
  (h6 : ct.total_units = 22) :
  ∃ (profit_A profit_B min_A : ℕ),
    profit_A = 3000 ∧
    profit_B = 5000 ∧
    min_A = 10 ∧
    2 * profit_A + 5 * profit_B = ct.profit_A_2_B_5 ∧
    profit_A + 2 * profit_B = ct.profit_A_1_B_2 ∧
    min_A * ct.price_A + (ct.total_units - min_A) * ct.price_B ≤ ct.total_budget :=
by sorry

end car_trading_theorem_l4134_413461


namespace equation_solution_l4134_413408

theorem equation_solution (x : ℂ) (h1 : x ≠ -2) (h2 : x ≠ 3) :
  (3*x - 6) / (x + 2) + (3*x^2 - 12) / (3 - x) = 3 ↔ x = -2 + 2*I ∨ x = -2 - 2*I :=
sorry

end equation_solution_l4134_413408


namespace secant_min_value_l4134_413412

/-- The secant function -/
noncomputable def sec (x : ℝ) : ℝ := 1 / Real.cos x

/-- The function y = a sec(bx) -/
noncomputable def f (a b x : ℝ) : ℝ := a * sec (b * x)

theorem secant_min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b x ≥ a) ∧ (∃ x, f a b x = a) →
  (∀ x, f a b x ≥ 3) ∧ (∃ x, f a b x = 3) →
  a = 3 :=
sorry

end secant_min_value_l4134_413412


namespace remainder_problem_l4134_413471

theorem remainder_problem : (8 * 7^19 + 1^19) % 9 = 3 := by
  sorry

end remainder_problem_l4134_413471


namespace exists_non_intersecting_circle_l4134_413483

-- Define the circular billiard table
def CircularBilliardTable := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ 1}

-- Define a trajectory of the ball
def Trajectory := Set (ℝ × ℝ)

-- Define the property of a trajectory following the laws of reflection
def FollowsReflectionLaws (t : Trajectory) : Prop := sorry

-- Define a circle inside the table
def InsideCircle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 ≤ radius^2 ∧ p ∈ CircularBilliardTable}

-- The main theorem
theorem exists_non_intersecting_circle :
  ∀ (start : ℝ × ℝ) (t : Trajectory),
    start ∈ CircularBilliardTable →
    FollowsReflectionLaws t →
    ∃ (center : ℝ × ℝ) (radius : ℝ),
      InsideCircle center radius ⊆ CircularBilliardTable ∧
      (InsideCircle center radius ∩ t = ∅) :=
by
  sorry

end exists_non_intersecting_circle_l4134_413483


namespace magic_king_episodes_l4134_413410

/-- Calculates the total number of episodes for a TV show with the given parameters -/
def total_episodes (total_seasons : ℕ) (episodes_first_half : ℕ) (episodes_second_half : ℕ) : ℕ :=
  let half_seasons := total_seasons / 2
  half_seasons * episodes_first_half + half_seasons * episodes_second_half

/-- Theorem stating that a show with 10 seasons, 20 episodes per season in the first half,
    and 25 episodes per season in the second half, has a total of 225 episodes -/
theorem magic_king_episodes :
  total_episodes 10 20 25 = 225 := by
  sorry

#eval total_episodes 10 20 25

end magic_king_episodes_l4134_413410


namespace cousin_distribution_count_l4134_413424

/-- The number of ways to distribute n indistinguishable objects into k distinguishable containers -/
def distribution_count (n k : ℕ) : ℕ :=
  sorry

/-- The number of cousins -/
def num_cousins : ℕ := 5

/-- The number of rooms -/
def num_rooms : ℕ := 5

/-- Theorem stating that the number of ways to distribute 5 cousins into 5 rooms is 37 -/
theorem cousin_distribution_count :
  distribution_count num_cousins num_rooms = 37 := by sorry

end cousin_distribution_count_l4134_413424


namespace sandwiches_available_l4134_413482

def initial_sandwiches : ℕ := 23
def sold_out_sandwiches : ℕ := 14

theorem sandwiches_available : initial_sandwiches - sold_out_sandwiches = 9 := by
  sorry

end sandwiches_available_l4134_413482


namespace max_sine_cosine_function_l4134_413470

theorem max_sine_cosine_function (a b : ℝ) :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x ≤ 4) ∧
  (a * Real.sin (π/3) + b * Real.cos (π/3) = 4) →
  a / b = Real.sqrt 3 := by
sorry

end max_sine_cosine_function_l4134_413470


namespace range_of_k_l4134_413418

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 2}
def B (k : ℝ) : Set ℝ := {x : ℝ | 2*k - 1 ≤ x ∧ x ≤ 2*k + 1}

-- State the theorem
theorem range_of_k (k : ℝ) : A ∩ B k = B k → -1 ≤ k ∧ k ≤ 1/2 := by
  sorry

end range_of_k_l4134_413418


namespace sum_of_parts_zero_l4134_413404

theorem sum_of_parts_zero : 
  let z : ℂ := (3 - Complex.I) / (2 + Complex.I)
  (z.re + z.im) = 0 := by
  sorry

end sum_of_parts_zero_l4134_413404


namespace complex_arithmetic_calculation_l4134_413453

theorem complex_arithmetic_calculation : 
  3 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 2400 := by
  sorry

end complex_arithmetic_calculation_l4134_413453


namespace parabola_sum_of_coefficients_l4134_413438

/-- A quadratic function with coefficients p, q, and r -/
def quadratic_function (p q r : ℝ) (x : ℝ) : ℝ := p * x^2 + q * x + r

theorem parabola_sum_of_coefficients 
  (p q r : ℝ) 
  (h_vertex : quadratic_function p q r 3 = 4)
  (h_symmetry : ∀ (x : ℝ), quadratic_function p q r (3 + x) = quadratic_function p q r (3 - x))
  (h_point1 : quadratic_function p q r 1 = 10)
  (h_point2 : quadratic_function p q r (-1) = 14) :
  p + q + r = 10 := by
  sorry

end parabola_sum_of_coefficients_l4134_413438


namespace inverse_variation_problem_l4134_413427

/-- Represents the inverse variation relationship between x^3 and ∛w -/
def inverse_variation (x w : ℝ) : Prop := ∃ k : ℝ, x^3 * w^(1/3) = k

/-- Given conditions and theorem statement -/
theorem inverse_variation_problem (x₀ w₀ x₁ w₁ : ℝ) 
  (h₀ : inverse_variation x₀ w₀)
  (h₁ : x₀ = 3)
  (h₂ : w₀ = 8)
  (h₃ : x₁ = 6)
  (h₄ : inverse_variation x₁ w₁) :
  w₁ = 1 / 64 := by
  sorry

end inverse_variation_problem_l4134_413427


namespace campers_rowing_difference_l4134_413487

theorem campers_rowing_difference (morning afternoon evening : ℕ) 
  (h1 : morning = 33) 
  (h2 : afternoon = 34) 
  (h3 : evening = 10) : 
  afternoon - evening = 24 := by
sorry

end campers_rowing_difference_l4134_413487


namespace min_value_of_expression_l4134_413455

theorem min_value_of_expression (m n : ℝ) (h1 : 2 * m + n = 2) (h2 : m * n > 0) :
  1 / m + 2 / n ≥ 4 ∧ (1 / m + 2 / n = 4 ↔ n = 2 * m ∧ n = 2) :=
sorry

end min_value_of_expression_l4134_413455


namespace third_number_proof_l4134_413479

def mean (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem third_number_proof (x : ℕ) (y : ℕ) :
  mean [28, x, y, 78, 104] = 90 →
  mean [128, 255, 511, 1023, x] = 423 →
  y = 42 := by
  sorry

end third_number_proof_l4134_413479


namespace prob_black_ball_is_one_fourth_l4134_413464

/-- Represents the number of black balls in the bag. -/
def black_balls : ℕ := 6

/-- Represents the number of red balls in the bag. -/
def red_balls : ℕ := 18

/-- Represents the total number of balls in the bag. -/
def total_balls : ℕ := black_balls + red_balls

/-- The probability of drawing a black ball from the bag. -/
def prob_black_ball : ℚ := black_balls / total_balls

theorem prob_black_ball_is_one_fourth : prob_black_ball = 1 / 4 := by
  sorry

end prob_black_ball_is_one_fourth_l4134_413464


namespace toy_cost_calculation_l4134_413476

theorem toy_cost_calculation (initial_amount : ℕ) (game_cost : ℕ) (num_toys : ℕ) :
  initial_amount = 83 →
  game_cost = 47 →
  num_toys = 9 →
  (initial_amount - game_cost) % num_toys = 0 →
  (initial_amount - game_cost) / num_toys = 4 :=
by
  sorry

end toy_cost_calculation_l4134_413476


namespace dog_grouping_theorem_l4134_413414

def number_of_dogs : ℕ := 10
def group_sizes : List ℕ := [3, 5, 2]

theorem dog_grouping_theorem :
  let remaining_dogs := number_of_dogs - 2  -- Fluffy and Nipper are pre-placed
  let ways_to_fill_fluffy_group := Nat.choose remaining_dogs (group_sizes[0] - 1)
  let remaining_after_fluffy := remaining_dogs - (group_sizes[0] - 1)
  let ways_to_fill_nipper_group := Nat.choose remaining_after_fluffy (group_sizes[1] - 1)
  ways_to_fill_fluffy_group * ways_to_fill_nipper_group = 420 :=
by
  sorry

end dog_grouping_theorem_l4134_413414


namespace prob_divisible_by_11_is_correct_l4134_413488

/-- The probability of reaching a number divisible by 11 in the described process -/
def prob_divisible_by_11 : ℚ := 11 / 20

/-- The process of building an integer as described in the problem -/
def build_integer (start : ℕ) (stop_condition : ℕ → Bool) : ℕ → ℚ := sorry

/-- The main theorem stating that the probability of reaching a number divisible by 11 is 11/20 -/
theorem prob_divisible_by_11_is_correct :
  build_integer 9 (λ n => n % 11 = 0 ∨ n % 11 = 1) 0 = prob_divisible_by_11 := by sorry

end prob_divisible_by_11_is_correct_l4134_413488


namespace tangent_line_condition_range_of_a_l4134_413420

noncomputable section

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x * Real.log x + a / x
def g (x : ℝ) : ℝ := 2 * x * Real.exp x - Real.log x - x - Real.log 2

-- Part 1: Tangent line condition
theorem tangent_line_condition (a : ℝ) :
  (∃ x₀ : ℝ, x₀ > 0 ∧ f a x₀ = x₀ ∧ (deriv (f a)) x₀ = 1) → a = Real.exp 1 / 2 :=
sorry

-- Part 2: Range of a
theorem range_of_a (a : ℝ) :
  (∀ x₁ > 0, ∃ x₂ > 0, f a x₁ ≥ g x₂) → a ≥ 1 :=
sorry

end

end tangent_line_condition_range_of_a_l4134_413420


namespace banana_cantaloupe_cost_l4134_413454

/-- Represents the cost of fruits in dollars -/
structure FruitCosts where
  apples : ℝ
  bananas : ℝ
  cantaloupe : ℝ
  dates : ℝ
  cherries : ℝ

/-- The conditions of the fruit purchase problem -/
def fruitProblemConditions (c : FruitCosts) : Prop :=
  c.apples + c.bananas + c.cantaloupe + c.dates + c.cherries = 30 ∧
  c.dates = 3 * c.apples ∧
  c.cantaloupe = c.apples - c.bananas ∧
  c.cherries = c.apples + c.bananas

/-- The theorem stating that under the given conditions, 
    the cost of bananas and cantaloupe is $5 -/
theorem banana_cantaloupe_cost (c : FruitCosts) 
  (h : fruitProblemConditions c) : 
  c.bananas + c.cantaloupe = 5 := by
  sorry

end banana_cantaloupe_cost_l4134_413454


namespace polly_cooking_time_l4134_413473

/-- Represents the cooking times for Polly in a week -/
structure CookingTimes where
  breakfast_time : ℕ  -- Time spent cooking breakfast daily
  lunch_time : ℕ      -- Time spent cooking lunch daily
  dinner_time_short : ℕ  -- Time spent cooking dinner on short days
  dinner_time_long : ℕ   -- Time spent cooking dinner on long days
  short_dinner_days : ℕ  -- Number of days with short dinner cooking time
  long_dinner_days : ℕ   -- Number of days with long dinner cooking time

/-- Calculates the total cooking time for a week given the cooking times -/
def total_cooking_time (times : CookingTimes) : ℕ :=
  7 * (times.breakfast_time + times.lunch_time) +
  times.short_dinner_days * times.dinner_time_short +
  times.long_dinner_days * times.dinner_time_long

/-- Theorem stating that Polly's total cooking time for the week is 305 minutes -/
theorem polly_cooking_time :
  ∀ (times : CookingTimes),
  times.breakfast_time = 20 ∧
  times.lunch_time = 5 ∧
  times.dinner_time_short = 10 ∧
  times.dinner_time_long = 30 ∧
  times.short_dinner_days = 4 ∧
  times.long_dinner_days = 3 →
  total_cooking_time times = 305 := by
  sorry


end polly_cooking_time_l4134_413473


namespace combination_98_96_l4134_413446

theorem combination_98_96 : Nat.choose 98 96 = 4753 := by
  sorry

end combination_98_96_l4134_413446


namespace inequality_solution_set_l4134_413495

theorem inequality_solution_set (x : ℝ) : 
  (3/20 : ℝ) + |x - 9/40| + |x + 1/8| < (1/2 : ℝ) ↔ -3/40 < x ∧ x < 11/40 := by
  sorry

end inequality_solution_set_l4134_413495


namespace tommy_balloons_l4134_413443

theorem tommy_balloons (initial : ℝ) : 
  initial + 34.5 - 12.75 = 60.75 → initial = 39 := by sorry

end tommy_balloons_l4134_413443


namespace annual_growth_rate_proof_l4134_413409

-- Define the initial number of students
def initial_students : ℕ := 200

-- Define the final number of students
def final_students : ℕ := 675

-- Define the number of years
def years : ℕ := 3

-- Define the growth rate as a real number between 0 and 1
def growth_rate : ℝ := 0.5

-- Theorem statement
theorem annual_growth_rate_proof :
  (initial_students : ℝ) * (1 + growth_rate)^years = final_students :=
sorry

end annual_growth_rate_proof_l4134_413409


namespace multiply_mixed_number_l4134_413440

theorem multiply_mixed_number : 7 * (9 + 2/5) = 329/5 := by
  sorry

end multiply_mixed_number_l4134_413440


namespace constant_expression_in_linear_system_l4134_413406

theorem constant_expression_in_linear_system (a k : ℝ) (x y : ℝ → ℝ) :
  (∀ a, x a + 2 * y a = -a + 1) →
  (∀ a, x a - 3 * y a = 4 * a + 6) →
  (∃ c, ∀ a, k * x a - y a = c) →
  k = -1 := by
sorry

end constant_expression_in_linear_system_l4134_413406


namespace unique_solution_for_all_y_l4134_413451

theorem unique_solution_for_all_y : ∃! x : ℝ, ∀ y : ℝ, 12 * x * y - 18 * y + 3 * x - 9 / 2 = 0 := by
  sorry

end unique_solution_for_all_y_l4134_413451


namespace chosen_number_proof_l4134_413474

theorem chosen_number_proof : ∃ x : ℝ, (x / 5) - 154 = 6 ∧ x = 800 := by
  sorry

end chosen_number_proof_l4134_413474


namespace remainder_97_103_times_7_mod_17_l4134_413467

theorem remainder_97_103_times_7_mod_17 : (97^103 * 7) % 17 = 13 := by
  sorry

end remainder_97_103_times_7_mod_17_l4134_413467


namespace intersection_cylinders_in_sphere_l4134_413419

/-- Theorem: Intersection of three perpendicular unit cylinders is contained in a sphere of radius √(3/2) -/
theorem intersection_cylinders_in_sphere (a b c d e f : ℝ) (x y z : ℝ) : 
  ((x - a)^2 + (y - b)^2 ≤ 1) →
  ((y - c)^2 + (z - d)^2 ≤ 1) →
  ((z - e)^2 + (x - f)^2 ≤ 1) →
  ∃ (center_x center_y center_z : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 + (z - center_z)^2 ≤ 3/2 :=
by sorry

end intersection_cylinders_in_sphere_l4134_413419


namespace not_in_second_quadrant_l4134_413490

/-- A linear function f(x) = x - 1 -/
def f (x : ℝ) : ℝ := x - 1

/-- The second quadrant of the Cartesian plane -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The linear function f(x) = x - 1 does not pass through the second quadrant -/
theorem not_in_second_quadrant :
  ∀ x : ℝ, ¬(second_quadrant x (f x)) :=
by sorry

end not_in_second_quadrant_l4134_413490


namespace music_class_size_l4134_413429

theorem music_class_size (total_students : ℕ) (art_music_overlap : ℕ) :
  total_students = 60 →
  art_music_overlap = 8 →
  ∃ (art_only music_only : ℕ),
    total_students = art_only + music_only + art_music_overlap ∧
    art_only + art_music_overlap = music_only + art_music_overlap + 10 →
    music_only + art_music_overlap = 33 :=
by sorry

end music_class_size_l4134_413429


namespace cookie_count_l4134_413468

theorem cookie_count (x y : ℕ) (hx : x = 137) (hy : y = 251) : x * y = 34387 := by
  sorry

end cookie_count_l4134_413468


namespace list_property_l4134_413472

theorem list_property (numbers : List ℝ) (n : ℝ) : 
  numbers.length = 21 →
  n ∈ numbers →
  n = 5 * ((numbers.sum - n) / 20) →
  n = 0.2 * numbers.sum →
  (numbers.filter (λ x => x ≠ n)).length = 20 := by
sorry

end list_property_l4134_413472


namespace exists_divisible_with_at_most_1988_ones_l4134_413460

/-- A natural number is representable with at most 1988 ones if its binary representation
    has at most 1988 ones. -/
def representable_with_at_most_1988_ones (n : ℕ) : Prop :=
  (n.digits 2).count 1 ≤ 1988

/-- For any natural number M, there exists a natural number N that is
    representable with at most 1988 ones and is divisible by M. -/
theorem exists_divisible_with_at_most_1988_ones (M : ℕ) :
  ∃ N : ℕ, representable_with_at_most_1988_ones N ∧ M ∣ N :=
by sorry


end exists_divisible_with_at_most_1988_ones_l4134_413460


namespace rectangle_area_rectangle_area_proof_l4134_413402

theorem rectangle_area (square_area : ℝ) (rectangle_breadth : ℝ) : ℝ :=
  let square_side : ℝ := Real.sqrt square_area
  let circle_radius : ℝ := square_side
  let rectangle_length : ℝ := (2 / 3) * circle_radius
  rectangle_length * rectangle_breadth

theorem rectangle_area_proof (h1 : square_area = 4761) (h2 : rectangle_breadth = 13) :
  rectangle_area square_area rectangle_breadth = 598 := by
  sorry

end rectangle_area_rectangle_area_proof_l4134_413402


namespace parabola_vertex_on_x_axis_l4134_413459

/-- A parabola with equation y = x^2 - 8x + m has its vertex on the x-axis if and only if m = 16 -/
theorem parabola_vertex_on_x_axis (m : ℝ) :
  (∃ x, x^2 - 8*x + m = 0 ∧ ∀ y, y^2 - 8*y + m ≥ x^2 - 8*x + m) ↔ m = 16 := by
  sorry

end parabola_vertex_on_x_axis_l4134_413459


namespace vector_a_magnitude_l4134_413450

def vector_a : ℝ × ℝ := (3, -2)

theorem vector_a_magnitude : ‖vector_a‖ = Real.sqrt 13 := by
  sorry

end vector_a_magnitude_l4134_413450


namespace polynomial_division_theorem_l4134_413466

theorem polynomial_division_theorem :
  let dividend : Polynomial ℚ := X^4 * 6 + X^3 * 9 - X^2 * 5 + X * 2 - 8
  let divisor : Polynomial ℚ := X * 3 + 4
  let quotient : Polynomial ℚ := X^3 * 2 - X^2 * 1 + X * 1 - 2
  let remainder : Polynomial ℚ := -8/3
  dividend = divisor * quotient + remainder := by sorry

end polynomial_division_theorem_l4134_413466


namespace min_roots_in_interval_l4134_413493

/-- A function satisfying the given symmetry conditions -/
def SymmetricFunction (g : ℝ → ℝ) : Prop :=
  (∀ x, g (3 + x) = g (3 - x)) ∧ (∀ x, g (5 + x) = g (5 - x))

/-- The theorem stating the minimum number of roots in the given interval -/
theorem min_roots_in_interval
  (g : ℝ → ℝ)
  (h_symmetric : SymmetricFunction g)
  (h_g1_zero : g 1 = 0) :
  ∃ (roots : Finset ℝ), 
    (∀ x ∈ roots, g x = 0 ∧ x ∈ Set.Icc (-1000) 1000) ∧
    roots.card ≥ 250 :=
  sorry

end min_roots_in_interval_l4134_413493


namespace reflection_sum_l4134_413456

/-- Reflects a point over the y-axis -/
def reflect_y (x y : ℝ) : ℝ × ℝ := (-x, y)

/-- Reflects a point over the x-axis -/
def reflect_x (x y : ℝ) : ℝ × ℝ := (x, -y)

/-- Sums the coordinates of a point -/
def sum_coordinates (p : ℝ × ℝ) : ℝ := p.1 + p.2

theorem reflection_sum (y : ℝ) :
  let C : ℝ × ℝ := (3, y)
  let D := reflect_y C.1 C.2
  let E := reflect_x D.1 D.2
  sum_coordinates C + sum_coordinates E = -6 := by
  sorry

end reflection_sum_l4134_413456


namespace range_of_m_l4134_413444

-- Define the conditions
def condition1 (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0

def condition2 (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0 ∧ m > 0

def p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 10

def q (x m : ℝ) : Prop := condition2 x m

-- State the theorem
theorem range_of_m : 
  (∀ x m : ℝ, condition1 x → (¬(p x) → ¬(q x m)) ∧ ∃ y : ℝ, ¬(p y) ∧ q y m) →
  (∀ m : ℝ, m ≥ 9) :=
sorry

end range_of_m_l4134_413444


namespace arithmetic_sequence_slope_l4134_413469

/-- An arithmetic sequence with sum of first n terms S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  sum_formula : ∀ n : ℕ, S n = n * (a 0 + a (n - 1)) / 2

/-- The slope of the line passing through P(n, a_n) and Q(n+2, a_{n+2}) is 4 -/
theorem arithmetic_sequence_slope (seq : ArithmeticSequence) 
    (h1 : seq.S 2 = 10) (h2 : seq.S 5 = 55) :
    ∀ n : ℕ+, (seq.a (n + 2) - seq.a n) / 2 = 4 := by
  sorry

end arithmetic_sequence_slope_l4134_413469


namespace triangle_centroid_inequality_locus_is_circle_l4134_413486

open Real

-- Define a triangle with vertices A, B, C and centroid G
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  G : ℝ × ℝ
  is_centroid : G = ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)

-- Define distance squared between two points
def dist_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

-- Define the theorem
theorem triangle_centroid_inequality (t : Triangle) (M : ℝ × ℝ) :
  dist_sq M t.A + dist_sq M t.B + dist_sq M t.C ≥ 
  dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C ∧
  (dist_sq M t.A + dist_sq M t.B + dist_sq M t.C = 
   dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C ↔ M = t.G) :=
sorry

-- Define the locus of points
def locus (t : Triangle) (k : ℝ) : Set (ℝ × ℝ) :=
  {M | dist_sq M t.A + dist_sq M t.B + dist_sq M t.C = k}

-- Define the theorem for the locus
theorem locus_is_circle (t : Triangle) (k : ℝ) 
  (h : k > dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C) :
  ∃ (r : ℝ), r > 0 ∧ locus t k = {M | dist_sq M t.G = r^2} ∧
  r^2 = (k - (dist_sq t.G t.A + dist_sq t.G t.B + dist_sq t.G t.C)) / 3 :=
sorry

end triangle_centroid_inequality_locus_is_circle_l4134_413486


namespace sector_area_l4134_413432

/-- The area of a sector with radius 2 and central angle 2π/3 is 4π/3 -/
theorem sector_area (r : ℝ) (θ : ℝ) (area : ℝ) : 
  r = 2 → θ = 2 * π / 3 → area = (1 / 2) * r^2 * θ → area = 4 * π / 3 := by
  sorry

end sector_area_l4134_413432
