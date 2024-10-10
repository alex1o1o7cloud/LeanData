import Mathlib

namespace consecutive_integers_product_336_sum_21_l645_64566

theorem consecutive_integers_product_336_sum_21 :
  ∃ (n : ℤ), (n - 1) * n * (n + 1) = 336 → (n - 1) + n + (n + 1) = 21 := by
  sorry

end consecutive_integers_product_336_sum_21_l645_64566


namespace dispatch_plans_count_l645_64520

/-- The number of vehicles in the fleet -/
def total_vehicles : ℕ := 7

/-- The number of vehicles to be dispatched -/
def dispatched_vehicles : ℕ := 4

/-- The number of ways to arrange vehicles A and B with A before B -/
def arrange_A_B : ℕ := 6

/-- The number of remaining vehicles after A and B are selected -/
def remaining_vehicles : ℕ := total_vehicles - 2

/-- The number of additional vehicles to be selected after A and B -/
def additional_vehicles : ℕ := dispatched_vehicles - 2

/-- Calculate the number of permutations of n elements taken r at a time -/
def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

theorem dispatch_plans_count :
  arrange_A_B * permutations remaining_vehicles additional_vehicles = 120 :=
sorry

end dispatch_plans_count_l645_64520


namespace eva_max_silver_tokens_l645_64552

/-- Represents the number of tokens Eva has -/
structure TokenCount where
  red : ℕ
  blue : ℕ
  silver : ℕ

/-- Represents an exchange booth -/
structure Booth where
  redIn : ℕ
  blueIn : ℕ
  silverOut : ℕ
  redOut : ℕ
  blueOut : ℕ

/-- The maximum number of silver tokens Eva can obtain -/
def maxSilverTokens (initial : TokenCount) (booth1 booth2 : Booth) : ℕ :=
  sorry

/-- Theorem stating that Eva can obtain at most 57 silver tokens -/
theorem eva_max_silver_tokens :
  let initial := TokenCount.mk 60 90 0
  let booth1 := Booth.mk 3 0 2 0 1
  let booth2 := Booth.mk 0 4 3 1 0
  maxSilverTokens initial booth1 booth2 = 57 :=
by sorry

end eva_max_silver_tokens_l645_64552


namespace custom_op_neg_two_neg_one_l645_64595

-- Define the custom operation
def customOp (a b : ℝ) : ℝ := a^2 - abs b

-- Theorem statement
theorem custom_op_neg_two_neg_one :
  customOp (-2) (-1) = 3 := by
  sorry

end custom_op_neg_two_neg_one_l645_64595


namespace max_leftover_grapes_l645_64554

theorem max_leftover_grapes :
  ∀ n : ℕ, ∃ q r : ℕ, n = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end max_leftover_grapes_l645_64554


namespace fraction_equality_l645_64540

theorem fraction_equality (a b : ℝ) (h : a ≠ b) :
  (a^2 - b^2) / (a - b)^2 = (a + b) / (a - b) := by sorry

end fraction_equality_l645_64540


namespace min_k_theorem_l645_64562

/-- The set S of powers of 1996 -/
def S : Set ℕ := {n : ℕ | ∃ m : ℕ, n = 1996^m}

/-- Definition of a valid sequence pair -/
def ValidSequencePair (k : ℕ) (a b : ℕ → ℕ) : Prop :=
  (∀ i ∈ Finset.range k, a i ∈ S ∧ b i ∈ S) ∧
  (∀ i ∈ Finset.range k, a i ≠ b i) ∧
  (∀ i ∈ Finset.range (k-1), a i ≤ a (i+1) ∧ b i ≤ b (i+1)) ∧
  (Finset.sum (Finset.range k) a = Finset.sum (Finset.range k) b)

/-- The theorem stating the minimum k -/
theorem min_k_theorem :
  (∃ k : ℕ, ∃ a b : ℕ → ℕ, ValidSequencePair k a b) ∧
  (∀ k < 1997, ¬∃ a b : ℕ → ℕ, ValidSequencePair k a b) ∧
  (∃ a b : ℕ → ℕ, ValidSequencePair 1997 a b) :=
sorry

end min_k_theorem_l645_64562


namespace smallest_prime_factor_of_1547_l645_64543

theorem smallest_prime_factor_of_1547 :
  Nat.minFac 1547 = 7 := by sorry

end smallest_prime_factor_of_1547_l645_64543


namespace scientific_notation_of_2150_l645_64503

def scientific_notation (x : ℝ) : ℝ × ℤ :=
  sorry

theorem scientific_notation_of_2150 :
  scientific_notation 2150 = (2.15, 3) := by
  sorry

end scientific_notation_of_2150_l645_64503


namespace gizmos_produced_l645_64548

/-- Represents the production scenario in a factory -/
structure ProductionScenario where
  a : ℝ  -- Time to produce a gadget
  b : ℝ  -- Time to produce a gizmo

/-- Checks if the production scenario satisfies the given conditions -/
def satisfies_conditions (s : ProductionScenario) : Prop :=
  s.a ≥ 0 ∧ s.b ≥ 0 ∧  -- Non-negative production times
  450 * s.a + 300 * s.b = 150 ∧  -- 150 workers in 1 hour
  360 * s.a + 450 * s.b = 180 ∧  -- 90 workers in 2 hours
  300 * s.a = 300  -- 75 workers produce 300 gadgets in 4 hours

/-- Theorem stating the number of gizmos produced by 75 workers in 4 hours -/
theorem gizmos_produced (s : ProductionScenario) 
  (h : satisfies_conditions s) : 
  75 * 4 / s.b = 150 := by
  sorry


end gizmos_produced_l645_64548


namespace vector_magnitude_l645_64538

theorem vector_magnitude (x : ℝ) : 
  let a : ℝ × ℝ := (1, x)
  let b : ℝ × ℝ := (-1, x)
  (2 • a - b) • b = 0 → ‖a‖ = 2 :=
by sorry

end vector_magnitude_l645_64538


namespace jack_marbles_l645_64500

/-- Calculates the final number of marbles Jack has after sharing and finding more -/
def final_marbles (initial : ℕ) (shared : ℕ) (multiplier : ℕ) : ℕ :=
  let remaining := initial - shared
  let found := remaining * multiplier
  remaining + found

/-- Theorem stating that Jack ends up with 232 marbles -/
theorem jack_marbles :
  final_marbles 62 33 7 = 232 := by
  sorry

end jack_marbles_l645_64500


namespace clubs_distribution_l645_64580

-- Define the set of cards
inductive Card : Type
| Hearts : Card
| Spades : Card
| Diamonds : Card
| Clubs : Card

-- Define the set of people
inductive Person : Type
| A : Person
| B : Person
| C : Person
| D : Person

-- Define a distribution as a function from Person to Card
def Distribution := Person → Card

-- Define the event "A gets the clubs"
def A_gets_clubs (d : Distribution) : Prop := d Person.A = Card.Clubs

-- Define the event "B gets the clubs"
def B_gets_clubs (d : Distribution) : Prop := d Person.B = Card.Clubs

-- State the theorem
theorem clubs_distribution :
  (∀ d : Distribution, ¬(A_gets_clubs d ∧ B_gets_clubs d)) ∧ 
  (∃ d : Distribution, ¬A_gets_clubs d ∧ ¬B_gets_clubs d) :=
sorry

end clubs_distribution_l645_64580


namespace simplify_tan_product_l645_64598

-- Define the tangent function
noncomputable def tan (x : Real) : Real := Real.tan x

-- State the theorem
theorem simplify_tan_product : 
  (1 + tan (10 * Real.pi / 180)) * (1 + tan (35 * Real.pi / 180)) = 2 := by
  -- Assuming the angle addition formula for tangent
  have angle_addition_formula : ∀ a b, 
    tan (a + b) = (tan a + tan b) / (1 - tan a * tan b) := by sorry
  
  -- Assuming tan 45° = 1
  have tan_45_deg : tan (45 * Real.pi / 180) = 1 := by sorry

  sorry -- The proof goes here

end simplify_tan_product_l645_64598


namespace roy_pens_total_l645_64521

theorem roy_pens_total (blue : ℕ) (black : ℕ) (red : ℕ) : 
  blue = 2 → 
  black = 2 * blue → 
  red = 2 * black - 2 → 
  blue + black + red = 12 := by
  sorry

end roy_pens_total_l645_64521


namespace bobby_total_candy_and_chocolate_l645_64505

def candy_initial : ℕ := 33
def candy_additional : ℕ := 4
def chocolate : ℕ := 14

theorem bobby_total_candy_and_chocolate :
  candy_initial + candy_additional + chocolate = 51 := by
  sorry

end bobby_total_candy_and_chocolate_l645_64505


namespace inscribed_squares_segment_product_l645_64544

theorem inscribed_squares_segment_product (c d : ℝ) : 
  (∃ (small_square_area large_square_area : ℝ),
    small_square_area = 9 ∧ 
    large_square_area = 18 ∧ 
    c + d = (large_square_area).sqrt ∧ 
    c^2 + d^2 = large_square_area) → 
  c * d = 0 := by
sorry

end inscribed_squares_segment_product_l645_64544


namespace abc_sum_in_base_7_l645_64547

theorem abc_sum_in_base_7 : ∃ (A B C : Nat), 
  A < 7 ∧ B < 7 ∧ C < 7 ∧  -- digits less than 7
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧  -- non-zero digits
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧  -- distinct digits
  Nat.Prime A ∧            -- A is prime
  (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A ∧  -- ABC₇ + BC₇ = ACA₇
  A + B + C = 13           -- sum is 16₇ in base 7, which is 13 in base 10
  := by sorry

end abc_sum_in_base_7_l645_64547


namespace car_cost_proof_l645_64512

/-- The cost of Gary's used car -/
def car_cost : ℝ := 6000

/-- The monthly payment difference between 2-year and 5-year loans -/
def monthly_difference : ℝ := 150

/-- The number of months in 2 years -/
def months_in_2_years : ℝ := 2 * 12

/-- The number of months in 5 years -/
def months_in_5_years : ℝ := 5 * 12

theorem car_cost_proof :
  (car_cost / months_in_2_years) - (car_cost / months_in_5_years) = monthly_difference :=
sorry

end car_cost_proof_l645_64512


namespace quadratic_factor_value_l645_64510

-- Define the polynomials
def f (x : ℝ) : ℝ := x^4 + 8*x^3 + 18*x^2 + 8*x + 35
def g (x : ℝ) : ℝ := 2*x^4 - 4*x^3 + x^2 + 26*x + 10

-- Define the quadratic polynomial q
def q (d e : ℤ) (x : ℝ) : ℝ := x^2 + d*x + e

-- State the theorem
theorem quadratic_factor_value (d e : ℤ) :
  (∃ (p₁ p₂ : ℝ → ℝ), f = q d e * p₁ ∧ g = q d e * p₂) →
  q d e 2 = 21 := by
  sorry

end quadratic_factor_value_l645_64510


namespace complement_intersection_problem_l645_64576

open Set

theorem complement_intersection_problem (I A B : Set ℕ) : 
  I = {0, 1, 2, 3, 4} →
  A = {0, 2, 3} →
  B = {1, 3, 4} →
  (I \ A) ∩ B = {1, 4} := by
  sorry

end complement_intersection_problem_l645_64576


namespace ellipse_properties_line_through_focus_l645_64546

/-- Ellipse C defined by the equation x²/2 + y² = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/2 + y^2 = 1

/-- Point on ellipse C -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse_C x y

/-- Line passing through a point with slope k -/
def line (k : ℝ) (x₀ y₀ : ℝ) (x y : ℝ) : Prop :=
  y - y₀ = k * (x - x₀)

theorem ellipse_properties :
  let a := Real.sqrt 2
  let b := 1
  let c := 1
  let e := c / a
  let left_focus := (-1, 0)
  (∀ x y, ellipse_C x y → x^2 / (a^2) + y^2 / (b^2) = 1) ∧
  (2 * a = 2 * Real.sqrt 2) ∧
  (2 * b = 2) ∧
  (e = Real.sqrt 2 / 2) ∧
  (left_focus.1 = -c ∧ left_focus.2 = 0) :=
by sorry

theorem line_through_focus (k : ℝ) :
  let left_focus := (-1, 0)
  ∃ A B : PointOnEllipse,
    line k left_focus.1 left_focus.2 A.x A.y ∧
    line k left_focus.1 left_focus.2 B.x B.y ∧
    (A.x - B.x)^2 + (A.y - B.y)^2 = (8 * Real.sqrt 2 / 7)^2 →
    k = Real.sqrt 3 ∨ k = -Real.sqrt 3 :=
by sorry

end ellipse_properties_line_through_focus_l645_64546


namespace solution_is_3x_l645_64549

-- Define the interval
def I : Set ℝ := Set.Icc (-1 : ℝ) 1

-- Define the integral equation
def integral_equation (φ : ℝ → ℝ) : Prop :=
  ∀ x ∈ I, φ x = x + ∫ t in I, x * t * φ t

-- State the theorem
theorem solution_is_3x :
  ∃ φ : ℝ → ℝ, integral_equation φ ∧ (∀ x ∈ I, φ x = 3 * x) :=
sorry

end solution_is_3x_l645_64549


namespace division_by_240_property_l645_64555

-- Define a function to check if a number has at least two digits
def hasAtLeastTwoDigits (n : ℕ) : Prop := n ≥ 10

-- Define the theorem
theorem division_by_240_property (a b : ℕ) 
  (ha : Nat.Prime a) (hb : Nat.Prime b) 
  (hta : hasAtLeastTwoDigits a) (htb : hasAtLeastTwoDigits b) 
  (hab : a > b) : 
  (∃ k : ℕ, a^4 - b^4 = 240 * k) ∧ 
  (∀ m : ℕ, m > 240 → ¬(∀ x y : ℕ, Nat.Prime x → Nat.Prime y → hasAtLeastTwoDigits x → hasAtLeastTwoDigits y → x > y → ∃ l : ℕ, x^4 - y^4 = m * l)) :=
by sorry


end division_by_240_property_l645_64555


namespace max_value_of_sum_products_l645_64579

theorem max_value_of_sum_products (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 →
  a * b + b * c + c * d + d * a ≤ 10000 := by
sorry

end max_value_of_sum_products_l645_64579


namespace second_chapter_pages_l645_64596

theorem second_chapter_pages (total_pages first_chapter third_chapter : ℕ) 
  (h1 : total_pages = 125)
  (h2 : first_chapter = 66)
  (h3 : third_chapter = 24) :
  total_pages - first_chapter - third_chapter = 59 :=
by
  sorry

end second_chapter_pages_l645_64596


namespace packets_to_fill_gunny_bag_l645_64593

/-- Represents the weight of a packet in ounces -/
def packet_weight : ℕ := 16 * 16 + 4

/-- Represents the capacity of the gunny bag in tons -/
def gunny_bag_capacity : ℕ := 13

/-- Conversion rate from tons to pounds -/
def tons_to_pounds : ℕ := 2500

/-- Conversion rate from pounds to ounces -/
def pounds_to_ounces : ℕ := 16

/-- Theorem stating the number of packets needed to fill the gunny bag -/
theorem packets_to_fill_gunny_bag : 
  (gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / packet_weight = 2000 := by
  sorry

#eval (gunny_bag_capacity * tons_to_pounds * pounds_to_ounces) / packet_weight

end packets_to_fill_gunny_bag_l645_64593


namespace no_real_a_for_unique_solution_l645_64557

theorem no_real_a_for_unique_solution : ¬∃ a : ℝ, ∃! x : ℝ, |x^2 + 4*a*x + 5*a| ≤ 3 := by
  sorry

end no_real_a_for_unique_solution_l645_64557


namespace collinear_vectors_m_value_l645_64591

theorem collinear_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, 3]
  let b : Fin 2 → ℝ := ![m, 2*m - 1]
  (∃ (k : ℝ), b = k • a) → m = -1 := by
sorry

end collinear_vectors_m_value_l645_64591


namespace apple_bag_cost_proof_l645_64594

/-- The cost of a bag of dozen apples -/
def apple_bag_cost : ℝ := 14

theorem apple_bag_cost_proof :
  let kiwi_cost : ℝ := 10
  let banana_cost : ℝ := 5
  let initial_money : ℝ := 50
  let subway_fare : ℝ := 3.5
  let max_apples : ℕ := 24
  apple_bag_cost = (initial_money - (kiwi_cost + banana_cost) - 2 * subway_fare) / (max_apples / 12) :=
by
  sorry

#check apple_bag_cost_proof

end apple_bag_cost_proof_l645_64594


namespace mary_apple_expense_l645_64502

theorem mary_apple_expense (total_spent berries_cost peaches_cost : ℚ)
  (h1 : total_spent = 34.72)
  (h2 : berries_cost = 11.08)
  (h3 : peaches_cost = 9.31) :
  total_spent - (berries_cost + peaches_cost) = 14.33 := by
sorry

end mary_apple_expense_l645_64502


namespace simplify_product_of_radicals_l645_64597

theorem simplify_product_of_radicals (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (45 * x) * Real.sqrt (20 * x) * Real.sqrt (30 * x) = 30 * x * Real.sqrt 30 := by
  sorry

end simplify_product_of_radicals_l645_64597


namespace min_tiles_for_room_l645_64523

/-- Represents the dimensions of a room in centimeters -/
structure Room where
  length : ℕ
  breadth : ℕ

/-- Represents a square tile with a given side length in centimeters -/
structure Tile where
  side : ℕ

/-- Calculates the number of tiles needed to cover a room, including wastage -/
def tilesNeeded (room : Room) (tile : Tile) : ℕ :=
  let roomArea := room.length * room.breadth
  let tileArea := tile.side * tile.side
  let baseTiles := (roomArea + tileArea - 1) / tileArea  -- Ceiling division
  let wastage := (baseTiles * 11 + 9) / 10  -- 10% wastage, rounded up
  baseTiles + wastage

/-- Theorem stating the minimum number of tiles required -/
theorem min_tiles_for_room (room : Room) (tile : Tile) :
  room.length = 888 ∧ room.breadth = 462 ∧ tile.side = 22 →
  tilesNeeded room tile ≥ 933 :=
by sorry

end min_tiles_for_room_l645_64523


namespace lcm_5_6_8_9_l645_64584

theorem lcm_5_6_8_9 : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end lcm_5_6_8_9_l645_64584


namespace days_worked_by_a_l645_64587

/-- Represents the number of days worked by person a -/
def days_a : ℕ := 16

/-- Represents the number of days worked by person b -/
def days_b : ℕ := 9

/-- Represents the number of days worked by person c -/
def days_c : ℕ := 4

/-- Represents the daily wage ratio of person a -/
def wage_ratio_a : ℚ := 3

/-- Represents the daily wage ratio of person b -/
def wage_ratio_b : ℚ := 4

/-- Represents the daily wage ratio of person c -/
def wage_ratio_c : ℚ := 5

/-- Represents the daily wage of person c -/
def wage_c : ℚ := 71.15384615384615

/-- Represents the total earnings of all three workers -/
def total_earnings : ℚ := 1480

/-- Theorem stating that given the conditions, the number of days worked by person a is 16 -/
theorem days_worked_by_a : 
  (days_a : ℚ) * (wage_ratio_a * wage_c / wage_ratio_c) + 
  (days_b : ℚ) * (wage_ratio_b * wage_c / wage_ratio_c) + 
  (days_c : ℚ) * wage_c = total_earnings :=
sorry

end days_worked_by_a_l645_64587


namespace set_operations_l645_64558

def I : Set Nat := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set Nat := {1, 2, 3, 4}
def B : Set Nat := {3, 5, 6, 7}

theorem set_operations :
  (A ∪ B = {1, 2, 3, 4, 5, 6, 7}) ∧
  (A ∩ (I \ B) = {1, 2, 4}) := by
sorry

end set_operations_l645_64558


namespace find_x_given_exponential_equation_l645_64571

theorem find_x_given_exponential_equation : ∃ x : ℝ, (2 : ℝ)^(x - 4) = 4^2 → x = 8 := by
  sorry

end find_x_given_exponential_equation_l645_64571


namespace sum_of_roots_quadratic_undefined_values_sum_l645_64504

theorem sum_of_roots_quadratic : 
  ∀ (a b c : ℝ) (y₁ y₂ : ℝ), 
  a ≠ 0 → 
  a * y₁^2 + b * y₁ + c = 0 → 
  a * y₂^2 + b * y₂ + c = 0 → 
  y₁ + y₂ = -b / a :=
sorry

theorem undefined_values_sum : 
  let y₁ := (3 + Real.sqrt 49) / 2
  let y₂ := (3 - Real.sqrt 49) / 2
  y₁^2 - 3*y₁ - 10 = 0 ∧ 
  y₂^2 - 3*y₂ - 10 = 0 ∧ 
  y₁ + y₂ = 3 :=
sorry

end sum_of_roots_quadratic_undefined_values_sum_l645_64504


namespace arrangement_of_cards_l645_64577

def number_of_arrangements (total_cards : ℕ) (interchangeable_cards : ℕ) : ℕ :=
  (total_cards.factorial) / (interchangeable_cards.factorial)

theorem arrangement_of_cards : number_of_arrangements 15 13 = 210 := by
  sorry

end arrangement_of_cards_l645_64577


namespace leak_empty_time_l645_64578

/-- Represents the time it takes for a leak to empty a full tank, given the filling times with and without the leak. -/
theorem leak_empty_time (fill_time : ℝ) (fill_time_with_leak : ℝ) (leak_empty_time : ℝ) : 
  fill_time > 0 ∧ fill_time_with_leak > fill_time →
  (1 / fill_time) - (1 / fill_time_with_leak) = 1 / leak_empty_time →
  fill_time = 6 →
  fill_time_with_leak = 9 →
  leak_empty_time = 18 := by
sorry

end leak_empty_time_l645_64578


namespace expand_product_l645_64501

theorem expand_product (x : ℝ) : (x + 4) * (x - 9) = x^2 - 5*x - 36 := by
  sorry

end expand_product_l645_64501


namespace composition_of_linear_functions_l645_64509

theorem composition_of_linear_functions (a b : ℝ) : 
  (∀ x : ℝ, (3 * (a * x + b) - 4) = 4 * x + 2) → 
  a + b = 10/3 := by
  sorry

end composition_of_linear_functions_l645_64509


namespace remainder_theorem_l645_64536

theorem remainder_theorem : (9^6 + 5^7 + 3^8) % 7 = 4 := by sorry

end remainder_theorem_l645_64536


namespace no_right_obtuse_triangle_l645_64534

-- Define a triangle type
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of triangles
def Triangle.isValid (t : Triangle) : Prop :=
  t.angle1 > 0 ∧ t.angle2 > 0 ∧ t.angle3 > 0 ∧ t.angle1 + t.angle2 + t.angle3 = 180

def Triangle.hasRightAngle (t : Triangle) : Prop :=
  t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

def Triangle.hasObtuseAngle (t : Triangle) : Prop :=
  t.angle1 > 90 ∨ t.angle2 > 90 ∨ t.angle3 > 90

-- Theorem: A valid triangle cannot be both right-angled and obtuse
theorem no_right_obtuse_triangle :
  ∀ t : Triangle, t.isValid → ¬(t.hasRightAngle ∧ t.hasObtuseAngle) :=
by
  sorry


end no_right_obtuse_triangle_l645_64534


namespace initial_pencils_equals_sum_l645_64575

/-- The number of pencils Ken had initially -/
def initial_pencils : ℕ := sorry

/-- The number of pencils Ken gave to Manny -/
def pencils_to_manny : ℕ := 10

/-- The number of pencils Ken gave to Nilo -/
def pencils_to_nilo : ℕ := pencils_to_manny + 10

/-- The number of pencils Ken kept for himself -/
def pencils_kept : ℕ := 20

/-- Theorem stating that the initial number of pencils is equal to the sum of
    pencils given to Manny, Nilo, and kept by Ken -/
theorem initial_pencils_equals_sum :
  initial_pencils = pencils_to_manny + pencils_to_nilo + pencils_kept :=
by sorry

end initial_pencils_equals_sum_l645_64575


namespace inequality_system_solution_l645_64508

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, (x + 4) / 3 > x / 2 + 1 ∧ x + a < 0 → x < 2) →
  (∀ x : ℝ, x < 2 → x + a < 0) →
  a ≤ -2 :=
by sorry

end inequality_system_solution_l645_64508


namespace nested_square_root_simplification_l645_64533

theorem nested_square_root_simplification :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * Real.sqrt (5 * Real.sqrt 5) := by
  sorry

end nested_square_root_simplification_l645_64533


namespace xy_value_l645_64569

theorem xy_value (x y : ℝ) (h : y = Real.sqrt (x - 3) + Real.sqrt (3 - x) - 2) : x * y = -6 := by
  sorry

end xy_value_l645_64569


namespace log_equality_l645_64511

theorem log_equality (y : ℝ) (m : ℝ) : 
  (Real.log 5 / Real.log 8 = y) → 
  (Real.log 125 / Real.log 2 = m * y) → 
  m = 9 := by
sorry

end log_equality_l645_64511


namespace inequality_proof_l645_64507

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a^2014 / (1 + 2*b*c)) + (b^2014 / (1 + 2*a*c)) + (c^2014 / (1 + 2*a*b)) ≥ 3 / (a*b + b*c + c*a) :=
by sorry

end inequality_proof_l645_64507


namespace brady_record_chase_l645_64541

/-- The minimum average yards per game needed to beat the record -/
def min_avg_yards_per_game (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ) : ℚ :=
  (current_record + 1 - current_yards) / games_left

theorem brady_record_chase (current_record : ℕ) (current_yards : ℕ) (games_left : ℕ)
  (h1 : current_record = 5999)
  (h2 : current_yards = 4200)
  (h3 : games_left = 6) :
  min_avg_yards_per_game current_record current_yards games_left = 300 := by
sorry

end brady_record_chase_l645_64541


namespace meeting_percentage_theorem_l645_64560

def work_day_hours : ℝ := 10
def first_meeting_minutes : ℝ := 45
def second_meeting_multiplier : ℝ := 3

def total_meeting_time : ℝ := first_meeting_minutes + second_meeting_multiplier * first_meeting_minutes
def work_day_minutes : ℝ := work_day_hours * 60

theorem meeting_percentage_theorem :
  (total_meeting_time / work_day_minutes) * 100 = 30 := by sorry

end meeting_percentage_theorem_l645_64560


namespace algebraic_expression_value_l645_64556

theorem algebraic_expression_value (x y : ℝ) 
  (hx : x = 1 / (Real.sqrt 3 - 2))
  (hy : y = 1 / (Real.sqrt 3 + 2)) :
  (x^2 + x*y + y^2) / (13 * (x + y)) = -(Real.sqrt 3) / 6 := by
  sorry

end algebraic_expression_value_l645_64556


namespace cube_root_of_eight_l645_64563

theorem cube_root_of_eight :
  (8 : ℝ) ^ (1/3 : ℝ) = 2 := by
  sorry

end cube_root_of_eight_l645_64563


namespace margaret_score_is_86_l645_64518

/-- Given an average test score, calculate Margaret's score based on the conditions -/
def margaret_score (average : ℝ) : ℝ :=
  let marco_score := average * 0.9
  marco_score + 5

/-- Theorem stating that Margaret's score is 86 given the conditions -/
theorem margaret_score_is_86 :
  margaret_score 90 = 86 := by
  sorry

end margaret_score_is_86_l645_64518


namespace sum_of_abc_l645_64506

theorem sum_of_abc (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c)
  (h4 : a^2 + b^2 + c^2 = 48) (h5 : a * b + b * c + c * a = 26) (h6 : a = 2 * b) :
  a + b + c = 6 + 2 * Real.sqrt 7 :=
by sorry

end sum_of_abc_l645_64506


namespace cubic_polynomials_common_roots_l645_64528

/-- 
Given two cubic polynomials x^3 + ax^2 + 17x + 10 and x^3 + bx^2 + 20x + 12 that have two distinct
common roots, prove that a = -6 and b = -7.
-/
theorem cubic_polynomials_common_roots (a b : ℝ) : 
  (∃ r s : ℝ, r ≠ s ∧ 
    r^3 + a*r^2 + 17*r + 10 = 0 ∧ 
    r^3 + b*r^2 + 20*r + 12 = 0 ∧
    s^3 + a*s^2 + 17*s + 10 = 0 ∧ 
    s^3 + b*s^2 + 20*s + 12 = 0) →
  a = -6 ∧ b = -7 := by
sorry

end cubic_polynomials_common_roots_l645_64528


namespace exists_universal_friend_l645_64513

-- Define a type for people
variable {Person : Type}

-- Define the friendship relation
variable (friends : Person → Person → Prop)

-- Define the property that every two people have exactly one friend in common
def one_common_friend (friends : Person → Person → Prop) : Prop :=
  ∀ a b : Person, a ≠ b →
    ∃! c : Person, friends a c ∧ friends b c

-- State the theorem
theorem exists_universal_friend
  [Finite Person]
  (h : one_common_friend friends) :
  ∃ x : Person, ∀ y : Person, y ≠ x → friends x y :=
sorry

end exists_universal_friend_l645_64513


namespace walts_age_l645_64590

theorem walts_age (walt_age music_teacher_age : ℕ) : 
  music_teacher_age = 3 * walt_age →
  (music_teacher_age + 12) = 2 * (walt_age + 12) →
  walt_age = 12 := by
  sorry

end walts_age_l645_64590


namespace salary_change_percentage_l645_64553

theorem salary_change_percentage (x : ℝ) : 
  (1 - x / 100) * (1 + x / 100) = 0.75 → x = 50 := by sorry

end salary_change_percentage_l645_64553


namespace green_beads_count_l645_64583

/-- The number of white beads in each necklace -/
def white_beads : ℕ := 6

/-- The number of orange beads in each necklace -/
def orange_beads : ℕ := 3

/-- The maximum number of necklaces that can be made -/
def max_necklaces : ℕ := 5

/-- The total number of beads available for each color -/
def total_beads : ℕ := 45

/-- The number of green beads in each necklace -/
def green_beads : ℕ := 9

theorem green_beads_count : 
  white_beads * max_necklaces ≤ total_beads ∧ 
  orange_beads * max_necklaces ≤ total_beads ∧ 
  green_beads * max_necklaces = total_beads := by
  sorry

end green_beads_count_l645_64583


namespace cone_base_radius_l645_64545

/-- A cone with surface area 3π and lateral surface that unfolds into a semicircle has a base radius of 1 -/
theorem cone_base_radius (r : ℝ) (l : ℝ) : 
  r > 0 → l > 0 → 
  l = 2 * r →  -- Lateral surface unfolds into a semicircle
  3 * π * r^2 = 3 * π →  -- Surface area is 3π
  r = 1 := by sorry

end cone_base_radius_l645_64545


namespace min_magnitude_sum_vectors_l645_64522

/-- Given two vectors a and b in a real inner product space, with magnitudes 8 and 12 respectively,
    the minimum value of the magnitude of their sum is 4. -/
theorem min_magnitude_sum_vectors {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (a b : V) (ha : ‖a‖ = 8) (hb : ‖b‖ = 12) : 
  ∃ (sum : V), sum = a + b ∧ ‖sum‖ = 4 ∧ ∀ (x : V), x = a + b → ‖x‖ ≥ 4 := by
  sorry

end min_magnitude_sum_vectors_l645_64522


namespace euler_line_equation_l645_64516

-- Define the triangle ABC
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (0, 4)

-- Define the property that AC = BC (isosceles triangle)
def is_isosceles (C : ℝ × ℝ) : Prop := dist A C = dist B C

-- Define the Euler line
def euler_line (C : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {P | P.1 - 2 * P.2 + 3 = 0}

-- Theorem statement
theorem euler_line_equation (C : ℝ × ℝ) (h : is_isosceles C) :
  euler_line C = {P : ℝ × ℝ | P.1 - 2 * P.2 + 3 = 0} :=
sorry

end euler_line_equation_l645_64516


namespace max_transitions_to_wiki_l645_64535

theorem max_transitions_to_wiki (channel_a channel_b channel_c : ℕ) :
  channel_a = 850 * 6 / 100 ∧
  channel_b = 1500 * 42 / 1000 ∧
  channel_c = 4536 / 72 →
  max channel_b channel_c = 63 :=
by
  sorry

end max_transitions_to_wiki_l645_64535


namespace crease_length_l645_64592

/-- Given a rectangular piece of paper 8 inches wide, when folded such that one corner
    touches the opposite side and forms an angle θ at the corner where the crease starts,
    the length L of the crease is equal to 8 cos(θ). -/
theorem crease_length (θ : Real) (L : Real) : L = 8 * Real.cos θ := by
  sorry

end crease_length_l645_64592


namespace relationship_A_and_p_l645_64530

theorem relationship_A_and_p (x y p : ℝ) (A : ℝ) 
  (h1 : A = (x^2 - 3*y^2) / (3*x^2 + y^2))
  (h2 : p*x*y / (x^2 - (2+p)*x*y + 2*p*y^2) - y / (x - 2*y) = 1/2)
  (h3 : x ≠ 0)
  (h4 : y ≠ 0)
  (h5 : x ≠ 2*y)
  (h6 : x ≠ p*y) :
  A = (9*p^2 - 3) / (27*p^2 + 1) :=
by sorry

end relationship_A_and_p_l645_64530


namespace last_number_is_one_l645_64564

/-- A sequence of 1999 numbers with specific properties -/
def SpecialSequence : Type :=
  { a : Fin 1999 → ℤ // 
    a 0 = 1 ∧ 
    ∀ i : Fin 1997, a (i + 1) = a i + a (i + 2) }

/-- The last number in the SpecialSequence is 1 -/
theorem last_number_is_one (seq : SpecialSequence) : 
  seq.val (Fin.last 1998) = 1 := by
  sorry

end last_number_is_one_l645_64564


namespace parallel_lines_equal_angles_l645_64599

-- Define the concept of lines
variable (Line : Type)

-- Define the concept of angles between lines
variable (angle : Line → Line → ℝ)

-- Define the concept of parallel lines
variable (parallel : Line → Line → Prop)

-- Theorem statement
theorem parallel_lines_equal_angles (a b c : Line) (θ : ℝ) :
  parallel a b → angle a c = θ → angle b c = θ := by sorry

end parallel_lines_equal_angles_l645_64599


namespace total_money_is_250_l645_64568

/-- The amount of money James owns -/
def james_money : ℕ := 145

/-- The difference between James' and Ali's money -/
def difference : ℕ := 40

/-- The amount of money Ali owns -/
def ali_money : ℕ := james_money - difference

/-- The total amount of money owned by James and Ali -/
def total_money : ℕ := james_money + ali_money

theorem total_money_is_250 : total_money = 250 := by sorry

end total_money_is_250_l645_64568


namespace cousins_assignment_count_l645_64525

/-- The number of ways to assign n indistinguishable objects to k indistinguishable containers -/
def assign_indistinguishable (n k : ℕ) : ℕ :=
  sorry

/-- There are 4 rooms available -/
def num_rooms : ℕ := 4

/-- There are 5 cousins to assign -/
def num_cousins : ℕ := 5

/-- The number of ways to assign the cousins to the rooms is 51 -/
theorem cousins_assignment_count : assign_indistinguishable num_cousins num_rooms = 51 := by
  sorry

end cousins_assignment_count_l645_64525


namespace range_of_a_l645_64519

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def B (a : ℝ) : Set ℝ := {x | 2^(1-x) + a ≤ 0 ∧ x^2 - 2*(a + 7)*x + 5 ≤ 0}

-- State the theorem
theorem range_of_a (a : ℝ) : A ⊆ B a → -4 ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l645_64519


namespace repeating_decimal_sum_l645_64567

theorem repeating_decimal_sum : 
  (2 : ℚ) / 9 + (3 : ℚ) / 99 + (4 : ℚ) / 9999 = (843 : ℚ) / 3333 := by
  sorry

end repeating_decimal_sum_l645_64567


namespace prob_2_to_4_value_l645_64574

/-- The probability distribution of a random variable ξ -/
def P (k : ℕ) : ℚ := 1 / 2^k

/-- The probability that 2 < ξ ≤ 4 -/
def prob_2_to_4 : ℚ := P 3 + P 4

theorem prob_2_to_4_value : prob_2_to_4 = 3/16 := by sorry

end prob_2_to_4_value_l645_64574


namespace product_not_in_set_l645_64586

def a (n : ℕ) : ℕ := n^2 + n + 1

theorem product_not_in_set : ∃ m k : ℕ, ¬∃ n : ℕ, a m * a k = a n := by
  sorry

end product_not_in_set_l645_64586


namespace theater_admission_revenue_l645_64581

/-- Calculates the total amount collected from theater admission tickets. -/
theorem theater_admission_revenue
  (total_persons : ℕ)
  (num_children : ℕ)
  (adult_price : ℚ)
  (child_price : ℚ)
  (h1 : total_persons = 280)
  (h2 : num_children = 80)
  (h3 : adult_price = 60 / 100)
  (h4 : child_price = 25 / 100) :
  (total_persons - num_children) * adult_price + num_children * child_price = 140 / 100 := by
  sorry

end theater_admission_revenue_l645_64581


namespace triangle_side_minimization_l645_64561

theorem triangle_side_minimization (t C : ℝ) (ht : t > 0) (hC : 0 < C ∧ C < π) :
  let min_c := 2 * Real.sqrt (t * Real.tan (C / 2))
  ∀ a b c : ℝ, (a > 0 ∧ b > 0 ∧ c > 0) →
    (1/2 * a * b * Real.sin C = t) →
    (c^2 = a^2 + b^2 - 2*a*b*Real.cos C) →
    c ≥ min_c ∧
    (c = min_c ↔ a = b) := by
  sorry

end triangle_side_minimization_l645_64561


namespace correct_substitution_l645_64589

/-- Given a system of equations { y = 1 - x, x - 2y = 4 }, 
    the correct substitution using the substitution method is x - 2 + 2x = 4 -/
theorem correct_substitution (x y : ℝ) : 
  (y = 1 - x ∧ x - 2*y = 4) → (x - 2 + 2*x = 4) :=
by sorry

end correct_substitution_l645_64589


namespace prob_product_div_by_3_l645_64550

/-- The number of sides on a standard die -/
def num_sides : ℕ := 6

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability of rolling a number not divisible by 3 on one die -/
def prob_not_div_by_3 : ℚ := 2/3

/-- The probability that the product of the numbers rolled on 5 dice is divisible by 3 -/
theorem prob_product_div_by_3 : 
  (1 - prob_not_div_by_3 ^ num_dice) = 211/243 := by sorry

end prob_product_div_by_3_l645_64550


namespace sum_of_squares_equals_5020030_l645_64524

def numbers : List Nat := [1000, 1001, 1002, 1003, 1004]

theorem sum_of_squares_equals_5020030 :
  (numbers.map (λ x => x * x)).sum = 5020030 := by
  sorry

end sum_of_squares_equals_5020030_l645_64524


namespace path_area_calculation_l645_64529

/-- Calculates the area of a path around a rectangular field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

theorem path_area_calculation (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.8) :
  path_area field_length field_width path_width = 759.36 := by
  sorry

#eval path_area 75 55 2.8

end path_area_calculation_l645_64529


namespace new_boys_in_classroom_l645_64515

/-- The number of new boys that joined a classroom --/
def new_boys (initial_size : ℕ) (initial_girls_percent : ℚ) (final_girls_percent : ℚ) : ℕ :=
  sorry

/-- Theorem stating the number of new boys that joined the classroom --/
theorem new_boys_in_classroom :
  new_boys 20 (40 / 100) (32 / 100) = 5 := by sorry

end new_boys_in_classroom_l645_64515


namespace imaginary_power_sum_l645_64582

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := by
  sorry

end imaginary_power_sum_l645_64582


namespace moss_pollen_radius_scientific_notation_l645_64517

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem moss_pollen_radius_scientific_notation :
  let r := 0.0000042
  let sn := toScientificNotation r
  sn.significand = 4.2 ∧ sn.exponent = -6 := by sorry

end moss_pollen_radius_scientific_notation_l645_64517


namespace average_problem_l645_64588

theorem average_problem (x : ℝ) : 
  (2 + 4 + 1 + 3 + x) / 5 = 3 → x = 5 := by
  sorry

end average_problem_l645_64588


namespace smallest_root_of_unity_for_equation_l645_64559

theorem smallest_root_of_unity_for_equation : ∃ (n : ℕ),
  (n > 0) ∧ 
  (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^n = 1) ∧
  (∀ m : ℕ, m > 0 → (∀ z : ℂ, z^6 - z^3 + 1 = 0 → z^m = 1) → m ≥ n) ∧
  n = 18 :=
sorry

end smallest_root_of_unity_for_equation_l645_64559


namespace two_lines_at_constant_distance_l645_64531

/-- A line in a plane -/
structure Line where
  -- Add necessary fields to define a line

/-- Distance between two lines in a plane -/
def distance (l1 l2 : Line) : ℝ :=
  sorry

/-- Theorem: There are exactly two lines at a constant distance of 2 from a given line -/
theorem two_lines_at_constant_distance (l : Line) :
  ∃! (pair : (Line × Line)), (distance l pair.1 = 2 ∧ distance l pair.2 = 2) :=
sorry

end two_lines_at_constant_distance_l645_64531


namespace arithmetic_sequence_problem_l645_64565

/-- An arithmetic sequence with specific conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_a4 : a 4 = 7)
  (h_sum : a 3 + a 6 = 16)
  (h_an : ∃ n : ℕ, a n = 31) :
  ∃ n : ℕ, a n = 31 ∧ n = 16 := by
  sorry

end arithmetic_sequence_problem_l645_64565


namespace hyperbola_eccentricity_l645_64514

/-- Definition of a hyperbola with foci and points -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- The conditions of the problem -/
def hyperbola_conditions (Γ : Hyperbola) : Prop :=
  Γ.a > 0 ∧ Γ.b > 0 ∧
  (Γ.C.2 = 0) ∧  -- C is on x-axis
  (Γ.C.1 - Γ.B.1, Γ.C.2 - Γ.B.2) = 3 • (Γ.F₂.1 - Γ.A.1, Γ.F₂.2 - Γ.A.2) ∧  -- CB = 3F₂A
  (∃ t : ℝ, t > 0 ∧ Γ.B.1 - Γ.F₂.1 = t * (Γ.F₁.1 - Γ.C.1) ∧ Γ.B.2 - Γ.F₂.2 = t * (Γ.F₁.2 - Γ.C.2))  -- BF₂ bisects ∠F₁BC

/-- The theorem to be proved -/
theorem hyperbola_eccentricity (Γ : Hyperbola) :
  hyperbola_conditions Γ → (Real.sqrt ((Γ.F₁.1 - Γ.F₂.1)^2 + (Γ.F₁.2 - Γ.F₂.2)^2) / (2 * Γ.a) = Real.sqrt 7) :=
by sorry

end hyperbola_eccentricity_l645_64514


namespace age_ratio_l645_64526

/-- Represents the ages of Sam, Sue, and Kendra -/
structure Ages where
  sam : ℕ
  sue : ℕ
  kendra : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : Ages) : Prop :=
  ages.sam = 2 * ages.sue ∧
  ages.kendra = 18 ∧
  ages.sam + ages.sue + ages.kendra + 9 = 36

/-- The theorem to prove -/
theorem age_ratio (ages : Ages) (h : satisfiesConditions ages) : 
  ages.kendra / ages.sam = 3 := by
  sorry

/-- Auxiliary lemma to help with division -/
lemma div_eq_of_mul_eq {a b c : ℕ} (hb : b ≠ 0) (h : a = b * c) : a / b = c := by
  sorry

end age_ratio_l645_64526


namespace circle_equation_with_center_and_tangent_line_l645_64585

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Represents a line in polar form ρ = a * sin(θ - α) + b -/
structure PolarLine where
  a : ℝ
  α : ℝ
  b : ℝ

/-- Represents a circle in polar form ρ = R * sin(θ - β) -/
structure PolarCircle where
  R : ℝ
  β : ℝ

def is_tangent (c : PolarCircle) (l : PolarLine) : Prop :=
  sorry

theorem circle_equation_with_center_and_tangent_line 
  (P : PolarPoint) 
  (l : PolarLine) 
  (h1 : P.r = 2 ∧ P.θ = π/3) 
  (h2 : l.a = 1 ∧ l.α = π/3 ∧ l.b = 2) : 
  ∃ (c : PolarCircle), c.R = 4 ∧ c.β = -π/6 ∧ is_tangent c l :=
sorry

end circle_equation_with_center_and_tangent_line_l645_64585


namespace circle_area_tripled_l645_64572

theorem circle_area_tripled (r n : ℝ) : 
  (π * (r + n)^2 = 3 * π * r^2) → (r = n * (Real.sqrt 3 - 1) / 2) :=
by sorry

end circle_area_tripled_l645_64572


namespace original_group_size_l645_64537

/-- Given a group of men working on a task, this theorem proves that the original number of men is 42, based on the conditions provided. -/
theorem original_group_size (total_days : ℕ) (remaining_days : ℕ) (absent_men : ℕ) : 
  (total_days = 17) → (remaining_days = 21) → (absent_men = 8) →
  ∃ (original_size : ℕ), 
    (original_size > absent_men) ∧ 
    (1 : ℚ) / (total_days * original_size) = (1 : ℚ) / (remaining_days * (original_size - absent_men)) ∧
    original_size = 42 :=
by sorry

end original_group_size_l645_64537


namespace arithmetic_sequence_sum_l645_64539

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 4 + a 6 + a 8 + a 10 = 80) →
  (a 1 + a 13 = 40) :=
by
  sorry

end arithmetic_sequence_sum_l645_64539


namespace angle_value_l645_64570

theorem angle_value (a : ℝ) : 
  (180 - a = 3 * (90 - a)) → a = 45 := by sorry

end angle_value_l645_64570


namespace fort_blocks_count_l645_64532

/-- Represents the dimensions of a fort --/
structure FortDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of blocks needed to construct a fort --/
def blocksNeeded (d : FortDimensions) (wallThickness : ℕ) (floorThickness : ℕ) : ℕ :=
  let outerVolume := d.length * d.width * d.height
  let innerLength := d.length - 2 * wallThickness
  let innerWidth := d.width - 2 * wallThickness
  let innerHeight := d.height - floorThickness
  let innerVolume := innerLength * innerWidth * innerHeight
  let topLayerVolume := d.length * d.width
  outerVolume - innerVolume + topLayerVolume

/-- Theorem stating that the number of blocks needed for the given fort is 912 --/
theorem fort_blocks_count :
  let fortDims : FortDimensions := ⟨15, 12, 7⟩
  blocksNeeded fortDims 2 1 = 912 := by
  sorry

end fort_blocks_count_l645_64532


namespace line_slope_intercept_sum_l645_64573

/-- Given a line with slope -4 passing through the point (5, 2),
    prove that the sum of the slope and y-intercept is 18. -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = -4 ∧ 2 = m * 5 + b → m + b = 18 := by
  sorry

end line_slope_intercept_sum_l645_64573


namespace select_students_result_l645_64542

/-- The number of ways to select 4 students from two classes, with 2 students from each class, 
    such that exactly 1 female student is among them. -/
def select_students (class_a_male class_a_female class_b_male class_b_female : ℕ) : ℕ :=
  Nat.choose class_a_male 1 * Nat.choose class_a_female 1 * Nat.choose class_b_male 2 +
  Nat.choose class_a_male 2 * Nat.choose class_b_male 1 * Nat.choose class_b_female 1

/-- Theorem stating that the number of ways to select 4 students from two classes, 
    with 2 students from each class, such that exactly 1 female student is among them, 
    is equal to 345, given the specific class compositions. -/
theorem select_students_result : select_students 5 3 6 2 = 345 := by
  sorry

end select_students_result_l645_64542


namespace jogging_distance_l645_64551

/-- Alice's jogging speed in miles per minute -/
def alice_speed : ℚ := 1 / 12

/-- Bob's jogging speed in miles per minute -/
def bob_speed : ℚ := 3 / 40

/-- Total jogging time in minutes -/
def total_time : ℕ := 120

/-- The distance between Alice and Bob after jogging for the total time -/
def distance_apart : ℚ := alice_speed * total_time + bob_speed * total_time

theorem jogging_distance : distance_apart = 19 := by sorry

end jogging_distance_l645_64551


namespace inscribed_sphere_slant_angle_l645_64527

/-- A sphere inscribed in a cone with ratio k of tangency circle radius to base radius -/
structure InscribedSphere (k : ℝ) where
  /-- The ratio of the radius of the circle of tangency to the radius of the base of the cone -/
  ratio : k > 0 ∧ k < 1

/-- The cosine of the angle between the slant height and the base of the cone -/
def slant_base_angle_cosine (s : InscribedSphere k) : ℝ := 1 - k

/-- Theorem: The cosine of the angle between the slant height and the base of the cone
    for a sphere inscribed in a cone with ratio k is 1 - k -/
theorem inscribed_sphere_slant_angle (k : ℝ) (s : InscribedSphere k) :
  slant_base_angle_cosine s = 1 - k := by sorry

end inscribed_sphere_slant_angle_l645_64527
