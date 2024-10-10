import Mathlib

namespace distribute_seven_to_twelve_l3975_397529

/-- The number of ways to distribute distinct items to recipients -/
def distribute_ways (n_items : ℕ) (n_recipients : ℕ) : ℕ :=
  n_recipients ^ n_items

/-- Theorem: The number of ways to distribute 7 distinct items to 12 recipients,
    where each recipient can receive multiple items, is equal to 12^7 -/
theorem distribute_seven_to_twelve :
  distribute_ways 7 12 = 35831808 := by
  sorry

end distribute_seven_to_twelve_l3975_397529


namespace triangle_sides_relation_l3975_397544

theorem triangle_sides_relation (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) (h7 : Real.cos (2 * Real.pi / 3) = -1/2) :
  a^2 + a*c + c^2 - b^2 = 0 := by
  sorry

end triangle_sides_relation_l3975_397544


namespace parallelogram_area_l3975_397553

-- Define the conversion factor
def inch_to_mm : ℝ := 25.4

-- Define the parallelogram's dimensions
def base_inches : ℝ := 18
def height_mm : ℝ := 25.4

-- Theorem statement
theorem parallelogram_area :
  (base_inches * (height_mm / inch_to_mm)) = 18 := by
  sorry

end parallelogram_area_l3975_397553


namespace visitors_previous_day_l3975_397548

/-- The number of visitors to Buckingham Palace over 25 days -/
def total_visitors : ℕ := 949

/-- The number of days over which visitors were counted -/
def total_days : ℕ := 25

/-- The number of visitors on the previous day -/
def previous_day_visitors : ℕ := 246

/-- Theorem stating that the number of visitors on the previous day was 246 -/
theorem visitors_previous_day : previous_day_visitors = 246 := by
  sorry

end visitors_previous_day_l3975_397548


namespace juice_bottles_theorem_l3975_397586

theorem juice_bottles_theorem (bottle_capacity : ℕ) (required_amount : ℕ) (min_bottles : ℕ) : 
  bottle_capacity = 15 →
  required_amount = 195 →
  min_bottles = 13 →
  (min_bottles * bottle_capacity ≥ required_amount ∧
   ∀ n : ℕ, n * bottle_capacity ≥ required_amount → n ≥ min_bottles) :=
by sorry

end juice_bottles_theorem_l3975_397586


namespace distance_to_big_rock_l3975_397526

/-- The distance to Big Rock given the rower's speed, river current, and round trip time -/
theorem distance_to_big_rock 
  (rower_speed : ℝ) 
  (river_current : ℝ) 
  (round_trip_time : ℝ) 
  (h1 : rower_speed = 7)
  (h2 : river_current = 1)
  (h3 : round_trip_time = 1) : 
  ∃ (distance : ℝ), distance = 24 / 7 ∧ 
    (distance / (rower_speed - river_current) + 
     distance / (rower_speed + river_current) = round_trip_time) :=
by sorry

end distance_to_big_rock_l3975_397526


namespace trigonometric_simplification_l3975_397564

theorem trigonometric_simplification :
  (Real.tan (20 * π / 180) + Real.tan (30 * π / 180) + Real.tan (60 * π / 180) + Real.tan (70 * π / 180)) /
  Real.cos (10 * π / 180) = (Real.sqrt 3 + 2) * Real.sqrt 3 / 2 := by
  sorry

end trigonometric_simplification_l3975_397564


namespace absolute_value_identity_l3975_397561

theorem absolute_value_identity (x : ℝ) (h : x = 2023) : 
  |‖x‖ - x| - ‖x‖ - x = 0 := by
  sorry

end absolute_value_identity_l3975_397561


namespace trigonometric_expression_equality_l3975_397543

theorem trigonometric_expression_equality : 
  (Real.sin (15 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (165 * π / 180) * Real.cos (105 * π / 180)) / 
  (Real.sin (25 * π / 180) * Real.cos (5 * π / 180) + 
   Real.cos (155 * π / 180) * Real.cos (95 * π / 180)) = 
  1 / (2 * Real.cos (25 * π / 180)) := by sorry

end trigonometric_expression_equality_l3975_397543


namespace vector_c_value_l3975_397520

/-- Given vectors a and b, if vector c satisfies the parallel and perpendicular conditions,
    then c equals the specified vector. -/
theorem vector_c_value (a b c : ℝ × ℝ) : 
  a = (1, 2) →
  b = (2, -3) →
  (∃ k : ℝ, c + a = k • b) →
  (c.1 * (a.1 + b.1) + c.2 * (a.2 + b.2) = 0) →
  c = (-7/9, -7/3) := by
  sorry

end vector_c_value_l3975_397520


namespace neg_two_plus_one_eq_neg_one_l3975_397505

theorem neg_two_plus_one_eq_neg_one : (-2) + 1 = -1 := by
  sorry

end neg_two_plus_one_eq_neg_one_l3975_397505


namespace arithmetic_sequence_sum_l3975_397569

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n, prove S_8 = 80 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2) →  -- Definition of S_n
  S 4 = 24 →                                                      -- Given condition
  a 8 = 17 →                                                      -- Given condition
  S 8 = 80 := by
sorry

end arithmetic_sequence_sum_l3975_397569


namespace smallest_sum_proof_l3975_397577

def is_valid_sum (a b c d e f : ℕ) : Prop :=
  a ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  b ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  c ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  d ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  e ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  f ∈ ({1, 2, 3, 7, 8, 9} : Set ℕ) ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
  c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
  d ≠ e ∧ d ≠ f ∧
  e ≠ f ∧
  100 ≤ a ∧ a ≤ 999 ∧
  100 ≤ d ∧ d ≤ 999 ∧
  (100 * a + 10 * b + c) + (100 * d + 10 * e + f) ≤ 1500

theorem smallest_sum_proof :
  ∀ a b c d e f : ℕ,
    is_valid_sum a b c d e f →
    (100 * a + 10 * b + c) + (100 * d + 10 * e + f) ≥ 417 :=
by sorry

end smallest_sum_proof_l3975_397577


namespace angle_rotation_l3975_397578

theorem angle_rotation (initial_angle rotation : ℝ) (h1 : initial_angle = 60) (h2 : rotation = 630) :
  (initial_angle + rotation) % 360 = 330 ∧ 360 - (initial_angle + rotation) % 360 = 30 := by
  sorry

end angle_rotation_l3975_397578


namespace susie_fish_count_l3975_397583

/-- The number of fish caught by each family member and the total number of filets --/
structure FishingTrip where
  ben_fish : ℕ
  judy_fish : ℕ
  billy_fish : ℕ
  jim_fish : ℕ
  susie_fish : ℕ
  thrown_back : ℕ
  total_filets : ℕ

/-- Theorem stating that Susie caught 3 fish given the conditions of the fishing trip --/
theorem susie_fish_count (trip : FishingTrip) 
  (h1 : trip.ben_fish = 4)
  (h2 : trip.judy_fish = 1)
  (h3 : trip.billy_fish = 3)
  (h4 : trip.jim_fish = 2)
  (h5 : trip.thrown_back = 3)
  (h6 : trip.total_filets = 24)
  (h7 : ∀ (fish : ℕ), fish * 2 = trip.total_filets → 
    fish = trip.ben_fish + trip.judy_fish + trip.billy_fish + trip.jim_fish + trip.susie_fish - trip.thrown_back) :
  trip.susie_fish = 3 := by
  sorry

end susie_fish_count_l3975_397583


namespace parkway_elementary_soccer_l3975_397560

theorem parkway_elementary_soccer (total_students : ℕ) (boys : ℕ) (soccer_players : ℕ) 
  (boys_soccer_percentage : ℚ) :
  total_students = 420 →
  boys = 296 →
  soccer_players = 250 →
  boys_soccer_percentage = 86 / 100 →
  (total_students - boys) - (soccer_players - (boys_soccer_percentage * soccer_players).floor) = 89 := by
  sorry

end parkway_elementary_soccer_l3975_397560


namespace crow_probability_l3975_397558

/-- Represents the number of crows of each color on each tree -/
structure CrowCounts where
  birchWhite : ℕ
  birchBlack : ℕ
  oakWhite : ℕ
  oakBlack : ℕ

/-- The probability of the number of white crows on the birch returning to its initial count -/
def probReturnToInitial (c : CrowCounts) : ℚ :=
  (c.birchBlack * (c.oakBlack + 1) + c.birchWhite * (c.oakWhite + 1)) / (50 * 51)

/-- The probability of the number of white crows on the birch changing -/
def probChange (c : CrowCounts) : ℚ :=
  (c.birchBlack * c.oakWhite + c.birchWhite * c.oakBlack) / (50 * 51)

theorem crow_probability (c : CrowCounts) 
  (h1 : c.birchWhite + c.birchBlack = 50)
  (h2 : c.oakWhite + c.oakBlack = 50)
  (h3 : c.birchWhite > 0)
  (h4 : c.birchBlack ≥ c.birchWhite)
  (h5 : c.oakBlack ≥ c.oakWhite - 1) :
  probReturnToInitial c > probChange c := by
  sorry

end crow_probability_l3975_397558


namespace flowchart_requirement_l3975_397546

-- Define the structure of a flowchart
structure Flowchart where
  boxes : Set (Operation)
  flowLines : Set (SequenceIndicator)

-- Define operations
inductive Operation
  | process : Operation
  | decision : Operation
  | inputOutput : Operation

-- Define sequence indicators
inductive SequenceIndicator
  | arrow : SequenceIndicator

-- Define the direction of flow
inductive FlowDirection
  | leftToRight : FlowDirection
  | topToBottom : FlowDirection

-- Define the general requirement for drawing a flowchart
def generalRequirement : (FlowDirection × FlowDirection) := (FlowDirection.leftToRight, FlowDirection.topToBottom)

-- Theorem: The general requirement for drawing a flowchart is from left to right, from top to bottom
theorem flowchart_requirement (f : Flowchart) : 
  generalRequirement = (FlowDirection.leftToRight, FlowDirection.topToBottom) := by
  sorry

end flowchart_requirement_l3975_397546


namespace percentage_difference_l3975_397550

theorem percentage_difference (A B : ℝ) (h1 : A > 0) (h2 : B > A) :
  let x := 100 * (B - A) / A
  B = A * (1 + x / 100) := by sorry

end percentage_difference_l3975_397550


namespace triangle_side_length_l3975_397567

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    if the area is √3, B = 60°, and a² + c² = 3ac, then b = 2√2. -/
theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) : 
  (1/2 : ℝ) * a * c * Real.sin B = Real.sqrt 3 →
  B = π / 3 →
  a^2 + c^2 = 3 * a * c →
  b = 2 * Real.sqrt 2 := by
  sorry


end triangle_side_length_l3975_397567


namespace not_tangent_implies_a_less_than_one_third_l3975_397508

/-- The function f(x) = x³ - 3ax --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*a*x

/-- The derivative of f(x) --/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 - 3*a

/-- Theorem stating that if the line x + y + m = 0 is not a tangent to y = f(x) for any m,
    then a < 1/3 --/
theorem not_tangent_implies_a_less_than_one_third (a : ℝ) :
  (∀ m : ℝ, ¬∃ x : ℝ, f_derivative a x = -1 ∧ f a x = -(x + m)) →
  a < 1/3 :=
sorry

end not_tangent_implies_a_less_than_one_third_l3975_397508


namespace units_digit_of_17_power_2007_l3975_397595

theorem units_digit_of_17_power_2007 : 17^2007 % 10 = 3 := by
  sorry

end units_digit_of_17_power_2007_l3975_397595


namespace solution_count_l3975_397514

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem solution_count (a : ℝ) :
  (∀ x > 1, f x ≠ (x - 1) * (a * x - a + 1)) ∨
  (a > 0 ∧ a < 1/2 ∧ (∀ x > 1, f x = (x - 1) * (a * x - a + 1) → 
    ∀ y > 1, y ≠ x → f y ≠ (y - 1) * (a * y - a + 1))) :=
by sorry

end solution_count_l3975_397514


namespace exactly_one_absent_l3975_397541

-- Define the three guests
variable (B K Z : Prop)

-- B: Baba Yaga comes to the festival
-- K: Koschei comes to the festival
-- Z: Zmey Gorynych comes to the festival

-- Define the conditions
axiom condition1 : ¬B → K
axiom condition2 : ¬K → Z
axiom condition3 : ¬Z → B
axiom at_least_one_absent : ¬B ∨ ¬K ∨ ¬Z

-- Theorem to prove
theorem exactly_one_absent : (¬B ∧ K ∧ Z) ∨ (B ∧ ¬K ∧ Z) ∨ (B ∧ K ∧ ¬Z) :=
sorry

end exactly_one_absent_l3975_397541


namespace arithmetic_progression_squares_l3975_397572

theorem arithmetic_progression_squares (a b c : ℝ) 
  (h : (1 / (a + b) + 1 / (b + c)) / 2 = 1 / (a + c)) : 
  a^2 + c^2 = 2 * b^2 := by
sorry

end arithmetic_progression_squares_l3975_397572


namespace factors_of_1320_l3975_397510

theorem factors_of_1320 : Nat.card (Nat.divisors 1320) = 32 := by
  sorry

end factors_of_1320_l3975_397510


namespace lighthouse_coverage_l3975_397524

/-- Represents a lighthouse with its illumination angle -/
structure Lighthouse where
  angle : ℝ

/-- Represents the Persian Gulf as a circle -/
def PersianGulf : ℝ := 360

/-- The number of lighthouses -/
def num_lighthouses : ℕ := 18

/-- The illumination angle of each lighthouse -/
def lighthouse_angle : ℝ := 20

/-- Proves that the lighthouses can cover the entire Persian Gulf -/
theorem lighthouse_coverage (lighthouses : Fin num_lighthouses → Lighthouse)
  (h1 : ∀ i, (lighthouses i).angle = lighthouse_angle)
  (h2 : lighthouse_angle * num_lighthouses = PersianGulf) :
  ∃ (arrangement : Fin num_lighthouses → ℝ),
    (∀ i, 0 ≤ arrangement i ∧ arrangement i < PersianGulf) ∧
    (∀ x, 0 ≤ x ∧ x < PersianGulf →
      ∃ i, x ∈ Set.Icc (arrangement i) ((arrangement i + (lighthouses i).angle) % PersianGulf)) :=
by sorry

end lighthouse_coverage_l3975_397524


namespace incorrect_statement_proof_l3975_397516

/-- Given non-empty sets A and B where A is not a subset of B, 
    prove that the statement "If x ∉ A, then x ∈ B is an impossible event" is false. -/
theorem incorrect_statement_proof 
  {α : Type*} (A B : Set α) (h_nonempty_A : A.Nonempty) (h_nonempty_B : B.Nonempty) 
  (h_not_subset : ¬(A ⊆ B)) :
  ¬(∀ x, x ∉ A → x ∉ B) :=
sorry

end incorrect_statement_proof_l3975_397516


namespace sam_distance_walked_sam_walks_25_miles_l3975_397533

-- Define the constants
def total_distance : ℝ := 55
def fred_speed : ℝ := 6
def sam_speed : ℝ := 5

-- Define the theorem
theorem sam_distance_walked : ℝ := by
  -- The distance Sam walks
  let d : ℝ := sam_speed * (total_distance / (fred_speed + sam_speed))
  -- Prove that d equals 25
  sorry

-- The main theorem
theorem sam_walks_25_miles :
  sam_distance_walked = 25 := by sorry

end sam_distance_walked_sam_walks_25_miles_l3975_397533


namespace inequalities_not_equivalent_l3975_397588

theorem inequalities_not_equivalent : 
  ¬(∀ x : ℝ, (x - 3) / (x^2 - 5*x + 6) < 2 ↔ 2*x^2 - 11*x + 15 > 0) :=
by sorry

end inequalities_not_equivalent_l3975_397588


namespace negation_of_positive_quadratic_inequality_l3975_397551

theorem negation_of_positive_quadratic_inequality :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x + 1 > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x + 1 ≤ 0) :=
by sorry

end negation_of_positive_quadratic_inequality_l3975_397551


namespace remainder_7n_mod_4_l3975_397552

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l3975_397552


namespace no_triangle_satisfies_equation_l3975_397568

-- Define a structure for a triangle
structure Triangle where
  x : ℝ
  y : ℝ
  z : ℝ
  x_pos : 0 < x
  y_pos : 0 < y
  z_pos : 0 < z
  triangle_ineq1 : x + y > z
  triangle_ineq2 : y + z > x
  triangle_ineq3 : z + x > y

-- Theorem statement
theorem no_triangle_satisfies_equation :
  ¬∃ t : Triangle, t.x^3 + t.y^3 + t.z^3 = (t.x + t.y) * (t.y + t.z) * (t.z + t.x) :=
sorry

end no_triangle_satisfies_equation_l3975_397568


namespace function_satisfies_condition_l3975_397511

-- Define the function y
def y (x : ℝ) : ℝ := x - 2

-- State the theorem
theorem function_satisfies_condition :
  y 1 = -1 := by sorry

end function_satisfies_condition_l3975_397511


namespace train_length_l3975_397562

/-- The length of a train given its speed, time to cross a bridge, and the bridge's length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 45 * (1000 / 3600) →
  crossing_time = 30 →
  bridge_length = 225 →
  train_speed * crossing_time - bridge_length = 150 := by
  sorry

end train_length_l3975_397562


namespace mandy_jackson_age_difference_l3975_397594

/-- Proves that Mandy is 10 years older than Jackson given the conditions of the problem -/
theorem mandy_jackson_age_difference :
  ∀ (mandy_age jackson_age adele_age : ℕ),
    jackson_age = 20 →
    adele_age = (3 * jackson_age) / 4 →
    mandy_age + jackson_age + adele_age + 30 = 95 →
    mandy_age > jackson_age →
    mandy_age - jackson_age = 10 := by
  sorry

end mandy_jackson_age_difference_l3975_397594


namespace points_are_collinear_l3975_397597

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Three points are collinear if the slope between any two pairs is equal -/
def collinear (p1 p2 p3 : Point) : Prop :=
  (p2.y - p1.y) * (p3.x - p2.x) = (p3.y - p2.y) * (p2.x - p1.x)

theorem points_are_collinear : 
  let p1 : Point := ⟨3, 1⟩
  let p2 : Point := ⟨6, 6.4⟩
  let p3 : Point := ⟨8, 10⟩
  collinear p1 p2 p3 := by
  sorry

end points_are_collinear_l3975_397597


namespace inequality_sine_square_l3975_397507

theorem inequality_sine_square (x : ℝ) (h : x ∈ Set.Ioo 0 (π / 2)) : 
  0 < (1 / Real.sin x ^ 2) - (1 / x ^ 2) ∧ (1 / Real.sin x ^ 2) - (1 / x ^ 2) < 1 := by
  sorry

end inequality_sine_square_l3975_397507


namespace number_problem_l3975_397585

theorem number_problem (N : ℝ) : 
  (1/2 : ℝ) * ((3/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * N) = 45 → 
  (65/100 : ℝ) * N = 585 := by
  sorry

end number_problem_l3975_397585


namespace unique_solution_for_2n_plus_1_eq_m2_l3975_397571

theorem unique_solution_for_2n_plus_1_eq_m2 :
  ∃! (m n : ℕ), 2^n + 1 = m^2 ∧ m = 3 ∧ n = 3 := by sorry

end unique_solution_for_2n_plus_1_eq_m2_l3975_397571


namespace complex_modulus_l3975_397549

theorem complex_modulus (z : ℂ) : z = (1 + I) / (2 - I) → Complex.abs z = Real.sqrt 10 / 5 := by
  sorry

end complex_modulus_l3975_397549


namespace coefficient_of_x_cubed_l3975_397575

def expression (x : ℝ) : ℝ :=
  4 * (x^2 - 2*x^3 + 2*x) + 2 * (x + 3*x^3 - 2*x^2 + 4*x^5 - x^3) - 6 * (2 + 2*x - 5*x^3 - 3*x^2 + x^4)

theorem coefficient_of_x_cubed : 
  (deriv (deriv (deriv expression))) 0 / 6 = 26 := by sorry

end coefficient_of_x_cubed_l3975_397575


namespace modulus_of_complex_number_l3975_397518

theorem modulus_of_complex_number (α : Real) (h : π < α ∧ α < 2*π) :
  Complex.abs (1 + Complex.cos α + Complex.I * Complex.sin α) = -2 * Real.cos (α/2) :=
sorry

end modulus_of_complex_number_l3975_397518


namespace alpha_plus_beta_equals_two_l3975_397532

theorem alpha_plus_beta_equals_two (α β : ℝ) 
  (h1 : α^3 - 3*α^2 + 5*α = 1) 
  (h2 : β^3 - 3*β^2 + 5*β = 5) : 
  α + β = 2 := by
sorry

end alpha_plus_beta_equals_two_l3975_397532


namespace revenue_change_l3975_397565

theorem revenue_change
  (initial_price initial_quantity : ℝ)
  (price_increase : ℝ)
  (quantity_decrease : ℝ)
  (h_price : price_increase = 0.4)
  (h_quantity : quantity_decrease = 0.2)
  : (1 + price_increase) * (1 - quantity_decrease) * initial_price * initial_quantity
    = 1.12 * initial_price * initial_quantity :=
by sorry

end revenue_change_l3975_397565


namespace extended_morse_code_symbols_l3975_397522

theorem extended_morse_code_symbols : 
  (Finset.range 5).sum (fun n => 2^(n+1)) = 62 := by
  sorry

end extended_morse_code_symbols_l3975_397522


namespace distance_difference_l3975_397576

/-- The line l -/
def line_l (x y : ℝ) : Prop := x - y - 1 = 0

/-- The ellipse C₁ -/
def ellipse_C₁ (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- Point F -/
def point_F : ℝ × ℝ := (1, 0)

/-- Point F₁ -/
def point_F₁ : ℝ × ℝ := (-1, 0)

/-- Theorem stating the difference of distances -/
theorem distance_difference (A B : ℝ × ℝ) 
  (h_line_A : line_l A.1 A.2)
  (h_line_B : line_l B.1 B.2)
  (h_ellipse_A : ellipse_C₁ A.1 A.2)
  (h_ellipse_B : ellipse_C₁ B.1 B.2)
  (h_above : A.2 > B.2) :
  |point_F₁.1 - A.1|^2 + |point_F₁.2 - A.2|^2 - 
  (|point_F₁.1 - B.1|^2 + |point_F₁.2 - B.2|^2) = (6 * Real.sqrt 2 / 7)^2 :=
sorry

end distance_difference_l3975_397576


namespace max_product_sum_300_l3975_397598

theorem max_product_sum_300 :
  ∀ x y : ℤ, x + y = 300 → x * y ≤ 22500 :=
by sorry

end max_product_sum_300_l3975_397598


namespace equation_proof_l3975_397527

theorem equation_proof : 121 + 2 * 11 * 8 + 64 = 361 := by
  sorry

end equation_proof_l3975_397527


namespace rachel_coloring_books_l3975_397512

/-- The number of pictures Rachel still has to color -/
def remaining_pictures (book1_pictures book2_pictures colored_pictures : ℕ) : ℕ :=
  book1_pictures + book2_pictures - colored_pictures

/-- Theorem: Rachel has 11 pictures left to color -/
theorem rachel_coloring_books :
  remaining_pictures 23 32 44 = 11 := by
  sorry

end rachel_coloring_books_l3975_397512


namespace one_intersection_values_l3975_397573

/-- The function f(x) parameterized by m -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 4) * x^2 - 2 * m * x - m - 6

/-- The discriminant of the quadratic function f(x) -/
def discriminant (m : ℝ) : ℝ := 4 * m^2 - 4 * (m - 4) * (-m - 6)

/-- Predicate to check if f(x) has only one intersection with x-axis -/
def has_one_intersection (m : ℝ) : Prop :=
  (m = 4) ∨ (discriminant m = 0)

/-- Theorem stating the values of m for which f(x) has one intersection with x-axis -/
theorem one_intersection_values :
  ∀ m : ℝ, has_one_intersection m ↔ m = -4 ∨ m = 3 ∨ m = 4 :=
sorry

end one_intersection_values_l3975_397573


namespace odd_even_function_sum_l3975_397570

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_even_function_sum (f : ℝ → ℝ) 
  (h_odd : is_odd f) 
  (h_even : is_even (fun x ↦ f (x + 2))) 
  (h_f1 : f 1 = 1) : 
  f 8 + f 17 = 1 := by sorry

end odd_even_function_sum_l3975_397570


namespace carrot_sticks_after_dinner_l3975_397530

-- Define the variables
def before_dinner : ℕ := 22
def total : ℕ := 37

-- Define the theorem
theorem carrot_sticks_after_dinner :
  total - before_dinner = 15 := by
  sorry

end carrot_sticks_after_dinner_l3975_397530


namespace prob_same_color_is_89_169_l3975_397555

def total_balls : ℕ := 13
def blue_balls : ℕ := 8
def yellow_balls : ℕ := 5

def prob_same_color : ℚ :=
  (blue_balls * blue_balls + yellow_balls * yellow_balls) / (total_balls * total_balls)

theorem prob_same_color_is_89_169 : prob_same_color = 89 / 169 := by
  sorry

end prob_same_color_is_89_169_l3975_397555


namespace stem_and_leaf_plot_preserves_information_l3975_397592

-- Define the different types of charts
inductive ChartType
  | BarChart
  | PieChart
  | LineChart
  | StemAndLeafPlot

-- Define a property for information preservation
def preserves_all_information (chart : ChartType) : Prop :=
  match chart with
  | ChartType.StemAndLeafPlot => True
  | _ => False

-- Theorem statement
theorem stem_and_leaf_plot_preserves_information :
  ∀ (chart : ChartType), preserves_all_information chart ↔ chart = ChartType.StemAndLeafPlot :=
by
  sorry


end stem_and_leaf_plot_preserves_information_l3975_397592


namespace square_diagonal_l3975_397521

theorem square_diagonal (area : ℝ) (side : ℝ) (diagonal : ℝ) 
  (h1 : area = 4802) 
  (h2 : area = side ^ 2) 
  (h3 : diagonal ^ 2 = 2 * side ^ 2) : 
  diagonal = Real.sqrt (2 * 4802) := by
sorry

end square_diagonal_l3975_397521


namespace z_max_min_difference_l3975_397534

theorem z_max_min_difference (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) : 
  let z := fun (a b : ℝ) => |a^2 - b^2| / (|a^2| + |b^2|)
  ∃ (max min : ℝ), 
    (∀ a b, a ≠ 0 → b ≠ 0 → a ≠ b → z a b ≤ max) ∧
    (∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ z a b = max) ∧
    (∀ a b, a ≠ 0 → b ≠ 0 → a ≠ b → min ≤ z a b) ∧
    (∃ a b, a ≠ 0 ∧ b ≠ 0 ∧ a ≠ b ∧ z a b = min) ∧
    max = 1 ∧ min = 0 ∧ max - min = 1 :=
by sorry

end z_max_min_difference_l3975_397534


namespace modular_congruence_solution_l3975_397506

theorem modular_congruence_solution (n : ℤ) : 
  0 ≤ n ∧ n < 103 ∧ (98 * n) % 103 = 33 % 103 → n % 103 = 87 % 103 := by
  sorry

end modular_congruence_solution_l3975_397506


namespace sum_after_100_operations_l3975_397540

/-- The operation that inserts the difference between each pair of neighboring numbers -/
def insertDifferences (s : List Int) : List Int :=
  sorry

/-- Applies the insertDifferences operation n times to a list -/
def applyNTimes (s : List Int) (n : Nat) : List Int :=
  sorry

/-- The sum of a list of integers -/
def listSum (s : List Int) : Int :=
  sorry

theorem sum_after_100_operations :
  let initialSequence : List Int := [1, 9, 8, 8]
  listSum (applyNTimes initialSequence 100) = 726 := by
  sorry

end sum_after_100_operations_l3975_397540


namespace tom_total_distance_l3975_397515

/-- Calculates the total distance covered by Tom given his swimming and running times and speeds. -/
theorem tom_total_distance (swim_time swim_speed : ℝ) (h1 : swim_time = 2) (h2 : swim_speed = 2)
  (h3 : swim_time > 0) (h4 : swim_speed > 0) : 
  let run_time := swim_time / 2
  let run_speed := 4 * swim_speed
  swim_time * swim_speed + run_time * run_speed = 12 := by
  sorry

#check tom_total_distance

end tom_total_distance_l3975_397515


namespace hyperbola_asymptotes_l3975_397531

/-- Given a hyperbola and a circle, prove the equations of the hyperbola's asymptotes -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∀ x y : ℝ, (x - 2)^2 + y^2 = 1 → 
    ∃ k : ℝ, (x = k ∧ y = k * (b / a)) ∨ (x = -k ∧ y = k * (b / a))) →
  ∃ c : ℝ, c^2 = 3 ∧ (∀ x y : ℝ, x + c * y = 0 ∨ x - c * y = 0) :=
by sorry

end hyperbola_asymptotes_l3975_397531


namespace binomial_theorem_and_sum_l3975_397589

def binomial_expansion (m : ℝ) : ℕ → ℝ
| 0 => 1
| 1 => 7 * m
| 2 => 21 * m^2
| 3 => 35 * m^3
| 4 => 35 * m^4
| 5 => 21 * m^5
| 6 => 7 * m^6
| 7 => m^7
| _ => 0

def a (m : ℝ) (i : ℕ) : ℝ := binomial_expansion m i

theorem binomial_theorem_and_sum (m : ℝ) :
  a m 3 = -280 →
  (m = -2 ∧ a m 1 + a m 3 + a m 5 + a m 7 = -1094) := by sorry

end binomial_theorem_and_sum_l3975_397589


namespace inequality_always_true_l3975_397587

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > d) :
  a + c > b + d := by
  sorry

end inequality_always_true_l3975_397587


namespace inequality_proof_l3975_397557

theorem inequality_proof (a b c : ℝ) 
  (ha : a = Real.log (1 + Real.exp 1))
  (hb : b = Real.sqrt (Real.exp 1))
  (hc : c = 2 * Real.exp 1 / 3) :
  c > b ∧ b > a :=
by sorry

end inequality_proof_l3975_397557


namespace roots_sum_product_equal_l3975_397528

theorem roots_sum_product_equal (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^2 - (m+2)*x + m^2 = 0 ∧ 
    y^2 - (m+2)*y + m^2 = 0 ∧ 
    x + y = x * y) → 
  m = 2 := by
  sorry

end roots_sum_product_equal_l3975_397528


namespace arctan_sum_equals_pi_fourth_l3975_397542

theorem arctan_sum_equals_pi_fourth (y : ℝ) : 
  2 * Real.arctan (1/3) + Real.arctan (1/7) + Real.arctan (1/y) = π/4 → y = 2 := by
  sorry

end arctan_sum_equals_pi_fourth_l3975_397542


namespace intersection_of_A_and_B_l3975_397502

-- Define the sets A and B
def A : Set (ℝ × ℝ) := {p | p.2 = p.1 + 1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 1}
def B : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 ∧ 0 ≤ p.1 ∧ p.1 ≤ 10}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {(1, 2)} := by sorry

end intersection_of_A_and_B_l3975_397502


namespace subtraction_of_fractions_l3975_397566

theorem subtraction_of_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by sorry

end subtraction_of_fractions_l3975_397566


namespace parallelogram_fourth_vertex_l3975_397563

/-- A parallelogram in 2D space --/
structure Parallelogram where
  a : ℝ × ℝ
  b : ℝ × ℝ
  c : ℝ × ℝ
  d : ℝ × ℝ

/-- The property that defines a parallelogram --/
def isParallelogram (p : Parallelogram) : Prop :=
  (p.a.1 + p.c.1 = p.b.1 + p.d.1) ∧ 
  (p.a.2 + p.c.2 = p.b.2 + p.d.2)

theorem parallelogram_fourth_vertex 
  (p : Parallelogram)
  (h1 : p.a = (-1, 0))
  (h2 : p.b = (3, 0))
  (h3 : p.c = (1, -5))
  (h4 : isParallelogram p) :
  p.d = (1, 5) ∨ p.d = (-3, -5) := by
  sorry

#check parallelogram_fourth_vertex

end parallelogram_fourth_vertex_l3975_397563


namespace quadratic_roots_sum_l3975_397539

theorem quadratic_roots_sum (a b c : ℝ) : 
  (∀ x : ℝ, a * (x^4 + x^2)^2 + b * (x^4 + x^2) + c ≥ a * (x^3 + 2)^2 + b * (x^3 + 2) + c) →
  (∃ r₁ r₂ : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = a * (x - r₁) * (x - r₂)) →
  r₁ + r₂ = -4 := by
sorry

end quadratic_roots_sum_l3975_397539


namespace plus_shape_perimeter_l3975_397593

/-- A shape formed by eight congruent squares arranged in a "plus" sign -/
structure PlusShape where
  /-- The side length of each square in the shape -/
  side_length : ℝ
  /-- The total area of the shape -/
  total_area : ℝ
  /-- The shape is formed by eight congruent squares -/
  area_eq : total_area = 8 * side_length ^ 2

/-- The perimeter of a PlusShape -/
def perimeter (shape : PlusShape) : ℝ := 12 * shape.side_length

theorem plus_shape_perimeter (shape : PlusShape) (h : shape.total_area = 648) :
  perimeter shape = 108 := by
  sorry

#check plus_shape_perimeter

end plus_shape_perimeter_l3975_397593


namespace circumcircle_tangent_to_excircle_l3975_397580

-- Define the points and circles
variable (A B C D E B₁ C₁ I J S : Point)
variable (Ω : Circle)

-- Define the convex quadrilateral ABCD
def is_convex_quadrilateral (A B C D : Point) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D E : Point) : Prop := sorry

-- Define the common excircle of triangles
def common_excircle (A B C D E : Point) (Ω : Circle) : Prop := sorry

-- Define tangent points
def tangent_points (A E D B₁ C₁ : Point) (Ω : Circle) : Prop := sorry

-- Define incircle centers
def incircle_centers (A B E C D I J : Point) : Prop := sorry

-- Define intersection of IC₁ and JB₁
def segments_intersect_at (I C₁ J B₁ S : Point) : Prop := sorry

-- Define S lying on Ω
def point_on_circle (S : Point) (Ω : Circle) : Prop := sorry

-- Define the circumcircle of a triangle
def circumcircle (A E D : Point) : Circle := sorry

-- Define tangency of circles
def circles_tangent (c₁ c₂ : Circle) : Prop := sorry

-- Theorem statement
theorem circumcircle_tangent_to_excircle 
  (h₁ : is_convex_quadrilateral A B C D)
  (h₂ : diagonals_intersect_at A B C D E)
  (h₃ : common_excircle A B C D E Ω)
  (h₄ : tangent_points A E D B₁ C₁ Ω)
  (h₅ : incircle_centers A B E C D I J)
  (h₆ : segments_intersect_at I C₁ J B₁ S)
  (h₇ : point_on_circle S Ω) :
  circles_tangent (circumcircle A E D) Ω :=
sorry

end circumcircle_tangent_to_excircle_l3975_397580


namespace incorrect_exponent_operation_l3975_397519

theorem incorrect_exponent_operation (a : ℝ) : (-a^2)^3 ≠ -a^5 := by
  sorry

end incorrect_exponent_operation_l3975_397519


namespace f_greater_than_exp_l3975_397537

open Set
open Function
open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- State the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x, deriv f x > f x)
variable (h3 : f 0 = 1)

-- Theorem statement
theorem f_greater_than_exp (x : ℝ) : f x > Real.exp x ↔ x > 0 := by
  sorry

end f_greater_than_exp_l3975_397537


namespace students_taking_one_subject_l3975_397596

theorem students_taking_one_subject (both : ℕ) (geometry : ℕ) (only_biology : ℕ)
  (h1 : both = 15)
  (h2 : geometry = 40)
  (h3 : only_biology = 20) :
  geometry - both + only_biology = 45 := by
  sorry

end students_taking_one_subject_l3975_397596


namespace tonis_dimes_l3975_397599

/-- Represents the savings of three kids in cents -/
structure Savings where
  teagan : ℕ  -- Teagan's savings in pennies
  rex : ℕ     -- Rex's savings in nickels
  toni : ℕ    -- Toni's savings in dimes
  total : ℕ   -- Total savings in cents

/-- Theorem stating that given the conditions, Toni saved 330 dimes -/
theorem tonis_dimes (s : Savings) : 
  s.teagan = 200 ∧ s.rex = 100 ∧ s.total = 4000 → s.toni = 330 := by
  sorry

#check tonis_dimes

end tonis_dimes_l3975_397599


namespace initial_boarders_count_prove_initial_boarders_count_l3975_397504

/-- Proves that the initial number of boarders is 120 given the conditions of the problem -/
theorem initial_boarders_count : ℕ → ℕ → Prop :=
  fun initial_boarders initial_day_students =>
    -- Initial ratio of boarders to day students is 2:5
    (initial_boarders : ℚ) / initial_day_students = 2 / 5 →
    -- After 30 new boarders join, the ratio becomes 1:2
    ((initial_boarders : ℚ) + 30) / initial_day_students = 1 / 2 →
    -- The initial number of boarders is 120
    initial_boarders = 120

-- The proof of the theorem
theorem prove_initial_boarders_count : ∃ (initial_boarders initial_day_students : ℕ),
  initial_boarders_count initial_boarders initial_day_students :=
by
  sorry -- Proof is omitted as per instructions

end initial_boarders_count_prove_initial_boarders_count_l3975_397504


namespace angle_property_l3975_397559

theorem angle_property (θ : Real) (h1 : 0 < θ ∧ θ < π) 
  (h2 : Real.sin θ * Real.cos θ = -1/8) : 
  Real.sin θ - Real.cos θ = Real.sqrt 5 / 2 := by
sorry

end angle_property_l3975_397559


namespace fuel_tank_capacity_l3975_397536

theorem fuel_tank_capacity : ∀ (x : ℚ), 
  (5 / 6 : ℚ) * x - (2 / 3 : ℚ) * x = 15 → x = 90 := by
  sorry

end fuel_tank_capacity_l3975_397536


namespace exp_gt_m_ln_plus_two_l3975_397523

theorem exp_gt_m_ln_plus_two (x m : ℝ) (hx : x > 0) (hm : 0 < m) (hm1 : m ≤ 1) :
  Real.exp x > m * (Real.log x + 2) := by
  sorry

end exp_gt_m_ln_plus_two_l3975_397523


namespace find_number_l3975_397556

theorem find_number : ∃ x : ℝ, 2.12 + 0.345 + x = 2.4690000000000003 ∧ x = 0.0040000000000003 := by
  sorry

end find_number_l3975_397556


namespace multiple_of_112_implies_multiple_of_8_l3975_397554

theorem multiple_of_112_implies_multiple_of_8 (n : ℤ) : 
  (∃ k : ℤ, 14 * n = 112 * k) → (∃ m : ℤ, n = 8 * m) := by
  sorry

end multiple_of_112_implies_multiple_of_8_l3975_397554


namespace arithmetic_calculations_l3975_397517

theorem arithmetic_calculations :
  ((-15) + (-23) - 26 - (-15) = -49) ∧
  ((-1/2 + 2/3 - 1/4) * (-24) = 2) ∧
  ((-24) / (-6) * (-1/4) = -1) ∧
  ((-1)^2024 - (-2)^3 - 3^2 + 2 / (2/3) * (3/2) = 5/2) := by
  sorry

end arithmetic_calculations_l3975_397517


namespace current_rate_calculation_l3975_397535

/-- Given a boat traveling downstream, calculate the rate of the current. -/
theorem current_rate_calculation 
  (boat_speed : ℝ) -- Speed of the boat in still water (km/hr)
  (distance : ℝ)   -- Distance traveled downstream (km)
  (time : ℝ)       -- Time traveled downstream (minutes)
  (h1 : boat_speed = 42)
  (h2 : distance = 36.67)
  (h3 : time = 44)
  : ∃ (current_rate : ℝ), current_rate = 8 ∧ 
    distance = (boat_speed + current_rate) * (time / 60) :=
by sorry


end current_rate_calculation_l3975_397535


namespace midpoint_trajectory_l3975_397581

/-- Given a point P (x_P, y_P) on the curve 2x^2 - y = 0 and a fixed point A (0, -1),
    the midpoint M (x, y) of AP satisfies the equation 8x^2 - 2y - 1 = 0 -/
theorem midpoint_trajectory (x_P y_P x y : ℝ) : 
  (2 * x_P^2 = y_P) →  -- P is on the curve 2x^2 - y = 0
  (x = x_P / 2) →      -- x-coordinate of midpoint
  (y = (y_P - 1) / 2)  -- y-coordinate of midpoint
  → 8 * x^2 - 2 * y - 1 = 0 := by sorry

end midpoint_trajectory_l3975_397581


namespace cos_24_cos_36_minus_sin_24_cos_54_l3975_397579

theorem cos_24_cos_36_minus_sin_24_cos_54 : 
  Real.cos (24 * π / 180) * Real.cos (36 * π / 180) - 
  Real.sin (24 * π / 180) * Real.cos (54 * π / 180) = 1 / 2 := by
  sorry

end cos_24_cos_36_minus_sin_24_cos_54_l3975_397579


namespace largest_common_divisor_408_310_l3975_397591

theorem largest_common_divisor_408_310 : Nat.gcd 408 310 = 2 := by
  sorry

end largest_common_divisor_408_310_l3975_397591


namespace ratio_equality_l3975_397500

theorem ratio_equality : ∃ x : ℚ, (2 / 5 : ℚ) / (3 / 7 : ℚ) = x / (1 / 2 : ℚ) ∧ x = 7 / 15 := by
  sorry

end ratio_equality_l3975_397500


namespace emilys_typing_speed_l3975_397584

/-- Emily's typing speed problem -/
theorem emilys_typing_speed : 
  ∀ (words_typed : ℕ) (hours_taken : ℕ),
  words_typed = 10800 ∧ hours_taken = 3 →
  words_typed / (hours_taken * 60) = 60 :=
by sorry

end emilys_typing_speed_l3975_397584


namespace angle_in_third_quadrant_l3975_397545

-- Define the quadrants
inductive Quadrant
  | first
  | second
  | third
  | fourth

-- Define a function to determine the quadrant of an angle
def angle_quadrant (α : Real) : Quadrant :=
  sorry

-- Theorem statement
theorem angle_in_third_quadrant (α : Real) 
  (h1 : Real.sin α * Real.cos α > 0) 
  (h2 : Real.sin α * Real.tan α < 0) : 
  angle_quadrant α = Quadrant.third :=
sorry

end angle_in_third_quadrant_l3975_397545


namespace fruit_purchase_problem_l3975_397538

theorem fruit_purchase_problem (x y : ℝ) :
  x ≥ 0 ∧ y ≥ 0 ∧ x + y = 7 ∧ 5 * x + 8 * y = 41 → x = 5 ∧ y = 2 := by
  sorry

end fruit_purchase_problem_l3975_397538


namespace probability_neither_red_nor_purple_l3975_397509

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 18
def yellow_balls : ℕ := 5
def red_balls : ℕ := 6
def purple_balls : ℕ := 9

theorem probability_neither_red_nor_purple :
  (total_balls - (red_balls + purple_balls)) / total_balls = 3 / 4 := by
  sorry

end probability_neither_red_nor_purple_l3975_397509


namespace death_rate_calculation_l3975_397513

/-- The average birth rate in people per two seconds -/
def average_birth_rate : ℕ := 10

/-- The population net increase in one day -/
def population_net_increase : ℕ := 345600

/-- The number of seconds in a day -/
def seconds_per_day : ℕ := 24 * 60 * 60

/-- The average death rate in people per two seconds -/
def average_death_rate : ℕ := 2

theorem death_rate_calculation :
  average_birth_rate - average_death_rate = 
    2 * (population_net_increase / seconds_per_day) :=
by sorry

end death_rate_calculation_l3975_397513


namespace sum_of_digits_in_multiple_of_72_l3975_397590

theorem sum_of_digits_in_multiple_of_72 (A B : ℕ) : 
  A < 10 → B < 10 → (A * 100000 + 44610 + B) % 72 = 0 → A + B = 12 := by
  sorry

end sum_of_digits_in_multiple_of_72_l3975_397590


namespace value_calculation_l3975_397525

theorem value_calculation : 0.833 * (-72.0) = -59.976 := by
  sorry

end value_calculation_l3975_397525


namespace arrangements_together_count_arrangements_alternate_count_arrangements_restricted_count_l3975_397503

-- Define the number of boys and girls
def num_boys : ℕ := 2
def num_girls : ℕ := 3
def total_students : ℕ := num_boys + num_girls

-- Define the functions for each arrangement scenario
def arrangements_together : ℕ := sorry

def arrangements_alternate : ℕ := sorry

def arrangements_restricted : ℕ := sorry

-- State the theorems to be proved
theorem arrangements_together_count : arrangements_together = 24 := by sorry

theorem arrangements_alternate_count : arrangements_alternate = 12 := by sorry

theorem arrangements_restricted_count : arrangements_restricted = 60 := by sorry

end arrangements_together_count_arrangements_alternate_count_arrangements_restricted_count_l3975_397503


namespace toy_store_revenue_ratio_l3975_397501

theorem toy_store_revenue_ratio :
  ∀ (december_revenue : ℚ),
  december_revenue > 0 →
  let november_revenue := (3 : ℚ) / 5 * december_revenue
  let january_revenue := (1 : ℚ) / 6 * november_revenue
  let average_revenue := (november_revenue + january_revenue) / 2
  december_revenue / average_revenue = 20 / 7 := by
sorry

end toy_store_revenue_ratio_l3975_397501


namespace simple_interest_rate_l3975_397547

theorem simple_interest_rate (P : ℝ) (h : P > 0) : 
  (∃ R : ℝ, R > 0 ∧ P + (P * R * 15) / 100 = 2 * P) → 
  (∃ R : ℝ, R > 0 ∧ P + (P * R * 15) / 100 = 2 * P ∧ R = 100 / 15) :=
by sorry

end simple_interest_rate_l3975_397547


namespace point_existence_and_uniqueness_l3975_397574

theorem point_existence_and_uniqueness :
  ∃! (x y : ℝ), 
    y = 8 ∧ 
    (x - 3)^2 + (y - 9)^2 = 12^2 ∧ 
    x^2 + y^2 = 14^2 ∧ 
    x > 3 := by
  sorry

end point_existence_and_uniqueness_l3975_397574


namespace cubic_kilometer_to_cubic_meters_l3975_397582

/-- Given that one kilometer equals 1000 meters, prove that one cubic kilometer equals 1,000,000,000 cubic meters. -/
theorem cubic_kilometer_to_cubic_meters :
  (1 : ℝ) * (1000 : ℝ)^3 = (1000000000 : ℝ) := by
  sorry

end cubic_kilometer_to_cubic_meters_l3975_397582
