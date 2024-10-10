import Mathlib

namespace perfect_square_trinomial_condition_l1084_108466

/-- A trinomial ax^2 + bx + c is a perfect square if there exist real numbers p and q
    such that ax^2 + bx + c = (px + q)^2 for all real x. -/
def is_perfect_square_trinomial (a b c : ℝ) : Prop :=
  ∃ p q : ℝ, ∀ x : ℝ, a * x^2 + b * x + c = (p * x + q)^2

/-- If x^2 + kx + 9 is a perfect square trinomial, then k = 6 or k = -6. -/
theorem perfect_square_trinomial_condition (k : ℝ) :
  is_perfect_square_trinomial 1 k 9 → k = 6 ∨ k = -6 := by
  sorry

end perfect_square_trinomial_condition_l1084_108466


namespace max_product_sum_l1084_108414

theorem max_product_sum (A M C : ℕ+) (h : A + M + C = 15) :
  (∀ A' M' C' : ℕ+, A' + M' + C' = 15 →
    A' * M' * C' + A' * M' + M' * C' + C' * A' + A' + M' + C' ≤
    A * M * C + A * M + M * C + C * A + A + M + C) →
  A * M * C + A * M + M * C + C * A + A + M + C = 215 :=
sorry

end max_product_sum_l1084_108414


namespace water_moles_theorem_l1084_108422

/-- Represents a chemical equation with reactants and products -/
structure ChemicalEquation where
  naoh_reactant : ℕ
  h2so4_reactant : ℕ
  na2so4_product : ℕ
  h2o_product : ℕ

/-- The balanced equation for the reaction -/
def balanced_equation : ChemicalEquation :=
  { naoh_reactant := 2
  , h2so4_reactant := 1
  , na2so4_product := 1
  , h2o_product := 2 }

/-- The number of moles of NaOH reacting -/
def naoh_moles : ℕ := 4

/-- The number of moles of H₂SO₄ reacting -/
def h2so4_moles : ℕ := 2

/-- Calculates the number of moles of water produced -/
def water_moles_produced (eq : ChemicalEquation) (naoh : ℕ) : ℕ :=
  (naoh * eq.h2o_product) / eq.naoh_reactant

/-- Theorem stating that 4 moles of water are produced -/
theorem water_moles_theorem :
  water_moles_produced balanced_equation naoh_moles = 4 :=
sorry

end water_moles_theorem_l1084_108422


namespace benjie_current_age_l1084_108493

/-- Benjie's age in years -/
def benjie_age : ℕ := 6

/-- Margo's age in years -/
def margo_age : ℕ := 1

/-- The age difference between Benjie and Margo in years -/
def age_difference : ℕ := 5

/-- The number of years until Margo is 4 years old -/
def years_until_margo_4 : ℕ := 3

theorem benjie_current_age :
  (benjie_age = margo_age + age_difference) ∧
  (margo_age + years_until_margo_4 = 4) →
  benjie_age = 6 := by sorry

end benjie_current_age_l1084_108493


namespace probability_divisor_of_12_l1084_108485

/-- A fair 6-sided die -/
def Die : Finset ℕ := {1, 2, 3, 4, 5, 6}

/-- The set of divisors of 12 that appear on the die -/
def DivisorsOf12OnDie : Finset ℕ := {1, 2, 3, 4, 6}

/-- The probability of an event on a fair die -/
def probability (event : Finset ℕ) : ℚ :=
  (event ∩ Die).card / Die.card

theorem probability_divisor_of_12 :
  probability DivisorsOf12OnDie = 5 / 6 := by
  sorry

end probability_divisor_of_12_l1084_108485


namespace find_certain_number_l1084_108444

theorem find_certain_number (N : ℚ) : (5/6 * N) - (5/16 * N) = 100 → N = 192 := by
  sorry

end find_certain_number_l1084_108444


namespace stream_speed_l1084_108492

/-- Given a canoe that rows upstream at 8 km/hr and downstream at 12 km/hr,
    prove that the speed of the stream is 2 km/hr. -/
theorem stream_speed (upstream_speed downstream_speed : ℝ)
  (h_upstream : upstream_speed = 8)
  (h_downstream : downstream_speed = 12) :
  ∃ (canoe_speed stream_speed : ℝ),
    canoe_speed - stream_speed = upstream_speed ∧
    canoe_speed + stream_speed = downstream_speed ∧
    stream_speed = 2 := by
  sorry

end stream_speed_l1084_108492


namespace maximize_product_l1084_108438

theorem maximize_product (A : ℝ) (h : A > 0) :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧
    a + b + c = A ∧
    ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x + y + z = A →
      x * y^2 * z^3 ≤ a * b^2 * c^3 ∧
    a = A / 6 ∧ b = A / 3 ∧ c = A / 2 :=
sorry

end maximize_product_l1084_108438


namespace octahedron_cube_volume_ratio_l1084_108487

/-- The volume ratio of an octahedron formed by connecting the centers of adjacent faces of a cube
    to the volume of the cube itself is 1/6, given that the cube has an edge length of 2 units. -/
theorem octahedron_cube_volume_ratio :
  let cube_edge : ℝ := 2
  let cube_volume : ℝ := cube_edge ^ 3
  let octahedron_edge : ℝ := Real.sqrt 8
  let octahedron_volume : ℝ := (Real.sqrt 2 / 3) * octahedron_edge ^ 3
  octahedron_volume / cube_volume = 1 / 6 := by
sorry

end octahedron_cube_volume_ratio_l1084_108487


namespace square_semicircle_perimeter_l1084_108400

theorem square_semicircle_perimeter (π : Real) (h : π > 0) : 
  let square_side : Real := 4 / π
  let semicircle_radius : Real := square_side / 2
  let num_semicircles : Nat := 4
  num_semicircles * (π * semicircle_radius) = 8 := by
  sorry

end square_semicircle_perimeter_l1084_108400


namespace air_conditioner_price_l1084_108470

/-- The selling price per unit of the air conditioner fan before the regulation. -/
def price_before : ℝ := 880

/-- The subsidy amount per unit after the regulation. -/
def subsidy : ℝ := 80

/-- The total amount spent on purchases after the regulation. -/
def total_spent : ℝ := 60000

/-- The ratio of units purchased after the regulation to before. -/
def purchase_ratio : ℝ := 1.1

theorem air_conditioner_price :
  (total_spent / (price_before - subsidy) = (total_spent / price_before) * purchase_ratio) ∧
  (price_before > 0) ∧ 
  (price_before > subsidy) := by sorry

end air_conditioner_price_l1084_108470


namespace pizzas_served_today_l1084_108456

/-- The number of pizzas served during lunch -/
def lunch_pizzas : ℕ := 9

/-- The number of pizzas served during dinner -/
def dinner_pizzas : ℕ := 6

/-- The total number of pizzas served today -/
def total_pizzas : ℕ := lunch_pizzas + dinner_pizzas

theorem pizzas_served_today : total_pizzas = 15 := by
  sorry

end pizzas_served_today_l1084_108456


namespace power_of_power_l1084_108424

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l1084_108424


namespace pipe_filling_time_l1084_108479

/-- Given two pipes A and B that can fill a tank, this theorem proves the time
    it takes for pipe B to fill the tank alone, given the times for pipe A
    and both pipes together. -/
theorem pipe_filling_time (time_A time_both : ℝ) (h1 : time_A = 10)
    (h2 : time_both = 20 / 3) : 
    (1 / time_A + 1 / 20 = 1 / time_both) := by
  sorry

#check pipe_filling_time

end pipe_filling_time_l1084_108479


namespace sum_of_reciprocals_l1084_108463

theorem sum_of_reciprocals (x y : ℚ) 
  (h1 : x ≠ 0) (h2 : y ≠ 0)
  (h3 : 1/x + 1/y = 4) (h4 : 1/x - 1/y = -8) : 
  x + y = -1/3 := by
  sorry

end sum_of_reciprocals_l1084_108463


namespace unique_solution_quadratic_l1084_108480

theorem unique_solution_quadratic (m : ℝ) : 
  (∃! x : ℝ, (x + 5) * (x + 2) = m + 3 * x) ↔ m = 6 :=
by sorry

end unique_solution_quadratic_l1084_108480


namespace log_101600_equals_2x_l1084_108413

theorem log_101600_equals_2x (x : ℝ) (h : Real.log 102 = x) : Real.log 101600 = 2 * x := by
  sorry

end log_101600_equals_2x_l1084_108413


namespace base7_subtraction_l1084_108420

/-- Represents a number in base 7 --/
def Base7 : Type := List Nat

/-- Converts a base 7 number to a natural number --/
def to_nat (b : Base7) : Nat :=
  b.foldr (fun digit acc => acc * 7 + digit) 0

/-- Subtracts two base 7 numbers --/
def subtract_base7 (a b : Base7) : Base7 :=
  sorry

theorem base7_subtraction :
  let a : Base7 := [1, 2, 1, 0, 0]
  let b : Base7 := [3, 6, 6, 6]
  subtract_base7 a b = [1, 1, 1, 1] := by sorry

end base7_subtraction_l1084_108420


namespace negation_of_existence_negation_of_proposition_l1084_108403

theorem negation_of_existence (p : ℝ → Prop) : 
  (¬∃ x, p x) ↔ (∀ x, ¬p x) :=
by sorry

theorem negation_of_proposition :
  (¬∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) :=
by sorry

end negation_of_existence_negation_of_proposition_l1084_108403


namespace l1_fixed_point_min_distance_intersection_l1084_108462

-- Define the lines and circle
def l1 (m : ℝ) (x y : ℝ) : Prop := m * x - (m + 1) * y - 2 = 0
def l2 (x y : ℝ) : Prop := x + 2 * y + 1 = 0
def l3 (x y : ℝ) : Prop := y = x - 2

def circle_C (center : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = 12}

-- Theorem 1: l1 always passes through (-2, -2)
theorem l1_fixed_point (m : ℝ) : l1 m (-2) (-2) := by sorry

-- Theorem 2: Minimum distance between intersection points
theorem min_distance_intersection :
  let center := (1, -1)  -- Intersection of l2 and l3
  ∃ (m : ℝ), ∀ (A B : ℝ × ℝ),
    A ∈ circle_C center → B ∈ circle_C center →
    l1 m A.1 A.2 → l1 m B.1 B.2 →
    (A.1 - B.1)^2 + (A.2 - B.2)^2 ≥ 8 := by sorry

end l1_fixed_point_min_distance_intersection_l1084_108462


namespace probability_green_is_25_56_l1084_108451

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from the given containers -/
def probability_green (containers : List Container) : ℚ :=
  let total_containers := containers.length
  let prob_green_per_container := containers.map (fun c => c.green / (c.red + c.green))
  (prob_green_per_container.sum) / total_containers

/-- The containers as described in the problem -/
def problem_containers : List Container :=
  [⟨8, 4⟩, ⟨2, 5⟩, ⟨2, 5⟩, ⟨4, 4⟩]

/-- The theorem stating the probability of selecting a green ball -/
theorem probability_green_is_25_56 :
  probability_green problem_containers = 25 / 56 := by
  sorry


end probability_green_is_25_56_l1084_108451


namespace tan_beta_value_l1084_108432

theorem tan_beta_value (α β : Real) 
  (h1 : Real.tan α = 1/3) 
  (h2 : Real.tan (α + β) = 1/2) : 
  Real.tan β = 1/7 := by
sorry

end tan_beta_value_l1084_108432


namespace trapezoid_area_l1084_108453

/-- The area of a trapezoid bounded by y=x, y=-x, x=10, and y=10 is 150 square units. -/
theorem trapezoid_area : Real := by
  -- Define the lines bounding the trapezoid
  let line1 : Real → Real := λ x => x
  let line2 : Real → Real := λ x => -x
  let line3 : Real → Real := λ _ => 10
  let vertical_line : Real := 10

  -- Define the trapezoid
  let trapezoid := {(x, y) : Real × Real | 
    (y = line1 x ∨ y = line2 x ∨ y = line3 x) ∧ 
    x ≤ vertical_line ∧ 
    y ≤ line3 x}

  -- Calculate the area of the trapezoid
  let area : Real := 150

  sorry -- Proof goes here

#check trapezoid_area

end trapezoid_area_l1084_108453


namespace black_go_stones_l1084_108471

theorem black_go_stones (total : ℕ) (difference : ℕ) (black : ℕ) (white : ℕ) : 
  total = 1256 → 
  difference = 408 → 
  total = black + white → 
  white = black + difference → 
  black = 424 := by
sorry

end black_go_stones_l1084_108471


namespace max_min_sum_xy_xz_yz_l1084_108445

theorem max_min_sum_xy_xz_yz (x y z : ℝ) (h : 3 * (x + y + z) = x^2 + y^2 + z^2) : 
  ∃ (M m : ℝ), (∀ t : ℝ, t = x*y + x*z + y*z → t ≤ M) ∧ 
                (∀ t : ℝ, t = x*y + x*z + y*z → m ≤ t) ∧ 
                M + 10*m = 27 := by
  sorry

end max_min_sum_xy_xz_yz_l1084_108445


namespace imaginary_part_of_z_l1084_108411

theorem imaginary_part_of_z (z : ℂ) (h : (Complex.I - 1) * z = Complex.I) : 
  z.im = -1/2 := by sorry

end imaginary_part_of_z_l1084_108411


namespace total_candies_l1084_108494

/-- The number of boxes Linda has -/
def x : ℕ := 3

/-- The number of candy bags Chloe has -/
def y : ℕ := 2

/-- The number of candy bars Olivia has -/
def z : ℕ := 5

/-- The number of candies in each of Linda's boxes -/
def candies_per_box : ℕ := 2

/-- The number of candies in each of Chloe's bags -/
def candies_per_bag : ℕ := 4

/-- The number of candies equivalent to each of Olivia's candy bars -/
def candies_per_bar : ℕ := 3

/-- The number of candies Linda has -/
def linda_candies : ℕ := 2 * x + 6

/-- The number of candies Chloe has -/
def chloe_candies : ℕ := 4 * y + 7

/-- The number of candies Olivia has -/
def olivia_candies : ℕ := 3 * z - 5

theorem total_candies : linda_candies + chloe_candies + olivia_candies = 37 := by
  sorry

end total_candies_l1084_108494


namespace zhaos_estimate_l1084_108476

theorem zhaos_estimate (x y ε : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : ε > 0) :
  (x + ε) - (y - 2*ε) > x - y :=
sorry

end zhaos_estimate_l1084_108476


namespace expression_equivalence_l1084_108425

variables (x y : ℝ)

def P : ℝ := 2*x + 3*y
def Q : ℝ := x - 2*y

theorem expression_equivalence :
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = (2*x + 3*y) / (2*x + 10*y) :=
by sorry

end expression_equivalence_l1084_108425


namespace sin_alpha_value_l1084_108460

theorem sin_alpha_value (α : Real) :
  let point : Real × Real := (2 * Real.sin (60 * π / 180), -2 * Real.cos (60 * π / 180))
  (∃ k : Real, k > 0 ∧ k * point.1 = Real.cos α ∧ k * point.2 = Real.sin α) →
  Real.sin α = -1/2 := by
sorry

end sin_alpha_value_l1084_108460


namespace two_digit_number_digit_difference_l1084_108417

theorem two_digit_number_digit_difference (x y : ℕ) : 
  x < 10 → y < 10 → (10 * x + y) - (10 * y + x) = 36 → x - y = 4 := by
  sorry

end two_digit_number_digit_difference_l1084_108417


namespace fraction_inequality_l1084_108465

theorem fraction_inequality (a b c d : ℝ) 
  (h1 : a > b) (h2 : b > 0) (h3 : d > c) (h4 : c > 0) : 
  a / c > b / d := by
  sorry

end fraction_inequality_l1084_108465


namespace dandelion_puff_distribution_l1084_108430

theorem dandelion_puff_distribution (total : ℕ) (given_away : ℕ) (friends : ℕ) 
  (h1 : total = 85)
  (h2 : given_away = 36)
  (h3 : friends = 5)
  (h4 : given_away < total) : 
  (total - given_away) / friends = (total - given_away) / (total - given_away) / friends :=
by sorry

end dandelion_puff_distribution_l1084_108430


namespace greatest_digit_sum_base_seven_l1084_108497

/-- Represents a positive integer in base 7 --/
def BaseSevenRepresentation := List Nat

/-- Converts a natural number to its base-seven representation --/
def toBaseSeven (n : Nat) : BaseSevenRepresentation :=
  sorry

/-- Calculates the sum of digits in a base-seven representation --/
def sumDigits (repr : BaseSevenRepresentation) : Nat :=
  sorry

/-- The upper bound for the problem --/
def upperBound : Nat := 2401

theorem greatest_digit_sum_base_seven :
  ∃ (max : Nat), ∀ (n : Nat), n < upperBound →
    sumDigits (toBaseSeven n) ≤ max ∧
    ∃ (m : Nat), m < upperBound ∧ sumDigits (toBaseSeven m) = max ∧
    max = 12 :=
  sorry

end greatest_digit_sum_base_seven_l1084_108497


namespace inequality_system_solution_l1084_108419

theorem inequality_system_solution :
  {x : ℝ | x + 2 > 3 * (2 - x) ∧ x < (x + 3) / 2} = {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end inequality_system_solution_l1084_108419


namespace no_solution_iff_a_less_than_one_l1084_108484

theorem no_solution_iff_a_less_than_one (a : ℝ) :
  (∀ x : ℝ, |x - 1| + x > a) ↔ a < 1 := by
sorry

end no_solution_iff_a_less_than_one_l1084_108484


namespace inequality_proof_l1084_108449

-- Define the function f
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- Define the set T
def T : Set ℝ := {a | -Real.sqrt 3 < a ∧ a < Real.sqrt 3}

-- Theorem statement
theorem inequality_proof (h : ∀ x : ℝ, ∀ a ∈ T, f x > a^2) (m n : ℝ) (hm : m ∈ T) (hn : n ∈ T) :
  Real.sqrt 3 * |m + n| < |m * n + 3| := by
  sorry

end inequality_proof_l1084_108449


namespace quadrilateral_point_D_l1084_108491

-- Define a structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a structure for a quadrilateral
structure Quadrilateral where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D

-- Define a property for parallel sides
def parallel_sides (q : Quadrilateral) : Prop :=
  (q.A.x - q.B.x) * (q.C.y - q.D.y) = (q.A.y - q.B.y) * (q.C.x - q.D.x) ∧
  (q.A.x - q.D.x) * (q.B.y - q.C.y) = (q.A.y - q.D.y) * (q.B.x - q.C.x)

-- Theorem statement
theorem quadrilateral_point_D (q : Quadrilateral) :
  q.A = Point2D.mk (-2) 0 ∧
  q.B = Point2D.mk 6 8 ∧
  q.C = Point2D.mk 8 6 ∧
  parallel_sides q →
  q.D = Point2D.mk 0 (-2) := by
  sorry

end quadrilateral_point_D_l1084_108491


namespace common_term_formula_l1084_108495

def x (n : ℕ) : ℕ := 2 * n - 1
def y (n : ℕ) : ℕ := n ^ 2

def is_common_term (m : ℕ) : Prop :=
  ∃ n k : ℕ, x n = m ∧ y k = m

def c (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem common_term_formula :
  ∀ n : ℕ, is_common_term (c n) ∧
  (∀ m : ℕ, m < c n → is_common_term m → ∃ k < n, c k = m) :=
sorry

end common_term_formula_l1084_108495


namespace two_thirds_of_number_is_fifty_l1084_108454

theorem two_thirds_of_number_is_fifty (y : ℝ) : (2 / 3 : ℝ) * y = 50 → y = 75 := by
  sorry

end two_thirds_of_number_is_fifty_l1084_108454


namespace triangle_folding_angle_range_l1084_108434

-- Define the triangle ABC
structure Triangle (A B C : ℝ × ℝ) : Prop where
  valid : A ≠ B ∧ B ≠ C ∧ C ≠ A

-- Define the angle between two vectors
def angle (v w : ℝ × ℝ) : ℝ := sorry

-- Define a point on a line segment
def pointOnSegment (P : ℝ × ℝ) (A B : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧ P = (t * A.1 + (1 - t) * B.1, t * A.2 + (1 - t) * B.2)

-- Define perpendicularity of two line segments
def perpendicular (AB CD : (ℝ × ℝ) × (ℝ × ℝ)) : Prop := sorry

theorem triangle_folding_angle_range 
  (A B C : ℝ × ℝ) 
  (h_triangle : Triangle A B C) 
  (h_angle_C : angle (B - C) (A - C) = π / 3) 
  (θ : ℝ) 
  (h_angle_BAC : angle (C - A) (B - A) = θ) :
  (∃ M : ℝ × ℝ, 
    pointOnSegment M B C ∧ 
    (∃ B' : ℝ × ℝ, perpendicular (A, B') (C, M))) →
  π / 6 < θ ∧ θ < 2 * π / 3 := by
  sorry

end triangle_folding_angle_range_l1084_108434


namespace angle_in_fourth_quadrant_l1084_108429

theorem angle_in_fourth_quadrant (α : Real) 
  (h1 : Real.sin α < 0) (h2 : Real.cos α > 0) : 
  α ∈ Set.Ioo (3 * Real.pi / 2) (2 * Real.pi) :=
sorry

end angle_in_fourth_quadrant_l1084_108429


namespace central_park_excess_cans_l1084_108416

def trash_can_problem (central_park : ℕ) (veterans_park : ℕ) : Prop :=
  -- Central Park had some more than half of the number of trash cans as in Veteran's Park
  central_park > veterans_park / 2 ∧
  -- Originally, there were 24 trash cans in Veteran's Park
  veterans_park = 24 ∧
  -- Half of the trash cans from Central Park were moved to Veteran's Park
  -- Now, there are 34 trash cans in Veteran's Park
  central_park / 2 + veterans_park = 34

theorem central_park_excess_cans :
  ∀ central_park veterans_park,
    trash_can_problem central_park veterans_park →
    central_park - veterans_park / 2 = 8 :=
by sorry

end central_park_excess_cans_l1084_108416


namespace olivias_paper_count_l1084_108440

/-- Calculates the total remaining pieces of paper given initial amounts and usage --/
def totalRemainingPieces (initialFolder1 initialFolder2 usedFolder1 usedFolder2 : ℕ) : ℕ :=
  (initialFolder1 - usedFolder1) + (initialFolder2 - usedFolder2)

/-- Theorem stating that given the initial conditions and usage, the total remaining pieces of paper is 130 --/
theorem olivias_paper_count :
  totalRemainingPieces 152 98 78 42 = 130 := by
  sorry

end olivias_paper_count_l1084_108440


namespace pancake_flour_calculation_l1084_108448

/-- Given a recipe for 20 pancakes requiring 3 cups of flour,
    prove that 27 cups of flour are needed for 180 pancakes. -/
theorem pancake_flour_calculation
  (original_pancakes : ℕ)
  (original_flour : ℕ)
  (desired_pancakes : ℕ)
  (h1 : original_pancakes = 20)
  (h2 : original_flour = 3)
  (h3 : desired_pancakes = 180) :
  (desired_pancakes / original_pancakes) * original_flour = 27 :=
by sorry

end pancake_flour_calculation_l1084_108448


namespace last_four_average_l1084_108478

/-- Given a list of seven real numbers where the average of all seven is 62
    and the average of the first three is 58, prove that the average of the
    last four numbers is 65. -/
theorem last_four_average (list : List ℝ) : 
  list.length = 7 →
  list.sum / 7 = 62 →
  (list.take 3).sum / 3 = 58 →
  (list.drop 3).sum / 4 = 65 := by
sorry

end last_four_average_l1084_108478


namespace selection_problem_l1084_108443

def number_of_ways_to_select (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem selection_problem (total_students : ℕ) (selected_students : ℕ) 
  (h_total : total_students = 10) 
  (h_selected : selected_students = 4) : 
  (number_of_ways_to_select 8 2) + (number_of_ways_to_select 8 3) = 84 := by
  sorry

#check selection_problem

end selection_problem_l1084_108443


namespace quadratic_function_inequality_l1084_108415

/-- Given a quadratic function y = (x - a)² + a - 1, where a is a constant,
    and (m, n) is a point on the graph with m > 0, prove that if m > 2a, then n > -5/4. -/
theorem quadratic_function_inequality (a m n : ℝ) : 
  m > 0 → 
  n = (m - a)^2 + a - 1 → 
  m > 2*a → 
  n > -5/4 := by sorry

end quadratic_function_inequality_l1084_108415


namespace trig_computation_l1084_108486

theorem trig_computation : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 3 / Real.sin (70 * π / 180) = 4 := by
  sorry

end trig_computation_l1084_108486


namespace smallest_factor_for_perfect_square_l1084_108437

theorem smallest_factor_for_perfect_square (n : ℕ) : n = 7 ↔ 
  (n > 0 ∧ 
   ∃ (m : ℕ), 1008 * n = m^2 ∧ 
   ∀ (k : ℕ), k > 0 → k < n → ¬∃ (l : ℕ), 1008 * k = l^2) := by
  sorry

end smallest_factor_for_perfect_square_l1084_108437


namespace work_completion_proof_l1084_108446

/-- The number of days it takes the first group to complete the work -/
def days_group1 : ℕ := 35

/-- The number of days it takes the second group to complete the work -/
def days_group2 : ℕ := 50

/-- The number of men in the second group -/
def men_group2 : ℕ := 7

/-- The number of men in the first group -/
def men_group1 : ℕ := men_group2 * days_group2 / days_group1

theorem work_completion_proof : men_group1 = 10 := by
  sorry

end work_completion_proof_l1084_108446


namespace negation_of_existence_negation_of_inequality_l1084_108467

theorem negation_of_existence (p : ℝ → Prop) :
  (¬∃ x : ℝ, p x) ↔ (∀ x : ℝ, ¬p x) :=
by sorry

theorem negation_of_inequality :
  (¬∃ x : ℝ, 2^x ≥ 2*x + 1) ↔ (∀ x : ℝ, 2^x < 2*x + 1) :=
by sorry

end negation_of_existence_negation_of_inequality_l1084_108467


namespace inequality_implies_upper_bound_l1084_108489

theorem inequality_implies_upper_bound (m : ℝ) :
  (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) →
  m ≤ 2 := by
  sorry

end inequality_implies_upper_bound_l1084_108489


namespace function_power_id_implies_bijective_l1084_108435

variable {X : Type*}

def compose_n_times {X : Type*} (f : X → X) : ℕ → (X → X)
  | 0 => id
  | n + 1 => f ∘ (compose_n_times f n)

theorem function_power_id_implies_bijective
  (f : X → X) (k : ℕ) (hk : k > 0) (h : compose_n_times f k = id) :
  Function.Bijective f :=
sorry

end function_power_id_implies_bijective_l1084_108435


namespace problem_1_problem_2_problem_3_problem_4_l1084_108477

-- Problem 1
theorem problem_1 : 3 * (13 / 15) - 2 * (13 / 14) + 5 * (2 / 15) - 1 * (1 / 14) = 5 := by sorry

-- Problem 2
theorem problem_2 : (1 / 9) / (2 / (3 / 4 - 2 / 3)) = 1 / 216 := by sorry

-- Problem 3
theorem problem_3 : 99 * 78.6 + 786 * 0.3 - 7.86 * 20 = 7860 := by sorry

-- Problem 4
theorem problem_4 : 2015 / (2015 * 2015 / 2016) = 2016 / 2017 := by sorry

end problem_1_problem_2_problem_3_problem_4_l1084_108477


namespace roots_of_polynomials_l1084_108469

theorem roots_of_polynomials (r : ℝ) : 
  r^2 - 2*r - 1 = 0 → r^5 - 29*r - 12 = 0 := by
  sorry

end roots_of_polynomials_l1084_108469


namespace no_solution_for_inequality_l1084_108472

theorem no_solution_for_inequality :
  ¬ ∃ x : ℝ, 3 * x^2 - x + 2 < 0 := by sorry

end no_solution_for_inequality_l1084_108472


namespace loss_per_meter_is_five_l1084_108412

/-- Calculates the loss per meter of cloth given the total cloth sold, total selling price, and cost price per meter. -/
def loss_per_meter (total_cloth : ℕ) (total_selling_price : ℕ) (cost_price_per_meter : ℕ) : ℕ :=
  let total_cost_price := total_cloth * cost_price_per_meter
  let total_loss := total_cost_price - total_selling_price
  total_loss / total_cloth

/-- Proves that the loss per meter of cloth is 5 rupees given the specific conditions. -/
theorem loss_per_meter_is_five :
  loss_per_meter 450 18000 45 = 5 := by
  sorry

#eval loss_per_meter 450 18000 45

end loss_per_meter_is_five_l1084_108412


namespace pink_highlighters_l1084_108408

theorem pink_highlighters (total : ℕ) (yellow : ℕ) (blue : ℕ) (h1 : yellow = 2) (h2 : blue = 5) (h3 : total = 11) :
  total - yellow - blue = 4 := by
  sorry

end pink_highlighters_l1084_108408


namespace line_equation_proof_l1084_108409

/-- Given a line defined by the equation (3, -4) · ((x, y) - (-2, 8)) = 0,
    prove that its slope-intercept form y = mx + b has m = 3/4 and b = 9.5 -/
theorem line_equation_proof :
  let line_eq := fun (x y : ℝ) => (3 * (x + 2) + (-4) * (y - 8) = 0)
  ∃ (m b : ℝ), m = 3/4 ∧ b = 9.5 ∧ ∀ x y, line_eq x y ↔ y = m * x + b :=
by sorry

end line_equation_proof_l1084_108409


namespace fraction_ordering_l1084_108455

theorem fraction_ordering : (8 : ℚ) / 25 < 6 / 17 ∧ 6 / 17 < 10 / 27 := by
  sorry

end fraction_ordering_l1084_108455


namespace shopping_theorem_l1084_108441

def shopping_scenario (initial_money : ℝ) : Prop :=
  let after_first_store := initial_money / 2 - 2000
  let after_second_store := after_first_store / 2 - 2000
  after_second_store = 0

theorem shopping_theorem : 
  ∃ (initial_money : ℝ), shopping_scenario initial_money ∧ initial_money = 12000 :=
sorry

end shopping_theorem_l1084_108441


namespace sum_of_coefficients_l1084_108431

theorem sum_of_coefficients (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ a₁₀ a₁₁ : ℝ) :
  (∀ x : ℝ, (x^2 + 1) * (x - 2)^9 = a₀ + a₁*(x-1) + a₂*(x-1)^2 + a₃*(x-1)^3 + a₄*(x-1)^4 + 
    a₅*(x-1)^5 + a₆*(x-1)^6 + a₇*(x-1)^7 + a₈*(x-1)^8 + a₉*(x-1)^9 + a₁₀*(x-1)^10 + a₁₁*(x-1)^11) →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ + a₈ + a₉ + a₁₀ + a₁₁ = 2 := by
sorry

end sum_of_coefficients_l1084_108431


namespace ratio_problem_l1084_108450

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end ratio_problem_l1084_108450


namespace original_vocabulary_l1084_108459

/-- The number of words learned per day -/
def words_per_day : ℕ := 10

/-- The number of days in 2 years -/
def days_in_two_years : ℕ := 365 * 2

/-- The percentage increase in vocabulary -/
def percentage_increase : ℚ := 1 / 2

theorem original_vocabulary (original : ℕ) : 
  (original : ℚ) + (original : ℚ) * percentage_increase = 
    (words_per_day * days_in_two_years : ℚ) → 
  original = 14600 := by sorry

end original_vocabulary_l1084_108459


namespace cliff_total_rocks_l1084_108442

/-- Represents the number of rocks in Cliff's collection --/
structure RockCollection where
  igneous : ℕ
  sedimentary : ℕ
  metamorphic : ℕ

/-- Conditions for Cliff's rock collection --/
def cliff_collection : RockCollection → Prop := fun r =>
  r.sedimentary = 2 * r.igneous ∧
  r.metamorphic = 2 * r.igneous ∧
  2 * r.igneous = 3 * 40 ∧
  r.sedimentary / 5 + r.metamorphic * 3 / 4 + 40 = (r.igneous + r.sedimentary + r.metamorphic) / 5

theorem cliff_total_rocks :
  ∀ r : RockCollection, cliff_collection r → r.igneous + r.sedimentary + r.metamorphic = 300 :=
by
  sorry

end cliff_total_rocks_l1084_108442


namespace factor_63x_plus_54_l1084_108404

theorem factor_63x_plus_54 : ∀ x : ℝ, 63 * x + 54 = 9 * (7 * x + 6) := by
  sorry

end factor_63x_plus_54_l1084_108404


namespace triangle_angle_c_two_thirds_pi_l1084_108406

theorem triangle_angle_c_two_thirds_pi
  (A B C : Real) (a b c : Real)
  (h1 : 0 < A ∧ 0 < B ∧ 0 < C)
  (h2 : A + B + C = π)
  (h3 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h4 : (a + b + c) * (Real.sin A + Real.sin B - Real.sin C) = a * Real.sin B) :
  C = 2 * π / 3 := by
  sorry

end triangle_angle_c_two_thirds_pi_l1084_108406


namespace cutlery_added_l1084_108498

def initial_forks : ℕ := 6

def initial_knives (forks : ℕ) : ℕ := forks + 9

def initial_spoons (knives : ℕ) : ℕ := 2 * knives

def initial_teaspoons (forks : ℕ) : ℕ := forks / 2

def total_initial_cutlery (forks knives spoons teaspoons : ℕ) : ℕ :=
  forks + knives + spoons + teaspoons

def final_total_cutlery : ℕ := 62

theorem cutlery_added :
  final_total_cutlery - total_initial_cutlery initial_forks
    (initial_knives initial_forks)
    (initial_spoons (initial_knives initial_forks))
    (initial_teaspoons initial_forks) = 8 := by
  sorry

end cutlery_added_l1084_108498


namespace ball_passing_game_l1084_108490

/-- Probability of the ball returning to player A after n passes in a three-player game --/
def P (n : ℕ) : ℚ :=
  1/3 - 1/3 * (-1/2)^(n-1)

theorem ball_passing_game :
  (P 2 = 1/2) ∧
  (∀ n : ℕ, P (n+1) = 1/2 * (1 - P n)) ∧
  (∀ n : ℕ, P n = 1/3 - 1/3 * (-1/2)^(n-1)) :=
by sorry

end ball_passing_game_l1084_108490


namespace larger_number_in_ratio_l1084_108461

theorem larger_number_in_ratio (a b : ℕ+) : 
  a.val * 5 = b.val * 2 →  -- ratio condition
  Nat.lcm a.val b.val = 160 →  -- LCM condition
  b = 160 := by  -- conclusion: larger number is 160
sorry

end larger_number_in_ratio_l1084_108461


namespace semicircle_area_ratio_l1084_108483

theorem semicircle_area_ratio (R : ℝ) (h : R > 0) :
  let r := (3 : ℝ) / 5 * R
  (π * r^2 / 2) / (π * R^2 / 2) = 9 / 25 := by sorry

end semicircle_area_ratio_l1084_108483


namespace lawyer_upfront_payment_l1084_108473

theorem lawyer_upfront_payment
  (hourly_rate : ℕ)
  (court_time : ℕ)
  (prep_time_multiplier : ℕ)
  (total_payment : ℕ)
  (h1 : hourly_rate = 100)
  (h2 : court_time = 50)
  (h3 : prep_time_multiplier = 2)
  (h4 : total_payment = 8000) :
  let prep_time := prep_time_multiplier * court_time
  let total_hours := court_time + prep_time
  let total_fee := hourly_rate * total_hours
  let johns_share := total_payment / 2
  let upfront_payment := johns_share
  upfront_payment = 4000 := by
sorry

end lawyer_upfront_payment_l1084_108473


namespace simplify_and_rationalize_l1084_108402

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 8) * (Real.sqrt 6 / Real.sqrt 9) = Real.sqrt 35 / 14 := by
  sorry

end simplify_and_rationalize_l1084_108402


namespace value_of_a_l1084_108452

theorem value_of_a (a : ℝ) : 4 ∈ ({a^2 - 3*a, a} : Set ℝ) → a = -1 := by
  sorry

end value_of_a_l1084_108452


namespace cell_count_after_3_hours_l1084_108421

/-- The number of cells after a given number of half-hour intervals, starting with one cell -/
def cell_count (n : ℕ) : ℕ := 2^n

/-- The number of half-hour intervals in 3 hours -/
def intervals_in_3_hours : ℕ := 6

theorem cell_count_after_3_hours :
  cell_count intervals_in_3_hours = 64 := by
  sorry

end cell_count_after_3_hours_l1084_108421


namespace square_plot_side_length_l1084_108464

theorem square_plot_side_length (area : ℝ) (side : ℝ) : 
  area = 2550.25 → side * side = area → side = 50.5 := by sorry

end square_plot_side_length_l1084_108464


namespace obtuse_minus_right_is_acute_straight_minus_acute_is_obtuse_l1084_108410

/-- Definition of an obtuse angle -/
def is_obtuse_angle (α : ℝ) : Prop := 90 < α ∧ α < 180

/-- Definition of a right angle -/
def is_right_angle (α : ℝ) : Prop := α = 90

/-- Definition of an acute angle -/
def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < 90

/-- Definition of a straight angle -/
def is_straight_angle (α : ℝ) : Prop := α = 180

/-- Theorem: When an obtuse angle is cut by a right angle, the remaining angle is acute -/
theorem obtuse_minus_right_is_acute (α β : ℝ) 
  (h1 : is_obtuse_angle α) (h2 : is_right_angle β) : 
  is_acute_angle (α - β) := by sorry

/-- Theorem: When a straight angle is cut by an acute angle, the remaining angle is obtuse -/
theorem straight_minus_acute_is_obtuse (α β : ℝ) 
  (h1 : is_straight_angle α) (h2 : is_acute_angle β) : 
  is_obtuse_angle (α - β) := by sorry

end obtuse_minus_right_is_acute_straight_minus_acute_is_obtuse_l1084_108410


namespace factorization_equality_l1084_108488

theorem factorization_equality (a b : ℝ) : (a^2 + b^2)^2 - 4*a^2*b^2 = (a + b)^2 * (a - b)^2 := by
  sorry

end factorization_equality_l1084_108488


namespace factorial_equality_l1084_108418

theorem factorial_equality : ∃ N : ℕ+, Nat.factorial 7 * Nat.factorial 11 = 18 * Nat.factorial N.val := by
  sorry

end factorial_equality_l1084_108418


namespace fibonacci_periodicity_l1084_108457

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

theorem fibonacci_periodicity :
  (∀ n, 10 ∣ (fib (n + 60) - fib n)) ∧
  (∀ k, 1 ≤ k → k < 60 → ∃ n, ¬(10 ∣ (fib (n + k) - fib n))) ∧
  (∀ n, 100 ∣ (fib (n + 300) - fib n)) ∧
  (∀ k, 1 ≤ k → k < 300 → ∃ n, ¬(100 ∣ (fib (n + k) - fib n))) :=
by sorry

end fibonacci_periodicity_l1084_108457


namespace polynomial_equality_l1084_108447

/-- Two distinct quadratic polynomials with leading coefficient 1 -/
def f (b c : ℝ) (x : ℝ) : ℝ := x^2 + b*x + c
def g (d e : ℝ) (x : ℝ) : ℝ := x^2 + d*x + e

/-- The theorem stating the point of equality for the polynomials -/
theorem polynomial_equality (b c d e : ℝ) 
  (h_distinct : b ≠ d ∨ c ≠ e)
  (h_sum_eq : f b c 1 + f b c 10 + f b c 100 = g d e 1 + g d e 10 + g d e 100) :
  f b c 37 = g d e 37 := by
  sorry

end polynomial_equality_l1084_108447


namespace x_squared_minus_y_squared_l1084_108436

theorem x_squared_minus_y_squared (x y : ℚ) 
  (h1 : x + y = 4/9) (h2 : x - y = 2/9) : x^2 - y^2 = 8/81 := by
  sorry

end x_squared_minus_y_squared_l1084_108436


namespace complex_fraction_simplification_l1084_108482

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_fraction_simplification :
  (1 + i) / (1 + i^3) = i :=
by
  sorry

end complex_fraction_simplification_l1084_108482


namespace rogers_broken_crayons_l1084_108428

/-- Given that Roger has 14 crayons in total, 2 new crayons, and 4 used crayons,
    prove that he has 8 broken crayons. -/
theorem rogers_broken_crayons :
  let total_crayons : ℕ := 14
  let new_crayons : ℕ := 2
  let used_crayons : ℕ := 4
  let broken_crayons : ℕ := total_crayons - new_crayons - used_crayons
  broken_crayons = 8 := by
  sorry

end rogers_broken_crayons_l1084_108428


namespace intersected_cells_count_l1084_108475

/-- Represents a grid cell -/
structure Cell where
  x : Int
  y : Int

/-- Represents a grid -/
structure Grid where
  width : Nat
  height : Nat

/-- Checks if a point (x, y) is inside the grid -/
def Grid.contains (g : Grid) (x y : Int) : Prop :=
  -g.width / 2 ≤ x ∧ x < g.width / 2 ∧ -g.height / 2 ≤ y ∧ y < g.height / 2

/-- Counts the number of cells intersected by the line y = mx -/
def countIntersectedCells (g : Grid) (m : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the number of cells intersected by y = 0.83x on a 60x70 grid is 108 -/
theorem intersected_cells_count :
  let g : Grid := { width := 60, height := 70 }
  let m : ℚ := 83 / 100
  countIntersectedCells g m = 108 := by
  sorry

end intersected_cells_count_l1084_108475


namespace quadratic_root_form_n_l1084_108499

def quadratic_equation (x : ℝ) : Prop := 3 * x^2 - 8 * x - 5 = 0

def root_form (x m n p : ℝ) : Prop :=
  x = (m + Real.sqrt n) / p ∨ x = (m - Real.sqrt n) / p

theorem quadratic_root_form_n :
  ∃ (m n p : ℕ+),
    (∀ x : ℝ, quadratic_equation x → root_form x m n p) ∧
    Nat.gcd m.val (Nat.gcd n.val p.val) = 1 ∧
    n = 124 :=
sorry

end quadratic_root_form_n_l1084_108499


namespace trigonometric_identity_l1084_108474

theorem trigonometric_identity : 
  1 / Real.cos (70 * π / 180) + Real.sqrt 3 / Real.sin (70 * π / 180) = 4 * Real.tan (10 * π / 180) := by
  sorry

end trigonometric_identity_l1084_108474


namespace parabola_focus_at_hyperbola_vertex_l1084_108423

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

-- Define the right vertex of the hyperbola
def right_vertex (x y : ℝ) : Prop := hyperbola x y ∧ y = 0 ∧ x > 0

-- Define the standard form of a parabola
def parabola (x y p : ℝ) : Prop := y^2 = 2 * p * x

-- Theorem statement
theorem parabola_focus_at_hyperbola_vertex :
  ∃ (x₀ y₀ : ℝ), right_vertex x₀ y₀ →
  ∃ (p : ℝ), p > 0 ∧ ∀ (x y : ℝ), parabola (x - x₀) y p ↔ y^2 = 16 * x :=
sorry

end parabola_focus_at_hyperbola_vertex_l1084_108423


namespace headlight_cost_is_180_l1084_108427

/-- Represents the scenario of Chris selling his car with two different offers --/
def car_sale_scenario (asking_price : ℝ) (maintenance_cost : ℝ) (headlight_cost : ℝ) : Prop :=
  let tire_cost := 3 * headlight_cost
  let first_offer := asking_price - maintenance_cost
  let second_offer := asking_price - (headlight_cost + tire_cost)
  (maintenance_cost = asking_price / 10) ∧
  (first_offer - second_offer = 200)

/-- Theorem stating that given the conditions, the headlight replacement cost is $180 --/
theorem headlight_cost_is_180 :
  car_sale_scenario 5200 520 180 :=
sorry

end headlight_cost_is_180_l1084_108427


namespace gala_trees_count_l1084_108458

/-- Represents an apple orchard with Fuji and Gala trees -/
structure Orchard where
  total : ℕ
  pure_fuji : ℕ
  pure_gala : ℕ
  cross_pollinated : ℕ

/-- Determines if an orchard satisfies the given conditions -/
def satisfies_conditions (o : Orchard) : Prop :=
  o.cross_pollinated = o.total / 10 ∧
  o.pure_fuji + o.cross_pollinated = 153 ∧
  o.pure_fuji = 3 * o.total / 4 ∧
  o.total = o.pure_fuji + o.pure_gala + o.cross_pollinated

theorem gala_trees_count (o : Orchard) :
  satisfies_conditions o → o.pure_gala = 45 := by
  sorry

end gala_trees_count_l1084_108458


namespace A_intersect_CᵣB_equals_zero_one_l1084_108407

-- Define the universal set
def 𝕌 : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {-1, 0, 1, 5}

-- Define set B
def B : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

-- Define the complement of B in ℝ
def CᵣB : Set ℝ := 𝕌 \ B

-- Theorem statement
theorem A_intersect_CᵣB_equals_zero_one : A ∩ CᵣB = {0, 1} := by sorry

end A_intersect_CᵣB_equals_zero_one_l1084_108407


namespace smallest_odd_abundant_number_l1084_108481

def is_abundant (n : ℕ) : Prop :=
  n < (Finset.sum (Finset.filter (λ x => x < n ∧ n % x = 0) (Finset.range n)) id)

def is_odd (n : ℕ) : Prop := n % 2 = 1

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_odd_abundant_number :
  (∀ n : ℕ, n < 945 → ¬(is_odd n ∧ is_abundant n ∧ is_composite n)) ∧
  (is_odd 945 ∧ is_abundant 945 ∧ is_composite 945) :=
sorry

end smallest_odd_abundant_number_l1084_108481


namespace circle_symmetry_range_l1084_108426

-- Define the circle equation
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 6*y + 5*a = 0

-- Define the symmetry line equation
def symmetry_line (x y b : ℝ) : Prop :=
  y = x + 2*b

-- Theorem statement
theorem circle_symmetry_range (a b : ℝ) :
  (∃ x y : ℝ, circle_equation x y a ∧ symmetry_line x y b) →
  a + b < 0 ∧ ∀ c, c < 0 → ∃ a' b', a' + b' = c ∧
    ∃ x y : ℝ, circle_equation x y a' ∧ symmetry_line x y b' :=
sorry

end circle_symmetry_range_l1084_108426


namespace largest_expression_l1084_108439

def expr_a : ℕ := 2 + 3 + 1 + 7
def expr_b : ℕ := 2 * 3 + 1 + 7
def expr_c : ℕ := 2 + 3 * 1 + 7
def expr_d : ℕ := 2 + 3 + 1 * 7
def expr_e : ℕ := 2 * 3 * 1 * 7

theorem largest_expression : 
  expr_e > expr_a ∧ 
  expr_e > expr_b ∧ 
  expr_e > expr_c ∧ 
  expr_e > expr_d := by
  sorry

end largest_expression_l1084_108439


namespace absolute_value_equation_unique_solution_l1084_108433

theorem absolute_value_equation_unique_solution :
  ∃! x : ℝ, |x - 10| = |x + 4| := by sorry

end absolute_value_equation_unique_solution_l1084_108433


namespace workshop_average_salary_l1084_108401

def total_workers : ℕ := 7
def technicians_salary : ℕ := 8000
def rest_salary : ℕ := 6000

theorem workshop_average_salary :
  (total_workers * technicians_salary) / total_workers = technicians_salary :=
by sorry

end workshop_average_salary_l1084_108401


namespace min_value_theorem_l1084_108405

theorem min_value_theorem (a : ℝ) (m n : ℝ) 
  (h1 : a > 0) 
  (h2 : a ≠ 1) 
  (h3 : 2*m - 1 + n = 0) :
  (4:ℝ)^m + 2^n ≥ 2*Real.sqrt 2 := by
  sorry

end min_value_theorem_l1084_108405


namespace cake_distribution_l1084_108496

theorem cake_distribution (total_pieces : ℕ) (eaten_percentage : ℚ) (num_sisters : ℕ) : 
  total_pieces = 240 →
  eaten_percentage = 60 / 100 →
  num_sisters = 3 →
  (total_pieces - (eaten_percentage * total_pieces).floor) / num_sisters = 32 := by
sorry

end cake_distribution_l1084_108496


namespace dogs_not_doing_anything_l1084_108468

theorem dogs_not_doing_anything (total : ℕ) (running : ℕ) (playing : ℕ) (barking : ℕ) : 
  total = 88 → 
  running = 12 → 
  playing = total / 2 → 
  barking = total / 4 → 
  total - (running + playing + barking) = 10 := by
  sorry

end dogs_not_doing_anything_l1084_108468
