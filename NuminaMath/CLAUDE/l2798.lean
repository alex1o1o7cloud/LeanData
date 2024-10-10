import Mathlib

namespace m_one_sufficient_not_necessary_l2798_279878

def is_perpendicular (m : ℝ) : Prop :=
  let line1_slope := -m
  let line2_slope := 1 / m
  line1_slope * line2_slope = -1

theorem m_one_sufficient_not_necessary :
  (∃ m : ℝ, m ≠ 1 ∧ is_perpendicular m) ∧
  (is_perpendicular 1) :=
sorry

end m_one_sufficient_not_necessary_l2798_279878


namespace quadratic_equation_solution_l2798_279861

theorem quadratic_equation_solution : 
  let f : ℝ → ℝ := λ x => 3 * x^2 + 6 * x - |-21 + 5|
  ∃ x₁ x₂ : ℝ, x₁ = -1 + Real.sqrt 19 ∧ x₂ = -1 - Real.sqrt 19 ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end quadratic_equation_solution_l2798_279861


namespace frame_interior_perimeter_l2798_279801

theorem frame_interior_perimeter
  (frame_width : ℝ)
  (frame_area : ℝ)
  (outer_edge : ℝ)
  (h1 : frame_width = 2)
  (h2 : frame_area = 60)
  (h3 : outer_edge = 10) :
  let inner_length := outer_edge - 2 * frame_width
  let inner_width := (frame_area / (outer_edge - inner_length)) - frame_width
  inner_length * 2 + inner_width * 2 = 22 := by
  sorry

end frame_interior_perimeter_l2798_279801


namespace minimum_race_distance_minimum_race_distance_rounded_l2798_279839

/-- The minimum distance a runner must travel in a race with specific conditions -/
theorem minimum_race_distance : ℝ :=
  let wall_distance : ℝ := 1500
  let a_to_first_wall : ℝ := 400
  let b_to_second_wall : ℝ := 600
  let total_vertical_distance : ℝ := a_to_first_wall + wall_distance + b_to_second_wall
  let minimum_distance : ℝ := (wall_distance ^ 2 + total_vertical_distance ^ 2).sqrt
  ⌊minimum_distance + 0.5⌋

/-- The minimum distance rounded to the nearest meter is 2915 -/
theorem minimum_race_distance_rounded : 
  ⌊minimum_race_distance + 0.5⌋ = 2915 := by sorry

end minimum_race_distance_minimum_race_distance_rounded_l2798_279839


namespace min_value_perpendicular_vectors_l2798_279811

theorem min_value_perpendicular_vectors (x y : ℝ) :
  let m : ℝ × ℝ := (x - 1, 1)
  let n : ℝ × ℝ := (1, y)
  (m.1 * n.1 + m.2 * n.2 = 0) →
  (∀ a b : ℝ, m = (a - 1, 1) ∧ n = (1, b) → 2^a + 2^b ≥ 2 * Real.sqrt 2) ∧
  (∃ a b : ℝ, m = (a - 1, 1) ∧ n = (1, b) ∧ 2^a + 2^b = 2 * Real.sqrt 2) :=
by sorry

end min_value_perpendicular_vectors_l2798_279811


namespace vector_perpendicular_condition_l2798_279841

/-- Given vectors a and b in R², if a + b is perpendicular to b, then the second component of a is 8. -/
theorem vector_perpendicular_condition (m : ℝ) : 
  let a : ℝ × ℝ := (1, m)
  let b : ℝ × ℝ := (3, -2)
  (a.1 + b.1) * b.1 + (a.2 + b.2) * b.2 = 0 → m = 8 := by
sorry

end vector_perpendicular_condition_l2798_279841


namespace otimes_nested_l2798_279872

/-- Custom binary operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^2 - 2*y

/-- Theorem stating the result of k ⊗ (k ⊗ k) -/
theorem otimes_nested (k : ℝ) : otimes k (otimes k k) = -k^2 + 4*k := by
  sorry

end otimes_nested_l2798_279872


namespace valid_colorings_l2798_279815

-- Define a color type
inductive Color
| A
| B
| C

-- Define a coloring function type
def Coloring := ℕ → Color

-- Define the condition for a valid coloring
def ValidColoring (f : Coloring) : Prop :=
  ∀ a b c : ℕ, 2000 * (a + b) = c →
    (f a = f b ∧ f b = f c) ∨
    (f a ≠ f b ∧ f b ≠ f c ∧ f a ≠ f c)

-- Define the two valid colorings
def AllSameColor : Coloring :=
  λ _ => Color.A

def ModuloThreeColoring : Coloring :=
  λ n => match n % 3 with
    | 1 => Color.A
    | 2 => Color.B
    | 0 => Color.C
    | _ => Color.A  -- This case is unreachable, but needed for exhaustiveness

-- State the theorem
theorem valid_colorings (f : Coloring) :
  ValidColoring f ↔ (f = AllSameColor ∨ f = ModuloThreeColoring) :=
sorry

end valid_colorings_l2798_279815


namespace basketball_probability_l2798_279853

theorem basketball_probability (p : ℝ) (n : ℕ) (h1 : p = 1/3) (h2 : n = 3) :
  (1 - p)^n + n * p * (1 - p)^(n-1) = 20/27 := by
  sorry

end basketball_probability_l2798_279853


namespace player_arrangement_count_l2798_279865

def num_players_alpha : ℕ := 4
def num_players_beta : ℕ := 4
def num_players_gamma : ℕ := 2
def total_players : ℕ := num_players_alpha + num_players_beta + num_players_gamma

theorem player_arrangement_count :
  (Nat.factorial 3) * (Nat.factorial num_players_alpha) * (Nat.factorial num_players_beta) * (Nat.factorial num_players_gamma) = 6912 :=
by sorry

end player_arrangement_count_l2798_279865


namespace least_possible_y_l2798_279857

theorem least_possible_y (x y z : ℤ) 
  (h_x_even : Even x)
  (h_y_odd : Odd y)
  (h_z_odd : Odd z)
  (h_y_minus_x : y - x > 5)
  (h_z_minus_x : ∀ w : ℤ, (Odd w ∧ w - x ≥ 9) → z - x ≤ w - x) : 
  y ≥ 7 := by
  sorry

end least_possible_y_l2798_279857


namespace prime_equality_l2798_279836

theorem prime_equality (p q r n : ℕ) : 
  Prime p → Prime q → Prime r → n > 0 →
  (∃ k₁ k₂ k₃ : ℕ, (p + n) = k₁ * q * r ∧ 
                   (q + n) = k₂ * r * p ∧ 
                   (r + n) = k₃ * p * q) →
  p = q ∧ q = r :=
sorry

end prime_equality_l2798_279836


namespace quadratic_inequality_solution_set_l2798_279870

theorem quadratic_inequality_solution_set (a : ℝ) :
  let S := {x : ℝ | (1 - a) * x^2 - 2 * x + 1 < 0}
  if a > 1 then
    S = {x : ℝ | x < (1 - Real.sqrt a) / (a - 1) ∨ x > (1 + Real.sqrt a) / (a - 1)}
  else if a = 1 then
    S = {x : ℝ | x > 1 / 2}
  else if 0 < a ∧ a < 1 then
    S = {x : ℝ | (1 - Real.sqrt a) / (1 - a) < x ∧ x < (1 + Real.sqrt a) / (1 - a)}
  else
    S = ∅ :=
by sorry

end quadratic_inequality_solution_set_l2798_279870


namespace inequality_system_solution_l2798_279897

theorem inequality_system_solution (a b : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 3 ↔ x - a < 1 ∧ x - 2*b > 3) → 
  a = 2 ∧ b = -2 := by
sorry

end inequality_system_solution_l2798_279897


namespace cos_BHD_value_l2798_279826

/-- A rectangular solid with specific angle conditions -/
structure RectangularSolid where
  /-- Angle DHG is 30 degrees -/
  angle_DHG : ℝ
  angle_DHG_eq : angle_DHG = 30 * π / 180
  /-- Angle FHB is 45 degrees -/
  angle_FHB : ℝ
  angle_FHB_eq : angle_FHB = 45 * π / 180

/-- The cosine of angle BHD in the rectangular solid -/
def cos_BHD (solid : RectangularSolid) : ℝ := sorry

/-- Theorem stating that the cosine of angle BHD is 5√2/12 -/
theorem cos_BHD_value (solid : RectangularSolid) : 
  cos_BHD solid = 5 * Real.sqrt 2 / 12 := by sorry

end cos_BHD_value_l2798_279826


namespace juice_cost_is_50_l2798_279818

/-- The cost of a candy bar in cents -/
def candy_cost : ℕ := 25

/-- The cost of a piece of chocolate in cents -/
def chocolate_cost : ℕ := 75

/-- The total cost in cents for the purchase -/
def total_cost : ℕ := 11 * 25

/-- The number of candy bars purchased -/
def num_candy : ℕ := 3

/-- The number of chocolate pieces purchased -/
def num_chocolate : ℕ := 2

/-- The number of juice packs purchased -/
def num_juice : ℕ := 1

theorem juice_cost_is_50 :
  ∃ (juice_cost : ℕ),
    juice_cost = 50 ∧
    total_cost = num_candy * candy_cost + num_chocolate * chocolate_cost + num_juice * juice_cost :=
by sorry

end juice_cost_is_50_l2798_279818


namespace f_lower_bound_l2798_279820

noncomputable section

variables (a x : ℝ)

def f (a x : ℝ) : ℝ := (1/2) * a * x^2 + (2*a - 1) * x - 2 * Real.log x

theorem f_lower_bound (ha : a > 0) (hx : x > 0) :
  f a x ≥ 4 - (5/(2*a)) := by sorry

end f_lower_bound_l2798_279820


namespace counterexample_existence_l2798_279802

theorem counterexample_existence : ∃ (S : Finset ℝ), 
  (Finset.card S = 25) ∧ 
  (∀ (a b c : ℝ), a ∈ S → b ∈ S → c ∈ S → 
    ∃ (d : ℝ), d ∈ S ∧ d ≠ a ∧ d ≠ b ∧ d ≠ c ∧ a + b + c + d > 0) ∧
  (Finset.sum S id ≤ 0) := by
  sorry

end counterexample_existence_l2798_279802


namespace milk_delivery_solution_l2798_279886

/-- Represents the milk delivery problem --/
def MilkDeliveryProblem (jarsPerCarton : ℕ) : Prop :=
  let usualCartons : ℕ := 50
  let actualCartons : ℕ := usualCartons - 20
  let damagedJarsInFiveCartons : ℕ := 5 * 3
  let totalDamagedJars : ℕ := damagedJarsInFiveCartons + jarsPerCarton
  let goodJars : ℕ := 565
  actualCartons * jarsPerCarton - totalDamagedJars = goodJars

/-- Theorem stating that the solution to the milk delivery problem is 20 jars per carton --/
theorem milk_delivery_solution : MilkDeliveryProblem 20 := by
  sorry

end milk_delivery_solution_l2798_279886


namespace ratio_of_divisors_sums_l2798_279819

def M : ℕ := 36 * 36 * 98 * 210

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 62 := by sorry

end ratio_of_divisors_sums_l2798_279819


namespace unique_perpendicular_projection_l2798_279868

-- Define the types for projections and points
def Projection : Type := ℝ → ℝ → ℝ
def Point : Type := ℝ × ℝ × ℝ

-- Define the given projections and intersection points
variable (g' g'' d'' : Projection)
variable (A' A'' : Point)

-- Define the perpendicularity condition
def perpendicular (l1 l2 : Projection) : Prop := sorry

-- Define the intersection condition
def intersect (l1 l2 : Projection) (p : Point) : Prop := sorry

-- Theorem statement
theorem unique_perpendicular_projection :
  ∃! d' : Projection,
    intersect g' d' A' ∧
    intersect g'' d'' A'' ∧
    perpendicular g' d' ∧
    perpendicular g'' d'' :=
sorry

end unique_perpendicular_projection_l2798_279868


namespace absolute_value_sum_simplification_l2798_279891

theorem absolute_value_sum_simplification (x : ℝ) : 
  |x - 1| + |x - 2| + |x + 3| = 
    if x < -3 then -3*x
    else if x < 1 then 6 - x
    else if x < 2 then 4 + x
    else 3*x := by sorry

end absolute_value_sum_simplification_l2798_279891


namespace negation_of_existence_negation_of_inequality_ge_negation_of_quadratic_inequality_l2798_279812

theorem negation_of_existence (P : ℕ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) := by sorry

theorem negation_of_inequality_ge (a b : ℝ) :
  (¬ (a ≥ b)) ↔ (a < b) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∃ x : ℕ, x^2 + 2*x ≥ 3) ↔ (∀ x : ℕ, x^2 + 2*x < 3) := by sorry

end negation_of_existence_negation_of_inequality_ge_negation_of_quadratic_inequality_l2798_279812


namespace lunch_probability_l2798_279866

def total_school_days : ℕ := 5

def ham_sandwich_days : ℕ := 3
def cake_days : ℕ := 1
def carrot_sticks_days : ℕ := 3

def prob_ham_sandwich : ℚ := ham_sandwich_days / total_school_days
def prob_cake : ℚ := cake_days / total_school_days
def prob_carrot_sticks : ℚ := carrot_sticks_days / total_school_days

theorem lunch_probability : 
  prob_ham_sandwich * prob_cake * prob_carrot_sticks = 3 / 125 := by
  sorry

end lunch_probability_l2798_279866


namespace planes_perpendicular_if_line_perpendicular_and_parallel_l2798_279883

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular_if_line_perpendicular_and_parallel
  (a : Line) (α β : Plane)
  (h1 : perpendicular a α)
  (h2 : parallel a β) :
  perpendicular_planes α β :=
sorry

end planes_perpendicular_if_line_perpendicular_and_parallel_l2798_279883


namespace rectangle_short_side_l2798_279896

/-- Proves that for a rectangle with perimeter 38 cm and long side 12 cm, the short side is 7 cm. -/
theorem rectangle_short_side (perimeter long_side short_side : ℝ) : 
  perimeter = 38 ∧ long_side = 12 ∧ perimeter = 2 * long_side + 2 * short_side → short_side = 7 :=
by
  sorry

end rectangle_short_side_l2798_279896


namespace handshake_count_l2798_279833

theorem handshake_count (n : ℕ) (h : n = 8) : 
  (n * (n - 2)) / 2 = 24 := by
  sorry

#check handshake_count

end handshake_count_l2798_279833


namespace total_cellphones_sold_l2798_279877

/-- Calculates the number of cell phones sold given initial and final inventories and damaged/defective phones. -/
def cellphonesSold (initialSamsung : ℕ) (finalSamsung : ℕ) (initialIphone : ℕ) (finalIphone : ℕ) (damagedSamsung : ℕ) (defectiveIphone : ℕ) : ℕ :=
  (initialSamsung - damagedSamsung - finalSamsung) + (initialIphone - defectiveIphone - finalIphone)

/-- Proves that the total number of cell phones sold is 4 given the inventory information. -/
theorem total_cellphones_sold :
  cellphonesSold 14 10 8 5 2 1 = 4 := by
  sorry

end total_cellphones_sold_l2798_279877


namespace sqrt_negative_a_squared_plus_one_undefined_l2798_279817

theorem sqrt_negative_a_squared_plus_one_undefined (a : ℝ) : ¬ ∃ (x : ℝ), x^2 = -a^2 - 1 := by
  sorry

end sqrt_negative_a_squared_plus_one_undefined_l2798_279817


namespace chord_length_is_three_l2798_279885

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

/-- The focus of the ellipse -/
def focus : ℝ × ℝ := (1, 0)

/-- The line passing through the focus and perpendicular to x-axis -/
def line (x : ℝ) : Prop := x = (focus.1)

/-- The chord length -/
def chord_length : ℝ := 3

/-- Theorem stating that the chord length cut by the line passing through
    the focus of the ellipse and perpendicular to the x-axis is equal to 3 -/
theorem chord_length_is_three :
  ∀ y₁ y₂ : ℝ,
  ellipse (focus.1) y₁ ∧ ellipse (focus.1) y₂ ∧ y₁ ≠ y₂ →
  |y₁ - y₂| = chord_length :=
sorry

end chord_length_is_three_l2798_279885


namespace empty_set_is_proposition_l2798_279893

-- Define what a proposition is
def is_proposition (s : String) : Prop := 
  ∃ (truth_value : Bool), (s = "true") ∨ (s = "false")

-- The statement we want to prove is a proposition
def empty_set_statement : String := "The empty set is a subset of any set"

-- Theorem statement
theorem empty_set_is_proposition : is_proposition empty_set_statement := by
  sorry


end empty_set_is_proposition_l2798_279893


namespace two_members_absent_l2798_279807

/-- Represents a trivia team with its total members and game performance -/
structure TriviaTeam where
  totalMembers : Float
  totalPoints : Float
  pointsPerMember : Float

/-- Calculates the number of members who didn't show up for a trivia game -/
def membersAbsent (team : TriviaTeam) : Float :=
  team.totalMembers - (team.totalPoints / team.pointsPerMember)

/-- Theorem stating that for the given trivia team, 2 members didn't show up -/
theorem two_members_absent (team : TriviaTeam) 
  (h1 : team.totalMembers = 5.0)
  (h2 : team.totalPoints = 6.0)
  (h3 : team.pointsPerMember = 2.0) : 
  membersAbsent team = 2 := by
  sorry

#eval membersAbsent { totalMembers := 5.0, totalPoints := 6.0, pointsPerMember := 2.0 }

end two_members_absent_l2798_279807


namespace total_prime_factors_l2798_279810

-- Define the expression
def expression := (4 : ℕ) ^ 13 * 7 ^ 5 * 11 ^ 2

-- Define the prime factorization of 4
axiom four_eq_two_squared : (4 : ℕ) = 2 ^ 2

-- Define 7 and 11 as prime numbers
axiom seven_prime : Nat.Prime 7
axiom eleven_prime : Nat.Prime 11

-- Theorem statement
theorem total_prime_factors : 
  (Nat.factors expression).length = 33 :=
sorry

end total_prime_factors_l2798_279810


namespace student_factor_problem_l2798_279804

theorem student_factor_problem (n : ℝ) (f : ℝ) : n = 124 → n * f - 138 = 110 → f = 2 := by
  sorry

end student_factor_problem_l2798_279804


namespace hypotenuse_length_hypotenuse_length_proof_l2798_279876

/-- Represents a right triangle with one leg of 10 inches and the angle opposite that leg being 60° --/
structure RightTriangle where
  leg : ℝ
  angle : ℝ
  leg_eq : leg = 10
  angle_eq : angle = 60

/-- Theorem stating that the hypotenuse of the described right triangle is (20√3)/3 inches --/
theorem hypotenuse_length (t : RightTriangle) : ℝ :=
  (20 * Real.sqrt 3) / 3

/-- Proof of the theorem --/
theorem hypotenuse_length_proof (t : RightTriangle) : 
  hypotenuse_length t = (20 * Real.sqrt 3) / 3 := by
  sorry

end hypotenuse_length_hypotenuse_length_proof_l2798_279876


namespace convex_quad_probability_l2798_279838

/-- The number of points on the circle -/
def n : ℕ := 8

/-- The number of chords to be selected -/
def k : ℕ := 4

/-- The total number of possible chords between n points -/
def total_chords : ℕ := n.choose 2

/-- The probability of forming a convex quadrilateral -/
def prob_convex_quad : ℚ := (n.choose k : ℚ) / (total_chords.choose k : ℚ)

/-- Theorem stating the probability of forming a convex quadrilateral -/
theorem convex_quad_probability : prob_convex_quad = 2 / 585 := by
  sorry

end convex_quad_probability_l2798_279838


namespace height_estimate_correct_l2798_279875

/-- Represents the regression line for student height based on foot length -/
structure HeightRegression where
  n : ℕ              -- number of students in the sample
  sum_x : ℝ          -- sum of foot lengths
  sum_y : ℝ          -- sum of heights
  slope : ℝ          -- slope of the regression line
  intercept : ℝ      -- y-intercept of the regression line

/-- Calculates the estimated height for a given foot length -/
def estimate_height (reg : HeightRegression) (x : ℝ) : ℝ :=
  reg.slope * x + reg.intercept

/-- Theorem stating that the estimated height for a foot length of 24 cm is 166 cm -/
theorem height_estimate_correct (reg : HeightRegression) : 
  reg.n = 10 ∧ 
  reg.sum_x = 225 ∧ 
  reg.sum_y = 1600 ∧ 
  reg.slope = 4 ∧
  reg.intercept = reg.sum_y / reg.n - reg.slope * (reg.sum_x / reg.n) →
  estimate_height reg 24 = 166 := by
  sorry

end height_estimate_correct_l2798_279875


namespace intersection_of_A_and_B_l2798_279835

def A : Set ℤ := {-1, 0, 1}
def B : Set ℤ := {0, 1, 2}

theorem intersection_of_A_and_B : A ∩ B = {0, 1} := by sorry

end intersection_of_A_and_B_l2798_279835


namespace smaller_cuboid_width_l2798_279846

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℝ :=
  d.length * d.width * d.height

theorem smaller_cuboid_width :
  let large_cuboid := CuboidDimensions.mk 12 14 10
  let num_smaller_cuboids : ℕ := 56
  let smaller_cuboid_length : ℝ := 5
  let smaller_cuboid_height : ℝ := 2
  let large_volume := cuboidVolume large_cuboid
  let smaller_volume := large_volume / num_smaller_cuboids
  smaller_volume / (smaller_cuboid_length * smaller_cuboid_height) = 3 := by
  sorry

#check smaller_cuboid_width

end smaller_cuboid_width_l2798_279846


namespace cars_meeting_time_l2798_279888

/-- Two cars driving towards each other meet after a certain time -/
theorem cars_meeting_time (speed1 : ℝ) (speed2 : ℝ) (distance : ℝ) : 
  speed1 = 100 →
  speed1 = 1.25 * speed2 →
  distance = 720 →
  (distance / (speed1 + speed2)) = 4 := by
sorry

end cars_meeting_time_l2798_279888


namespace min_sum_of_coefficients_l2798_279860

theorem min_sum_of_coefficients (a b : ℕ+) (h : 2 * a * 2 + b * 1 = 13) : 
  ∃ (m n : ℕ+), 2 * m * 2 + n * 1 = 13 ∧ m + n ≤ a + b ∧ m + n = 4 := by
  sorry

end min_sum_of_coefficients_l2798_279860


namespace perimeter_of_AMN_l2798_279899

-- Define the triangle ABC
structure Triangle :=
  (AB BC CA : ℝ)

-- Define the properties of triangle AMN
structure TriangleAMN (ABC : Triangle) :=
  (M : ℝ) -- Distance BM
  (N : ℝ) -- Distance CN
  (parallel_to_BC : True) -- MN is parallel to BC

-- Theorem statement
theorem perimeter_of_AMN (ABC : Triangle) (AMN : TriangleAMN ABC) :
  ABC.AB = 26 ∧ ABC.BC = 17 ∧ ABC.CA = 19 →
  (ABC.AB - AMN.M) + (ABC.CA - AMN.N) + 
    ((AMN.M / ABC.AB) * ABC.BC) = 45 :=
sorry

end perimeter_of_AMN_l2798_279899


namespace mike_books_before_sale_l2798_279895

def books_before_sale (books_bought books_after : ℕ) : ℕ :=
  books_after - books_bought

theorem mike_books_before_sale :
  books_before_sale 21 56 = 35 := by
  sorry

end mike_books_before_sale_l2798_279895


namespace other_root_of_complex_quadratic_l2798_279832

theorem other_root_of_complex_quadratic (z : ℂ) :
  z^2 = -20 + 15*I ∧ (4 + 3*I)^2 = -20 + 15*I →
  (-4 - 3*I)^2 = -20 + 15*I :=
by sorry

end other_root_of_complex_quadratic_l2798_279832


namespace guessing_game_score_sum_l2798_279880

/-- The guessing game score problem -/
theorem guessing_game_score_sum :
  ∀ (hajar_score farah_score : ℕ),
  hajar_score = 24 →
  farah_score - hajar_score = 21 →
  farah_score > hajar_score →
  hajar_score + farah_score = 69 :=
by
  sorry

end guessing_game_score_sum_l2798_279880


namespace fraction_problem_l2798_279800

theorem fraction_problem (x : ℝ) (f : ℝ) (h1 : x = 140) (h2 : 0.65 * x = f * x - 21) : f = 0.8 := by
  sorry

end fraction_problem_l2798_279800


namespace cosine_power_relation_l2798_279874

theorem cosine_power_relation (z : ℂ) (α : ℝ) (h : z + 1/z = 2 * Real.cos α) :
  ∀ n : ℕ, z^n + 1/z^n = 2 * Real.cos (n * α) := by
  sorry

end cosine_power_relation_l2798_279874


namespace mitten_knitting_time_l2798_279898

/-- Represents the time (in hours) to knit each item -/
structure KnittingTimes where
  hat : ℝ
  scarf : ℝ
  sweater : ℝ
  sock : ℝ
  mitten : ℝ

/-- Represents the number of each item in a set -/
structure SetComposition where
  hats : ℕ
  scarves : ℕ
  sweaters : ℕ
  mittens : ℕ
  socks : ℕ

def numGrandchildren : ℕ := 3

def knittingTimes : KnittingTimes := {
  hat := 2,
  scarf := 3,
  sweater := 6,
  sock := 1.5,
  mitten := 0  -- We'll solve for this
}

def setComposition : SetComposition := {
  hats := 1,
  scarves := 1,
  sweaters := 1,
  mittens := 2,
  socks := 2
}

def totalTime : ℝ := 48

theorem mitten_knitting_time :
  ∃ (mittenTime : ℝ),
    mittenTime > 0 ∧
    (let kt := { knittingTimes with mitten := mittenTime };
     (kt.hat * setComposition.hats +
      kt.scarf * setComposition.scarves +
      kt.sweater * setComposition.sweaters +
      kt.mitten * setComposition.mittens +
      kt.sock * setComposition.socks) * numGrandchildren = totalTime) ∧
    mittenTime = 1 := by sorry

end mitten_knitting_time_l2798_279898


namespace total_score_l2798_279825

-- Define the players and their scores
def Alex : ℕ := 18
def Sam : ℕ := Alex / 2
def Jon : ℕ := 2 * Sam + 3
def Jack : ℕ := Jon + 5
def Tom : ℕ := Jon + Jack - 4

-- State the theorem
theorem total_score : Alex + Sam + Jon + Jack + Tom = 117 := by
  sorry

end total_score_l2798_279825


namespace abc_value_l2798_279867

theorem abc_value (a b c : ℂ) 
  (eq1 : a * b + 4 * b = -16)
  (eq2 : b * c + 4 * c = -16)
  (eq3 : c * a + 4 * a = -16) :
  a * b * c = 64 := by
sorry

end abc_value_l2798_279867


namespace tangent_line_equation_l2798_279863

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - x + 3

-- Define the point of tangency
def point : ℝ × ℝ := (1, 3)

-- State the theorem
theorem tangent_line_equation :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, y = f x → (a * x + b * y + c = 0 ↔ (x, y) = point ∨ 
      ∃ h > 0, ∀ t : ℝ, 0 < |t| → |t| < h → 
        (a * (point.1 + t) + b * f (point.1 + t) + c) * (a * point.1 + b * point.2 + c) > 0)) ∧
    a = 2 ∧ b = -1 ∧ c = 1 :=
sorry

end tangent_line_equation_l2798_279863


namespace parallel_lines_theorem_l2798_279856

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)

-- State the theorem
theorem parallel_lines_theorem 
  (m n : Line) (α β : Plane) 
  (hm_neq_n : m ≠ n)
  (hα_neq_β : α ≠ β)
  (hm_parallel_β : parallel_line_plane m β)
  (hm_in_α : contained_in m α)
  (hα_intersect_β : intersect α β n) :
  parallel m n :=
sorry

end parallel_lines_theorem_l2798_279856


namespace joshua_justin_ratio_l2798_279852

def total_amount : ℝ := 40
def joshua_share : ℝ := 30

theorem joshua_justin_ratio :
  ∃ (k : ℝ), k > 0 ∧ joshua_share = k * (total_amount - joshua_share) →
  joshua_share / (total_amount - joshua_share) = 3 := by
  sorry

end joshua_justin_ratio_l2798_279852


namespace expansion_no_constant_term_l2798_279834

def has_no_constant_term (n : ℕ+) : Prop :=
  ∀ k : ℕ, k ≤ n → (1 + k - 4 * (k / 4) ≠ 0 ∧ 2 + k - 4 * (k / 4) ≠ 0)

theorem expansion_no_constant_term (n : ℕ+) (h : 2 ≤ n ∧ n ≤ 7) :
  has_no_constant_term n ↔ n = 5 := by
  sorry

end expansion_no_constant_term_l2798_279834


namespace triangle_possibilities_l2798_279814

-- Define a matchstick as a unit length
def matchstick_length : ℝ := 1

-- Define the total number of matchsticks
def total_matchsticks : ℕ := 12

-- Define a function to check if three lengths can form a triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Define the types of triangles
def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∧ a ≠ c) ∨ (b = c ∧ b ≠ a) ∨ (c = a ∧ c ≠ b)

def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

def is_right_angled (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ b^2 + c^2 = a^2 ∨ c^2 + a^2 = b^2

-- Theorem statement
theorem triangle_possibilities :
  ∃ (a b c : ℝ),
    a + b + c = total_matchsticks * matchstick_length ∧
    is_triangle a b c ∧
    (is_isosceles a b c ∧
     ∃ (d e f : ℝ), d + e + f = total_matchsticks * matchstick_length ∧
       is_triangle d e f ∧ is_equilateral d e f ∧
     ∃ (g h i : ℝ), g + h + i = total_matchsticks * matchstick_length ∧
       is_triangle g h i ∧ is_right_angled g h i) :=
by sorry

end triangle_possibilities_l2798_279814


namespace speed_time_relationship_l2798_279827

theorem speed_time_relationship (t v : ℝ) : t = 5 * v^2 ∧ t = 20 → v = 2 := by
  sorry

end speed_time_relationship_l2798_279827


namespace sin_two_pi_thirds_l2798_279849

theorem sin_two_pi_thirds : Real.sin (2 * Real.pi / 3) = Real.sqrt 3 / 2 := by
  sorry

end sin_two_pi_thirds_l2798_279849


namespace school_gender_ratio_l2798_279816

theorem school_gender_ratio (num_boys : ℕ) (num_girls : ℕ) : 
  num_boys = 80 →
  num_boys * 13 = num_girls * 5 →
  num_girls > num_boys →
  num_girls - num_boys = 128 :=
by
  sorry

end school_gender_ratio_l2798_279816


namespace book_sale_gain_percentage_l2798_279864

/-- Represents the problem of calculating the gain percentage on a book sale --/
theorem book_sale_gain_percentage 
  (total_cost : ℝ) 
  (cost_book1 : ℝ) 
  (loss_percentage : ℝ) 
  (total_cost_eq : total_cost = 360) 
  (cost_book1_eq : cost_book1 = 210) 
  (loss_percentage_eq : loss_percentage = 15) 
  (cost_book2_eq : total_cost = cost_book1 + cost_book2) 
  (same_selling_price : 
    cost_book1 * (1 - loss_percentage / 100) = 
    cost_book2 * (1 + gain_percentage / 100)) : 
  gain_percentage = 19 := by sorry

end book_sale_gain_percentage_l2798_279864


namespace factory_reorganization_l2798_279830

theorem factory_reorganization (workshop1 workshop2 : ℕ) : 
  (workshop1 / 2 + workshop2 / 3 = (workshop1 / 3 + workshop2 / 2) * 8 / 7) →
  (workshop1 + workshop2 - (workshop1 / 2 + workshop2 / 3 + workshop1 / 3 + workshop2 / 2) = 120) →
  (workshop1 = 480 ∧ workshop2 = 240) := by
  sorry

end factory_reorganization_l2798_279830


namespace least_square_tiles_l2798_279829

def room_length : ℕ := 720
def room_width : ℕ := 432

theorem least_square_tiles (l w : ℕ) (h1 : l = room_length) (h2 : w = room_width) :
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧
    l % tile_size = 0 ∧ 
    w % tile_size = 0 ∧
    (l / tile_size) * (w / tile_size) = 15 ∧
    ∀ (other_size : ℕ), 
      (other_size > 0 ∧ l % other_size = 0 ∧ w % other_size = 0) →
      (l / other_size) * (w / other_size) ≥ 15 := by
  sorry

#check least_square_tiles

end least_square_tiles_l2798_279829


namespace inequality_product_l2798_279843

theorem inequality_product (a b c d : ℝ) 
  (h1 : a < b) (h2 : b < 0) 
  (h3 : c < d) (h4 : d < 0) : 
  a * c > b * d := by
  sorry

end inequality_product_l2798_279843


namespace infinitely_many_primes_6n_plus_5_l2798_279869

theorem infinitely_many_primes_6n_plus_5 : 
  Set.Infinite {p : ℕ | Nat.Prime p ∧ ∃ n : ℕ, p = 6 * n + 5} := by
  sorry

end infinitely_many_primes_6n_plus_5_l2798_279869


namespace abs_diff_one_if_sum_one_l2798_279889

theorem abs_diff_one_if_sum_one (a b : ℤ) (h : |a| + |b| = 1) : |a - b| = 1 := by
  sorry

end abs_diff_one_if_sum_one_l2798_279889


namespace solution_set_quadratic_inequality_l2798_279873

theorem solution_set_quadratic_inequality :
  let f : ℝ → ℝ := λ x => -x^2 + 3*x - 2
  {x : ℝ | f x ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} := by
sorry

end solution_set_quadratic_inequality_l2798_279873


namespace sampling_methods_correct_l2798_279858

-- Define the sampling methods
inductive SamplingMethod
  | SimpleRandom
  | Stratified
  | Systematic

-- Define a scenario
structure Scenario where
  total_population : ℕ
  sample_size : ℕ
  has_distinct_groups : Bool
  is_ordered : Bool

-- Define a function to determine the most suitable sampling method
def most_suitable_method (s : Scenario) : SamplingMethod :=
  if s.total_population ≤ 15 then SamplingMethod.SimpleRandom
  else if s.has_distinct_groups then SamplingMethod.Stratified
  else if s.is_ordered then SamplingMethod.Systematic
  else SamplingMethod.SimpleRandom

-- Theorem to prove
theorem sampling_methods_correct :
  (most_suitable_method ⟨15, 5, false, false⟩ = SamplingMethod.SimpleRandom) ∧
  (most_suitable_method ⟨240, 20, true, false⟩ = SamplingMethod.Stratified) ∧
  (most_suitable_method ⟨950, 25, false, true⟩ = SamplingMethod.Systematic) :=
by sorry

end sampling_methods_correct_l2798_279858


namespace students_after_three_stops_l2798_279887

/-- Calculates the number of students on the bus after three stops --/
def studentsOnBusAfterThreeStops (initial : ℕ) 
  (firstOff firstOn : ℕ) 
  (secondOff secondOn : ℕ) 
  (thirdOff thirdOn : ℕ) : ℕ :=
  initial - firstOff + firstOn - secondOff + secondOn - thirdOff + thirdOn

/-- Theorem stating the number of students on the bus after three stops --/
theorem students_after_three_stops :
  studentsOnBusAfterThreeStops 10 3 4 2 5 6 3 = 11 := by
  sorry

end students_after_three_stops_l2798_279887


namespace binary_representation_properties_l2798_279809

def has_exactly_three_ones (n : ℕ) : Prop :=
  (n.digits 2).count 1 = 3

def is_multiple_of_617 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 617 * k

theorem binary_representation_properties (n : ℕ) 
  (h1 : is_multiple_of_617 n) 
  (h2 : has_exactly_three_ones n) : 
  ((n.digits 2).length ≥ 9) ∧ 
  ((n.digits 2).length = 10 → Even n) :=
sorry

end binary_representation_properties_l2798_279809


namespace geometric_sequence_sixth_term_l2798_279840

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_sixth_term
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_roots : a 2 ^ 2 - 13 * a 2 + 14 = 0 ∧ a 10 ^ 2 - 13 * a 10 + 14 = 0) :
  a 6 = Real.sqrt 14 := by
sorry

end geometric_sequence_sixth_term_l2798_279840


namespace price_change_theorem_l2798_279824

theorem price_change_theorem (initial_price : ℝ) (h_pos : initial_price > 0) :
  let price_after_increase := initial_price * (1 + 0.34)
  let price_after_first_discount := price_after_increase * (1 - 0.10)
  let final_price := price_after_first_discount * (1 - 0.15)
  let percentage_change := (final_price - initial_price) / initial_price * 100
  percentage_change = 2.51 := by sorry

end price_change_theorem_l2798_279824


namespace green_hat_cost_l2798_279859

/-- Proves that the cost of each green hat is $7 given the conditions of the problem -/
theorem green_hat_cost (total_hats : ℕ) (blue_hat_cost : ℕ) (total_price : ℕ) (green_hats : ℕ) :
  total_hats = 85 →
  blue_hat_cost = 6 →
  total_price = 550 →
  green_hats = 40 →
  (total_hats - green_hats) * blue_hat_cost + green_hats * 7 = total_price :=
by sorry

end green_hat_cost_l2798_279859


namespace circle_area_from_circumference_l2798_279882

theorem circle_area_from_circumference (circumference : ℝ) (area : ℝ) :
  circumference = 18 →
  area = 81 / Real.pi :=
by
  sorry

end circle_area_from_circumference_l2798_279882


namespace hyperbola_equivalence_l2798_279847

theorem hyperbola_equivalence (x y : ℝ) :
  (4 * x^2 * y^2 = 4 * x * y + 3) ↔ (x * y = 3/2 ∨ x * y = -1/2) := by
  sorry

end hyperbola_equivalence_l2798_279847


namespace hidden_dots_count_l2798_279808

def standard_die_sum : ℕ := 21

def visible_faces : List ℕ := [1, 2, 3, 3, 4, 5, 6, 6, 6]

def total_dice : ℕ := 4

theorem hidden_dots_count :
  (total_dice * standard_die_sum) - (visible_faces.sum) = 48 := by
  sorry

end hidden_dots_count_l2798_279808


namespace flour_per_batch_correct_l2798_279894

/-- The number of cups of flour required for one batch of cookies. -/
def flour_per_batch : ℝ := 2

/-- The number of batches Gigi has baked. -/
def baked_batches : ℕ := 3

/-- The total amount of flour in Gigi's bag. -/
def total_flour : ℝ := 20

/-- The number of additional batches Gigi could make with remaining flour. -/
def future_batches : ℕ := 7

/-- Theorem stating that the amount of flour per batch is correct given the conditions. -/
theorem flour_per_batch_correct :
  flour_per_batch * (baked_batches + future_batches : ℝ) = total_flour :=
by sorry

end flour_per_batch_correct_l2798_279894


namespace quadratic_no_solution_l2798_279844

-- Define the quadratic function
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_no_solution (a b c : ℝ) (h_a : a ≠ 0) :
  f a b c 0 = 2 →
  f a b c 1 = 1 →
  f a b c 2 = 2 →
  f a b c 3 = 5 →
  f a b c 4 = 10 →
  ∀ x, f a b c x ≠ 0 :=
by
  sorry

end quadratic_no_solution_l2798_279844


namespace three_integer_chords_l2798_279850

/-- A circle with a point P inside --/
structure CircleWithPoint where
  radius : ℝ
  distanceToCenter : ℝ

/-- Count of integer-length chords through P --/
def integerChordCount (c : CircleWithPoint) : ℕ :=
  sorry

theorem three_integer_chords (c : CircleWithPoint) 
  (h1 : c.radius = 12)
  (h2 : c.distanceToCenter = 5) : 
  integerChordCount c = 3 := by
  sorry

end three_integer_chords_l2798_279850


namespace profit_range_max_avg_profit_l2798_279892

/-- Cumulative profit function -/
def profit (x : ℕ) : ℚ :=
  -1/2 * x^2 + 60*x - 800

/-- Average daily profit function -/
def avgProfit (x : ℕ) : ℚ :=
  profit x / x

theorem profit_range (x : ℕ) (hx : x > 0) :
  profit x > 800 ↔ x > 40 ∧ x < 80 :=
sorry

theorem max_avg_profit :
  ∃ (x : ℕ), x > 0 ∧ ∀ (y : ℕ), y > 0 → avgProfit x ≥ avgProfit y ∧ x = 400 :=
sorry

end profit_range_max_avg_profit_l2798_279892


namespace chapatis_ordered_l2798_279828

/-- The number of chapatis ordered by Alok -/
def chapatis : ℕ := sorry

/-- The cost of each chapati in rupees -/
def chapati_cost : ℕ := 6

/-- The cost of each plate of rice in rupees -/
def rice_cost : ℕ := 45

/-- The cost of each plate of mixed vegetable in rupees -/
def vegetable_cost : ℕ := 70

/-- The cost of each ice-cream cup in rupees -/
def icecream_cost : ℕ := 40

/-- The number of plates of rice ordered -/
def rice_plates : ℕ := 5

/-- The number of plates of mixed vegetable ordered -/
def vegetable_plates : ℕ := 7

/-- The number of ice-cream cups ordered -/
def icecream_cups : ℕ := 6

/-- The total amount paid in rupees -/
def total_paid : ℕ := 1051

theorem chapatis_ordered : chapatis = 16 := by
  sorry

end chapatis_ordered_l2798_279828


namespace sum_of_solutions_x_minus_4_squared_equals_16_l2798_279862

theorem sum_of_solutions_x_minus_4_squared_equals_16 : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 4)^2 = 16 ∧ (x₂ - 4)^2 = 16 ∧ x₁ + x₂ = 8 :=
by sorry

end sum_of_solutions_x_minus_4_squared_equals_16_l2798_279862


namespace right_triangle_area_l2798_279823

/-- The area of a right triangle with legs 18 and 80 is 720 -/
theorem right_triangle_area : 
  ∀ (a b c : ℝ), 
  a = 18 → b = 80 → c = 82 →
  a^2 + b^2 = c^2 →
  (1/2) * a * b = 720 := by
  sorry

end right_triangle_area_l2798_279823


namespace bird_cage_problem_l2798_279821

theorem bird_cage_problem (initial_birds : ℕ) (final_birds : ℕ) : 
  initial_birds = 60 → 
  final_birds = 8 → 
  ∃ (remaining_after_second : ℕ),
    remaining_after_second = initial_birds * (2/3) * (3/5) ∧
    (2/3 : ℚ) = (remaining_after_second - final_birds) / remaining_after_second :=
by sorry

end bird_cage_problem_l2798_279821


namespace book_cost_calculation_l2798_279831

theorem book_cost_calculation (initial_amount : ℕ) (num_books : ℕ) (remaining_amount : ℕ) :
  initial_amount = 79 →
  num_books = 9 →
  remaining_amount = 16 →
  (initial_amount - remaining_amount) / num_books = 7 :=
by sorry

end book_cost_calculation_l2798_279831


namespace square_of_two_minus_x_l2798_279871

theorem square_of_two_minus_x (x : ℝ) : (2 - x)^2 = 4 - 4*x + x^2 := by
  sorry

end square_of_two_minus_x_l2798_279871


namespace unique_three_digit_factorial_sum_l2798_279879

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def digit_factorial_sum (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  factorial hundreds + factorial tens + factorial ones

theorem unique_three_digit_factorial_sum :
  ∃! n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 
  (∃ d, d ∈ [n / 100, (n / 10) % 10, n % 10] ∧ d = 6) ∧
  n = digit_factorial_sum n :=
by
  sorry

end unique_three_digit_factorial_sum_l2798_279879


namespace blue_pen_cost_is_ten_cents_l2798_279848

/-- The cost of a blue pen given the conditions of Maci's pen purchase. -/
def blue_pen_cost (blue_pens red_pens : ℕ) (total_cost : ℚ) : ℚ :=
  total_cost / (blue_pens + 2 * red_pens)

/-- Theorem stating that the cost of a blue pen is $0.10 under the given conditions. -/
theorem blue_pen_cost_is_ten_cents :
  blue_pen_cost 10 15 4 = 1/10 := by
  sorry

end blue_pen_cost_is_ten_cents_l2798_279848


namespace circle_line_distance_range_l2798_279851

/-- The range of c for which there are four points on the circle x^2 + y^2 = 4
    at a distance of 1 from the line 12x - 5y + c = 0 is (-13, 13) -/
theorem circle_line_distance_range :
  ∀ c : ℝ,
  (∃ (points : Finset (ℝ × ℝ)),
    points.card = 4 ∧
    (∀ (x y : ℝ), (x, y) ∈ points →
      x^2 + y^2 = 4 ∧
      (|12*x - 5*y + c| / Real.sqrt (12^2 + (-5)^2) = 1))) ↔
  -13 < c ∧ c < 13 :=
by sorry

end circle_line_distance_range_l2798_279851


namespace tangent_properties_l2798_279845

/-- The function f(x) = x^3 - x -/
def f (x : ℝ) : ℝ := x^3 - x

/-- The function g(x) = x^2 + a, where a is a parameter -/
def g (a : ℝ) (x : ℝ) : ℝ := x^2 + a

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 * x^2 - 1

/-- The derivative of g(x) -/
def g' (x : ℝ) : ℝ := 2 * x

/-- The tangent line of f at x₁ is also the tangent line of g -/
def tangent_condition (a : ℝ) (x₁ : ℝ) : Prop :=
  ∃ x₂ : ℝ, f' x₁ = g' x₂ ∧ f x₁ + f' x₁ * (x₂ - x₁) = g a x₂

theorem tangent_properties :
  (tangent_condition 3 (-1)) ∧
  (∀ a : ℝ, (∃ x₁ : ℝ, tangent_condition a x₁) → a ≥ -1) :=
sorry

end tangent_properties_l2798_279845


namespace power_six_expression_l2798_279855

theorem power_six_expression (m n : ℕ) (P Q : ℕ) 
  (h1 : P = 2^m) (h2 : Q = 5^n) : 
  6^(m+n) = P * 2^n * 3^(m+n) := by
  sorry

end power_six_expression_l2798_279855


namespace x_intercepts_count_l2798_279805

theorem x_intercepts_count : 
  let f (x : ℝ) := (x - 3) * (x^2 + 4*x + 4)
  ∃ (a b : ℝ), a ≠ b ∧ 
    (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b) :=
by sorry

end x_intercepts_count_l2798_279805


namespace quadratic_roots_sum_equality_l2798_279813

theorem quadratic_roots_sum_equality (b₁ b₂ b₃ : ℝ) : ∃ (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ),
  (x₁ = (-b₁ + 1) / 2 ∧ y₁ = (-b₁ - 1) / 2) ∧
  (x₂ = (-b₂ + 2) / 2 ∧ y₂ = (-b₂ - 2) / 2) ∧
  (x₃ = (-b₃ + 3) / 2 ∧ y₃ = (-b₃ - 3) / 2) ∧
  x₁ + x₂ + x₃ = y₁ + y₂ + y₃ :=
by sorry

end quadratic_roots_sum_equality_l2798_279813


namespace sphere_surface_area_for_given_prism_l2798_279884

/-- A right square prism with all vertices on the surface of a sphere -/
structure PrismOnSphere where
  height : ℝ
  volume : ℝ
  prism_on_sphere : Bool

/-- The surface area of a sphere given a PrismOnSphere -/
def sphere_surface_area (p : PrismOnSphere) : ℝ := sorry

theorem sphere_surface_area_for_given_prism :
  ∀ p : PrismOnSphere,
    p.height = 4 ∧ 
    p.volume = 16 ∧ 
    p.prism_on_sphere = true →
    sphere_surface_area p = 24 * Real.pi :=
by sorry

end sphere_surface_area_for_given_prism_l2798_279884


namespace evaluate_expression_l2798_279881

theorem evaluate_expression : (30 - (3030 - 303)) * (3030 - (303 - 30)) = -7435969 := by
  sorry

end evaluate_expression_l2798_279881


namespace unique_arithmetic_triangle_l2798_279806

/-- A triangle with integer angles in arithmetic progression -/
structure ArithmeticTriangle where
  a : ℕ
  d : ℕ
  sum_180 : a + (a + d) + (a + 2*d) = 180
  distinct : a ≠ a + d ∧ a ≠ a + 2*d ∧ a + d ≠ a + 2*d

/-- Theorem stating there's exactly one valid arithmetic triangle with possibly zero angle -/
theorem unique_arithmetic_triangle : 
  ∃! t : ArithmeticTriangle, t.a = 0 ∨ t.a + t.d = 0 ∨ t.a + 2*t.d = 0 :=
sorry

end unique_arithmetic_triangle_l2798_279806


namespace natural_numbers_difference_l2798_279890

theorem natural_numbers_difference (a b : ℕ) : 
  a + b = 20250 → 
  b % 15 = 0 → 
  a = b / 3 → 
  b - a = 10130 := by
sorry

end natural_numbers_difference_l2798_279890


namespace bicycle_price_problem_l2798_279854

theorem bicycle_price_problem (cost_price_A : ℝ) : 
  let selling_price_B := cost_price_A * 1.25
  let selling_price_C := selling_price_B * 1.5
  selling_price_C = 225 → cost_price_A = 120 := by
sorry

end bicycle_price_problem_l2798_279854


namespace no_odd_integer_solution_l2798_279837

theorem no_odd_integer_solution :
  ¬∃ (x y z : ℤ), Odd x ∧ Odd y ∧ Odd z ∧ (x + y)^2 + (x + z)^2 = (y + z)^2 := by
  sorry

end no_odd_integer_solution_l2798_279837


namespace log_ratio_squared_l2798_279803

theorem log_ratio_squared (x y : ℝ) (hx : x > 0) (hy : y > 0) (hx1 : x ≠ 1) (hy1 : y ≠ 1) 
  (h1 : Real.log x / Real.log 3 = Real.log 81 / Real.log y) (h2 : x * y = 243) : 
  (Real.log (x / y) / Real.log 3)^2 = 9 := by
  sorry

end log_ratio_squared_l2798_279803


namespace max_guaranteed_amount_100_cards_l2798_279842

/-- Represents a set of bank cards with amounts from 1 to n rubles -/
def BankCards (n : ℕ) := Finset (Fin n)

/-- The strategy of requesting a fixed amount from each card -/
def Strategy (n : ℕ) := ℕ

/-- The amount guaranteed to be collected given a strategy -/
def guaranteedAmount (n : ℕ) (s : Strategy n) : ℕ := sorry

/-- The maximum guaranteed amount that can be collected -/
def maxGuaranteedAmount (n : ℕ) : ℕ := sorry

theorem max_guaranteed_amount_100_cards :
  maxGuaranteedAmount 100 = 2550 := by sorry

end max_guaranteed_amount_100_cards_l2798_279842


namespace condition_analysis_l2798_279822

theorem condition_analysis (a : ℝ) : 
  (∀ a, a > 1 → 1/a < 1) ∧ 
  (∃ a, 1/a < 1 ∧ a ≤ 1) := by
  sorry

end condition_analysis_l2798_279822
