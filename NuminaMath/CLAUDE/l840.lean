import Mathlib

namespace baker_cake_difference_l840_84083

/-- Given the initial number of cakes, number of cakes sold, and number of cakes bought,
    prove that the difference between cakes bought and sold is 63. -/
theorem baker_cake_difference (initial : ℕ) (sold : ℕ) (bought : ℕ)
  (h1 : initial = 13)
  (h2 : sold = 91)
  (h3 : bought = 154) :
  bought - sold = 63 := by
  sorry

end baker_cake_difference_l840_84083


namespace octagon_trapezoid_area_l840_84025

/-- The area of a trapezoid formed by four consecutive vertices of a regular octagon --/
theorem octagon_trapezoid_area (side_length : ℝ) (h : side_length = 6) :
  let diagonal_ratio : ℝ := Real.sqrt (4 + 2 * Real.sqrt 2)
  let height : ℝ := side_length * diagonal_ratio * (Real.sqrt (2 - Real.sqrt 2) / 2)
  let area : ℝ := side_length * height
  area = 18 * Real.sqrt (16 - 4 * Real.sqrt 2) :=
by sorry


end octagon_trapezoid_area_l840_84025


namespace geometric_sequence_common_ratio_l840_84020

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 2 →                        -- given a₂ = 2
  a 5 = 1/4 →                      -- given a₅ = 1/4
  q = 1/2 :=                       -- prove q = 1/2
by
  sorry

end geometric_sequence_common_ratio_l840_84020


namespace college_student_count_l840_84069

/-- Represents the number of students in a college -/
structure College where
  boys : ℕ
  girls : ℕ

/-- The total number of students in the college -/
def College.total (c : College) : ℕ := c.boys + c.girls

/-- A college with a 5:7 ratio of boys to girls and 140 girls -/
def myCollege : College where
  girls := 140
  boys := 140 * 5 / 7

theorem college_student_count : myCollege.total = 240 := by
  sorry

end college_student_count_l840_84069


namespace first_number_is_55_l840_84026

def problem (x : ℝ) : Prop :=
  let known_numbers : List ℝ := [48, 507, 2, 684, 42]
  let all_numbers : List ℝ := x :: known_numbers
  (List.sum all_numbers) / 6 = 223

theorem first_number_is_55 : 
  ∃ (x : ℝ), problem x ∧ x = 55 :=
sorry

end first_number_is_55_l840_84026


namespace arithmetic_geometric_means_sum_l840_84047

/-- Given real numbers a, b, c in geometric progression and non-zero real numbers x, y
    that are arithmetic means of a, b and b, c respectively, prove that a/x + b/y = 2 -/
theorem arithmetic_geometric_means_sum (a b c x y : ℝ) 
  (hgp : b^2 = a*c)  -- geometric progression condition
  (hx : x ≠ 0)       -- x is non-zero
  (hy : y ≠ 0)       -- y is non-zero
  (hax : 2*x = a + b)  -- x is arithmetic mean of a and b
  (hby : 2*y = b + c)  -- y is arithmetic mean of b and c
  : a/x + b/y = 2 := by
  sorry

end arithmetic_geometric_means_sum_l840_84047


namespace prime_equation_solution_l840_84078

theorem prime_equation_solution (p q r : ℕ) (A : ℕ) 
  (h_prime_p : Nat.Prime p) 
  (h_prime_q : Nat.Prime q) 
  (h_prime_r : Nat.Prime r) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ q ≠ r)
  (h_equation : 2*p*q*r + 50*p*q = 7*p*q*r + 55*p*r ∧ 
                7*p*q*r + 55*p*r = 8*p*q*r + 12*q*r ∧
                8*p*q*r + 12*q*r = A)
  (h_positive : A > 0) : 
  A = 1980 := by
sorry

end prime_equation_solution_l840_84078


namespace fescue_percentage_in_y_l840_84002

/-- Represents the composition of a seed mixture -/
structure SeedMixture where
  ryegrass : ℝ
  bluegrass : ℝ
  fescue : ℝ

/-- The combined mixture of X and Y -/
def combinedMixture (x y : SeedMixture) (xProportion : ℝ) : SeedMixture :=
  { ryegrass := x.ryegrass * xProportion + y.ryegrass * (1 - xProportion)
  , bluegrass := x.bluegrass * xProportion + y.bluegrass * (1 - xProportion)
  , fescue := x.fescue * xProportion + y.fescue * (1 - xProportion) }

/-- The theorem stating the percentage of fescue in mixture Y -/
theorem fescue_percentage_in_y
  (x : SeedMixture)
  (y : SeedMixture)
  (h1 : x.ryegrass = 0.4)
  (h2 : x.bluegrass = 0.6)
  (h3 : x.fescue = 0)
  (h4 : y.ryegrass = 0.25)
  (h5 : x.ryegrass + x.bluegrass + x.fescue = 1)
  (h6 : y.ryegrass + y.bluegrass + y.fescue = 1)
  (h7 : (combinedMixture x y 0.4667).ryegrass = 0.32) :
  y.fescue = 0.75 := by
  sorry

end fescue_percentage_in_y_l840_84002


namespace seventh_group_selection_l840_84082

/-- Represents a systematic sampling method for a class of students. -/
structure SystematicSampling where
  total_students : ℕ
  num_groups : ℕ
  group_size : ℕ
  third_group_selection : ℕ

/-- Calculates the number drawn from a specific group in a systematic sampling method. -/
def number_drawn (s : SystematicSampling) (group : ℕ) : ℕ :=
  (group - 1) * s.group_size + (s.third_group_selection - ((3 - 1) * s.group_size))

/-- Theorem stating that if the number drawn from the third group is 13,
    then the number drawn from the seventh group is 33. -/
theorem seventh_group_selection
  (s : SystematicSampling)
  (h1 : s.total_students = 50)
  (h2 : s.num_groups = 10)
  (h3 : s.group_size = s.total_students / s.num_groups)
  (h4 : s.third_group_selection = 13) :
  number_drawn s 7 = 33 := by
  sorry

end seventh_group_selection_l840_84082


namespace perpendicular_to_plane_implies_parallel_l840_84079

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_to_plane_implies_parallel 
  (a b : Line) (α : Plane) :
  perpendicular a α → perpendicular b α → parallel a b :=
sorry

end perpendicular_to_plane_implies_parallel_l840_84079


namespace solve_salt_merchant_problem_l840_84092

def salt_merchant_problem (initial_purchase : ℝ) (profit1 : ℝ) (profit2 : ℝ) : Prop :=
  let revenue1 := initial_purchase + profit1
  let profit_rate := profit2 / revenue1
  profit_rate * initial_purchase = profit1 ∧ profit1 = 100 ∧ profit2 = 120

theorem solve_salt_merchant_problem :
  ∃ (initial_purchase : ℝ), salt_merchant_problem initial_purchase 100 120 ∧ initial_purchase = 500 :=
by
  sorry

end solve_salt_merchant_problem_l840_84092


namespace venue_cost_venue_cost_is_10000_l840_84088

/-- Calculates the venue cost for John's wedding --/
theorem venue_cost (cost_per_guest : ℕ) (john_guests : ℕ) (wife_extra_percent : ℕ) (total_cost : ℕ) : ℕ :=
  let wife_guests := john_guests + (wife_extra_percent * john_guests) / 100
  let guest_cost := cost_per_guest * wife_guests
  total_cost - guest_cost

/-- Proves that the venue cost is $10,000 given the specified conditions --/
theorem venue_cost_is_10000 :
  venue_cost 500 50 60 50000 = 10000 := by
  sorry

end venue_cost_venue_cost_is_10000_l840_84088


namespace original_room_population_l840_84074

theorem original_room_population (x : ℚ) : 
  (1 / 2 : ℚ) * x = 18 →
  (2 / 3 : ℚ) * x - (1 / 4 : ℚ) * ((2 / 3 : ℚ) * x) = 18 →
  x = 36 := by sorry

end original_room_population_l840_84074


namespace theater_occupancy_l840_84055

theorem theater_occupancy (total_seats empty_seats : ℕ) 
  (h1 : total_seats = 750) 
  (h2 : empty_seats = 218) : 
  total_seats - empty_seats = 532 := by
  sorry

end theater_occupancy_l840_84055


namespace wang_elevator_journey_l840_84058

def floor_movements : List Int := [6, -3, 10, -8, 12, -7, -10]
def floor_height : ℝ := 3
def electricity_per_meter : ℝ := 0.2

theorem wang_elevator_journey :
  (List.sum floor_movements = 0) ∧
  (List.sum (List.map Int.natAbs floor_movements) * floor_height * electricity_per_meter = 33.6) := by
  sorry

end wang_elevator_journey_l840_84058


namespace trajectory_equation_of_midpoints_l840_84029

/-- Given three real numbers forming an arithmetic sequence and equations of a line and parabola,
    prove the trajectory equation of the midpoints of the intercepted chords. -/
theorem trajectory_equation_of_midpoints
  (a b c : ℝ)
  (h_arithmetic : c = 2*b - a) -- arithmetic sequence condition
  (h_line : ∀ x y, b*x + a*y + c = 0 → (x : ℝ) = x ∧ (y : ℝ) = y) -- line equation
  (h_parabola : ∀ x y, y^2 = -1/2*x → (x : ℝ) = x ∧ (y : ℝ) = y) -- parabola equation
  : ∃ (x y : ℝ), x + 1 = -(2*y - 1)^2 ∧ y ≠ 1 :=
sorry

end trajectory_equation_of_midpoints_l840_84029


namespace exists_integers_for_n_squared_and_cubed_l840_84038

theorem exists_integers_for_n_squared_and_cubed (n : ℕ) : 
  (∃ a b : ℤ, n^2 = a + b ∧ n^3 = a^2 + b^2) ↔ n = 0 ∨ n = 1 ∨ n = 2 := by
sorry

end exists_integers_for_n_squared_and_cubed_l840_84038


namespace people_per_cubic_yard_l840_84040

theorem people_per_cubic_yard (people_per_yard : ℕ) : 
  (9000 * people_per_yard - 6400 * people_per_yard = 208000) → 
  people_per_yard = 80 := by
sorry

end people_per_cubic_yard_l840_84040


namespace product_value_l840_84056

theorem product_value : 
  (1 / 4 : ℚ) * 8 * (1 / 16 : ℚ) * 32 * (1 / 64 : ℚ) * 128 * (1 / 256 : ℚ) * 512 * (1 / 1024 : ℚ) * 2048 = 32 := by
  sorry

end product_value_l840_84056


namespace sum_of_digits_Q_is_six_l840_84084

-- Define R_k as a function that takes k and returns the integer with k ones in base 10
def R (k : ℕ) : ℕ := (10^k - 1) / 9

-- Define Q as R_30 / R_5
def Q : ℕ := R 30 / R 5

-- Function to calculate the sum of digits of a natural number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

-- Theorem stating that the sum of digits of Q is 6
theorem sum_of_digits_Q_is_six : sum_of_digits Q = 6 := by
  sorry

end sum_of_digits_Q_is_six_l840_84084


namespace modified_cube_edge_count_l840_84070

/-- Represents a modified cube with smaller cubes removed from corners -/
structure ModifiedCube where
  initialSideLength : ℕ
  removedCubeSideLength : ℕ
  numCornersRemoved : ℕ

/-- Calculates the number of edges in the modified cube -/
def edgeCount (cube : ModifiedCube) : ℕ :=
  12 + 6 * cube.numCornersRemoved

/-- Theorem stating that a cube of side length 5 with 1x1 cubes removed from 4 corners has 36 edges -/
theorem modified_cube_edge_count :
  let cube : ModifiedCube := {
    initialSideLength := 5,
    removedCubeSideLength := 1,
    numCornersRemoved := 4
  }
  edgeCount cube = 36 := by sorry

end modified_cube_edge_count_l840_84070


namespace gcd_of_g_103_104_l840_84021

/-- The function g as defined in the problem -/
def g (x : ℤ) : ℤ := x^2 - x + 2025

/-- The theorem stating that the GCD of g(103) and g(104) is 2 -/
theorem gcd_of_g_103_104 : Int.gcd (g 103) (g 104) = 2 := by sorry

end gcd_of_g_103_104_l840_84021


namespace download_calculation_l840_84048

/-- Calculates the number of songs that can be downloaded given internet speed, song size, and time. -/
def songs_downloaded (internet_speed : ℕ) (song_size : ℕ) (time_minutes : ℕ) : ℕ :=
  (internet_speed * 60 * time_minutes) / song_size

/-- Theorem stating that with given conditions, 7200 songs can be downloaded. -/
theorem download_calculation :
  let internet_speed : ℕ := 20  -- MBps
  let song_size : ℕ := 5        -- MB
  let time_minutes : ℕ := 30    -- half an hour
  songs_downloaded internet_speed song_size time_minutes = 7200 := by
sorry

end download_calculation_l840_84048


namespace intersection_range_l840_84093

-- Define the endpoints of the line segment
def A : ℝ × ℝ := (3, 2)
def B : ℝ × ℝ := (2, 3)

-- Define the line equation
def line_equation (k : ℝ) (x : ℝ) : ℝ := k * (x - 1)

-- Define the condition for intersection
def intersects (k : ℝ) : Prop :=
  ∃ x y, x ≥ min A.1 B.1 ∧ x ≤ max A.1 B.1 ∧
         y ≥ min A.2 B.2 ∧ y ≤ max A.2 B.2 ∧
         y = line_equation k x

-- Theorem statement
theorem intersection_range :
  {k : ℝ | intersects k} = {k : ℝ | 1 ≤ k ∧ k ≤ 3} :=
sorry

end intersection_range_l840_84093


namespace modular_inverse_of_3_mod_23_l840_84035

theorem modular_inverse_of_3_mod_23 : ∃ x : ℕ, x ≤ 22 ∧ (3 * x) % 23 = 1 :=
by
  use 8
  sorry

end modular_inverse_of_3_mod_23_l840_84035


namespace bee_count_l840_84014

theorem bee_count (flowers : ℕ) (bees : ℕ) : 
  flowers = 5 → bees = flowers - 2 → bees = 3 := by sorry

end bee_count_l840_84014


namespace polynomial_roots_condition_l840_84007

open Real

/-- The polynomial in question -/
def polynomial (q x : ℝ) : ℝ := x^4 + 2*q*x^3 + 3*x^2 + 2*q*x + 2

/-- Predicate for a number being a root of the polynomial -/
def is_root (q x : ℝ) : Prop := polynomial q x = 0

/-- Theorem stating the condition for the polynomial to have at least two distinct negative real roots with product 2 -/
theorem polynomial_roots_condition (q : ℝ) : 
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ x ≠ y ∧ x * y = 2 ∧ is_root q x ∧ is_root q y) ↔ q < -7 * sqrt 2 / 4 := by sorry

end polynomial_roots_condition_l840_84007


namespace complex_expression_evaluation_l840_84003

theorem complex_expression_evaluation : 
  (0.027)^(-1/3 : ℝ) - (-1/7)^(-2 : ℝ) + (25/9 : ℝ)^(1/2 : ℝ) - (Real.sqrt 2 - 1)^(0 : ℝ) = -45 := by
  sorry

end complex_expression_evaluation_l840_84003


namespace mass_of_man_is_60kg_l840_84042

/-- The mass of a man who causes a boat to sink by a certain amount. -/
def mass_of_man (boat_length boat_breadth sink_depth water_density : ℝ) : ℝ :=
  boat_length * boat_breadth * sink_depth * water_density

/-- Theorem stating that the mass of the man is 60 kg under the given conditions. -/
theorem mass_of_man_is_60kg :
  mass_of_man 3 2 0.01 1000 = 60 := by sorry

end mass_of_man_is_60kg_l840_84042


namespace combined_tower_height_l840_84027

/-- The height of Grace's tower in inches -/
def grace_height : ℕ := 40

/-- The ratio of Grace's tower height to Clyde's tower height -/
def grace_to_clyde_ratio : ℕ := 8

/-- The ratio of Sarah's tower height to Clyde's tower height -/
def sarah_to_clyde_ratio : ℕ := 2

/-- Theorem stating the combined height of all three towers -/
theorem combined_tower_height : 
  grace_height + (grace_height / grace_to_clyde_ratio) * (1 + sarah_to_clyde_ratio) = 55 := by
  sorry

end combined_tower_height_l840_84027


namespace john_remaining_cards_l840_84022

def cards_per_deck : ℕ := 52
def half_full_decks : ℕ := 3
def full_decks : ℕ := 3
def discarded_cards : ℕ := 34

theorem john_remaining_cards : 
  cards_per_deck * full_decks + (cards_per_deck / 2) * half_full_decks - discarded_cards = 200 := by
  sorry

end john_remaining_cards_l840_84022


namespace yogurt_production_cost_l840_84034

/-- The price of fruit per kilogram that satisfies the yogurt production constraints -/
def fruit_price : ℝ := 2

/-- The cost of milk per liter -/
def milk_cost : ℝ := 1.5

/-- The number of liters of milk needed for one batch of yogurt -/
def milk_per_batch : ℝ := 10

/-- The number of kilograms of fruit needed for one batch of yogurt -/
def fruit_per_batch : ℝ := 3

/-- The cost to produce three batches of yogurt -/
def cost_three_batches : ℝ := 63

theorem yogurt_production_cost :
  fruit_price * fruit_per_batch * 3 + milk_cost * milk_per_batch * 3 = cost_three_batches :=
sorry

end yogurt_production_cost_l840_84034


namespace factorial_difference_quotient_l840_84095

theorem factorial_difference_quotient : (Nat.factorial 11 - Nat.factorial 10) / Nat.factorial 9 = 100 := by
  sorry

end factorial_difference_quotient_l840_84095


namespace point_B_in_first_quadrant_l840_84031

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def is_in_second_quadrant (p : Point2D) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Definition of the first quadrant -/
def is_in_first_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- The theorem to be proved -/
theorem point_B_in_first_quadrant (A : Point2D) (hA : is_in_second_quadrant A) :
  let B : Point2D := ⟨-2 * A.x, (1/3) * A.y⟩
  is_in_first_quadrant B := by
  sorry

end point_B_in_first_quadrant_l840_84031


namespace similar_triangles_sequence_l840_84081

/-- Given a sequence of six similar right triangles with vertex A, where AB = 24 and AC = 54,
    prove that the length of AD (hypotenuse of the third triangle) is 36. -/
theorem similar_triangles_sequence (a b x c d : ℝ) : 
  (24 : ℝ) / a = a / b ∧ 
  a / b = b / x ∧ 
  b / x = x / c ∧ 
  x / c = c / d ∧ 
  c / d = d / 54 → 
  x = 36 := by sorry

end similar_triangles_sequence_l840_84081


namespace manuscript_solution_l840_84060

/-- Represents the problem of determining the number of pages in a manuscript. -/
def ManuscriptProblem (copies : ℕ) (printCost : ℚ) (bindCost : ℚ) (totalCost : ℚ) : Prop :=
  ∃ (pages : ℕ),
    (copies : ℚ) * printCost * (pages : ℚ) + (copies : ℚ) * bindCost = totalCost ∧
    pages = 400

/-- The solution to the manuscript problem. -/
theorem manuscript_solution :
  ManuscriptProblem 10 (5/100) 5 250 := by
  sorry

#check manuscript_solution

end manuscript_solution_l840_84060


namespace unique_c_value_l840_84011

/-- A polynomial has exactly one real root if and only if its discriminant is zero -/
def has_one_real_root (b c : ℝ) : Prop :=
  b ^ 2 = 4 * c

/-- The product of all possible values of c satisfying the conditions -/
def product_of_c_values (b c : ℝ) : ℝ :=
  -- This is a placeholder; the actual computation would be more complex
  1

theorem unique_c_value (b c : ℝ) 
  (h1 : has_one_real_root b c)
  (h2 : b = c^2 + 1) :
  product_of_c_values b c = 1 := by
  sorry

#check unique_c_value

end unique_c_value_l840_84011


namespace unique_arrangement_l840_84085

/-- Represents the three types of people in the problem -/
inductive PersonType
  | TruthTeller
  | Liar
  | Diplomat

/-- Represents the three positions -/
inductive Position
  | Left
  | Middle
  | Right

/-- A person's statement about another person's type -/
structure Statement where
  speaker : Position
  subject : Position
  claimedType : PersonType

/-- The arrangement of people -/
structure Arrangement where
  left : PersonType
  middle : PersonType
  right : PersonType

def isConsistent (arr : Arrangement) (statements : List Statement) : Prop :=
  ∀ s ∈ statements,
    (s.speaker = Position.Left ∧ arr.left = PersonType.TruthTeller) ∨
    (s.speaker = Position.Left ∧ arr.left = PersonType.Diplomat) ∨
    (s.speaker = Position.Middle ∧ arr.middle = PersonType.Liar) ∨
    (s.speaker = Position.Right ∧ arr.right = PersonType.TruthTeller) →
      ((s.subject = Position.Middle ∧ s.claimedType = arr.middle) ∨
       (s.subject = Position.Right ∧ s.claimedType = arr.right))

def problemStatements : List Statement :=
  [ ⟨Position.Left, Position.Middle, PersonType.TruthTeller⟩,
    ⟨Position.Middle, Position.Middle, PersonType.Diplomat⟩,
    ⟨Position.Right, Position.Middle, PersonType.Liar⟩ ]

theorem unique_arrangement :
  ∃! arr : Arrangement,
    arr.left = PersonType.Diplomat ∧
    arr.middle = PersonType.Liar ∧
    arr.right = PersonType.TruthTeller ∧
    isConsistent arr problemStatements :=
  sorry

end unique_arrangement_l840_84085


namespace equation_solutions_l840_84046

theorem equation_solutions : 
  (∃ (S₁ : Set ℝ), S₁ = {x : ℝ | x * (x + 2) = 2 * x + 4} ∧ S₁ = {-2, 2}) ∧
  (∃ (S₂ : Set ℝ), S₂ = {x : ℝ | 3 * x^2 - x - 2 = 0} ∧ S₂ = {1, -2/3}) :=
by sorry

end equation_solutions_l840_84046


namespace parabola_properties_l840_84071

theorem parabola_properties (a b c m : ℝ) (ha : a ≠ 0) (hm : -2 < m ∧ m < -1)
  (h_downward : a < 0)
  (h_root1 : a * 1^2 + b * 1 + c = 0)
  (h_root2 : a * m^2 + b * m + c = 0) :
  abc > 0 ∧ a - b + c > 0 ∧ a * (m + 1) - b + c > 0 := by
  sorry

#check parabola_properties

end parabola_properties_l840_84071


namespace triangle_side_a_value_l840_84091

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
theorem triangle_side_a_value (A B C : Real) (a b c : Real) : 
  -- Given conditions
  (Real.tan A = 2 * Real.tan B) →
  (b = Real.sqrt 2) →
  -- Assuming the area is at its maximum (we can't directly express this in Lean without additional setup)
  -- Conclusion
  (a = Real.sqrt 5) :=
by
  sorry

end triangle_side_a_value_l840_84091


namespace red_blood_cell_surface_area_calculation_l840_84072

/-- The sum of the surface areas of all red blood cells in a normal adult body. -/
def red_blood_cell_surface_area (body_surface_area : ℝ) : ℝ :=
  2000 * body_surface_area

/-- Theorem: The sum of the surface areas of all red blood cells in an adult body
    with a body surface area of 1800 cm² is 3.6 × 10⁶ cm². -/
theorem red_blood_cell_surface_area_calculation :
  red_blood_cell_surface_area 1800 = 3.6 * (10 ^ 6) := by
  sorry

end red_blood_cell_surface_area_calculation_l840_84072


namespace hexagon_extension_l840_84005

/-- Regular hexagon ABCDEF with side length 3 -/
structure RegularHexagon :=
  (A B C D E F : ℝ × ℝ)
  (side_length : ℝ)
  (is_regular : side_length = 3)

/-- Point Y is on the extension of AB such that AY = 2AB -/
def extend_side (h : RegularHexagon) (Y : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ Y = h.A + t • (h.B - h.A) ∧ dist h.A Y = 2 * h.side_length

/-- The length of FY -/
def FY_length (h : RegularHexagon) (Y : ℝ × ℝ) : ℝ :=
  dist h.F Y

theorem hexagon_extension (h : RegularHexagon) (Y : ℝ × ℝ) 
  (ext : extend_side h Y) : FY_length h Y = 15 * Real.sqrt 3 / 4 := by
  sorry

end hexagon_extension_l840_84005


namespace quadratic_roots_to_coefficients_l840_84044

theorem quadratic_roots_to_coefficients 
  (a b p q : ℝ) 
  (h1 : Complex.I ^ 2 = -1) 
  (h2 : (2 + a * Complex.I) ^ 2 + p * (2 + a * Complex.I) + q = 0) 
  (h3 : (b + Complex.I) ^ 2 + p * (b + Complex.I) + q = 0) : 
  p = -4 ∧ q = 5 := by
  sorry

end quadratic_roots_to_coefficients_l840_84044


namespace largest_power_dividing_product_l840_84067

def pow (n : ℕ) : ℕ :=
  sorry

def product_pow : ℕ :=
  sorry

theorem largest_power_dividing_product :
  (∃ m : ℕ, (2310 : ℕ)^m ∣ product_pow ∧ 
    ∀ k : ℕ, (2310 : ℕ)^k ∣ product_pow → k ≤ m) ∧
  (∃ m : ℕ, (2310 : ℕ)^m ∣ product_pow ∧ m = 319) :=
by sorry

end largest_power_dividing_product_l840_84067


namespace angle4_value_l840_84019

-- Define the angles
def angle1 : ℝ := 50
def angle2 : ℝ := 110
def angle3 : ℝ := 35
def angle4 : ℝ := 35
def angle5 : ℝ := 60
def angle6 : ℝ := 70

-- State the theorem
theorem angle4_value :
  angle1 + angle2 = 180 ∧
  angle3 = angle4 ∧
  angle1 = 50 ∧
  angle5 = 60 ∧
  angle1 + angle5 + angle6 = 180 ∧
  angle2 + angle6 = 180 ∧
  angle3 + angle4 = 180 - angle2 →
  angle4 = 35 := by sorry

end angle4_value_l840_84019


namespace triathlon_bicycle_speed_triathlon_solution_l840_84065

theorem triathlon_bicycle_speed 
  (total_time : ℝ) 
  (swim_speed swim_distance : ℝ) 
  (run_speed run_distance : ℝ) 
  (bike_distance : ℝ) : ℝ :=
  let swim_time := swim_distance / swim_speed
  let run_time := run_distance / run_speed
  let remaining_time := total_time - (swim_time + run_time)
  bike_distance / remaining_time

theorem triathlon_solution :
  triathlon_bicycle_speed 3 1 0.5 8 4 20 = 10 := by
  sorry

end triathlon_bicycle_speed_triathlon_solution_l840_84065


namespace same_color_probability_l840_84037

-- Define the number of red and blue plates
def red_plates : ℕ := 5
def blue_plates : ℕ := 4

-- Define the total number of plates
def total_plates : ℕ := red_plates + blue_plates

-- Define the function to calculate combinations
def choose (n k : ℕ) : ℕ := (n.factorial) / (k.factorial * (n - k).factorial)

-- Theorem statement
theorem same_color_probability :
  (choose red_plates 2 + choose blue_plates 2) / choose total_plates 2 = 4 / 9 := by
  sorry

end same_color_probability_l840_84037


namespace evaluate_expression_l840_84061

theorem evaluate_expression : (2^2010 * 3^2012) / 6^2011 = 3/2 := by
  sorry

end evaluate_expression_l840_84061


namespace optimal_pool_dimensions_l840_84090

/-- Represents the dimensions and cost of a rectangular pool -/
structure Pool :=
  (length : ℝ)
  (width : ℝ)
  (depth : ℝ)
  (bottomCost : ℝ)
  (wallCost : ℝ)

/-- Calculates the total cost of the pool -/
def totalCost (p : Pool) : ℝ :=
  p.bottomCost * p.length * p.width + p.wallCost * 2 * (p.length + p.width) * p.depth

/-- Theorem stating the optimal dimensions and minimum cost of the pool -/
theorem optimal_pool_dimensions :
  ∀ p : Pool,
  p.depth = 2 ∧
  p.length * p.width * p.depth = 18 ∧
  p.bottomCost = 200 ∧
  p.wallCost = 150 →
  ∃ (minCost : ℝ),
    minCost = 7200 ∧
    totalCost p ≥ minCost ∧
    (totalCost p = minCost ↔ p.length = 3 ∧ p.width = 3) :=
by
  sorry

end optimal_pool_dimensions_l840_84090


namespace parallelogram_base_l840_84041

theorem parallelogram_base (area height : ℝ) (h1 : area = 704) (h2 : height = 22) :
  area / height = 32 :=
by sorry

end parallelogram_base_l840_84041


namespace perpendicular_line_through_point_l840_84068

/-- Given a line L1 with equation 3x - 6y = 9 and a point P (2, -3),
    prove that the line L2 with equation y = -2x + 1 is perpendicular to L1 and passes through P. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y => 3 * x - 6 * y = 9
  let L2 : ℝ → ℝ → Prop := λ x y => y = -2 * x + 1
  let P : ℝ × ℝ := (2, -3)
  (∀ x y, L1 x y ↔ y = (1/2) * x - 3/2) →  -- L1 in slope-intercept form
  (L2 P.1 P.2) →  -- L2 passes through P
  (∀ m1 m2, m1 = 1/2 ∧ m2 = -2 → m1 * m2 = -1) →  -- Perpendicular slopes multiply to -1
  ∃ m b, ∀ x y, L2 x y ↔ y = m * x + b ∧ m * (1/2) = -1  -- L2 is perpendicular to L1
  := by sorry

end perpendicular_line_through_point_l840_84068


namespace max_elevation_is_550_l840_84004

/-- The elevation function for a vertically projected particle -/
def elevation (t : ℝ) : ℝ := 200 * t - 20 * t^2 + 50

/-- The time at which the maximum elevation occurs -/
def max_time : ℝ := 5

theorem max_elevation_is_550 :
  ∃ (t : ℝ), ∀ (t' : ℝ), elevation t ≥ elevation t' ∧ elevation t = 550 :=
sorry

end max_elevation_is_550_l840_84004


namespace clares_money_l840_84054

/-- The amount of money Clare's mother gave her --/
def money_from_mother : ℕ := sorry

/-- The number of loaves of bread Clare bought --/
def bread_count : ℕ := 4

/-- The number of cartons of milk Clare bought --/
def milk_count : ℕ := 2

/-- The cost of one loaf of bread in dollars --/
def bread_cost : ℕ := 2

/-- The cost of one carton of milk in dollars --/
def milk_cost : ℕ := 2

/-- The amount of money Clare has left after shopping --/
def money_left : ℕ := 35

/-- Theorem stating that the amount of money Clare's mother gave her is $47 --/
theorem clares_money : money_from_mother = 47 := by
  sorry

end clares_money_l840_84054


namespace prism_volume_l840_84050

/-- The volume of a right rectangular prism with given face areas -/
theorem prism_volume (a b c : ℝ) 
  (h₁ : a * b = 18) 
  (h₂ : b * c = 12) 
  (h₃ : a * c = 8) : 
  a * b * c = 24 * Real.sqrt 3 := by
  sorry

end prism_volume_l840_84050


namespace max_sum_of_digits_is_24_l840_84018

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Calculates the sum of digits for a Time24 -/
def sumOfDigitsTime24 (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The maximum sum of digits in a 24-hour format digital watch display -/
def maxSumOfDigits : Nat := 24

theorem max_sum_of_digits_is_24 :
  ∀ t : Time24, sumOfDigitsTime24 t ≤ maxSumOfDigits :=
by sorry

end max_sum_of_digits_is_24_l840_84018


namespace parabola_p_value_l840_84077

/-- Given a parabola y^2 = 2px with directrix x = -2, prove that p = 4 -/
theorem parabola_p_value (y x p : ℝ) : 
  (y^2 = 2*p*x) → -- Parabola equation
  (-p/2 = -2) →   -- Directrix equation (transformed)
  p = 4 := by
sorry

end parabola_p_value_l840_84077


namespace second_derivative_zero_not_implies_extreme_point_l840_84032

open Real

-- Define the function f(x) = x^3
def f (x : ℝ) := x^3

-- Define what it means for a point to be an extreme point
def is_extreme_point (f : ℝ → ℝ) (x₀ : ℝ) :=
  ∀ x, |x - x₀| < 1 → f x ≤ f x₀ ∨ f x ≥ f x₀

-- State the theorem
theorem second_derivative_zero_not_implies_extreme_point :
  ∃ x₀ : ℝ, (deriv (deriv f)) x₀ = 0 ∧ ¬(is_extreme_point f x₀) := by
  sorry


end second_derivative_zero_not_implies_extreme_point_l840_84032


namespace integer_solution_l840_84016

theorem integer_solution (x : ℤ) : 
  x + 15 ≥ 16 ∧ -3*x ≥ -15 → x ∈ ({1, 2, 3, 4, 5} : Set ℤ) := by
sorry

end integer_solution_l840_84016


namespace stock_percentage_l840_84087

/-- Calculate the percentage of a stock given income, stock price, and total investment. -/
theorem stock_percentage (income : ℚ) (stock_price : ℚ) (total_investment : ℚ) :
  income = 450 →
  stock_price = 108 →
  total_investment = 4860 →
  (income / total_investment) * 100 = (450 : ℚ) / 4860 * 100 := by
  sorry

end stock_percentage_l840_84087


namespace z_axis_symmetry_of_M_l840_84063

/-- A point in 3D Cartesian space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The z-axis symmetry operation on a 3D point -/
def zAxisSymmetry (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

/-- The original point M -/
def M : Point3D :=
  { x := 3, y := -4, z := 5 }

/-- The expected symmetric point -/
def SymmetricPoint : Point3D :=
  { x := -3, y := 4, z := 5 }

theorem z_axis_symmetry_of_M :
  zAxisSymmetry M = SymmetricPoint := by sorry

end z_axis_symmetry_of_M_l840_84063


namespace laura_weekly_mileage_l840_84008

-- Define the distances
def school_round_trip : ℕ := 20
def supermarket_extra_distance : ℕ := 10

-- Define the number of trips
def school_trips_per_week : ℕ := 5
def supermarket_trips_per_week : ℕ := 2

-- Calculate the total weekly mileage
def total_weekly_mileage : ℕ :=
  (school_round_trip * school_trips_per_week) +
  ((school_round_trip / 2 + supermarket_extra_distance) * 2 * supermarket_trips_per_week)

-- Theorem to prove
theorem laura_weekly_mileage :
  total_weekly_mileage = 180 := by
  sorry

end laura_weekly_mileage_l840_84008


namespace unique_solution_symmetric_difference_l840_84066

variable {U : Type*} -- Universe set

def symmetric_difference (A B : Set U) : Set U := (A \ B) ∪ (B \ A)

theorem unique_solution_symmetric_difference
  (A B X : Set U)
  (h1 : X ∩ (A ∪ B) = X)
  (h2 : A ∩ (B ∪ X) = A)
  (h3 : B ∩ (A ∪ X) = B)
  (h4 : X ∩ A ∩ B = ∅) :
  X = symmetric_difference A B ∧ 
  (∀ Y : Set U, Y ∩ (A ∪ B) = Y → A ∩ (B ∪ Y) = A → B ∩ (A ∪ Y) = B → Y ∩ A ∩ B = ∅ → Y = X) :=
by sorry

end unique_solution_symmetric_difference_l840_84066


namespace sqrt_product_simplification_l840_84052

theorem sqrt_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (50 * q) * Real.sqrt (10 * q) * Real.sqrt (15 * q) = 10 * q * Real.sqrt (15 * q) := by
  sorry

end sqrt_product_simplification_l840_84052


namespace parabola_properties_l840_84028

-- Define the parabola
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 2 * a * x + 3

-- Theorem statement
theorem parabola_properties (a : ℝ) (h : a ≠ 0) :
  -- 1. Axis of symmetry
  (∀ x : ℝ, parabola a x = parabola a (2 - x)) ∧
  -- 2. Vertex on x-axis after shifting
  ((∃ x : ℝ, parabola a x - 3 * |a| = 0 ∧
             ∀ y : ℝ, parabola a y - 3 * |a| ≥ 0) ↔ (a = 3/4 ∨ a = -3/2)) ∧
  -- 3. Range of a for given points
  (∀ y₁ y₂ : ℝ, y₁ > y₂ → parabola a a = y₁ → parabola a 2 = y₂ → a > 2) :=
by sorry

end parabola_properties_l840_84028


namespace spelling_bee_contestants_l840_84097

theorem spelling_bee_contestants (initial_students : ℕ) : 
  (initial_students : ℚ) * (1 - 0.6) * (1 / 2) * (1 / 4) = 15 → 
  initial_students = 300 := by
sorry

end spelling_bee_contestants_l840_84097


namespace residue_negative_1234_mod_32_l840_84099

theorem residue_negative_1234_mod_32 : Int.mod (-1234) 32 = 14 := by
  sorry

end residue_negative_1234_mod_32_l840_84099


namespace quadratic_roots_differ_by_two_l840_84039

/-- For a quadratic equation ax^2 + bx + c = 0 where a ≠ 0, 
    if the roots of the equation differ by 2, then c = (b^2 / (4a)) - a -/
theorem quadratic_roots_differ_by_two (a b c : ℝ) (ha : a ≠ 0) :
  (∃ x y : ℝ, x - y = 2 ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) →
  c = (b^2 / (4 * a)) - a := by
sorry

end quadratic_roots_differ_by_two_l840_84039


namespace gcd_of_360_and_504_l840_84017

theorem gcd_of_360_and_504 : Nat.gcd 360 504 = 72 := by sorry

end gcd_of_360_and_504_l840_84017


namespace triangle_base_length_l840_84096

theorem triangle_base_length 
  (area : ℝ) 
  (height : ℝ) 
  (h1 : area = 25) 
  (h2 : height = 5) : 
  area = (base * height) / 2 → base = 10 := by
  sorry

end triangle_base_length_l840_84096


namespace irrational_sqrt_3_rational_others_l840_84073

-- Define rationality
def IsRational (x : ℝ) : Prop := ∃ (p q : ℤ), q ≠ 0 ∧ x = p / q

-- Define irrationality
def IsIrrational (x : ℝ) : Prop := ¬ IsRational x

-- Theorem statement
theorem irrational_sqrt_3_rational_others :
  IsIrrational (Real.sqrt 3) ∧
  IsRational 0 ∧
  IsRational (-2) ∧
  IsRational (1/2) := by
  sorry

end irrational_sqrt_3_rational_others_l840_84073


namespace min_slope_is_three_l840_84089

-- Define the function
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 10

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem stating that the minimum slope of tangents is 3
theorem min_slope_is_three :
  ∃ (x : ℝ), ∀ (y : ℝ), f' x ≤ f' y ∧ f' x = 3 :=
sorry

end min_slope_is_three_l840_84089


namespace complement_intersection_theorem_l840_84053

def U : Set ℤ := {x : ℤ | x^2 - 5*x - 6 < 0}

def A : Set ℤ := {x : ℤ | -1 < x ∧ x ≤ 2}

def B : Set ℤ := {2, 3, 5}

theorem complement_intersection_theorem :
  (U \ A) ∩ B = {3, 5} := by sorry

end complement_intersection_theorem_l840_84053


namespace unique_k_solution_l840_84013

def f (n : ℤ) : ℤ := 
  if n % 2 = 1 then n + 3 else n / 2

theorem unique_k_solution : 
  ∃! k : ℤ, k % 2 = 1 ∧ f (f (f k)) = 27 ∧ k = 105 := by
  sorry

end unique_k_solution_l840_84013


namespace second_store_earns_more_l840_84075

/-- Represents the total value of goods sold by each department store -/
def total_goods_value : ℕ := 1000000

/-- Represents the discount rate offered by the first department store -/
def discount_rate : ℚ := 1/10

/-- Represents the number of lottery tickets given per 100 yuan spent -/
def tickets_per_hundred : ℕ := 1

/-- Represents the total number of lottery tickets -/
def total_tickets : ℕ := 10000

/-- Represents the number of first prizes -/
def first_prize_count : ℕ := 5

/-- Represents the value of each first prize -/
def first_prize_value : ℕ := 1000

/-- Represents the number of second prizes -/
def second_prize_count : ℕ := 10

/-- Represents the value of each second prize -/
def second_prize_value : ℕ := 500

/-- Represents the number of third prizes -/
def third_prize_count : ℕ := 20

/-- Represents the value of each third prize -/
def third_prize_value : ℕ := 200

/-- Represents the number of fourth prizes -/
def fourth_prize_count : ℕ := 40

/-- Represents the value of each fourth prize -/
def fourth_prize_value : ℕ := 100

/-- Represents the number of fifth prizes -/
def fifth_prize_count : ℕ := 1000

/-- Represents the value of each fifth prize -/
def fifth_prize_value : ℕ := 10

/-- Calculates the earnings of the first department store -/
def first_store_earnings : ℚ := total_goods_value * (1 - discount_rate)

/-- Calculates the total prize value for the second department store -/
def total_prize_value : ℕ := 
  first_prize_count * first_prize_value +
  second_prize_count * second_prize_value +
  third_prize_count * third_prize_value +
  fourth_prize_count * fourth_prize_value +
  fifth_prize_count * fifth_prize_value

/-- Calculates the earnings of the second department store -/
def second_store_earnings : ℕ := total_goods_value - total_prize_value

/-- Theorem stating that the second department store earns at least 72,000 yuan more than the first -/
theorem second_store_earns_more :
  (second_store_earnings : ℚ) - first_store_earnings ≥ 72000 := by
  sorry

end second_store_earns_more_l840_84075


namespace largest_divisor_of_five_consecutive_integers_l840_84006

theorem largest_divisor_of_five_consecutive_integers :
  ∀ n : ℤ, ∃ m : ℤ, m ≥ 120 ∧ (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) →
  120 ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4)) :=
by sorry

end largest_divisor_of_five_consecutive_integers_l840_84006


namespace blast_distance_problem_l840_84094

/-- The distance traveled by sound in a given time -/
def sound_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- The problem statement -/
theorem blast_distance_problem (man_speed : ℝ) (sound_speed : ℝ) (total_time : ℝ) (blast_interval : ℝ) :
  sound_speed = 330 →
  total_time = 30 * 60 + 12 →
  blast_interval = 30 * 60 →
  sound_distance sound_speed (total_time - blast_interval) = 3960 := by
  sorry

end blast_distance_problem_l840_84094


namespace inequality_solution_set_l840_84033

-- Define the solution set
def solution_set : Set ℝ := {x | -1 < x ∧ x ≤ 2}

-- Define the inequality
def inequality (x : ℝ) : Prop := (2 - x) / (x + 1) ≥ 0

-- Theorem statement
theorem inequality_solution_set :
  ∀ x : ℝ, x ∈ solution_set ↔ inequality x :=
sorry

end inequality_solution_set_l840_84033


namespace square_division_rectangle_perimeter_l840_84023

/-- Given a square with perimeter 120 units divided into four congruent rectangles,
    the perimeter of one of these rectangles is 90 units. -/
theorem square_division_rectangle_perimeter :
  ∀ (s : ℝ),
  s > 0 →
  4 * s = 120 →
  2 * (s + s / 2) = 90 :=
by
  sorry

end square_division_rectangle_perimeter_l840_84023


namespace complex_magnitude_calculation_l840_84059

theorem complex_magnitude_calculation : 
  Complex.abs (6 - 3 * Complex.I) * Complex.abs (6 + 3 * Complex.I) - 2 * Complex.abs (5 - Complex.I) = 45 - 2 * Real.sqrt 26 := by
  sorry

end complex_magnitude_calculation_l840_84059


namespace intersection_A_B_l840_84049

def A : Set ℕ := {1, 3, 5, 7, 9}
def B : Set ℕ := {x | 2 ≤ x ∧ x ≤ 5}

theorem intersection_A_B : A ∩ B = {3, 5} := by sorry

end intersection_A_B_l840_84049


namespace quadratic_has_real_roots_rhombus_area_when_m_neg_seven_l840_84043

-- Define the quadratic equation
def quadratic_equation (x m : ℝ) : ℝ := 2 * x^2 + (m - 2) * x - m

-- Theorem 1: The equation always has real roots
theorem quadratic_has_real_roots (m : ℝ) :
  ∃ x : ℝ, quadratic_equation x m = 0 :=
sorry

-- Theorem 2: Area of rhombus when m = -7
theorem rhombus_area_when_m_neg_seven :
  let m : ℝ := -7
  let root1 : ℝ := (9 + Real.sqrt 25) / 4
  let root2 : ℝ := (9 - Real.sqrt 25) / 4
  quadratic_equation root1 m = 0 ∧
  quadratic_equation root2 m = 0 →
  (1 / 2) * root1 * root2 = 7 / 4 :=
sorry

end quadratic_has_real_roots_rhombus_area_when_m_neg_seven_l840_84043


namespace range_of_a_l840_84064

-- Define the quadratic function
def f (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * (a - 1) * x - 4

-- Define the solution set of the inequality
def solution_set (a : ℝ) : Set ℝ := {x | f a x ≥ 0}

-- State the theorem
theorem range_of_a : 
  (∀ a : ℝ, solution_set a = ∅) ↔ (∀ a : ℝ, -3 < a ∧ a ≤ 1) :=
sorry

end range_of_a_l840_84064


namespace monogram_count_l840_84015

def alphabet_size : ℕ := 26

theorem monogram_count : (alphabet_size.choose 2) = 325 := by sorry

end monogram_count_l840_84015


namespace least_common_period_l840_84001

-- Define the property for function f
def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 5) + f (x - 5) = f x

-- Define the concept of a period for a function
def is_period (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x : ℝ, f (x + p) = f x

-- State the theorem
theorem least_common_period :
  ∃ p : ℝ, p > 0 ∧
    (∀ f : ℝ → ℝ, satisfies_condition f → is_period f p) ∧
    (∀ q : ℝ, q > 0 → (∀ f : ℝ → ℝ, satisfies_condition f → is_period f q) → p ≤ q) ∧
    p = 30 :=
  sorry

end least_common_period_l840_84001


namespace cost_of_bread_and_drinks_l840_84080

/-- The cost of buying bread and drinks -/
theorem cost_of_bread_and_drinks 
  (a b : ℝ) 
  (h1 : a ≥ 0) 
  (h2 : b ≥ 0) : 
  a + 2 * b = (1 : ℝ) * a + (2 : ℝ) * b := by sorry

end cost_of_bread_and_drinks_l840_84080


namespace inscribed_rectangle_circle_circumference_l840_84012

theorem inscribed_rectangle_circle_circumference :
  ∀ (rectangle_width rectangle_height circle_circumference : ℝ),
  rectangle_width = 9 →
  rectangle_height = 12 →
  circle_circumference = π * (rectangle_width^2 + rectangle_height^2).sqrt →
  circle_circumference = 15 * π :=
by sorry

end inscribed_rectangle_circle_circumference_l840_84012


namespace collinear_vectors_y_value_l840_84030

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, b = (k * a.1, k * a.2)

/-- Given vectors a and b, prove that if they are collinear, then y = -2 -/
theorem collinear_vectors_y_value :
  let a : ℝ × ℝ := (-3, 1)
  let b : ℝ × ℝ := (6, y)
  collinear a b → y = -2 := by
  sorry

end collinear_vectors_y_value_l840_84030


namespace expression_simplification_l840_84076

theorem expression_simplification (x : ℝ) (hx : x^2 - 2*x = 0) (hx_nonzero : x ≠ 0) :
  (1 + 1 / (x - 1)) / (x / (x^2 - 1)) = 3 := by
  sorry

end expression_simplification_l840_84076


namespace isosceles_triangle_base_angle_l840_84045

theorem isosceles_triangle_base_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  (α = β ∨ α = γ ∨ β = γ) →  -- The triangle is isosceles
  (α = 110 ∨ β = 110 ∨ γ = 110) →  -- One of the angles is 110°
  (α = 35 ∨ β = 35 ∨ γ = 35) :=  -- One of the base angles is 35°
by sorry

end isosceles_triangle_base_angle_l840_84045


namespace min_sum_of_reciprocals_l840_84009

theorem min_sum_of_reciprocals (a b : ℝ) : 
  a > 0 → b > 0 → (2 / a + 2 / b = 1) → a + b ≥ 8 := by sorry

end min_sum_of_reciprocals_l840_84009


namespace school_store_sale_l840_84024

/-- The number of pencils sold in a school store sale -/
def pencils_sold (first_two : ℕ) (next_six : ℕ) (last_two : ℕ) : ℕ :=
  2 * first_two + 6 * next_six + 2 * last_two

/-- Theorem: Given the conditions of the pencil sale, 24 pencils were sold -/
theorem school_store_sale : pencils_sold 2 3 1 = 24 := by
  sorry

end school_store_sale_l840_84024


namespace triangle_double_angle_sine_sum_l840_84000

/-- For angles α, β, and γ of a triangle, sin 2α + sin 2β + sin 2γ = 4 sin α sin β sin γ -/
theorem triangle_double_angle_sine_sum (α β γ : ℝ) 
  (h : α + β + γ = Real.pi) : 
  Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ) = 
  4 * Real.sin α * Real.sin β * Real.sin γ := by
  sorry

end triangle_double_angle_sine_sum_l840_84000


namespace difference_of_squares_l840_84062

theorem difference_of_squares (a b : ℝ) :
  ∃ (p q : ℝ), (a - 2*b) * (a + 2*b) = (p + q) * (p - q) ∧
                (-a + b) * (-a - b) = (p + q) * (p - q) ∧
                (-a - 1) * (1 - a) = (p + q) * (p - q) ∧
                ¬(∃ (r s : ℝ), (-x + y) * (x - y) = (r + s) * (r - s)) :=
by sorry


end difference_of_squares_l840_84062


namespace range_of_a_given_proposition_l840_84036

theorem range_of_a_given_proposition (a : ℝ) : 
  (∃ x ∈ Set.Icc 1 2, x^2 + 2*x + a ≥ 0) → a ≥ -8 :=
by sorry

end range_of_a_given_proposition_l840_84036


namespace gcd_problem_l840_84098

theorem gcd_problem (x : ℤ) (h : ∃ k : ℤ, x = 17248 * k) :
  Int.gcd ((5*x+4)*(8*x+1)*(11*x+6)*(3*x+9)) x = 24 := by
  sorry

end gcd_problem_l840_84098


namespace sum_abs_coefficients_f6_l840_84051

def polynomial_sequence : ℕ → (ℝ → ℝ) 
  | 0 => λ x => 1
  | n + 1 => λ x => (x^2 - 1) * (polynomial_sequence n x) - 2*x

def sum_abs_coefficients (f : ℝ → ℝ) : ℝ := sorry

theorem sum_abs_coefficients_f6 : 
  sum_abs_coefficients (polynomial_sequence 6) = 190 := by sorry

end sum_abs_coefficients_f6_l840_84051


namespace one_in_M_l840_84010

def M : Set ℕ := {1, 2, 3}

theorem one_in_M : 1 ∈ M := by
  sorry

end one_in_M_l840_84010


namespace correct_count_of_students_using_both_colors_l840_84057

/-- The number of students using both green and red colors in a painting activity. -/
def students_using_both_colors (total_students green_users red_users : ℕ) : ℕ :=
  green_users + red_users - total_students

/-- Theorem stating that the number of students using both colors is correct. -/
theorem correct_count_of_students_using_both_colors
  (total_students green_users red_users : ℕ)
  (h1 : total_students = 70)
  (h2 : green_users = 52)
  (h3 : red_users = 56) :
  students_using_both_colors total_students green_users red_users = 38 := by
  sorry

end correct_count_of_students_using_both_colors_l840_84057


namespace angle_ratio_3_4_5_not_right_triangle_l840_84086

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (sum_angles : A + B + C = Real.pi)
  (side_angle_correspondence : True)  -- This is a placeholder for the side-angle correspondence

/-- A right triangle is a triangle with one right angle (π/2) -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

/-- The condition that angle ratios are 3:4:5 -/
def angle_ratio_3_4_5 (t : Triangle) : Prop :=
  ∃ (k : ℝ), k > 0 ∧ t.A = 3 * k ∧ t.B = 4 * k ∧ t.C = 5 * k

/-- Theorem: The condition ∠A:∠B:∠C = 3:4:5 cannot determine △ABC to be a right triangle -/
theorem angle_ratio_3_4_5_not_right_triangle :
  ∃ (t : Triangle), angle_ratio_3_4_5 t ∧ ¬(is_right_triangle t) :=
sorry

end angle_ratio_3_4_5_not_right_triangle_l840_84086
