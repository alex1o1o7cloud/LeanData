import Mathlib

namespace texas_maine_plate_difference_l2948_294826

/-- The number of possible choices for a letter in a license plate. -/
def letter_choices : ℕ := 26

/-- The number of possible choices for a number in a license plate. -/
def number_choices : ℕ := 10

/-- The number of possible license plates in Texas format (LLNNNNL). -/
def texas_plates : ℕ := letter_choices^3 * number_choices^4

/-- The number of possible license plates in Maine format (LLLNNN). -/
def maine_plates : ℕ := letter_choices^3 * number_choices^3

/-- The difference in the number of possible license plates between Texas and Maine. -/
def plate_difference : ℕ := texas_plates - maine_plates

theorem texas_maine_plate_difference :
  plate_difference = 158184000 := by
  sorry

end texas_maine_plate_difference_l2948_294826


namespace inverse_exists_mod_prime_wilsons_theorem_l2948_294872

-- Define primality
def isPrime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

-- Part 1: Inverse exists for non-zero elements modulo prime
theorem inverse_exists_mod_prime (p k : ℕ) (hp : isPrime p) (hk : ¬(p ∣ k)) :
  ∃ l : ℕ, k * l ≡ 1 [ZMOD p] :=
sorry

-- Part 2: Wilson's theorem
theorem wilsons_theorem (n : ℕ) :
  isPrime n ↔ (Nat.factorial (n - 1)) ≡ -1 [ZMOD n] :=
sorry

end inverse_exists_mod_prime_wilsons_theorem_l2948_294872


namespace parabola_distance_theorem_l2948_294822

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define point B
def B : ℝ × ℝ := (3, 0)

-- Theorem statement
theorem parabola_distance_theorem 
  (A : ℝ × ℝ) -- Point A
  (h1 : parabola A.1 A.2) -- A is on the parabola
  (h2 : ‖A - focus‖ = ‖B - focus‖) -- |AF| = |BF|
  : ‖A - B‖ = 2 * Real.sqrt 2 := by sorry

end parabola_distance_theorem_l2948_294822


namespace arthurs_walk_l2948_294801

/-- Arthur's walk problem -/
theorem arthurs_walk (blocks_west blocks_south : ℕ) (block_length : ℚ) :
  blocks_west = 8 →
  blocks_south = 10 →
  block_length = 1/4 →
  (blocks_west + blocks_south : ℚ) * block_length = 4.5 := by
  sorry

end arthurs_walk_l2948_294801


namespace points_collinear_iff_k_eq_one_l2948_294883

-- Define the vectors
def OA : ℝ × ℝ := (1, -3)
def OB : ℝ × ℝ := (2, -1)
def OC (k : ℝ) : ℝ × ℝ := (k + 1, k - 2)

-- Define collinearity condition
def areCollinear (A B C : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, C.1 - A.1 = t * (B.1 - A.1) ∧ C.2 - A.2 = t * (B.2 - A.2)

-- Theorem statement
theorem points_collinear_iff_k_eq_one :
  ∀ k : ℝ, areCollinear OA OB (OC k) ↔ k = 1 := by sorry

end points_collinear_iff_k_eq_one_l2948_294883


namespace monica_subjects_l2948_294843

theorem monica_subjects (monica marius millie : ℕ) 
  (h1 : millie = marius + 3)
  (h2 : marius = monica + 4)
  (h3 : monica + marius + millie = 41) : 
  monica = 10 :=
sorry

end monica_subjects_l2948_294843


namespace tan_seven_pi_fourth_l2948_294846

theorem tan_seven_pi_fourth : Real.tan (7 * π / 4) = -1 := by
  sorry

end tan_seven_pi_fourth_l2948_294846


namespace brandon_skittles_proof_l2948_294850

def brandon_initial_skittles (skittles_lost : ℕ) (final_skittles : ℕ) : ℕ :=
  final_skittles + skittles_lost

theorem brandon_skittles_proof :
  brandon_initial_skittles 9 87 = 96 :=
by sorry

end brandon_skittles_proof_l2948_294850


namespace triangle_side_decomposition_l2948_294810

/-- Given a triangle with side lengths a, b, and c, there exist positive numbers x, y, and z
    such that a = y + z, b = x + z, and c = x + y -/
theorem triangle_side_decomposition (a b c : ℝ) (h_triangle : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧ a = y + z ∧ b = x + z ∧ c = x + y :=
sorry

end triangle_side_decomposition_l2948_294810


namespace workshop_novelists_l2948_294879

theorem workshop_novelists (total : ℕ) (ratio_novelists : ℕ) (ratio_poets : ℕ) 
  (h1 : total = 24)
  (h2 : ratio_novelists = 5)
  (h3 : ratio_poets = 3) :
  (total * ratio_novelists) / (ratio_novelists + ratio_poets) = 15 := by
  sorry

end workshop_novelists_l2948_294879


namespace sunflower_majority_on_day_two_l2948_294863

/-- Represents the proportion of sunflower seeds in the feeder on a given day -/
def sunflower_proportion (day : ℕ) : ℝ :=
  1 - (0.7 : ℝ) ^ day

/-- The day when more than half of the seeds are sunflower seeds -/
def target_day : ℕ := 2

theorem sunflower_majority_on_day_two :
  sunflower_proportion target_day > (1/2 : ℝ) :=
by sorry

#check sunflower_majority_on_day_two

end sunflower_majority_on_day_two_l2948_294863


namespace children_left_on_bus_l2948_294851

theorem children_left_on_bus (initial_children : Nat) (children_off : Nat) : 
  initial_children = 43 → children_off = 22 → initial_children - children_off = 21 := by
  sorry

end children_left_on_bus_l2948_294851


namespace railroad_cars_theorem_l2948_294808

/-- Sum of binary digits of a natural number -/
def sumBinaryDigits (n : ℕ) : ℕ := sorry

/-- Minimum number of tries needed to determine if there is an equal number of both types of cars -/
def minTries (totalCars : ℕ) : ℕ := totalCars - sumBinaryDigits totalCars

theorem railroad_cars_theorem :
  let totalCars : ℕ := 2022
  minTries totalCars = 2014 := by sorry

end railroad_cars_theorem_l2948_294808


namespace aunt_angela_nieces_l2948_294827

theorem aunt_angela_nieces (total_jellybeans : ℕ) (num_nephews : ℕ) (jellybeans_per_child : ℕ) 
  (h1 : total_jellybeans = 70)
  (h2 : num_nephews = 3)
  (h3 : jellybeans_per_child = 14) :
  total_jellybeans / jellybeans_per_child - num_nephews = 2 :=
by sorry

end aunt_angela_nieces_l2948_294827


namespace linear_system_integer_solution_l2948_294841

theorem linear_system_integer_solution (a b : ℤ) :
  ∃ (x y z t : ℤ), x + y + 2*z + 2*t = a ∧ 2*x - 2*y + z - t = b := by
sorry

end linear_system_integer_solution_l2948_294841


namespace projection_of_two_vectors_l2948_294891

/-- Given two vectors that project to the same vector, find the projection --/
theorem projection_of_two_vectors (v₁ v₂ v : ℝ × ℝ) (h : v ≠ (0, 0)) :
  let p := (14/73, 214/73)
  (∃ (k₁ k₂ : ℝ), v₁ - k₁ • v = p ∧ v₂ - k₂ • v = p) →
  v₁ = (5, -2) →
  v₂ = (2, 6) →
  (∃ (k : ℝ), v₁ - k • v = p ∧ v₂ - k • v = p) :=
by sorry


end projection_of_two_vectors_l2948_294891


namespace complex_modulus_l2948_294868

theorem complex_modulus (z : ℂ) (h : z * (1 + Complex.I) = 2 * Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_modulus_l2948_294868


namespace order_of_rational_numbers_l2948_294889

theorem order_of_rational_numbers
  (a b c d : ℚ)
  (sum_eq : a + b = c + d)
  (ineq_1 : a + d < b + c)
  (ineq_2 : c < d) :
  b > d ∧ d > c ∧ c > a :=
sorry

end order_of_rational_numbers_l2948_294889


namespace mary_snake_count_l2948_294820

/-- The number of breeding balls -/
def num_breeding_balls : ℕ := 3

/-- The number of snakes in each breeding ball -/
def snakes_per_ball : ℕ := 8

/-- The number of additional pairs of snakes -/
def num_snake_pairs : ℕ := 6

/-- The total number of snakes Mary saw -/
def total_snakes : ℕ := num_breeding_balls * snakes_per_ball + 2 * num_snake_pairs

theorem mary_snake_count : total_snakes = 36 := by
  sorry

end mary_snake_count_l2948_294820


namespace ball_trajectory_l2948_294803

/-- A rectangle with side lengths 2a and 2b -/
structure Rectangle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  a : ℝ
  b : ℝ
  h : (5 : ℝ) * a = (3 : ℝ) * b

/-- The angle at which the ball is hit from corner A -/
def hitAngle (α : ℝ) := α

/-- The ball hits three different sides before reaching the center -/
def hitsThreeSides (rect : Rectangle ℝ) (α : ℝ) : Prop :=
  ∃ (p q r : ℝ × ℝ), 
    p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    (p.1 = 0 ∨ p.1 = 2*rect.a ∨ p.2 = 0 ∨ p.2 = 2*rect.b) ∧
    (q.1 = 0 ∨ q.1 = 2*rect.a ∨ q.2 = 0 ∨ q.2 = 2*rect.b) ∧
    (r.1 = 0 ∨ r.1 = 2*rect.a ∨ r.2 = 0 ∨ r.2 = 2*rect.b)

theorem ball_trajectory (rect : Rectangle ℝ) (α : ℝ) :
  hitsThreeSides rect α ↔ Real.tan α = 9/25 := by sorry

end ball_trajectory_l2948_294803


namespace max_fleas_l2948_294849

/-- Represents a flea's direction of movement --/
inductive Direction
| Up
| Down
| Left
| Right

/-- Represents a position on the board --/
structure Position :=
  (x : Nat)
  (y : Nat)

/-- Represents a flea on the board --/
structure Flea :=
  (position : Position)
  (direction : Direction)

/-- Represents the board state --/
def BoardState := List Flea

/-- The size of the board --/
def boardSize : Nat := 10

/-- The duration of observation in minutes --/
def observationTime : Nat := 60

/-- Function to update a flea's position based on its direction --/
def updateFleaPosition (f : Flea) : Flea :=
  sorry

/-- Function to check if two fleas occupy the same position --/
def fleaCollision (f1 f2 : Flea) : Bool :=
  sorry

/-- Function to simulate one minute of flea movement --/
def simulateMinute (state : BoardState) : BoardState :=
  sorry

/-- Function to simulate the entire observation period --/
def simulateObservation (initialState : BoardState) : Bool :=
  sorry

/-- Theorem stating the maximum number of fleas --/
theorem max_fleas : 
  ∀ (initialState : BoardState),
    simulateObservation initialState → List.length initialState ≤ 40 :=
  sorry

end max_fleas_l2948_294849


namespace min_box_value_l2948_294884

theorem min_box_value (a b Box : ℤ) : 
  a ≠ b ∧ a ≠ Box ∧ b ≠ Box →
  (∀ x, (a*x + b)*(b*x + a) = 30*x^2 + Box*x + 30) →
  a * b = 30 →
  Box = a^2 + b^2 →
  (∀ a' b' Box' : ℤ, 
    a' ≠ b' ∧ a' ≠ Box' ∧ b' ≠ Box' →
    (∀ x, (a'*x + b')*(b'*x + a') = 30*x^2 + Box'*x + 30) →
    a' * b' = 30 →
    Box' = a'^2 + b'^2 →
    Box ≤ Box') →
  Box = 61 := by
sorry

end min_box_value_l2948_294884


namespace ryan_study_time_l2948_294892

/-- Ryan's daily study hours for English -/
def english_hours : ℕ := 6

/-- Ryan's daily study hours for Chinese -/
def chinese_hours : ℕ := 7

/-- Number of days Ryan studies -/
def study_days : ℕ := 5

/-- Total study hours for both languages over the given period -/
def total_study_hours : ℕ := (english_hours + chinese_hours) * study_days

theorem ryan_study_time : total_study_hours = 65 := by
  sorry

end ryan_study_time_l2948_294892


namespace parabola_area_theorem_l2948_294809

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- Point on a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: Given a parabola y^2 = 2px (p > 0) with a point A(m,1) on it,
    and a point B on the directrix such that AB is perpendicular to the directrix,
    if the area of triangle AOB (where O is the origin) is 1/2, then p = 1. -/
theorem parabola_area_theorem (C : Parabola) (A : Point) (m : ℝ) :
  A.x = m →
  A.y = 1 →
  A.y^2 = 2 * C.p * A.x →
  (∃ B : Point, B.y = -C.p/2 ∧ (A.x - B.x) * (A.y - B.y) = 0) →
  (1/2 * m * 1 + 1/2 * (C.p/2) * 1 = 1/2) →
  C.p = 1 := by
  sorry

end parabola_area_theorem_l2948_294809


namespace triangle_side_length_l2948_294878

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  c = 4 * Real.sqrt 2 →
  B = π / 4 →
  S = 2 →
  S = (1 / 2) * a * c * Real.sin B →
  b^2 = a^2 + c^2 - 2 * a * c * Real.cos B →
  b = 5 := by
sorry

end triangle_side_length_l2948_294878


namespace pascal_interior_sum_l2948_294842

def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

theorem pascal_interior_sum :
  (interior_sum 4 = 6) →
  (interior_sum 5 = 14) →
  (∀ k ≥ 3, interior_sum k = 2^(k-1) - 2) →
  interior_sum 9 = 254 :=
by sorry

end pascal_interior_sum_l2948_294842


namespace quadratic_function_value_l2948_294840

/-- A quadratic function f(x) = ax^2 + bx + 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + 1

/-- Theorem: If f(1) = 3 and f(2) = 6, then f(3) = 10 -/
theorem quadratic_function_value (a b : ℝ) :
  f a b 1 = 3 → f a b 2 = 6 → f a b 3 = 10 := by
  sorry

end quadratic_function_value_l2948_294840


namespace relationship_abc_l2948_294829

theorem relationship_abc (a b c : ℝ) : 
  a = 2 → b = Real.log 9 → c = 2 * Real.sin (9 * π / 5) → a > b ∧ b > c := by
  sorry

end relationship_abc_l2948_294829


namespace quadratic_sum_of_constants_l2948_294897

/-- Given a quadratic expression x^2 - 20x + 100 that can be written in the form (x + b)^2 + c,
    prove that b + c = -10 -/
theorem quadratic_sum_of_constants (b c : ℝ) : 
  (∀ x, x^2 - 20*x + 100 = (x + b)^2 + c) → b + c = -10 := by
  sorry

end quadratic_sum_of_constants_l2948_294897


namespace unique_plane_through_parallel_lines_l2948_294877

-- Define a type for points in 3D space
variable (Point : Type)

-- Define a type for lines in 3D space
variable (Line : Type)

-- Define a type for planes in 3D space
variable (Plane : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Define a relation for a line being contained in a plane
variable (contains : Plane → Line → Prop)

-- Theorem: Through two parallel lines, there is exactly one plane
theorem unique_plane_through_parallel_lines 
  (l1 l2 : Line) 
  (h : parallel l1 l2) : 
  ∃! p : Plane, contains p l1 ∧ contains p l2 :=
sorry

end unique_plane_through_parallel_lines_l2948_294877


namespace prob_all_odd_is_one_42_l2948_294807

/-- The number of slips in the hat -/
def total_slips : ℕ := 10

/-- The number of odd-numbered slips in the hat -/
def odd_slips : ℕ := 5

/-- The number of slips drawn -/
def drawn_slips : ℕ := 4

/-- The probability of drawing all odd-numbered slips -/
def prob_all_odd : ℚ := (odd_slips : ℚ) / total_slips *
                        (odd_slips - 1) / (total_slips - 1) *
                        (odd_slips - 2) / (total_slips - 2) *
                        (odd_slips - 3) / (total_slips - 3)

theorem prob_all_odd_is_one_42 : prob_all_odd = 1 / 42 := by
  sorry

end prob_all_odd_is_one_42_l2948_294807


namespace cos_sin_sum_equals_half_l2948_294830

theorem cos_sin_sum_equals_half : 
  Real.cos (263 * π / 180) * Real.cos (203 * π / 180) + 
  Real.sin (83 * π / 180) * Real.sin (23 * π / 180) = 1 / 2 := by
  sorry

end cos_sin_sum_equals_half_l2948_294830


namespace investment_ratio_l2948_294858

/-- 
Given two investors p and q who divide their profit in the ratio 4:5,
prove that if p invested 52000, then q invested 65000.
-/
theorem investment_ratio (p q : ℕ) (h1 : p = 52000) : 
  (p : ℚ) / q = 4 / 5 → q = 65000 := by
sorry

end investment_ratio_l2948_294858


namespace imaginary_part_of_one_plus_i_fifth_power_l2948_294871

theorem imaginary_part_of_one_plus_i_fifth_power (i : ℂ) :
  i * i = -1 →
  Complex.im ((1 : ℂ) + i)^5 = -4 := by sorry

end imaginary_part_of_one_plus_i_fifth_power_l2948_294871


namespace club_officer_selection_l2948_294848

theorem club_officer_selection (n : ℕ) (k : ℕ) (h1 : n = 20) (h2 : k = 3) :
  n * (n - 1) * (n - 2) = 6840 :=
sorry

end club_officer_selection_l2948_294848


namespace sqrt_a_div_sqrt_b_l2948_294861

theorem sqrt_a_div_sqrt_b (a b : ℝ) (h : (1/3)^2 + (1/4)^2 = (37*a/100/b) * ((1/5)^2 + (1/6)^2)) :
  Real.sqrt a / Real.sqrt b = 50/19 := by
  sorry

end sqrt_a_div_sqrt_b_l2948_294861


namespace sum_of_w_and_y_is_eight_l2948_294874

theorem sum_of_w_and_y_is_eight (W X Y Z : ℤ) : 
  W ∈ ({1, 2, 3, 5} : Set ℤ) →
  X ∈ ({1, 2, 3, 5} : Set ℤ) →
  Y ∈ ({1, 2, 3, 5} : Set ℤ) →
  Z ∈ ({1, 2, 3, 5} : Set ℤ) →
  W ≠ X → W ≠ Y → W ≠ Z → X ≠ Y → X ≠ Z → Y ≠ Z →
  (W : ℚ) / (X : ℚ) - (Y : ℚ) / (Z : ℚ) = 1 →
  W + Y = 8 := by
sorry

end sum_of_w_and_y_is_eight_l2948_294874


namespace fraction_equation_solution_l2948_294802

theorem fraction_equation_solution (n : ℚ) :
  (1 : ℚ) / (n + 1) + (2 : ℚ) / (n + 1) + n / (n + 1) = 4 → n = -1/3 := by
  sorry

end fraction_equation_solution_l2948_294802


namespace sixth_term_of_arithmetic_sequence_l2948_294899

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sixth_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (is_arithmetic : arithmetic_sequence a)
  (first_term : a 1 = 2)
  (sum_of_three : a 1 + a 2 + a 3 = 12) :
  a 6 = 12 := by
sorry

end sixth_term_of_arithmetic_sequence_l2948_294899


namespace tenth_meeting_position_l2948_294873

/-- Represents a robot on a circular track -/
structure Robot where
  speed : ℝ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Represents the state of the system -/
structure State where
  robotA : Robot
  robotB : Robot
  position : ℝ  -- Position on the track (0 ≤ position < 8)
  meetings : ℕ

/-- Updates the state after a meeting -/
def updateState (s : State) : State :=
  { s with
    robotB := { s.robotB with direction := !s.robotB.direction }
    meetings := s.meetings + 1
  }

/-- Simulates the movement of robots until they meet 10 times -/
def simulate (initialState : State) : ℝ :=
  sorry

theorem tenth_meeting_position (initialA initialB : Robot) :
  let initialState : State :=
    { robotA := initialA
      robotB := initialB
      position := 0
      meetings := 0
    }
  simulate initialState = 0 :=
sorry

end tenth_meeting_position_l2948_294873


namespace min_tiles_to_cover_min_tiles_for_given_dimensions_l2948_294869

/-- Represents the dimensions of a rectangle in inches -/
structure Rectangle where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangle in square inches -/
def area (r : Rectangle) : ℕ := r.length * r.width

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Theorem: The minimum number of 3x5 inch tiles needed to cover a 3x4 foot region is 116 -/
theorem min_tiles_to_cover (tile : Rectangle) (region : Rectangle) : ℕ :=
  let tileArea := area tile
  let regionArea := area { length := feetToInches region.length, width := feetToInches region.width }
  ((regionArea + tileArea - 1) / tileArea : ℕ)

/-- Main theorem statement -/
theorem min_tiles_for_given_dimensions : 
  min_tiles_to_cover { length := 3, width := 5 } { length := 3, width := 4 } = 116 := by
  sorry

end min_tiles_to_cover_min_tiles_for_given_dimensions_l2948_294869


namespace ellipse_equation_l2948_294814

/-- An ellipse centered at the origin -/
structure Ellipse where
  equation : ℝ → ℝ → Prop

/-- A hyperbola -/
structure Hyperbola where
  equation : ℝ → ℝ → Prop

/-- The eccentricity of a conic section -/
def eccentricity (c : ℝ) : ℝ := c

/-- Theorem: Given an ellipse centered at the origin sharing a common focus with 
    the hyperbola 2x^2 - 2y^2 = 1, and their eccentricities being reciprocal to 
    each other, the equation of the ellipse is x^2/2 + y^2 = 1 -/
theorem ellipse_equation 
  (e : Ellipse) 
  (h : Hyperbola) 
  (h_eq : h.equation = fun x y => 2 * x^2 - 2 * y^2 = 1) 
  (common_focus : ∃ (f : ℝ × ℝ), f ∈ {p | ∃ (x y : ℝ), p = (x, y) ∧ h.equation x y} ∧ 
                                 f ∈ {p | ∃ (x y : ℝ), p = (x, y) ∧ e.equation x y})
  (reciprocal_eccentricity : ∃ (e_ecc h_ecc : ℝ), 
    eccentricity e_ecc * eccentricity h_ecc = 1) :
  e.equation = fun x y => x^2 / 2 + y^2 = 1 :=
sorry

end ellipse_equation_l2948_294814


namespace berts_spending_l2948_294880

/-- Bert's spending problem -/
theorem berts_spending (n : ℝ) : 
  (1/2) * ((2/3) * n - 7) = 10.5 → n = 42 := by
  sorry

end berts_spending_l2948_294880


namespace max_value_of_expression_l2948_294895

theorem max_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  let A := ((x - y) * Real.sqrt (x^2 + y^2) + 
            (y - z) * Real.sqrt (y^2 + z^2) + 
            (z - x) * Real.sqrt (z^2 + x^2) + 
            Real.sqrt 2) / 
           ((x - y)^2 + (y - z)^2 + (z - x)^2 + 2)
  A ≤ 1 / Real.sqrt 2 := by
  sorry

end max_value_of_expression_l2948_294895


namespace pencil_pen_cost_l2948_294800

theorem pencil_pen_cost (p q : ℝ) 
  (h1 : 3 * p + 2 * q = 4.20)
  (h2 : 4 * p + 3 * q = 5.60) : 
  p + q = 1.40 := by
  sorry

end pencil_pen_cost_l2948_294800


namespace cuboid_volume_example_l2948_294817

/-- A cuboid with given base area and height -/
structure Cuboid where
  base_area : ℝ
  height : ℝ

/-- The volume of a cuboid -/
def volume (c : Cuboid) : ℝ := c.base_area * c.height

/-- Theorem: The volume of a cuboid with base area 14 cm² and height 13 cm is 182 cm³ -/
theorem cuboid_volume_example : 
  let c : Cuboid := { base_area := 14, height := 13 }
  volume c = 182 := by
  sorry

end cuboid_volume_example_l2948_294817


namespace log_product_equals_two_thirds_l2948_294815

theorem log_product_equals_two_thirds : 
  Real.log 2 / Real.log 3 * Real.log 9 / Real.log 8 = 2 / 3 := by
  sorry

end log_product_equals_two_thirds_l2948_294815


namespace equality_iff_inequality_holds_l2948_294835

theorem equality_iff_inequality_holds (x y : ℝ) : x = y ↔ x * y ≥ ((x + y) / 2)^2 := by
  sorry

end equality_iff_inequality_holds_l2948_294835


namespace ma_xiaohu_speed_ma_xiaohu_speed_proof_l2948_294819

/-- Proves that Ma Xiaohu's speed is 80 meters per minute given the problem conditions -/
theorem ma_xiaohu_speed : ℝ → Prop :=
  fun (x : ℝ) ↦
    let total_distance : ℝ := 1800
    let catch_up_distance : ℝ := 200
    let father_delay : ℝ := 10
    let father_speed : ℝ := 2 * x
    let ma_distance : ℝ := total_distance - catch_up_distance
    let ma_time : ℝ := ma_distance / x
    let father_time : ℝ := ma_distance / father_speed
    ma_time - father_time = father_delay → x = 80

/-- Proof of the theorem -/
theorem ma_xiaohu_speed_proof : ma_xiaohu_speed 80 := by
  sorry

end ma_xiaohu_speed_ma_xiaohu_speed_proof_l2948_294819


namespace expression_evaluation_l2948_294844

theorem expression_evaluation : 3^(1^(2^3)) + ((3^1)^2)^2 = 84 := by
  sorry

end expression_evaluation_l2948_294844


namespace maintenance_check_increase_l2948_294847

theorem maintenance_check_increase (original_time new_time : ℕ) 
  (h1 : original_time = 30) 
  (h2 : new_time = 60) : 
  (new_time - original_time) / original_time * 100 = 100 :=
by sorry

end maintenance_check_increase_l2948_294847


namespace bowTie_equation_solution_l2948_294812

-- Define the bow tie operation
noncomputable def bowTie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b + Real.sqrt (b + Real.sqrt (b + Real.sqrt b)))

-- Theorem statement
theorem bowTie_equation_solution (h : ℝ) :
  bowTie 5 h = 7 → h = 2 := by sorry

end bowTie_equation_solution_l2948_294812


namespace min_value_expression_min_value_achievable_l2948_294876

theorem min_value_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  8 * a^4 + 12 * b^4 + 40 * c^4 + 2 * d^2 + 1 / (5 * a * b * c * d) ≥ 4 * Real.sqrt 10 / 5 :=
by sorry

theorem min_value_achievable :
  ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  8 * a^4 + 12 * b^4 + 40 * c^4 + 2 * d^2 + 1 / (5 * a * b * c * d) = 4 * Real.sqrt 10 / 5 :=
by sorry

end min_value_expression_min_value_achievable_l2948_294876


namespace three_lines_intersection_l2948_294816

/-- Three lines intersect at a single point if and only if k = -7 -/
theorem three_lines_intersection (x y k : ℝ) : 
  (∃! p : ℝ × ℝ, (y = 7*x + 5 ∧ y = -3*x - 35 ∧ y = 4*x + k) → p.1 = x ∧ p.2 = y) ↔ k = -7 :=
by sorry

end three_lines_intersection_l2948_294816


namespace range_of_x_minus_2y_l2948_294893

theorem range_of_x_minus_2y (x y : ℝ) 
  (hx : -1 ≤ x ∧ x < 2) (hy : 0 < y ∧ y ≤ 1) : 
  -3 ≤ x - 2*y ∧ x - 2*y < 2 := by
  sorry

end range_of_x_minus_2y_l2948_294893


namespace production_time_calculation_l2948_294836

/-- The number of days it takes for a given number of machines to produce a certain amount of product P -/
def production_time (num_machines : ℕ) (units : ℝ) : ℝ := sorry

/-- The number of units produced by a given number of machines in a certain number of days -/
def units_produced (num_machines : ℕ) (days : ℝ) : ℝ := sorry

theorem production_time_calculation :
  let d := production_time 5 x
  let x : ℝ := units_produced 5 d
  units_produced 20 2 = 2 * x →
  d = 4 := by sorry

end production_time_calculation_l2948_294836


namespace at_least_two_correct_coats_l2948_294865

theorem at_least_two_correct_coats (n : ℕ) (h : n = 5) : 
  (Finset.sum (Finset.range (n - 1)) (λ k => (n.choose (k + 2)) * ((n - k - 2).factorial))) = 31 := by
  sorry

end at_least_two_correct_coats_l2948_294865


namespace sum_of_specific_repeating_decimals_l2948_294855

/-- Definition of a repeating decimal with a 3-digit repetend -/
def repeating_decimal (a b c : ℕ) : ℚ := (a * 100 + b * 10 + c) / 999

/-- The sum of two specific repeating decimals equals 161/999 -/
theorem sum_of_specific_repeating_decimals : 
  repeating_decimal 1 3 7 + repeating_decimal 0 2 4 = 161 / 999 := by sorry

end sum_of_specific_repeating_decimals_l2948_294855


namespace rhombus_perimeter_l2948_294824

/-- The perimeter of a rhombus with diagonals of 12 inches and 16 inches is 40 inches. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 12) (h2 : d2 = 16) : 
  4 * Real.sqrt ((d1/2)^2 + (d2/2)^2) = 40 :=
by sorry

end rhombus_perimeter_l2948_294824


namespace triangular_prism_theorem_l2948_294811

theorem triangular_prism_theorem (V k : ℝ) (S H : Fin 4 → ℝ) : 
  (∀ i : Fin 4, S i = (i.val + 1 : ℕ) * k) →
  (∀ i : Fin 4, V = (1/3) * S i * H i) →
  H 0 + 2 * H 1 + 3 * H 2 + 4 * H 3 = 3 * V / k :=
by sorry

end triangular_prism_theorem_l2948_294811


namespace number_exceeds_fraction_by_40_l2948_294831

theorem number_exceeds_fraction_by_40 (x : ℝ) : x = (3 / 8) * x + 40 → x = 64 := by
  sorry

end number_exceeds_fraction_by_40_l2948_294831


namespace team_a_faster_by_three_hours_l2948_294875

/-- Proves that Team A finishes 3 hours faster than Team W in a 300-mile race -/
theorem team_a_faster_by_three_hours 
  (course_length : ℝ) 
  (speed_w : ℝ) 
  (speed_difference : ℝ) : 
  course_length = 300 → 
  speed_w = 20 → 
  speed_difference = 5 → 
  (course_length / speed_w) - (course_length / (speed_w + speed_difference)) = 3 := by
  sorry

#check team_a_faster_by_three_hours

end team_a_faster_by_three_hours_l2948_294875


namespace probability_of_specific_selection_l2948_294828

/-- The number of shirts in the drawer -/
def num_shirts : ℕ := 6

/-- The number of pairs of shorts in the drawer -/
def num_shorts : ℕ := 8

/-- The number of pairs of socks in the drawer -/
def num_socks : ℕ := 7

/-- The number of jackets in the drawer -/
def num_jackets : ℕ := 3

/-- The total number of clothing items in the drawer -/
def total_items : ℕ := num_shirts + num_shorts + num_socks + num_jackets

/-- The number of items to be selected -/
def items_to_select : ℕ := 4

theorem probability_of_specific_selection :
  (num_shirts : ℚ) * num_shorts * num_socks * num_jackets /
  (total_items.choose items_to_select) = 144 / 1815 :=
sorry

end probability_of_specific_selection_l2948_294828


namespace expand_product_l2948_294845

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12*x + 27 := by
  sorry

end expand_product_l2948_294845


namespace smallest_integer_y_smallest_integer_is_negative_six_l2948_294838

theorem smallest_integer_y (y : ℤ) : (10 + 3 * y ≤ -8) ↔ (y ≤ -6) :=
  sorry

theorem smallest_integer_is_negative_six : ∃ (y : ℤ), (10 + 3 * y ≤ -8) ∧ (∀ (z : ℤ), (10 + 3 * z ≤ -8) → z ≥ y) ∧ y = -6 :=
  sorry

end smallest_integer_y_smallest_integer_is_negative_six_l2948_294838


namespace more_b_shoes_than_a_l2948_294837

/-- Given the conditions about shoe boxes, prove that there are 640 more pairs of (B) shoes than (A) shoes. -/
theorem more_b_shoes_than_a : 
  ∀ (pairs_per_box : ℕ) (num_a_boxes : ℕ) (num_b_boxes : ℕ),
  pairs_per_box = 20 →
  num_a_boxes = 8 →
  num_b_boxes = 5 * num_a_boxes →
  num_b_boxes * pairs_per_box - num_a_boxes * pairs_per_box = 640 :=
by
  sorry

#check more_b_shoes_than_a

end more_b_shoes_than_a_l2948_294837


namespace soup_ingredients_weights_l2948_294853

/-- Represents the ingredients of the soup --/
structure SoupIngredients where
  fat : ℝ
  onion : ℝ
  potatoes : ℝ
  grain : ℝ
  water : ℝ

/-- The conditions of the soup recipe --/
def SoupConditions (s : SoupIngredients) : Prop :=
  s.water = s.grain + s.potatoes + s.onion + s.fat ∧
  s.grain = s.potatoes + s.onion + s.fat ∧
  s.potatoes = s.onion + s.fat ∧
  s.fat = s.onion / 2 ∧
  s.water + s.grain + s.potatoes + s.onion + s.fat = 12

/-- The theorem stating the correct weights of the ingredients --/
theorem soup_ingredients_weights :
  ∃ (s : SoupIngredients),
    SoupConditions s ∧
    s.fat = 0.5 ∧
    s.onion = 1 ∧
    s.potatoes = 1.5 ∧
    s.grain = 3 ∧
    s.water = 6 :=
  sorry

end soup_ingredients_weights_l2948_294853


namespace potassium_dichromate_oxidizes_Br_and_I_l2948_294806

/-- Standard reduction potential for I₂ + 2e⁻ → 2I⁻ -/
def E_I₂ : ℝ := 0.54

/-- Standard reduction potential for Cr₂O₇²⁻ + 14H⁺ + 6e⁻ → 2Cr³⁺ + 7H₂O -/
def E_Cr₂O₇ : ℝ := 1.33

/-- Standard oxidation potential for 2Br⁻ - 2e⁻ → Br₂ -/
def E_Br : ℝ := 1.07

/-- Standard oxidation potential for 2I⁻ - 2e⁻ → I₂ -/
def E_I : ℝ := 0.54

/-- A reaction is spontaneous if its cell potential is positive -/
def is_spontaneous (cell_potential : ℝ) : Prop := cell_potential > 0

/-- Theorem: Potassium dichromate can oxidize both Br⁻ and I⁻ -/
theorem potassium_dichromate_oxidizes_Br_and_I :
  is_spontaneous (E_Cr₂O₇ - E_Br) ∧ is_spontaneous (E_Cr₂O₇ - E_I) := by
  sorry


end potassium_dichromate_oxidizes_Br_and_I_l2948_294806


namespace percentage_of_invalid_votes_l2948_294882

theorem percentage_of_invalid_votes
  (total_votes : ℕ)
  (candidate_a_percentage : ℚ)
  (candidate_a_votes : ℕ)
  (h1 : total_votes = 560000)
  (h2 : candidate_a_percentage = 80 / 100)
  (h3 : candidate_a_votes = 380800) :
  (total_votes - (candidate_a_votes / candidate_a_percentage)) / total_votes = 15 / 100 :=
sorry

end percentage_of_invalid_votes_l2948_294882


namespace divide_powers_l2948_294839

theorem divide_powers (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  2 * x^2 * y^3 / (x * y^2) = 2 * x * y := by
  sorry

end divide_powers_l2948_294839


namespace rectangular_screen_area_l2948_294888

/-- Proves that a rectangular screen with width-to-height ratio of 3:2 and diagonal length of 65 cm has an area of 1950 cm². -/
theorem rectangular_screen_area (width height diagonal : ℝ) : 
  width / height = 3 / 2 →
  width^2 + height^2 = diagonal^2 →
  diagonal = 65 →
  width * height = 1950 := by
sorry

end rectangular_screen_area_l2948_294888


namespace rain_forest_animal_count_l2948_294866

/-- The number of animals in the Rain Forest exhibit -/
def rain_forest_animals : ℕ := 7

/-- The number of animals in the Reptile House -/
def reptile_house_animals : ℕ := 16

/-- Theorem stating the relationship between the number of animals in the Rain Forest exhibit and the Reptile House -/
theorem rain_forest_animal_count : 
  reptile_house_animals = 3 * rain_forest_animals - 5 ∧ 
  rain_forest_animals = 7 := by
  sorry

end rain_forest_animal_count_l2948_294866


namespace charlotte_tuesday_poodles_l2948_294856

/-- Represents the schedule and constraints for Charlotte's dog walking --/
structure DogWalkingSchedule where
  monday_poodles : Nat
  monday_chihuahuas : Nat
  wednesday_labradors : Nat
  poodle_time : Nat
  chihuahua_time : Nat
  labrador_time : Nat
  total_time : Nat

/-- Calculates the number of poodles Charlotte can walk on Tuesday --/
def tuesday_poodles (schedule : DogWalkingSchedule) : Nat :=
  let monday_time := schedule.monday_poodles * schedule.poodle_time + 
                     schedule.monday_chihuahuas * schedule.chihuahua_time
  let tuesday_chihuahua_time := schedule.monday_chihuahuas * schedule.chihuahua_time
  let wednesday_time := schedule.wednesday_labradors * schedule.labrador_time
  let available_time := schedule.total_time - monday_time - tuesday_chihuahua_time - wednesday_time
  available_time / schedule.poodle_time

/-- Theorem stating that given the schedule constraints, Charlotte can walk 4 poodles on Tuesday --/
theorem charlotte_tuesday_poodles : 
  ∀ (schedule : DogWalkingSchedule), 
  schedule.monday_poodles = 4 ∧ 
  schedule.monday_chihuahuas = 2 ∧ 
  schedule.wednesday_labradors = 4 ∧ 
  schedule.poodle_time = 2 ∧ 
  schedule.chihuahua_time = 1 ∧ 
  schedule.labrador_time = 3 ∧ 
  schedule.total_time = 32 → 
  tuesday_poodles schedule = 4 := by
  sorry


end charlotte_tuesday_poodles_l2948_294856


namespace rug_overlap_problem_l2948_294896

/-- Given three rugs with a combined area of 200 square meters, prove that the area
    covered by exactly two layers of rug is 5 square meters when:
    1. The rugs cover a floor area of 138 square meters when overlapped.
    2. The area covered by exactly some layers of rug is 24 square meters.
    3. The area covered by three layers of rug is 19 square meters. -/
theorem rug_overlap_problem (total_area : ℝ) (covered_area : ℝ) (some_layers_area : ℝ) (three_layers_area : ℝ)
    (h1 : total_area = 200)
    (h2 : covered_area = 138)
    (h3 : some_layers_area = 24)
    (h4 : three_layers_area = 19) :
    total_area - (covered_area + some_layers_area) = 5 := by
  sorry

end rug_overlap_problem_l2948_294896


namespace system_solvable_iff_a_in_range_l2948_294833

-- Define the system of equations
def system (a b x y : ℝ) : Prop :=
  x^2 + y^2 + 2*a*(a - x - y) = 64 ∧
  y = 8 * Real.sin (x - 2*b) - 6 * Real.cos (x - 2*b)

-- Theorem statement
theorem system_solvable_iff_a_in_range :
  ∀ a : ℝ, (∃ b x y : ℝ, system a b x y) ↔ -18 ≤ a ∧ a ≤ 18 := by
  sorry

end system_solvable_iff_a_in_range_l2948_294833


namespace baker_remaining_cakes_l2948_294859

theorem baker_remaining_cakes (total_cakes friend_bought : ℕ) 
  (h1 : total_cakes = 155)
  (h2 : friend_bought = 140) :
  total_cakes - friend_bought = 15 := by
  sorry

end baker_remaining_cakes_l2948_294859


namespace odd_divides_power_factorial_minus_one_l2948_294823

theorem odd_divides_power_factorial_minus_one (n : ℕ) (h_pos : 0 < n) (h_odd : Odd n) :
  n ∣ 2^(n.factorial) - 1 := by
  sorry

end odd_divides_power_factorial_minus_one_l2948_294823


namespace f_at_one_plus_sqrt_two_l2948_294867

-- Define the function f
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

-- State the theorem
theorem f_at_one_plus_sqrt_two : f (1 + Real.sqrt 2) = 4 * Real.sqrt 2 := by
  sorry

end f_at_one_plus_sqrt_two_l2948_294867


namespace quadratic_inequality_sum_l2948_294825

/-- Given a quadratic inequality ax^2 - 3ax - 6 < 0 with solution set {x | x < 1 or x > b}, 
    prove that a + b = -1 -/
theorem quadratic_inequality_sum (a b : ℝ) : 
  (∀ x, ax^2 - 3*a*x - 6 < 0 ↔ x < 1 ∨ x > b) → 
  a + b = -1 := by
sorry

end quadratic_inequality_sum_l2948_294825


namespace dinner_cost_bret_dinner_cost_l2948_294886

theorem dinner_cost (people : ℕ) (main_meal_cost appetizer_cost : ℚ) 
  (appetizers : ℕ) (tip_percentage : ℚ) (rush_fee : ℚ) : ℚ :=
  let main_meals_total := people * main_meal_cost
  let appetizers_total := appetizers * appetizer_cost
  let subtotal := main_meals_total + appetizers_total
  let tip := tip_percentage * subtotal
  let total := subtotal + tip + rush_fee
  total

theorem bret_dinner_cost : 
  dinner_cost 4 12 6 2 (20/100) 5 = 77 := by
  sorry

end dinner_cost_bret_dinner_cost_l2948_294886


namespace profit_starts_third_year_option_one_more_profitable_l2948_294894

/-- Represents the financial model of the fishing boat -/
structure FishingBoat where
  initialCost : ℕ
  firstYearExpenses : ℕ
  annualExpenseIncrease : ℕ
  annualIncome : ℕ

/-- Calculates the cumulative profit after n years -/
def cumulativeProfit (boat : FishingBoat) (n : ℕ) : ℤ :=
  n * boat.annualIncome - boat.initialCost - boat.firstYearExpenses
    - (n - 1) * boat.annualExpenseIncrease * n / 2

/-- Calculates the average profit after n years -/
def averageProfit (boat : FishingBoat) (n : ℕ) : ℚ :=
  (cumulativeProfit boat n : ℚ) / n

/-- The boat configuration from the problem -/
def problemBoat : FishingBoat :=
  { initialCost := 980000
    firstYearExpenses := 120000
    annualExpenseIncrease := 40000
    annualIncome := 500000 }

theorem profit_starts_third_year :
  ∀ n : ℕ, n < 3 → cumulativeProfit problemBoat n ≤ 0
  ∧ cumulativeProfit problemBoat 3 > 0 := by sorry

theorem option_one_more_profitable :
  let optionOne := cumulativeProfit problemBoat 7 + 260000
  let optionTwo := cumulativeProfit problemBoat 10 + 80000
  optionOne = optionTwo ∧ 7 < 10 := by sorry

end profit_starts_third_year_option_one_more_profitable_l2948_294894


namespace weighted_power_inequality_l2948_294834

theorem weighted_power_inequality (a b c : ℝ) (n : ℕ) (p q r : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hn : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p := by
  sorry

end weighted_power_inequality_l2948_294834


namespace store_profit_theorem_l2948_294818

/-- Represents the store's sales and profit model -/
structure Store where
  initial_profit_per_unit : ℝ
  initial_daily_sales : ℝ
  sales_increase_per_yuan : ℝ

/-- Calculates the new daily sales after a price reduction -/
def new_daily_sales (s : Store) (price_reduction : ℝ) : ℝ :=
  s.initial_daily_sales + s.sales_increase_per_yuan * price_reduction

/-- Calculates the daily profit after a price reduction -/
def daily_profit (s : Store) (price_reduction : ℝ) : ℝ :=
  (s.initial_profit_per_unit - price_reduction) * (new_daily_sales s price_reduction)

/-- The store model based on the given conditions -/
def my_store : Store := {
  initial_profit_per_unit := 60,
  initial_daily_sales := 40,
  sales_increase_per_yuan := 2
}

theorem store_profit_theorem (s : Store) :
  (new_daily_sales s 10 = 60) ∧
  (∃ x : ℝ, x = 30 ∧ daily_profit s x = 3000) ∧
  (¬ ∃ y : ℝ, daily_profit s y = 3300) := by
  sorry

#check store_profit_theorem my_store

end store_profit_theorem_l2948_294818


namespace coupon_value_l2948_294805

def vacuum_cost : ℝ := 250
def dishwasher_cost : ℝ := 450
def total_cost_after_coupon : ℝ := 625

theorem coupon_value : 
  vacuum_cost + dishwasher_cost - total_cost_after_coupon = 75 := by
  sorry

end coupon_value_l2948_294805


namespace trig_simplification_l2948_294890

theorem trig_simplification :
  (Real.sin (30 * π / 180) + Real.sin (40 * π / 180)) /
  (Real.cos (30 * π / 180) + Real.cos (40 * π / 180)) =
  Real.tan (35 * π / 180) := by sorry

end trig_simplification_l2948_294890


namespace union_equals_universe_l2948_294813

def U : Finset ℕ := {2, 3, 4, 5, 6}
def M : Finset ℕ := {3, 4, 5}
def N : Finset ℕ := {2, 4, 5, 6}

theorem union_equals_universe : M ∪ N = U := by
  sorry

end union_equals_universe_l2948_294813


namespace product_first_three_terms_l2948_294885

def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem product_first_three_terms 
  (a : ℕ → ℕ) 
  (h1 : arithmetic_sequence a 2)
  (h2 : a 7 = 20) : 
  a 1 * a 2 * a 3 = 960 := by
sorry

end product_first_three_terms_l2948_294885


namespace c_investment_is_2000_l2948_294898

/-- Represents the investment and profit distribution in a business partnership --/
structure BusinessPartnership where
  a_investment : ℕ
  b_investment : ℕ
  c_investment : ℕ
  total_profit : ℕ
  b_profit_share : ℕ
  a_c_profit_diff : ℕ

/-- Calculates the profit share for a given investment --/
def profit_share (investment total_investment total_profit : ℕ) : ℕ :=
  (investment * total_profit) / total_investment

/-- Theorem stating that C's investment is 2000 given the problem conditions --/
theorem c_investment_is_2000 (bp : BusinessPartnership)
  (h1 : bp.a_investment = 6000)
  (h2 : bp.b_investment = 8000)
  (h3 : bp.b_profit_share = 1000)
  (h4 : bp.a_c_profit_diff = 500)
  (h5 : bp.b_profit_share = profit_share bp.b_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit)
  (h6 : bp.a_c_profit_diff = profit_share bp.a_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit -
                             profit_share bp.c_investment (bp.a_investment + bp.b_investment + bp.c_investment) bp.total_profit) :
  bp.c_investment = 2000 := by
  sorry


end c_investment_is_2000_l2948_294898


namespace right_rectangular_prism_volume_l2948_294887

theorem right_rectangular_prism_volume 
  (a b c : ℝ) 
  (h1 : a * b = 36) 
  (h2 : a * c = 48) 
  (h3 : b * c = 72) : 
  a * b * c = 168 := by
sorry

end right_rectangular_prism_volume_l2948_294887


namespace triangle_and_circle_symmetry_l2948_294870

-- Define the point A
def A : ℝ × ℝ := (4, -3)

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Define the vector OA
def OA : ℝ × ℝ := A

-- Define the vector AB
def AB : ℝ × ℝ := (6, 8)

-- Define point B
def B : ℝ × ℝ := (A.1 + AB.1, A.2 + AB.2)

-- Define the original circle
def original_circle (x y : ℝ) : Prop :=
  x^2 - 6*x + y^2 + 2*y = 0

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 3)^2 = 10

-- Theorem statement
theorem triangle_and_circle_symmetry :
  -- A is the right-angle vertex of triangle OAB
  (OA.1 * AB.1 + OA.2 * AB.2 = 0) →
  -- |AB| = 2|OA|
  (AB.1^2 + AB.2^2 = 4 * (OA.1^2 + OA.2^2)) →
  -- The ordinate of point B is greater than 0
  (B.2 > 0) →
  -- AB has coordinates (6, 8)
  (AB = (6, 8)) ∧
  -- The equation of the symmetric circle is correct
  (∀ x y, symmetric_circle x y ↔
    ∃ x' y', original_circle x' y' ∧
      -- x and y are symmetric to x' and y' with respect to line OB
      ((x + x') / 2 = B.1 * ((y + y') / 2) / B.2)) :=
sorry

end triangle_and_circle_symmetry_l2948_294870


namespace probability_sum_less_than_product_l2948_294852

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

def valid_number (n : ℕ) : Prop := is_even n ∧ 0 < n ∧ n ≤ 10

def valid_pair (a b : ℕ) : Prop := valid_number a ∧ valid_number b ∧ a + b < a * b

def total_pairs : ℕ := 25

def valid_pairs : ℕ := 16

theorem probability_sum_less_than_product :
  (valid_pairs : ℚ) / total_pairs = 16 / 25 := by sorry

end probability_sum_less_than_product_l2948_294852


namespace conic_section_eccentricity_l2948_294864

/-- Given that real numbers 4, m, 9 form a geometric sequence,
    prove that the eccentricity of the conic section x^2/m + y^2 = 1
    is either √30/6 or √7 -/
theorem conic_section_eccentricity (m : ℝ) 
  (h_geom_seq : (4 : ℝ) * 9 = m^2) :
  let e := if m > 0 
    then Real.sqrt (1 - m / 6) / Real.sqrt (m / 6)
    else Real.sqrt (1 + 6 / m) / 1
  e = Real.sqrt 30 / 6 ∨ e = Real.sqrt 7 := by
  sorry

end conic_section_eccentricity_l2948_294864


namespace geometric_sequence_property_l2948_294854

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 5 * a 7 * a 9 * a 11 = 243 →
  a 9^2 / a 11 = 3 := by
  sorry

end geometric_sequence_property_l2948_294854


namespace maria_score_l2948_294821

def test_score (total_questions : ℕ) (correct_points : ℕ) (incorrect_deduction : ℕ) (correct_answers : ℕ) : ℤ :=
  (correct_answers * correct_points : ℤ) - ((total_questions - correct_answers) * incorrect_deduction)

theorem maria_score :
  test_score 30 20 5 19 = 325 := by
  sorry

end maria_score_l2948_294821


namespace f_30_value_l2948_294860

/-- A function from positive integers to positive integers satisfying certain properties -/
def special_function (f : ℕ+ → ℕ+) : Prop :=
  (∀ n : ℕ+, f (n + 1) > f n) ∧ 
  (∀ m n : ℕ+, f (m * n) = f m * f n) ∧
  (∀ m n : ℕ+, m ≠ n → m ^ m.val = n ^ n.val → (f m = n ∨ f n = m))

theorem f_30_value (f : ℕ+ → ℕ+) (h : special_function f) : f 30 = 900 := by
  sorry

end f_30_value_l2948_294860


namespace pen_count_l2948_294881

theorem pen_count (num_pencils : ℕ) (max_students : ℕ) (num_pens : ℕ) : 
  num_pencils = 520 →
  max_students = 40 →
  num_pencils % max_students = 0 →
  num_pens % max_students = 0 →
  (num_pencils / max_students = num_pens / max_students) →
  num_pens = 520 := by
sorry

end pen_count_l2948_294881


namespace stacy_berries_l2948_294832

theorem stacy_berries (steve_initial : ℕ) (steve_takes : ℕ) (difference : ℕ) : 
  steve_initial = 21 → steve_takes = 4 → difference = 7 → 
  ∃ stacy_initial : ℕ, stacy_initial = 32 ∧ 
    steve_initial + steve_takes = stacy_initial - difference :=
by sorry

end stacy_berries_l2948_294832


namespace at_least_one_greater_than_one_l2948_294857

theorem at_least_one_greater_than_one (a b : ℝ) :
  (a + b > 2 → max a b > 1) ∧ (a * b > 1 → max a b > 1) := by
  sorry

end at_least_one_greater_than_one_l2948_294857


namespace special_square_PT_l2948_294862

/-- A square with side length 2 and special points T and U -/
structure SpecialSquare where
  -- Point P is at (0, 0), Q at (2, 0), R at (2, 2), and S at (0, 2)
  T : ℝ × ℝ  -- Point on PQ
  U : ℝ × ℝ  -- Point on SQ
  h_T_on_PQ : T.1 ∈ Set.Icc 0 2 ∧ T.2 = 0
  h_U_on_SQ : U.1 = 2 ∧ U.2 ∈ Set.Icc 0 2
  h_PT_eq_QU : T.1 = 2 - U.2  -- PT = QU
  h_fold : (2 - T.1)^2 + T.1^2 = 8  -- Condition for PR and SR to coincide with RQ when folded

theorem special_square_PT (s : SpecialSquare) : s.T.1 = Real.sqrt 2 / 2 := by
  sorry

end special_square_PT_l2948_294862


namespace square_plus_product_equals_zero_l2948_294804

theorem square_plus_product_equals_zero : (-2)^2 + (-2) * 2 = 0 := by
  sorry

end square_plus_product_equals_zero_l2948_294804
