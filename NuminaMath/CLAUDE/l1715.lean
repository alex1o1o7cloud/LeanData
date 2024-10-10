import Mathlib

namespace vector_projection_and_magnitude_l1715_171518

/-- Given vectors a and b in R², if the projection of a in its direction is -√2,
    then the second component of b is 4 and the magnitude of b is 2√5. -/
theorem vector_projection_and_magnitude (a b : ℝ × ℝ) :
  a = (1, -1) →
  b.1 = 2 →
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (a.1^2 + a.2^2) = -Real.sqrt 2 →
  b.2 = 4 ∧ Real.sqrt (b.1^2 + b.2^2) = 2 * Real.sqrt 5 := by
  sorry

end vector_projection_and_magnitude_l1715_171518


namespace spherical_sector_central_angle_l1715_171574

theorem spherical_sector_central_angle (R : ℝ) (α : ℝ) :
  R > 0 →
  (∃ r m : ℝ, R * π * r = 2 * R * π * m ∧ 
              R^2 = r^2 + (R - m)^2 ∧ 
              0 < m ∧ m < R) →
  α = 2 * Real.arccos (3/5) :=
sorry

end spherical_sector_central_angle_l1715_171574


namespace sum_of_radii_l1715_171519

/-- The sum of radii of circles tangent to x and y axes and externally tangent to a circle at (5,0) with radius 1.5 -/
theorem sum_of_radii : ∃ (r₁ r₂ : ℝ),
  r₁ > 0 ∧ r₂ > 0 ∧
  (r₁ - 5)^2 + r₁^2 = (r₁ + 1.5)^2 ∧
  (r₂ - 5)^2 + r₂^2 = (r₂ + 1.5)^2 ∧
  r₁ + r₂ = 13 :=
by sorry


end sum_of_radii_l1715_171519


namespace circle_center_radius_sum_l1715_171596

/-- Given a circle with equation x^2 - 8x + y^2 + 16y = -100, 
    prove that the sum of the x-coordinate of the center, 
    the y-coordinate of the center, and the radius is -4 + 2√5 -/
theorem circle_center_radius_sum :
  ∃ (c d s : ℝ), 
    (∀ (x y : ℝ), x^2 - 8*x + y^2 + 16*y = -100 ↔ (x - c)^2 + (y - d)^2 = s^2) ∧
    c + d + s = -4 + 2 * Real.sqrt 5 := by
  sorry

end circle_center_radius_sum_l1715_171596


namespace descending_order_inequality_l1715_171568

theorem descending_order_inequality (x y : ℝ) (hx : x < 0) (hy : -1 < y ∧ y < 0) :
  x * y > x * y^2 ∧ x * y^2 > x := by
  sorry

end descending_order_inequality_l1715_171568


namespace max_lateral_surface_area_triangular_prism_l1715_171566

/-- Given a triangular prism with perimeter 12, prove that its maximum lateral surface area is 6 -/
theorem max_lateral_surface_area_triangular_prism :
  ∀ x y : ℝ, x > 0 → y > 0 → 6 * x + 3 * y = 12 →
  3 * x * y ≤ 6 :=
by sorry

end max_lateral_surface_area_triangular_prism_l1715_171566


namespace smallest_x_absolute_value_equation_l1715_171535

theorem smallest_x_absolute_value_equation :
  let f : ℝ → ℝ := λ x ↦ |2 * x + 5|
  ∃ x : ℝ, f x = 18 ∧ ∀ y : ℝ, f y = 18 → x ≤ y :=
by
  sorry

end smallest_x_absolute_value_equation_l1715_171535


namespace john_spent_625_l1715_171508

/-- The amount John spent on his purchases with a coupon -/
def total_spent (vacuum_cost dishwasher_cost coupon_value : ℕ) : ℕ :=
  vacuum_cost + dishwasher_cost - coupon_value

/-- Theorem stating that John spent $625 on his purchases -/
theorem john_spent_625 :
  total_spent 250 450 75 = 625 := by
  sorry

end john_spent_625_l1715_171508


namespace inequality_solution_set_l1715_171569

def solution_set (x : ℝ) : Prop := x < 1/3 ∨ x > 2

theorem inequality_solution_set :
  ∀ x : ℝ, (3*x - 1)/(x - 2) > 0 ↔ solution_set x :=
sorry

end inequality_solution_set_l1715_171569


namespace students_neither_music_nor_art_l1715_171562

theorem students_neither_music_nor_art 
  (total : ℕ) 
  (music : ℕ) 
  (art : ℕ) 
  (both : ℕ) 
  (h1 : total = 500) 
  (h2 : music = 50) 
  (h3 : art = 20) 
  (h4 : both = 10) : 
  total - (music + art - both) = 440 := by
  sorry

end students_neither_music_nor_art_l1715_171562


namespace angle_sum_identity_l1715_171528

theorem angle_sum_identity (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 + 2 * Real.cos α * Real.cos β * Real.cos γ = 1 := by
  sorry

end angle_sum_identity_l1715_171528


namespace builder_total_payment_l1715_171550

/-- Calculates the total amount paid for a purchase of drill bits, hammers, and a toolbox with specific taxes and discounts. -/
def total_amount_paid (drill_bit_sets : ℕ) (drill_bit_price : ℚ) (drill_bit_tax : ℚ)
                      (hammers : ℕ) (hammer_price : ℚ) (hammer_discount : ℚ)
                      (toolbox_price : ℚ) (toolbox_tax : ℚ) : ℚ :=
  let drill_bits_cost := drill_bit_sets * drill_bit_price * (1 + drill_bit_tax)
  let hammers_cost := hammers * hammer_price * (1 - hammer_discount)
  let toolbox_cost := toolbox_price * (1 + toolbox_tax)
  drill_bits_cost + hammers_cost + toolbox_cost

/-- The total amount paid by the builder is $84.55. -/
theorem builder_total_payment :
  total_amount_paid 5 6 (10/100) 3 8 (5/100) 25 (15/100) = 8455/100 := by
  sorry

end builder_total_payment_l1715_171550


namespace business_profit_l1715_171567

def total_subscription : ℕ := 50000
def a_more_than_b : ℕ := 4000
def b_more_than_c : ℕ := 5000
def a_profit : ℕ := 29400

theorem business_profit :
  ∃ (c_subscription : ℕ),
    let b_subscription := c_subscription + b_more_than_c
    let a_subscription := b_subscription + a_more_than_b
    a_subscription + b_subscription + c_subscription = total_subscription →
    (a_profit * total_subscription) / a_subscription = 70000 :=
sorry

end business_profit_l1715_171567


namespace gem_stone_necklaces_count_gem_stone_necklaces_count_proof_l1715_171588

theorem gem_stone_necklaces_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_earnings cost_per_necklace bead_necklaces gem_stone_necklaces =>
    total_earnings = cost_per_necklace * (bead_necklaces + gem_stone_necklaces) →
    cost_per_necklace = 6 →
    bead_necklaces = 3 →
    total_earnings = 36 →
    gem_stone_necklaces = 3

-- Proof
theorem gem_stone_necklaces_count_proof :
  gem_stone_necklaces_count 36 6 3 3 := by
  sorry

end gem_stone_necklaces_count_gem_stone_necklaces_count_proof_l1715_171588


namespace music_club_members_not_playing_l1715_171555

theorem music_club_members_not_playing (total_members guitar_players piano_players both_players : ℕ) 
  (h1 : total_members = 80)
  (h2 : guitar_players = 45)
  (h3 : piano_players = 30)
  (h4 : both_players = 18) :
  total_members - (guitar_players + piano_players - both_players) = 23 := by
  sorry

end music_club_members_not_playing_l1715_171555


namespace length_AC_is_12_l1715_171583

/-- Two circles in a plane with given properties -/
structure TwoCircles where
  A : ℝ × ℝ  -- Center of larger circle
  B : ℝ × ℝ  -- Center of smaller circle
  C : ℝ × ℝ  -- Point on line segment AB
  rA : ℝ     -- Radius of larger circle
  rB : ℝ     -- Radius of smaller circle

/-- The theorem to be proved -/
theorem length_AC_is_12 (circles : TwoCircles)
  (h1 : circles.rA = 12)
  (h2 : circles.rB = 7)
  (h3 : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ circles.C = (1 - t) • circles.A + t • circles.B)
  (h4 : ‖circles.C - circles.B‖ = circles.rB) :
  ‖circles.A - circles.C‖ = 12 :=
sorry

end length_AC_is_12_l1715_171583


namespace sum_and_count_theorem_l1715_171525

def sum_of_integers (a b : ℕ) : ℕ := (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ := (b - a) / 2 + 1

theorem sum_and_count_theorem :
  let x := sum_of_integers 10 20
  let y := count_even_integers 10 20
  x + y = 171 := by sorry

end sum_and_count_theorem_l1715_171525


namespace equal_perimeter_triangles_l1715_171592

theorem equal_perimeter_triangles (a b c x y : ℝ) : 
  a = 7 → b = 12 → c = 9 → x = 2 → y = 7 → x + y = c →
  (a + x + (b - a)) = (b + y + (b - a)) := by sorry

end equal_perimeter_triangles_l1715_171592


namespace geometric_sequence_seventh_term_l1715_171590

/-- Given a geometric sequence of positive integers with first term 3 and sixth term 729,
    prove that the seventh term is 2187. -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) / a n = a 2 / a 1) →  -- Geometric sequence condition
  a 1 = 3 →                             -- First term is 3
  a 6 = 729 →                           -- Sixth term is 729
  (∀ n, a n > 0) →                      -- All terms are positive
  a 7 = 2187 := by
sorry

end geometric_sequence_seventh_term_l1715_171590


namespace boys_playing_both_sports_l1715_171510

theorem boys_playing_both_sports (total : ℕ) (basketball : ℕ) (football : ℕ) (neither : ℕ) :
  total = 30 →
  basketball = 18 →
  football = 21 →
  neither = 4 →
  basketball + football - (total - neither) = 13 := by
  sorry

end boys_playing_both_sports_l1715_171510


namespace leaf_travel_11_gusts_l1715_171501

/-- The net distance traveled by a leaf after a number of wind gusts -/
def leaf_travel (gusts : ℕ) (forward : ℕ) (backward : ℕ) : ℤ :=
  (gusts * forward : ℤ) - (gusts * backward : ℤ)

/-- Theorem: The leaf travels 33 feet after 11 gusts of wind -/
theorem leaf_travel_11_gusts :
  leaf_travel 11 5 2 = 33 := by
  sorry

end leaf_travel_11_gusts_l1715_171501


namespace triangle_side_length_l1715_171529

def isOnParabola (p : ℝ × ℝ) : Prop := p.2 = -(p.1^2)

def isIsoscelesRightTriangle (p q : ℝ × ℝ) : Prop :=
  p.1^2 + p.2^2 = q.1^2 + q.2^2 ∧ p.1 * q.1 + p.2 * q.2 = 0

theorem triangle_side_length
  (p q : ℝ × ℝ)
  (h1 : isOnParabola p)
  (h2 : isOnParabola q)
  (h3 : isIsoscelesRightTriangle p q)
  : Real.sqrt (p.1^2 + p.2^2) = Real.sqrt 2 := by
  sorry

end triangle_side_length_l1715_171529


namespace three_number_problem_l1715_171513

theorem three_number_problem (a b c : ℝ) 
  (sum_30 : a + b + c = 30)
  (first_twice_sum : a = 2 * (b + c))
  (second_five_third : b = 5 * c)
  (sum_first_third : a + c = 22) :
  a * b * c = 2500 / 9 := by
sorry

end three_number_problem_l1715_171513


namespace trig_equation_solution_l1715_171577

theorem trig_equation_solution (x : ℝ) : 
  2 * Real.cos x + Real.cos (3 * x) + Real.cos (5 * x) = 0 →
  ∃ (n : ℤ), x = n * (π / 4) ∧ n % 4 ≠ 0 :=
by sorry

end trig_equation_solution_l1715_171577


namespace ant_final_position_l1715_171539

/-- Represents the position of the ant on a 2D plane -/
structure Position where
  x : Int
  y : Int

/-- Represents the direction the ant is facing -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the state of the ant at any given moment -/
structure AntState where
  pos : Position
  dir : Direction
  moveCount : Nat

/-- The movement function for the ant -/
def move (state : AntState) : AntState :=
  sorry

/-- The main theorem stating the final position of the ant -/
theorem ant_final_position :
  let initial_state : AntState :=
    { pos := { x := -25, y := 25 }
    , dir := Direction.North
    , moveCount := 0
    }
  let final_state := (move^[1010]) initial_state
  final_state.pos = { x := 1491, y := -481 } :=
sorry

end ant_final_position_l1715_171539


namespace y_intercept_not_z_l1715_171520

/-- For a line ax + by - z = 0 where b ≠ 0, the y-intercept is not equal to z -/
theorem y_intercept_not_z (a b z : ℝ) (h : b ≠ 0) :
  ∃ (y_intercept : ℝ), y_intercept = z / b ∧ y_intercept ≠ z := by
  sorry

end y_intercept_not_z_l1715_171520


namespace principal_calculation_l1715_171553

/-- The principal amount in dollars -/
def principal : ℝ := sorry

/-- The compounded amount after 7 years -/
def compounded_amount : ℝ := sorry

/-- The difference between compounded amount and principal -/
def difference : ℝ := 5000

/-- The interest rate for the first 2 years -/
def rate1 : ℝ := 0.03

/-- The interest rate for the next 3 years -/
def rate2 : ℝ := 0.04

/-- The interest rate for the last 2 years -/
def rate3 : ℝ := 0.05

/-- The number of years for each interest rate period -/
def years1 : ℕ := 2
def years2 : ℕ := 3
def years3 : ℕ := 2

theorem principal_calculation :
  principal * (1 + rate1) ^ years1 * (1 + rate2) ^ years2 * (1 + rate3) ^ years3 = principal + difference :=
by sorry

end principal_calculation_l1715_171553


namespace journey_distance_l1715_171504

/-- Represents the journey from John's house to the conference center -/
structure Journey where
  initial_speed : ℝ             -- Initial speed in miles per hour
  initial_distance : ℝ          -- Distance covered in the first hour
  late_time : ℝ                 -- Time he would be late if continued at initial speed
  speed_increase : ℝ            -- Increase in speed for the rest of the journey
  early_time : ℝ                -- Time he arrives early after increasing speed

/-- Calculates the total distance of the journey -/
def calculate_distance (j : Journey) : ℝ :=
  sorry

/-- Theorem stating that the distance to the conference center is 191.25 miles -/
theorem journey_distance (j : Journey) 
  (h1 : j.initial_speed = 45)
  (h2 : j.initial_distance = 45)
  (h3 : j.late_time = 0.75)
  (h4 : j.speed_increase = 20)
  (h5 : j.early_time = 0.25) :
  calculate_distance j = 191.25 :=
sorry

end journey_distance_l1715_171504


namespace probability_less_than_5_is_17_18_l1715_171524

/-- A square in the 2D plane --/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point (x,y) in the given square satisfies x + y < 5 --/
def probabilityLessThan5 (s : Square) : ℝ :=
  sorry

/-- The specific square with vertices (0,0), (0,3), (3,3), and (3,0) --/
def specificSquare : Square :=
  { bottomLeft := (0, 0), sideLength := 3 }

theorem probability_less_than_5_is_17_18 :
  probabilityLessThan5 specificSquare = 17 / 18 :=
sorry

end probability_less_than_5_is_17_18_l1715_171524


namespace laptop_price_proof_l1715_171549

/-- The original sticker price of a laptop -/
def sticker_price : ℝ := 1004

/-- The discount rate at store A -/
def discount_A : ℝ := 0.20

/-- The rebate amount at store A -/
def rebate_A : ℝ := 120

/-- The discount rate at store B -/
def discount_B : ℝ := 0.30

/-- The tax rate applied at both stores -/
def tax_rate : ℝ := 0.07

/-- The price difference between stores A and B -/
def price_difference : ℝ := 21

theorem laptop_price_proof :
  let price_A := (sticker_price * (1 - discount_A) - rebate_A) * (1 + tax_rate)
  let price_B := sticker_price * (1 - discount_B) * (1 + tax_rate)
  price_B - price_A = price_difference :=
by sorry

end laptop_price_proof_l1715_171549


namespace biography_increase_l1715_171548

theorem biography_increase (B : ℝ) (N : ℝ) (h1 : B > 0) (h2 : N > 0) : 
  (0.20 * B + N = 0.32 * (B + N)) → 
  ((N / (0.20 * B)) = 15 / 17) := by
sorry

end biography_increase_l1715_171548


namespace largest_quantity_l1715_171516

theorem largest_quantity (a b c d : ℝ) : 
  (a + 2 = b - 1) ∧ (b - 1 = c + 3) ∧ (c + 3 = d - 4) →
  (d > a) ∧ (d > b) ∧ (d > c) := by
sorry

end largest_quantity_l1715_171516


namespace only_negative_number_l1715_171599

def is_negative (x : ℝ) : Prop := x < 0

theorem only_negative_number (a b c d : ℝ) 
  (ha : a = -2) (hb : b = 0) (hc : c = 1) (hd : d = 3) : 
  is_negative a ∧ ¬is_negative b ∧ ¬is_negative c ∧ ¬is_negative d :=
sorry

end only_negative_number_l1715_171599


namespace complex_equation_solution_l1715_171575

theorem complex_equation_solution (m : ℝ) : 
  let z₁ : ℂ := m^2 - 3*m + m^2*Complex.I
  let z₂ : ℂ := 4 + (5*m + 6)*Complex.I
  z₁ - z₂ = 0 → m = -1 := by
sorry

end complex_equation_solution_l1715_171575


namespace problem_1_l1715_171533

-- Define the sets A and B
def A : Set ℝ := {x | x^2 + 3*x + 2 = 0}
def B (m : ℝ) : Set ℝ := {x | x^2 + (m+1)*x + m = 0}

-- State the theorem
theorem problem_1 (m : ℝ) : (A.compl ∩ B m = ∅) → (m = 1 ∨ m = 2) :=
sorry

end problem_1_l1715_171533


namespace rectangle_sides_l1715_171531

theorem rectangle_sides (area : ℝ) (perimeter : ℝ) : area = 12 ∧ perimeter = 26 →
  ∃ (length width : ℝ), length * width = area ∧ 2 * (length + width) = perimeter ∧
  ((length = 12 ∧ width = 1) ∨ (length = 1 ∧ width = 12)) := by
  sorry

end rectangle_sides_l1715_171531


namespace correct_sample_size_l1715_171526

/-- Represents a school in the sampling problem -/
structure School where
  students : ℕ

/-- Represents the sampling data for two schools -/
structure SamplingData where
  schoolA : School
  schoolB : School
  sampleA : ℕ

/-- Calculates the proportional sample size for the second school -/
def calculateSampleB (data : SamplingData) : ℕ :=
  (data.schoolB.students * data.sampleA) / data.schoolA.students

/-- Theorem stating the correct sample size for School B -/
theorem correct_sample_size (data : SamplingData) 
    (h1 : data.schoolA.students = 800)
    (h2 : data.schoolB.students = 500)
    (h3 : data.sampleA = 48) :
  calculateSampleB data = 30 := by
  sorry

#eval calculateSampleB { 
  schoolA := { students := 800 }, 
  schoolB := { students := 500 }, 
  sampleA := 48 
}

end correct_sample_size_l1715_171526


namespace hexagon_arithmetic_progression_angle_l1715_171540

theorem hexagon_arithmetic_progression_angle (a d : ℝ) :
  (∀ i : Fin 6, 0 ≤ i.val → i.val < 6 → 0 < a + i.val * d) →
  (6 * a + 15 * d = 720) →
  ∃ i : Fin 6, a + i.val * d = 240 :=
sorry

end hexagon_arithmetic_progression_angle_l1715_171540


namespace union_of_M_and_N_l1715_171534

def M : Set ℝ := {x | x^2 - x - 12 = 0}
def N : Set ℝ := {x | x^2 + 3*x = 0}

theorem union_of_M_and_N : M ∪ N = {0, -3, 4} := by
  sorry

end union_of_M_and_N_l1715_171534


namespace triangle_half_angle_sine_product_l1715_171544

theorem triangle_half_angle_sine_product (A B C : ℝ) (h_triangle : A + B + C = π) :
  Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) < 1 / 4 := by sorry

end triangle_half_angle_sine_product_l1715_171544


namespace diamond_equal_forms_intersecting_lines_l1715_171584

/-- The diamond operation -/
def diamond (a b : ℝ) : ℝ := a^3 * b - a * b^3

/-- The set of points (x, y) where x ⋄ y = y ⋄ x -/
def diamond_equal_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | diamond p.1 p.2 = diamond p.2 p.1}

/-- The set of points on the lines y = x and y = -x -/
def intersecting_lines : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = p.1 ∨ p.2 = -p.1}

theorem diamond_equal_forms_intersecting_lines :
  diamond_equal_set = intersecting_lines :=
sorry

end diamond_equal_forms_intersecting_lines_l1715_171584


namespace x_intercepts_count_l1715_171597

theorem x_intercepts_count (x : ℝ) : 
  ∃! x, (x - 5) * (x^2 + x + 1) = 0 :=
by sorry

end x_intercepts_count_l1715_171597


namespace sum_largest_smallest_special_digits_l1715_171554

def largest_three_digit (hundreds : Nat) (ones : Nat) : Nat :=
  hundreds * 100 + 90 + ones

def smallest_three_digit (hundreds : Nat) (ones : Nat) : Nat :=
  hundreds * 100 + ones

theorem sum_largest_smallest_special_digits :
  largest_three_digit 2 7 + smallest_three_digit 2 7 = 504 := by
  sorry

end sum_largest_smallest_special_digits_l1715_171554


namespace distance_difference_l1715_171543

/-- The width of a street in Simplifiedtown -/
def street_width : ℝ := 30

/-- The length of one side of a square block in Simplifiedtown -/
def block_side_length : ℝ := 400

/-- The distance Sarah runs from the block's inner edge -/
def sarah_distance : ℝ := 400

/-- The distance Maude runs from the block's inner edge -/
def maude_distance : ℝ := block_side_length + street_width

/-- The theorem stating the difference in distance run by Maude and Sarah -/
theorem distance_difference :
  4 * maude_distance - 4 * sarah_distance = 120 :=
sorry

end distance_difference_l1715_171543


namespace largest_mersenne_prime_under_200_l1715_171560

-- Define what a Mersenne number is
def mersenne_number (n : ℕ) : ℕ := 2^n - 1

-- Define what a Mersenne prime is
def is_mersenne_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ n : ℕ, Nat.Prime n ∧ p = mersenne_number n

-- Theorem statement
theorem largest_mersenne_prime_under_200 :
  (∀ p : ℕ, is_mersenne_prime p ∧ p < 200 → p ≤ 127) ∧
  is_mersenne_prime 127 := by sorry

end largest_mersenne_prime_under_200_l1715_171560


namespace quadratic_shift_l1715_171570

/-- Represents a quadratic function of the form y = (x + a)² + b -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ

/-- Shifts a quadratic function horizontally -/
def shift_horizontal (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a - d, b := f.b }

/-- Shifts a quadratic function vertically -/
def shift_vertical (f : QuadraticFunction) (d : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b - d }

/-- The main theorem stating that shifting y = (x + 1)² + 3 by 2 units right and 1 unit down
    results in y = (x - 1)² + 2 -/
theorem quadratic_shift :
  let f := QuadraticFunction.mk 1 3
  let g := shift_vertical (shift_horizontal f 2) 1
  g = QuadraticFunction.mk (-1) 2 := by sorry

end quadratic_shift_l1715_171570


namespace group_size_calculation_l1715_171563

theorem group_size_calculation (average_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) :
  average_increase = 6.2 →
  old_weight = 76 →
  new_weight = 119.4 →
  (new_weight - old_weight) / average_increase = 7 :=
by
  sorry

end group_size_calculation_l1715_171563


namespace dennis_initial_money_dennis_initial_money_proof_l1715_171587

/-- Proves that Dennis's initial amount of money equals $50, given the conditions of his purchase and change received. -/
theorem dennis_initial_money : ℕ → Prop :=
  fun initial : ℕ =>
    let shirt_cost : ℕ := 27
    let change_bills : ℕ := 2 * 10
    let change_coins : ℕ := 3
    let total_change : ℕ := change_bills + change_coins
    initial = shirt_cost + total_change ∧ initial = 50

/-- The theorem holds for the specific case where Dennis's initial money is 50. -/
theorem dennis_initial_money_proof : dennis_initial_money 50 := by
  sorry

#check dennis_initial_money
#check dennis_initial_money_proof

end dennis_initial_money_dennis_initial_money_proof_l1715_171587


namespace zainab_hourly_wage_l1715_171578

/-- Zainab's work schedule and earnings -/
structure WorkSchedule where
  daysPerWeek : ℕ
  hoursPerDay : ℕ
  totalWeeks : ℕ
  totalEarnings : ℕ

/-- Calculate hourly wage given a work schedule -/
def hourlyWage (schedule : WorkSchedule) : ℚ :=
  schedule.totalEarnings / (schedule.daysPerWeek * schedule.hoursPerDay * schedule.totalWeeks)

/-- Zainab's specific work schedule -/
def zainabSchedule : WorkSchedule :=
  { daysPerWeek := 3
  , hoursPerDay := 4
  , totalWeeks := 4
  , totalEarnings := 96 }

/-- Theorem: Zainab's hourly wage is $2 -/
theorem zainab_hourly_wage :
  hourlyWage zainabSchedule = 2 := by
  sorry

end zainab_hourly_wage_l1715_171578


namespace division_value_problem_l1715_171547

theorem division_value_problem (x : ℝ) : 
  (1152 / x) - 189 = 3 → x = 6 := by
sorry

end division_value_problem_l1715_171547


namespace asymptotic_lines_of_hyperbola_l1715_171593

/-- The asymptotic lines of the hyperbola x²/9 - y² = 1 are y = ±x/3 -/
theorem asymptotic_lines_of_hyperbola :
  let hyperbola := {(x, y) : ℝ × ℝ | x^2/9 - y^2 = 1}
  let asymptotic_lines := {(x, y) : ℝ × ℝ | y = x/3 ∨ y = -x/3}
  (∀ (p : ℝ × ℝ), p ∈ asymptotic_lines ↔ 
    (∃ (ε : ℝ → ℝ), (∀ t, t ≠ 0 → |ε t| < |t|) ∧
      ∀ t : ℝ, t ≠ 0 → (t*p.1 + ε t, t*p.2 + ε t) ∈ hyperbola)) :=
by sorry

end asymptotic_lines_of_hyperbola_l1715_171593


namespace inheritance_problem_l1715_171572

theorem inheritance_problem (x : ℝ) : 
  (100 + (1/10) * (x - 100) = 200 + (1/10) * (x - (100 + (1/10) * (x - 100)) - 200)) →
  x = 8100 := by
sorry

end inheritance_problem_l1715_171572


namespace sunflower_height_l1715_171527

/-- The height of sunflowers from Packet B in inches -/
def height_B : ℝ := 160

/-- The percentage difference between Packet A and Packet B sunflowers -/
def percentage_difference : ℝ := 0.2

/-- The height of sunflowers from Packet A in inches -/
def height_A : ℝ := height_B * (1 + percentage_difference)

theorem sunflower_height : height_A = 192 := by
  sorry

end sunflower_height_l1715_171527


namespace widget_purchase_l1715_171530

theorem widget_purchase (W : ℝ) (h1 : 6 * W = 8 * (W - 2)) : 6 * W = 48 := by
  sorry

end widget_purchase_l1715_171530


namespace constant_term_binomial_expansion_l1715_171582

theorem constant_term_binomial_expansion :
  let n : ℕ := 8
  let a : ℚ := 5
  let b : ℚ := 2
  (Nat.choose n (n / 2)) * (a ^ (n / 2)) * (b ^ (n / 2)) = 700000 :=
by
  sorry

end constant_term_binomial_expansion_l1715_171582


namespace a_greater_than_c_greater_than_b_l1715_171541

theorem a_greater_than_c_greater_than_b :
  let a := 0.6 * Real.exp 0.4
  let b := 2 - Real.log 4
  let c := Real.exp 1 - 2
  a > c ∧ c > b :=
by sorry

end a_greater_than_c_greater_than_b_l1715_171541


namespace painting_time_equation_l1715_171500

/-- Represents the time (in hours) it takes for a person to paint a room alone -/
structure PaintTime where
  hours : ℝ
  hours_positive : hours > 0

/-- Represents the painting scenario with Doug and Dave -/
structure PaintingScenario where
  doug_time : PaintTime
  dave_time : PaintTime
  doug_start_time : ℝ
  dave_join_time : ℝ
  total_time : ℝ
  doug_start_first : doug_start_time = 0
  dave_joins_later : dave_join_time > doug_start_time

/-- The main theorem stating the equation that the total painting time satisfies -/
theorem painting_time_equation (scenario : PaintingScenario) 
  (h1 : scenario.doug_time.hours = 3)
  (h2 : scenario.dave_time.hours = 4)
  (h3 : scenario.dave_join_time = 1) :
  (scenario.total_time - 1) * (7/12 : ℝ) = 2/3 := by
  sorry

end painting_time_equation_l1715_171500


namespace decimal_to_percentage_l1715_171589

theorem decimal_to_percentage (x : ℚ) : x = 2.08 → (x * 100 : ℚ) = 208 := by
  sorry

end decimal_to_percentage_l1715_171589


namespace safe_opening_l1715_171585

theorem safe_opening (a b c n m k : ℕ) :
  ∃ (x y z : ℕ), ∃ (w : ℕ),
    (a^n * b^m * c^k = w^3) ∨
    (a^n * b^y * c^z = w^3) ∨
    (a^x * b^m * c^z = w^3) ∨
    (a^x * b^y * c^k = w^3) ∨
    (x^n * b^m * c^z = w^3) ∨
    (x^n * b^y * c^k = w^3) ∨
    (x^y * b^m * c^k = w^3) ∨
    (a^x * y^m * c^z = w^3) ∨
    (a^x * y^z * c^k = w^3) ∨
    (x^n * y^m * c^z = w^3) ∨
    (x^n * y^z * c^k = w^3) ∨
    (x^y * z^m * c^k = w^3) ∨
    (a^x * y^m * z^k = w^3) ∨
    (x^n * y^m * z^k = w^3) ∨
    (x^y * b^m * z^k = w^3) ∨
    (x^y * z^m * c^k = w^3) :=
by sorry


end safe_opening_l1715_171585


namespace line_segment_endpoint_l1715_171598

/-- Given a line segment from (1, 3) to (-7, y) with length 12 and y > 0, prove y = 3 + 4√5 -/
theorem line_segment_endpoint (y : ℝ) (h1 : y > 0) 
  (h2 : Real.sqrt (((-7) - 1)^2 + (y - 3)^2) = 12) : y = 3 + 4 * Real.sqrt 5 := by
  sorry

end line_segment_endpoint_l1715_171598


namespace small_box_dimension_l1715_171521

/-- Given a large rectangular box and smaller boxes, proves the dimensions of the smaller boxes. -/
theorem small_box_dimension (large_length large_width large_height : ℕ)
                             (small_length small_height : ℕ)
                             (max_boxes : ℕ)
                             (h1 : large_length = 12)
                             (h2 : large_width = 14)
                             (h3 : large_height = 16)
                             (h4 : small_length = 3)
                             (h5 : small_height = 2)
                             (h6 : max_boxes = 64) :
  ∃ (small_width : ℕ), small_width = 7 ∧
    max_boxes * (small_length * small_width * small_height) = 
    large_length * large_width * large_height :=
by sorry

end small_box_dimension_l1715_171521


namespace arithmetic_sequence_length_l1715_171580

/-- The number of terms in the arithmetic sequence 2.5, 6.5, 10.5, ..., 54.5, 58.5 -/
def sequence_length : ℕ := 15

/-- The first term of the sequence -/
def a₁ : ℚ := 2.5

/-- The last term of the sequence -/
def aₙ : ℚ := 58.5

/-- The common difference of the sequence -/
def d : ℚ := 4

theorem arithmetic_sequence_length :
  sequence_length = (aₙ - a₁) / d + 1 := by sorry

end arithmetic_sequence_length_l1715_171580


namespace necessary_not_sufficient_condition_l1715_171595

theorem necessary_not_sufficient_condition (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + 2 * (1 - m) * x + 3 > 0) →
  (m > 0 ∧ ∃ m' > 0, ¬(∀ x : ℝ, (m' - 1) * x^2 + 2 * (1 - m') * x + 3 > 0)) :=
by sorry

end necessary_not_sufficient_condition_l1715_171595


namespace fraction_problem_l1715_171514

theorem fraction_problem (a b c : ℕ) : 
  a ≤ 10 ∧ b ≤ 10 ∧ c ≤ 10 →
  (3 * a + 2 : ℚ) / 3 = (4 * b + 3 : ℚ) / 4 ∧ 
  (3 * a + 2 : ℚ) / 3 = (5 * c + 3 : ℚ) / 5 →
  (2 * a + b : ℚ) / c = 19 / 4 := by
sorry

#eval (19 : ℚ) / 4  -- This should output 4.75

end fraction_problem_l1715_171514


namespace apple_picking_theorem_l1715_171559

/-- Represents the number of apples of each color in the bin -/
structure AppleBin :=
  (red : ℕ)
  (green : ℕ)
  (yellow : ℕ)
  (blue : ℕ)
  (orange : ℕ)

/-- The minimum number of apples needed to guarantee a specific count of one color -/
def minApplesToGuarantee (bin : AppleBin) (targetCount : ℕ) : ℕ :=
  sorry

theorem apple_picking_theorem (bin : AppleBin) (h : bin = ⟨32, 24, 22, 15, 14⟩) :
  minApplesToGuarantee bin 18 = 81 :=
sorry

end apple_picking_theorem_l1715_171559


namespace sufficient_not_necessary_condition_l1715_171558

theorem sufficient_not_necessary_condition (a : ℝ) :
  (a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  ¬(∃ x : ℝ, a * x^2 + x + 1 ≥ 0 → a ≥ 0) :=
by sorry

end sufficient_not_necessary_condition_l1715_171558


namespace certain_number_is_three_l1715_171532

theorem certain_number_is_three (a b x : ℝ) 
  (h1 : 2 * a = x * b) 
  (h2 : a * b ≠ 0) 
  (h3 : (a / 3) / (b / 2) = 1) : 
  x = 3 := by
sorry

end certain_number_is_three_l1715_171532


namespace cricket_team_left_handed_fraction_l1715_171586

/-- Proves that the fraction of left-handed non-throwers is 1/3 given the conditions of the cricket team -/
theorem cricket_team_left_handed_fraction 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (right_handed : ℕ) 
  (h1 : total_players = 61) 
  (h2 : throwers = 37) 
  (h3 : right_handed = 53) 
  (h4 : throwers ≤ right_handed) : 
  (total_players - throwers - (right_handed - throwers)) / (total_players - throwers) = 1 / 3 :=
by sorry

end cricket_team_left_handed_fraction_l1715_171586


namespace train_length_l1715_171537

/-- The length of a train given its speed and the time it takes to cross a platform -/
theorem train_length (train_speed : ℝ) (platform_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 * 1000 / 3600 →
  platform_length = 270 →
  crossing_time = 26 →
  train_speed * crossing_time - platform_length = 250 := by
  sorry

#check train_length

end train_length_l1715_171537


namespace rental_cost_equality_l1715_171556

/-- The daily rate for Sunshine Car Rentals in dollars -/
def sunshine_daily_rate : ℝ := 17.99

/-- The per-mile rate for Sunshine Car Rentals in dollars -/
def sunshine_mile_rate : ℝ := 0.18

/-- The daily rate for City Rentals in dollars -/
def city_daily_rate : ℝ := 18.95

/-- The per-mile rate for City Rentals in dollars -/
def city_mile_rate : ℝ := 0.16

/-- The mileage at which the cost is the same for both rental companies -/
def equal_cost_mileage : ℝ := 48

theorem rental_cost_equality :
  sunshine_daily_rate + sunshine_mile_rate * equal_cost_mileage =
  city_daily_rate + city_mile_rate * equal_cost_mileage :=
by sorry

end rental_cost_equality_l1715_171556


namespace john_payment_is_1200_l1715_171505

/-- Calculates John's payment for renting a camera -/
def johnPayment (cameraValue : ℝ) (rentalRatePerWeek : ℝ) (rentalWeeks : ℕ) (friendContributionRate : ℝ) : ℝ :=
  let totalRental := cameraValue * rentalRatePerWeek * rentalWeeks
  let friendContribution := totalRental * friendContributionRate
  totalRental - friendContribution

/-- Theorem stating that John's payment is $1200 given the problem conditions -/
theorem john_payment_is_1200 :
  johnPayment 5000 0.1 4 0.4 = 1200 := by
  sorry

end john_payment_is_1200_l1715_171505


namespace zoo_animals_l1715_171507

/-- The number of animals in a zoo satisfies certain conditions. -/
theorem zoo_animals (parrots snakes monkeys elephants zebras : ℕ) :
  parrots = 8 →
  snakes = 3 * parrots →
  monkeys = 2 * snakes →
  elephants = (parrots + snakes) / 2 →
  zebras + 35 = monkeys →
  elephants - zebras = 3 := by
  sorry

end zoo_animals_l1715_171507


namespace farmer_land_ownership_l1715_171564

theorem farmer_land_ownership (total_land : ℝ) : 
  (0.9 * total_land * 0.1 = 540) →
  total_land = 6000 := by
  sorry

end farmer_land_ownership_l1715_171564


namespace turtle_count_l1715_171517

theorem turtle_count (T : ℕ) : 
  (T + (3 * T - 2)) / 2 = 17 → T = 9 := by
  sorry

end turtle_count_l1715_171517


namespace crayons_in_boxes_l1715_171506

/-- Given a number of crayons per box and a number of boxes, 
    calculate the total number of crayons -/
def total_crayons (crayons_per_box : ℕ) (num_boxes : ℕ) : ℕ :=
  crayons_per_box * num_boxes

/-- Theorem stating that with 8 crayons per box and 10 boxes, 
    the total number of crayons is 80 -/
theorem crayons_in_boxes : total_crayons 8 10 = 80 := by
  sorry

end crayons_in_boxes_l1715_171506


namespace min_cube_volume_for_pyramid_l1715_171503

/-- Represents a square-based pyramid -/
structure Pyramid where
  height : ℝ
  baseLength : ℝ

/-- Represents a cube -/
structure Cube where
  sideLength : ℝ

/-- Calculates the volume of a cube -/
def cubeVolume (c : Cube) : ℝ := c.sideLength ^ 3

/-- Checks if a pyramid fits inside a cube -/
def pyramidFitsInCube (p : Pyramid) (c : Cube) : Prop :=
  c.sideLength ≥ p.height ∧ c.sideLength ≥ p.baseLength

theorem min_cube_volume_for_pyramid (p : Pyramid) (h1 : p.height = 18) (h2 : p.baseLength = 15) :
  ∃ (c : Cube), pyramidFitsInCube p c ∧ cubeVolume c = 5832 ∧
  ∀ (c' : Cube), pyramidFitsInCube p c' → cubeVolume c' ≥ 5832 := by
  sorry

end min_cube_volume_for_pyramid_l1715_171503


namespace rectangle_and_parallelogram_area_l1715_171579

-- Define the shapes and their properties
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side ^ 2

structure Circle where
  radius : ℝ

structure Rectangle where
  length : ℝ
  breadth : ℝ
  area : ℝ
  area_eq : area = length * breadth

structure Parallelogram where
  base : ℝ
  height : ℝ
  diagonal : ℝ
  area : ℝ
  area_eq : area = base * height

-- Define the problem
def problem (s : Square) (c : Circle) (r : Rectangle) (p : Parallelogram) : Prop :=
  s.area = 3600 ∧
  s.side = c.radius ∧
  r.length = 2/5 * c.radius ∧
  r.breadth = 10 ∧
  r.breadth = 1/2 * p.diagonal ∧
  p.base = 20 * Real.sqrt 3 ∧
  p.height = r.breadth

-- Theorem to prove
theorem rectangle_and_parallelogram_area 
  (s : Square) (c : Circle) (r : Rectangle) (p : Parallelogram) 
  (h : problem s c r p) : 
  r.area = 240 ∧ p.area = 200 * Real.sqrt 3 := by
  sorry

end rectangle_and_parallelogram_area_l1715_171579


namespace sufficient_but_not_necessary_l1715_171591

theorem sufficient_but_not_necessary : 
  (∀ x : ℝ, x > 3 → x ≥ 3) ∧ 
  (∃ x : ℝ, x ≥ 3 ∧ ¬(x > 3)) := by
  sorry

end sufficient_but_not_necessary_l1715_171591


namespace quadratic_polynomials_sum_nonnegative_l1715_171557

theorem quadratic_polynomials_sum_nonnegative 
  (b c p q m₁ m₂ k₁ k₂ : ℝ) 
  (hf : ∀ x, x^2 + b*x + c = (x - m₁) * (x - m₂))
  (hg : ∀ x, x^2 + p*x + q = (x - k₁) * (x - k₂)) :
  (k₁^2 + b*k₁ + c) + (k₂^2 + b*k₂ + c) + 
  (m₁^2 + p*m₁ + q) + (m₂^2 + p*m₂ + q) ≥ 0 :=
by sorry

end quadratic_polynomials_sum_nonnegative_l1715_171557


namespace intersection_chord_length_l1715_171542

/-- Given a line and a circle that intersect to form a chord of length √3, 
    prove that the parameter 'a' in the circle equation is 0. -/
theorem intersection_chord_length (a : ℝ) : 
  (∃ (x y : ℝ), (8*x - 6*y - 3 = 0) ∧ 
                (x^2 + y^2 - 2*x + a = 0) ∧ 
                (∃ (x' y' : ℝ), (x' ≠ x ∨ y' ≠ y) ∧ 
                                (8*x' - 6*y' - 3 = 0) ∧ 
                                (x'^2 + y'^2 - 2*x' + a = 0) ∧ 
                                ((x - x')^2 + (y - y')^2 = 3))) →
  a = 0 := by
sorry

end intersection_chord_length_l1715_171542


namespace multiply_powers_l1715_171571

theorem multiply_powers (x y : ℝ) : 3 * x^2 * (-2 * x * y^3) = -6 * x^3 * y^3 := by
  sorry

end multiply_powers_l1715_171571


namespace arithmetic_sequence_count_l1715_171512

theorem arithmetic_sequence_count : ∀ (a₁ d aₙ : ℝ) (n : ℕ),
  a₁ = 1.5 ∧ d = 4 ∧ aₙ = 45.5 ∧ aₙ = a₁ + (n - 1) * d →
  n = 12 := by
  sorry

end arithmetic_sequence_count_l1715_171512


namespace original_average_from_doubled_l1715_171536

theorem original_average_from_doubled (n : ℕ) (A : ℚ) (h1 : n = 10) (h2 : 2 * A = 80) : A = 40 := by
  sorry

end original_average_from_doubled_l1715_171536


namespace probability_alternating_colors_is_correct_l1715_171552

def total_balls : ℕ := 12
def white_balls : ℕ := 6
def black_balls : ℕ := 6

def alternating_sequence : List Bool := [true, false, true, false, true, false, true, false, true, false, true, false]

def probability_alternating_colors : ℚ :=
  1 / (total_balls.choose white_balls)

theorem probability_alternating_colors_is_correct :
  probability_alternating_colors = 1 / 924 :=
by sorry

end probability_alternating_colors_is_correct_l1715_171552


namespace rosencrantz_win_probability_value_l1715_171538

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the state of the game -/
inductive GameState
| InProgress
| RosencrantzWins
| GuildensternWins

/-- Represents the game rules -/
def game_rules : List CoinFlip → GameState :=
  sorry

/-- The probability of Rosencrantz winning the game -/
def rosencrantz_win_probability : ℚ :=
  sorry

/-- Theorem stating the probability of Rosencrantz winning -/
theorem rosencrantz_win_probability_value :
  rosencrantz_win_probability = (2^2009 - 1) / (3 * 2^2008 - 1) :=
sorry

end rosencrantz_win_probability_value_l1715_171538


namespace tangent_line_to_circle_l1715_171545

theorem tangent_line_to_circle (r : ℝ) : 
  r > 0 → 
  (∀ x y : ℝ, x^2 + y^2 = r^2 → (x + y = r + 1 → 
    ∀ ε > 0, ∃ δ > 0, ∀ x' y' : ℝ, 
      (x' - x)^2 + (y' - y)^2 < δ^2 → 
      ((x'^2 + y'^2 - r^2) * ((x' + y') - (r + 1)) ≥ 0))) →
  r = 1 + Real.sqrt 2 := by
sorry

end tangent_line_to_circle_l1715_171545


namespace shell_addition_problem_l1715_171551

/-- Calculates the final addition of shells given the initial amount, additions, and removals. -/
def final_addition (initial : ℕ) (first_addition : ℕ) (removal : ℕ) (total : ℕ) : ℕ :=
  total - (initial + first_addition - removal)

/-- Proves that the final addition of shells is 16 pounds given the problem conditions. -/
theorem shell_addition_problem :
  final_addition 5 9 2 28 = 16 := by
  sorry

end shell_addition_problem_l1715_171551


namespace corresponding_angles_equal_if_then_form_l1715_171511

/-- Two angles are corresponding if they occupy the same relative position when a line intersects two other lines. -/
def are_corresponding (α β : Angle) : Prop := sorry

/-- Rewrite the statement "corresponding angles are equal" in if-then form -/
theorem corresponding_angles_equal_if_then_form :
  (∀ α β : Angle, are_corresponding α β → α = β) ↔
  (∀ α β : Angle, are_corresponding α β → α = β) :=
by sorry

end corresponding_angles_equal_if_then_form_l1715_171511


namespace point_coordinates_wrt_origin_l1715_171523

/-- The coordinates of a point (3, -2) with respect to the origin are (3, -2). -/
theorem point_coordinates_wrt_origin :
  let p : ℝ × ℝ := (3, -2)
  p = (3, -2) :=
by
  sorry

end point_coordinates_wrt_origin_l1715_171523


namespace smallest_integer_solution_l1715_171546

theorem smallest_integer_solution (m : ℚ) : 
  (∃ x : ℤ, (3 * (x + 1) - 2 ≤ 4 * (x - 3) + 1) ∧ 
    (∀ y : ℤ, 3 * (y + 1) - 2 ≤ 4 * (y - 3) + 1 → x ≤ y) ∧
    ((1 : ℚ) / 2 * x - m = 5)) → 
  m = 1 := by
sorry

end smallest_integer_solution_l1715_171546


namespace certain_percentage_problem_l1715_171594

theorem certain_percentage_problem (x : ℝ) : x = 12 → (x / 100) * 24.2 = 0.1 * 14.2 + 1.484 := by
  sorry

end certain_percentage_problem_l1715_171594


namespace last_three_digits_of_power_l1715_171561

theorem last_three_digits_of_power (N : ℕ) : 
  N = 2002^2001 → 2003^N ≡ 241 [ZMOD 1000] := by
  sorry

end last_three_digits_of_power_l1715_171561


namespace bird_speed_theorem_l1715_171581

theorem bird_speed_theorem (d t : ℝ) 
  (h1 : d = 40 * (t + 1/20))
  (h2 : d = 60 * (t - 1/20)) :
  d / t = 48 := by
  sorry

end bird_speed_theorem_l1715_171581


namespace classrooms_needed_l1715_171573

/-- Given a school with 390 students and classrooms that hold 30 students each,
    prove that 13 classrooms are needed. -/
theorem classrooms_needed (total_students : Nat) (students_per_classroom : Nat) :
  total_students = 390 →
  students_per_classroom = 30 →
  (total_students + students_per_classroom - 1) / students_per_classroom = 13 := by
  sorry

end classrooms_needed_l1715_171573


namespace square_area_on_parabola_and_line_square_area_on_parabola_and_line_is_36_l1715_171509

theorem square_area_on_parabola_and_line : ℝ → Prop :=
  fun area =>
    ∃ (x₁ x₂ : ℝ),
      -- The endpoints lie on the parabola y = x^2 + 4x + 3
      8 = x₁^2 + 4*x₁ + 3 ∧
      8 = x₂^2 + 4*x₂ + 3 ∧
      -- The side length is the absolute difference between x-coordinates
      area = (x₁ - x₂)^2 ∧
      -- The area of the square is 36
      area = 36

-- The proof of the theorem
theorem square_area_on_parabola_and_line_is_36 :
  square_area_on_parabola_and_line 36 := by
  sorry

end square_area_on_parabola_and_line_square_area_on_parabola_and_line_is_36_l1715_171509


namespace square_sum_equals_90_l1715_171522

theorem square_sum_equals_90 (x y : ℝ) (h1 : x + 3 * y = 6) (h2 : x * y = -9) :
  x^2 + 9 * y^2 = 90 := by
  sorry

end square_sum_equals_90_l1715_171522


namespace ratio_evaluation_l1715_171565

theorem ratio_evaluation : (2^2023 * 3^2025) / 6^2024 = 3/2 := by
  sorry

end ratio_evaluation_l1715_171565


namespace initial_number_of_boys_l1715_171502

theorem initial_number_of_boys (initial_girls : ℕ) (boys_dropped : ℕ) (girls_dropped : ℕ) (remaining_total : ℕ) : 
  initial_girls = 10 →
  boys_dropped = 4 →
  girls_dropped = 3 →
  remaining_total = 17 →
  ∃ initial_boys : ℕ, 
    initial_boys - boys_dropped + (initial_girls - girls_dropped) = remaining_total ∧
    initial_boys = 14 :=
by sorry

end initial_number_of_boys_l1715_171502


namespace ratio_of_linear_system_l1715_171515

theorem ratio_of_linear_system (x y c d : ℝ) (h1 : 3 * x + 2 * y = c) (h2 : 4 * y - 6 * x = d) (h3 : d ≠ 0) :
  c / d = -1 / 3 := by
sorry

end ratio_of_linear_system_l1715_171515


namespace complex_fraction_sum_l1715_171576

theorem complex_fraction_sum (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) : 
  (x^6 + y^6) / (x^6 - y^6) + (x^6 - y^6) / (x^6 + y^6) = 41 / 20 := by
  sorry

end complex_fraction_sum_l1715_171576
