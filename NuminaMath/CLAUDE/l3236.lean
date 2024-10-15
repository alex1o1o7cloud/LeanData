import Mathlib

namespace NUMINAMATH_CALUDE_circles_intersect_l3236_323679

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_O₂ (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 4

-- Define the center and radius of each circle
def center_O₁ : ℝ × ℝ := (0, 0)
def center_O₂ : ℝ × ℝ := (3, 0)
def radius_O₁ : ℝ := 2
def radius_O₂ : ℝ := 2

-- Define the distance between centers
def distance_between_centers : ℝ := 3

-- Theorem stating that the circles intersect
theorem circles_intersect :
  distance_between_centers > abs (radius_O₁ - radius_O₂) ∧
  distance_between_centers < radius_O₁ + radius_O₂ :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l3236_323679


namespace NUMINAMATH_CALUDE_parallel_line_plane_conditions_l3236_323690

-- Define the types for lines and planes
def Line : Type := ℝ → ℝ → ℝ → Prop
def Plane : Type := ℝ → ℝ → ℝ → Prop

-- Define the parallel relation
def parallel (x y : Line) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Define the subset relation for a line in a plane
def line_in_plane (l : Line) (p : Plane) : Prop := sorry

theorem parallel_line_plane_conditions
  (a b : Line) (α : Plane) (h : line_in_plane a α) :
  ¬(∀ (h1 : parallel a b), parallel_line_plane b α) ∧
  ¬(∀ (h2 : parallel_line_plane b α), parallel a b) :=
sorry

end NUMINAMATH_CALUDE_parallel_line_plane_conditions_l3236_323690


namespace NUMINAMATH_CALUDE_yz_circle_radius_l3236_323666

/-- A sphere intersecting two planes -/
structure IntersectingSphere where
  /-- Center of the circle in xy-plane -/
  xy_center : ℝ × ℝ × ℝ
  /-- Radius of the circle in xy-plane -/
  xy_radius : ℝ
  /-- Center of the circle in yz-plane -/
  yz_center : ℝ × ℝ × ℝ

/-- Theorem: The radius of the circle formed by the intersection of the sphere and the yz-plane -/
theorem yz_circle_radius (s : IntersectingSphere) 
  (h_xy : s.xy_center = (3, 5, -2) ∧ s.xy_radius = 3)
  (h_yz : s.yz_center = (-2, 5, 3)) :
  ∃ r : ℝ, r = Real.sqrt 46 ∧ 
  r = Real.sqrt ((Real.sqrt 50 : ℝ) ^ 2 - 2 ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_yz_circle_radius_l3236_323666


namespace NUMINAMATH_CALUDE_exception_pair_of_equations_other_pairs_valid_l3236_323684

theorem exception_pair_of_equations (x : ℝ) : 
  (∃ y, y = x ∧ y = x - 2 ∧ x^2 - 2*x = 0) ↔ False :=
by sorry

theorem other_pairs_valid (x : ℝ) :
  ((∃ y, y = x^2 ∧ y = 2*x ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 2*x ∧ y = 0 ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 2*x + 1 ∧ y = 1 ∧ x^2 - 2*x = 0) ∨
   (∃ y, y = x^2 - 1 ∧ y = 2*x - 1 ∧ x^2 - 2*x = 0)) ↔ True :=
by sorry

end NUMINAMATH_CALUDE_exception_pair_of_equations_other_pairs_valid_l3236_323684


namespace NUMINAMATH_CALUDE_smallest_sum_of_three_primes_l3236_323661

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

def digits_used_once (a b c : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ),
    d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ d1 ≠ d5 ∧ d1 ≠ d6 ∧ d1 ≠ d7 ∧ d1 ≠ d8 ∧ d1 ≠ d9 ∧
    d2 ≠ d3 ∧ d2 ≠ d4 ∧ d2 ≠ d5 ∧ d2 ≠ d6 ∧ d2 ≠ d7 ∧ d2 ≠ d8 ∧ d2 ≠ d9 ∧
    d3 ≠ d4 ∧ d3 ≠ d5 ∧ d3 ≠ d6 ∧ d3 ≠ d7 ∧ d3 ≠ d8 ∧ d3 ≠ d9 ∧
    d4 ≠ d5 ∧ d4 ≠ d6 ∧ d4 ≠ d7 ∧ d4 ≠ d8 ∧ d4 ≠ d9 ∧
    d5 ≠ d6 ∧ d5 ≠ d7 ∧ d5 ≠ d8 ∧ d5 ≠ d9 ∧
    d6 ≠ d7 ∧ d6 ≠ d8 ∧ d6 ≠ d9 ∧
    d7 ≠ d8 ∧ d7 ≠ d9 ∧
    d8 ≠ d9 ∧
    d1 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d2 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d3 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d4 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d5 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d6 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d7 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d8 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    d9 ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] ∧
    a = d1 * 100 + d2 * 10 + d3 ∧
    b = d4 * 100 + d5 * 10 + d6 ∧
    c = d7 * 100 + d8 * 10 + d9

theorem smallest_sum_of_three_primes :
  ∀ a b c : ℕ,
    is_prime a ∧ is_prime b ∧ is_prime c ∧
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    digits_used_once a b c →
    a + b + c ≥ 999 ∧
    (∃ x y z : ℕ, is_prime x ∧ is_prime y ∧ is_prime z ∧
      x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
      digits_used_once x y z ∧
      x + y + z = 999) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_three_primes_l3236_323661


namespace NUMINAMATH_CALUDE_vector_AB_coordinates_and_magnitude_l3236_323642

def OA : ℝ × ℝ := (1, 2)
def OB : ℝ × ℝ := (3, 1)

def AB : ℝ × ℝ := (OB.1 - OA.1, OB.2 - OA.2)

theorem vector_AB_coordinates_and_magnitude :
  AB = (2, -1) ∧ Real.sqrt ((AB.1)^2 + (AB.2)^2) = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_AB_coordinates_and_magnitude_l3236_323642


namespace NUMINAMATH_CALUDE_packets_needed_l3236_323609

/-- Calculates the total number of packets needed for seedlings --/
def total_packets (oak_seedlings maple_seedlings pine_seedlings : ℕ) 
                  (oak_per_packet maple_per_packet pine_per_packet : ℕ) : ℕ :=
  (oak_seedlings / oak_per_packet) + 
  (maple_seedlings / maple_per_packet) + 
  (pine_seedlings / pine_per_packet)

/-- Theorem stating that the total number of packets needed is 395 --/
theorem packets_needed : 
  total_packets 420 825 2040 7 5 12 = 395 := by
  sorry

#eval total_packets 420 825 2040 7 5 12

end NUMINAMATH_CALUDE_packets_needed_l3236_323609


namespace NUMINAMATH_CALUDE_last_box_contents_l3236_323654

-- Define the total number of bars for each type of chocolate
def total_A : ℕ := 853845
def total_B : ℕ := 537896
def total_C : ℕ := 729763

-- Define the box capacity for each type of chocolate
def capacity_A : ℕ := 9
def capacity_B : ℕ := 11
def capacity_C : ℕ := 15

-- Theorem to prove the number of bars in the last partially filled box for each type
theorem last_box_contents :
  (total_A % capacity_A = 4) ∧
  (total_B % capacity_B = 3) ∧
  (total_C % capacity_C = 8) := by
  sorry

end NUMINAMATH_CALUDE_last_box_contents_l3236_323654


namespace NUMINAMATH_CALUDE_rectangle_shorter_side_l3236_323663

/-- A rectangle with perimeter 60 and area 221 has a shorter side of length 13 -/
theorem rectangle_shorter_side : ∃ (x y : ℝ), 
  x > 0 ∧ y > 0 ∧ 
  2 * x + 2 * y = 60 ∧ 
  x * y = 221 ∧ 
  min x y = 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_shorter_side_l3236_323663


namespace NUMINAMATH_CALUDE_greatest_piece_length_l3236_323629

theorem greatest_piece_length (a b c : ℕ) (ha : a = 45) (hb : b = 75) (hc : c = 90) :
  Nat.gcd a (Nat.gcd b c) = 15 := by
  sorry

end NUMINAMATH_CALUDE_greatest_piece_length_l3236_323629


namespace NUMINAMATH_CALUDE_space_shuttle_speed_km_per_second_l3236_323602

def orbit_speed_km_per_hour : ℝ := 43200

theorem space_shuttle_speed_km_per_second :
  orbit_speed_km_per_hour / 3600 = 12 := by
  sorry

end NUMINAMATH_CALUDE_space_shuttle_speed_km_per_second_l3236_323602


namespace NUMINAMATH_CALUDE_perfect_square_binomial_l3236_323659

theorem perfect_square_binomial (a b : ℝ) : ∃ (x : ℝ), a^2 + 2*a*b + b^2 = x^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_binomial_l3236_323659


namespace NUMINAMATH_CALUDE_min_value_theorem_l3236_323644

theorem min_value_theorem (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_cond : x + y + z + w = 2)
  (prod_cond : x * y * z * w = 1/16) :
  (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 → 
    a + b + c + d = 2 → a * b * c * d = 1/16 → 
    (x + y + z) / (x * y * z * w) ≤ (a + b + c) / (a * b * c * d)) →
  (x + y + z) / (x * y * z * w) = 24 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3236_323644


namespace NUMINAMATH_CALUDE_second_circle_radius_l3236_323636

/-- Two circles are externally tangent, with one having radius 2. Their common tangent intersects
    another common tangent at a point 4 units away from the point of tangency. -/
structure TangentCircles where
  r : ℝ
  R : ℝ
  tangent_length : ℝ
  h_r : r = 2
  h_tangent : tangent_length = 4

/-- The radius of the second circle is 8. -/
theorem second_circle_radius (tc : TangentCircles) : tc.R = 8 := by sorry

end NUMINAMATH_CALUDE_second_circle_radius_l3236_323636


namespace NUMINAMATH_CALUDE_bird_count_difference_l3236_323612

theorem bird_count_difference (monday_count tuesday_count wednesday_count : ℕ) : 
  monday_count = 70 →
  tuesday_count = monday_count / 2 →
  monday_count + tuesday_count + wednesday_count = 148 →
  wednesday_count - tuesday_count = 8 := by
sorry

end NUMINAMATH_CALUDE_bird_count_difference_l3236_323612


namespace NUMINAMATH_CALUDE_min_value_2m_plus_n_solution_set_f_gt_5_l3236_323669

-- Define the function f
def f (x m n : ℝ) : ℝ := |x + m| + |2*x - n|

-- Theorem for part I
theorem min_value_2m_plus_n (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, f x m n ≥ 1) → 2*m + n ≥ 2 :=
sorry

-- Theorem for part II
theorem solution_set_f_gt_5 :
  {x : ℝ | f x 2 3 > 5} = {x : ℝ | x < 0 ∨ x > 2} :=
sorry

end NUMINAMATH_CALUDE_min_value_2m_plus_n_solution_set_f_gt_5_l3236_323669


namespace NUMINAMATH_CALUDE_min_box_value_l3236_323649

theorem min_box_value (a b Box : ℤ) :
  (∀ x, (a * x + b) * (b * x + a) = 26 * x^2 + Box * x + 26) →
  a ≠ b ∧ b ≠ Box ∧ a ≠ Box →
  (∃ a' b' Box' : ℤ, 
    (∀ x, (a' * x + b') * (b' * x + a') = 26 * x^2 + Box' * x + 26) ∧
    a' ≠ b' ∧ b' ≠ Box' ∧ a' ≠ Box' ∧
    Box' < Box) →
  Box ≥ 173 :=
by sorry

end NUMINAMATH_CALUDE_min_box_value_l3236_323649


namespace NUMINAMATH_CALUDE_existence_of_unsolvable_linear_system_l3236_323641

theorem existence_of_unsolvable_linear_system :
  ∃ (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ),
    (∀ x y : ℝ, a₁ * x + b₁ * y ≠ c₁ ∨ a₂ * x + b₂ * y ≠ c₂) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_unsolvable_linear_system_l3236_323641


namespace NUMINAMATH_CALUDE_smallest_y_with_remainders_l3236_323606

theorem smallest_y_with_remainders : ∃! y : ℕ, 
  y > 0 ∧ 
  y % 6 = 5 ∧ 
  y % 7 = 6 ∧ 
  y % 8 = 7 ∧
  ∀ z : ℕ, z > 0 ∧ z % 6 = 5 ∧ z % 7 = 6 ∧ z % 8 = 7 → y ≤ z :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_y_with_remainders_l3236_323606


namespace NUMINAMATH_CALUDE_boys_candies_order_independent_l3236_323622

/-- Represents a child's gender -/
inductive Gender
| Boy
| Girl

/-- Represents a child with their gender -/
structure Child where
  gender : Gender

/-- Represents the state of the candy distribution process -/
structure CandyState where
  remaining_candies : ℕ
  remaining_children : List Child

/-- Represents the result of a candy distribution process -/
structure DistributionResult where
  boys_candies : ℕ
  girls_candies : ℕ

/-- Function to distribute candies according to the rules -/
def distributeCandies (initial_state : CandyState) : DistributionResult :=
  sorry

/-- Theorem stating that the number of candies taken by boys is independent of the order -/
theorem boys_candies_order_independent
  (children : List Child)
  (perm : List Child)
  (h : perm.Perm children) :
  (distributeCandies { remaining_candies := 2021, remaining_children := children }).boys_candies =
  (distributeCandies { remaining_candies := 2021, remaining_children := perm }).boys_candies :=
  sorry

end NUMINAMATH_CALUDE_boys_candies_order_independent_l3236_323622


namespace NUMINAMATH_CALUDE_expression_simplification_l3236_323628

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2 + 1) :
  (1 - 1 / (m + 1)) * ((m^2 - 1) / m) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l3236_323628


namespace NUMINAMATH_CALUDE_triangle_isosceles_if_bcosC_eq_CcosB_l3236_323600

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the cosine of an angle in a triangle
def cos_angle (t : Triangle) (angle : Fin 3) : ℝ :=
  sorry

-- Define the length of a side in a triangle
def side_length (t : Triangle) (side : Fin 3) : ℝ :=
  sorry

-- Define an isosceles triangle
def is_isosceles (t : Triangle) : Prop :=
  sorry

-- Theorem statement
theorem triangle_isosceles_if_bcosC_eq_CcosB (t : Triangle) :
  side_length t 1 * cos_angle t 2 = side_length t 2 * cos_angle t 1 →
  is_isosceles t :=
sorry

end NUMINAMATH_CALUDE_triangle_isosceles_if_bcosC_eq_CcosB_l3236_323600


namespace NUMINAMATH_CALUDE_problem_statement_l3236_323667

theorem problem_statement (x y : ℝ) (h1 : x + y = 7) (h2 : x * y = 10) : 3 * x^2 + 3 * y^2 = 87 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3236_323667


namespace NUMINAMATH_CALUDE_rhombus_inscribed_circle_area_ratio_l3236_323656

theorem rhombus_inscribed_circle_area_ratio (d₁ d₂ : ℝ) (h : d₁ / d₂ = 3 / 4) :
  let r := d₁ * d₂ / (2 * Real.sqrt ((d₁/2)^2 + (d₂/2)^2))
  (d₁ * d₂ / 2) / (π * r^2) = 25 / (6 * π) := by
  sorry

end NUMINAMATH_CALUDE_rhombus_inscribed_circle_area_ratio_l3236_323656


namespace NUMINAMATH_CALUDE_median_and_mode_are_23_l3236_323676

/-- Represents a shoe size distribution --/
structure ShoeSizeDistribution where
  sizes : List Nat
  frequencies : List Nat
  total_students : Nat

/-- Calculates the median of a shoe size distribution --/
def median (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- Calculates the mode of a shoe size distribution --/
def mode (dist : ShoeSizeDistribution) : Nat :=
  sorry

/-- The given shoe size distribution --/
def class_distribution : ShoeSizeDistribution :=
  { sizes := [20, 21, 22, 23, 24],
    frequencies := [2, 8, 9, 19, 2],
    total_students := 40 }

theorem median_and_mode_are_23 :
  median class_distribution = 23 ∧ mode class_distribution = 23 := by
  sorry

end NUMINAMATH_CALUDE_median_and_mode_are_23_l3236_323676


namespace NUMINAMATH_CALUDE_prime_square_sum_equation_l3236_323693

theorem prime_square_sum_equation :
  ∀ (a b c k : ℕ),
    Prime a ∧ Prime b ∧ Prime c ∧ k > 0 ∧
    a^2 + b^2 + 16*c^2 = 9*k^2 + 1 →
    ((a = 3 ∧ b = 3 ∧ c = 2 ∧ k = 3) ∨
     (a = 3 ∧ b = 37 ∧ c = 3 ∧ k = 13) ∨
     (a = 37 ∧ b = 3 ∧ c = 3 ∧ k = 13) ∨
     (a = 3 ∧ b = 17 ∧ c = 3 ∧ k = 7) ∨
     (a = 17 ∧ b = 3 ∧ c = 3 ∧ k = 7)) :=
by
  sorry

end NUMINAMATH_CALUDE_prime_square_sum_equation_l3236_323693


namespace NUMINAMATH_CALUDE_max_queens_8x8_l3236_323681

/-- Represents a chessboard configuration -/
def ChessBoard := Fin 8 → Fin 8

/-- Checks if two positions are on the same diagonal -/
def onSameDiagonal (p1 p2 : Fin 8 × Fin 8) : Prop :=
  (p1.1 : ℤ) - (p2.1 : ℤ) = (p1.2 : ℤ) - (p2.2 : ℤ) ∨
  (p1.1 : ℤ) - (p2.1 : ℤ) = (p2.2 : ℤ) - (p1.2 : ℤ)

/-- Checks if a chessboard configuration is valid (no queens attack each other) -/
def isValidConfiguration (board : ChessBoard) : Prop :=
  ∀ i j : Fin 8, i ≠ j →
    board i ≠ board j ∧
    ¬onSameDiagonal (i, board i) (j, board j)

/-- The theorem stating that the maximum number of non-attacking queens on an 8x8 chessboard is 8 -/
theorem max_queens_8x8 :
  (∃ (board : ChessBoard), isValidConfiguration board) ∧
  (∀ (n : ℕ) (f : Fin n → Fin 8 × Fin 8),
    (∀ i j : Fin n, i ≠ j → f i ≠ f j ∧ ¬onSameDiagonal (f i) (f j)) →
    n ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_max_queens_8x8_l3236_323681


namespace NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3236_323638

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 56 ways to distribute 5 indistinguishable balls into 4 distinguishable boxes -/
theorem distribute_five_balls_four_boxes :
  distribute_balls 5 4 = 56 := by
  sorry

end NUMINAMATH_CALUDE_distribute_five_balls_four_boxes_l3236_323638


namespace NUMINAMATH_CALUDE_total_dimes_l3236_323639

-- Define the initial number of dimes Melanie had
def initial_dimes : Nat := 19

-- Define the number of dimes given by her dad
def dimes_from_dad : Nat := 39

-- Define the number of dimes given by her mother
def dimes_from_mom : Nat := 25

-- Theorem to prove the total number of dimes
theorem total_dimes : 
  initial_dimes + dimes_from_dad + dimes_from_mom = 83 := by
  sorry

end NUMINAMATH_CALUDE_total_dimes_l3236_323639


namespace NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l3236_323670

theorem sum_of_sqrt_inequality (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 1) : 
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) > 2 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sqrt_inequality_l3236_323670


namespace NUMINAMATH_CALUDE_circle_equation_k_value_l3236_323665

theorem circle_equation_k_value (x y k : ℝ) : 
  (∀ x y, x^2 + 8*x + y^2 + 14*y - k = 0 ↔ (x + 4)^2 + (y + 7)^2 = 25) → 
  k = -40 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_k_value_l3236_323665


namespace NUMINAMATH_CALUDE_absolute_value_equality_implies_midpoint_l3236_323617

theorem absolute_value_equality_implies_midpoint (x : ℚ) :
  |x - 2| = |x - 5| → x = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equality_implies_midpoint_l3236_323617


namespace NUMINAMATH_CALUDE_no_solution_to_system_l3236_323650

theorem no_solution_to_system :
  ¬ ∃ (x y z : ℝ), 
    (x^2 - 3*x*y + 2*y^2 - z^2 = 31) ∧
    (-x^2 + 6*y*z + 2*z^2 = 44) ∧
    (x^2 + x*y + 8*z^2 = 100) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_system_l3236_323650


namespace NUMINAMATH_CALUDE_function_increasing_implies_a_leq_neg_two_l3236_323631

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 3

-- State the theorem
theorem function_increasing_implies_a_leq_neg_two :
  ∀ a : ℝ, (∀ x y : ℝ, -2 < x ∧ x < y ∧ y < 2 → f a x < f a y) → a ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_function_increasing_implies_a_leq_neg_two_l3236_323631


namespace NUMINAMATH_CALUDE_gambler_initial_games_gambler_initial_games_proof_l3236_323608

theorem gambler_initial_games : ℝ → Prop :=
  fun x =>
    let initial_win_rate : ℝ := 0.4
    let new_win_rate : ℝ := 0.8
    let additional_games : ℝ := 30
    let final_win_rate : ℝ := 0.6
    (initial_win_rate * x + new_win_rate * additional_games) / (x + additional_games) = final_win_rate →
    x = 30

theorem gambler_initial_games_proof : ∃ x : ℝ, gambler_initial_games x := by
  sorry

end NUMINAMATH_CALUDE_gambler_initial_games_gambler_initial_games_proof_l3236_323608


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3236_323699

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  ArithmeticSequence a → a 2 = 5 → a 6 = 33 → a 3 + a 5 = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3236_323699


namespace NUMINAMATH_CALUDE_bd_length_l3236_323605

-- Define the triangle ABC
structure Triangle (α : Type*) [NormedAddCommGroup α] [InnerProductSpace ℝ α] :=
  (A B C : α)

-- Define the properties of the triangle
def IsoscelesTriangle {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) : Prop :=
  ‖t.A - t.C‖ = ‖t.B - t.C‖

-- Define point D on AB
def PointOnLine {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (A B D : α) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • A + t • B

-- Main theorem
theorem bd_length {α : Type*} [NormedAddCommGroup α] [InnerProductSpace ℝ α] (t : Triangle α) (D : α) :
  IsoscelesTriangle t →
  PointOnLine t.A t.B D →
  ‖t.A - t.C‖ = 10 →
  ‖t.A - D‖ = 12 →
  ‖t.C - D‖ = 4 →
  ‖t.B - D‖ = 7 := by
  sorry


end NUMINAMATH_CALUDE_bd_length_l3236_323605


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_sum_l3236_323625

theorem equilateral_triangle_area_sum : 
  let triangle1_side : ℝ := 2
  let triangle2_side : ℝ := 3
  let new_triangle_side : ℝ := Real.sqrt 13
  let area (side : ℝ) : ℝ := (Real.sqrt 3 / 4) * side^2
  area new_triangle_side = area triangle1_side + area triangle2_side :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_sum_l3236_323625


namespace NUMINAMATH_CALUDE_shiela_paper_stars_l3236_323682

/-- The number of classmates Shiela has -/
def num_classmates : ℕ := 9

/-- The number of stars Shiela places in each bottle -/
def stars_per_bottle : ℕ := 5

/-- The total number of paper stars Shiela prepared -/
def total_stars : ℕ := num_classmates * stars_per_bottle

/-- Theorem stating that the total number of paper stars Shiela prepared is 45 -/
theorem shiela_paper_stars : total_stars = 45 := by
  sorry

end NUMINAMATH_CALUDE_shiela_paper_stars_l3236_323682


namespace NUMINAMATH_CALUDE_polynomial_expansion_l3236_323618

theorem polynomial_expansion (s : ℝ) :
  (3 * s^3 - 4 * s^2 + 5 * s - 2) * (2 * s^2 - 3 * s + 4) =
  6 * s^5 - 17 * s^4 + 34 * s^3 - 35 * s^2 + 26 * s - 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l3236_323618


namespace NUMINAMATH_CALUDE_smallest_absolute_value_l3236_323610

theorem smallest_absolute_value : ∀ x : ℝ, |0| ≤ |x| := by
  sorry

end NUMINAMATH_CALUDE_smallest_absolute_value_l3236_323610


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3236_323620

/-- The line equation passing through a fixed point for all values of parameter a -/
def line_equation (a x y : ℝ) : Prop :=
  (a - 1) * x - y + 2 * a + 1 = 0

/-- Theorem stating that the line passes through the point (-2, 3) for all values of a -/
theorem line_passes_through_fixed_point :
  ∀ a : ℝ, line_equation a (-2) 3 := by
sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l3236_323620


namespace NUMINAMATH_CALUDE_transportation_budget_degrees_l3236_323619

theorem transportation_budget_degrees (salaries research_dev utilities equipment supplies : ℝ)
  (h1 : salaries = 60)
  (h2 : research_dev = 9)
  (h3 : utilities = 5)
  (h4 : equipment = 4)
  (h5 : supplies = 2)
  (h6 : salaries + research_dev + utilities + equipment + supplies < 100) :
  let transportation := 100 - (salaries + research_dev + utilities + equipment + supplies)
  360 * (transportation / 100) = 72 := by
  sorry

end NUMINAMATH_CALUDE_transportation_budget_degrees_l3236_323619


namespace NUMINAMATH_CALUDE_range_of_expression_l3236_323688

-- Define the conditions
def condition1 (x y : ℝ) : Prop := -1 < x + y ∧ x + y < 4
def condition2 (x y : ℝ) : Prop := 2 < x - y ∧ x - y < 3

-- Define the expression we're interested in
def expression (x y : ℝ) : ℝ := 3*x + 2*y

-- State the theorem
theorem range_of_expression (x y : ℝ) 
  (h1 : condition1 x y) (h2 : condition2 x y) :
  -3/2 < expression x y ∧ expression x y < 23/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3236_323688


namespace NUMINAMATH_CALUDE_milk_sharing_l3236_323694

theorem milk_sharing (don_milk : ℚ) (rachel_portion : ℚ) (rachel_milk : ℚ) : 
  don_milk = 3 / 7 → 
  rachel_portion = 1 / 2 → 
  rachel_milk = rachel_portion * don_milk → 
  rachel_milk = 3 / 14 := by
sorry

end NUMINAMATH_CALUDE_milk_sharing_l3236_323694


namespace NUMINAMATH_CALUDE_constant_in_toll_formula_l3236_323689

/-- The toll formula for a truck using a certain bridge -/
def toll_formula (constant : ℝ) (x : ℕ) : ℝ :=
  1.50 + constant * (x - 2)

/-- The number of axles on an 18-wheel truck -/
def axles_18_wheel_truck : ℕ := 9

/-- The toll for an 18-wheel truck -/
def toll_18_wheel_truck : ℝ := 5

theorem constant_in_toll_formula :
  ∃ (constant : ℝ), 
    toll_formula constant axles_18_wheel_truck = toll_18_wheel_truck ∧ 
    constant = 0.50 := by
  sorry

end NUMINAMATH_CALUDE_constant_in_toll_formula_l3236_323689


namespace NUMINAMATH_CALUDE_line_AB_passes_through_fixed_point_l3236_323653

-- Define the hyperbola D
def hyperbolaD (x y : ℝ) : Prop := y^2/2 - x^2 = 1/3

-- Define the parabola C
def parabolaC (x y : ℝ) : Prop := x^2 = 4*y

-- Define the point P on parabola C
def P : ℝ × ℝ := (2, 1)

-- Define a point on parabola C
def pointOnParabolaC (x y : ℝ) : Prop := parabolaC x y

-- Define the perpendicular condition for PA and PB
def perpendicularCondition (x1 y1 x2 y2 : ℝ) : Prop :=
  ((y1 - 1) / (x1 - 2)) * ((y2 - 1) / (x2 - 2)) = -1

-- The main theorem
theorem line_AB_passes_through_fixed_point :
  ∀ (x1 y1 x2 y2 : ℝ),
  pointOnParabolaC x1 y1 →
  pointOnParabolaC x2 y2 →
  perpendicularCondition x1 y1 x2 y2 →
  ∃ (t : ℝ), t ∈ (Set.Icc 0 1) ∧ 
  (t * x1 + (1 - t) * x2 = -2) ∧
  (t * y1 + (1 - t) * y2 = 5) :=
sorry

end NUMINAMATH_CALUDE_line_AB_passes_through_fixed_point_l3236_323653


namespace NUMINAMATH_CALUDE_min_tiles_for_floor_l3236_323640

/-- Represents the dimensions of a rectangular shape in inches -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Calculates the area of a rectangular shape given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Calculates the number of tiles needed to cover a floor -/
def tilesNeeded (floorDim : Dimensions) (tileDim : Dimensions) : ℕ :=
  (area floorDim) / (area tileDim)

theorem min_tiles_for_floor : 
  let tileDim : Dimensions := ⟨3, 4⟩
  let floorDimFeet : Dimensions := ⟨2, 5⟩
  let floorDimInches : Dimensions := ⟨feetToInches floorDimFeet.length, feetToInches floorDimFeet.width⟩
  tilesNeeded floorDimInches tileDim = 120 := by
  sorry

end NUMINAMATH_CALUDE_min_tiles_for_floor_l3236_323640


namespace NUMINAMATH_CALUDE_linear_function_composition_l3236_323672

/-- A linear function from ℝ to ℝ -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x + b

theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x - 1) →
  (∀ x, f x = 2 * x - 1/3) ∨ (∀ x, f x = -2 * x + 1) :=
by sorry

end NUMINAMATH_CALUDE_linear_function_composition_l3236_323672


namespace NUMINAMATH_CALUDE_rectangle_area_l3236_323668

theorem rectangle_area (k : ℕ+) : 
  let square_side : ℝ := (16 : ℝ).sqrt
  let rectangle_length : ℝ := k * square_side
  let rectangle_breadth : ℝ := 11
  rectangle_length * rectangle_breadth = 220 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l3236_323668


namespace NUMINAMATH_CALUDE_weight_distribution_problem_l3236_323615

theorem weight_distribution_problem :
  ∃! (a b c : ℕ), a + b + c = 100 ∧ a + 10 * b + 50 * c = 500 ∧ (a, b, c) = (60, 39, 1) := by
  sorry

end NUMINAMATH_CALUDE_weight_distribution_problem_l3236_323615


namespace NUMINAMATH_CALUDE_projection_problem_l3236_323674

/-- Given a projection that takes (3, 6) to (9/5, 18/5), prove that it takes (1, -1) to (-1/5, -2/5) -/
theorem projection_problem (proj : ℝ × ℝ → ℝ × ℝ) 
  (h : proj (3, 6) = (9/5, 18/5)) : 
  proj (1, -1) = (-1/5, -2/5) := by
  sorry

end NUMINAMATH_CALUDE_projection_problem_l3236_323674


namespace NUMINAMATH_CALUDE_win_sector_area_l3236_323616

/-- Given a circular spinner with radius 15 cm and a probability of winning of 1/3,
    the area of the WIN sector is 75π square centimeters. -/
theorem win_sector_area (radius : ℝ) (win_prob : ℝ) (win_area : ℝ) : 
  radius = 15 → 
  win_prob = 1/3 → 
  win_area = win_prob * π * radius^2 →
  win_area = 75 * π := by
sorry

end NUMINAMATH_CALUDE_win_sector_area_l3236_323616


namespace NUMINAMATH_CALUDE_insurance_payment_percentage_l3236_323647

theorem insurance_payment_percentage
  (total_cost : ℝ)
  (individual_payment_percentage : ℝ)
  (individual_payment : ℝ)
  (h1 : total_cost = 110000)
  (h2 : individual_payment_percentage = 20)
  (h3 : individual_payment = 22000)
  (h4 : individual_payment = (individual_payment_percentage / 100) * total_cost) :
  100 - individual_payment_percentage = 80 := by
  sorry

end NUMINAMATH_CALUDE_insurance_payment_percentage_l3236_323647


namespace NUMINAMATH_CALUDE_even_numbers_average_21_l3236_323621

/-- The sum of the first n even numbers -/
def sumFirstEvenNumbers (n : ℕ) : ℕ := n * (n + 1)

/-- The average of the first n even numbers -/
def averageFirstEvenNumbers (n : ℕ) : ℚ := (sumFirstEvenNumbers n : ℚ) / n

theorem even_numbers_average_21 :
  ∃ n : ℕ, n > 0 ∧ averageFirstEvenNumbers n = 21 :=
sorry

end NUMINAMATH_CALUDE_even_numbers_average_21_l3236_323621


namespace NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3236_323675

theorem count_integers_satisfying_inequality :
  ∃! (S : Finset ℤ), (∀ n ∈ S, (3 * (n - 1) * (n + 5) : ℤ) < 0) ∧ S.card = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_satisfying_inequality_l3236_323675


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3236_323601

theorem inequality_solution_set (x : ℝ) :
  (x - 1) * (x + 2) < 0 ↔ -2 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3236_323601


namespace NUMINAMATH_CALUDE_first_set_cost_l3236_323678

/-- The cost of a football in dollars -/
def football_cost : ℝ := 35

/-- The cost of a soccer ball in dollars -/
def soccer_cost : ℝ := 50

/-- The cost of 2 footballs and 3 soccer balls in dollars -/
def two_footballs_three_soccer_cost : ℝ := 220

theorem first_set_cost : 3 * football_cost + soccer_cost = 155 :=
  by sorry

end NUMINAMATH_CALUDE_first_set_cost_l3236_323678


namespace NUMINAMATH_CALUDE_line_equation_l3236_323643

/-- Given a line with slope -2 and y-intercept 4, its equation is 2x+y-4=0 -/
theorem line_equation (x y : ℝ) : 
  (∃ (m b : ℝ), m = -2 ∧ b = 4 ∧ y = m * x + b) → 2 * x + y - 4 = 0 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_l3236_323643


namespace NUMINAMATH_CALUDE_leak_emptying_time_l3236_323673

theorem leak_emptying_time (fill_time_no_leak fill_time_with_leak : ℝ) 
  (h1 : fill_time_no_leak = 8)
  (h2 : fill_time_with_leak = 12) :
  let fill_rate := 1 / fill_time_no_leak
  let combined_rate := 1 / fill_time_with_leak
  let leak_rate := fill_rate - combined_rate
  24 = 1 / leak_rate := by
sorry

end NUMINAMATH_CALUDE_leak_emptying_time_l3236_323673


namespace NUMINAMATH_CALUDE_sequences_theorem_l3236_323651

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

def geometric_sequence (b : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, q > 0 ∧ ∀ n : ℕ, b (n + 1) = q * b n

def sum_arithmetic (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n : ℚ) * (a 1 + a n) / 2

theorem sequences_theorem (a b : ℕ → ℚ) :
  arithmetic_sequence a →
  geometric_sequence b →
  b 1 = 2 →
  b 2 + b 3 = 12 →
  b 3 = a 4 - 2 * a 1 →
  sum_arithmetic a 11 = 11 * b 4 →
  (∀ n : ℕ, n > 0 → a n = 3 * n - 2) ∧
  (∀ n : ℕ, n > 0 → b n = 2^n) ∧
  (∀ n : ℕ, n > 0 → 
    (Finset.range n).sum (λ i => a (2 * (i + 1)) * b (2 * i + 1)) = 
      (3 * n - 2) / 3 * 4^(n + 1) + 8 / 3) := by
  sorry

end NUMINAMATH_CALUDE_sequences_theorem_l3236_323651


namespace NUMINAMATH_CALUDE_triangle_area_l3236_323630

theorem triangle_area (A B C : Real) (a b c : Real) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  A = π/3 ∧
  c = 4 ∧
  b = 2 * Real.sqrt 3 →
  (1/2) * b * c * Real.sin A = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3236_323630


namespace NUMINAMATH_CALUDE_decimal_representation_of_fraction_l3236_323652

theorem decimal_representation_of_fraction (n d : ℕ) (h : d ≠ 0) :
  (n : ℚ) / d = 0.36 ↔ n = 9 ∧ d = 25 :=
sorry

end NUMINAMATH_CALUDE_decimal_representation_of_fraction_l3236_323652


namespace NUMINAMATH_CALUDE_train_speed_l3236_323603

/-- Proves that a train crossing a bridge has a specific speed -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) :
  train_length = 110 ∧
  bridge_length = 112 ∧
  crossing_time = 11.099112071034318 →
  (train_length + bridge_length) / crossing_time * 3.6 = 72 := by
sorry

end NUMINAMATH_CALUDE_train_speed_l3236_323603


namespace NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3236_323632

theorem complex_number_in_first_quadrant 
  (m n : ℝ) 
  (h : (m : ℂ) / (1 + Complex.I) = 1 - n * Complex.I) : 
  m > 0 ∧ n > 0 := by
sorry

end NUMINAMATH_CALUDE_complex_number_in_first_quadrant_l3236_323632


namespace NUMINAMATH_CALUDE_weight_ratio_proof_l3236_323655

/-- Prove the ratio of weight added back to initial weight lost --/
theorem weight_ratio_proof (initial_weight final_weight : ℕ) 
  (first_loss third_loss final_gain : ℕ) (weight_added : ℕ) : 
  initial_weight = 99 →
  final_weight = 81 →
  first_loss = 12 →
  third_loss = 3 * first_loss →
  final_gain = 6 →
  initial_weight - first_loss + weight_added - third_loss + final_gain = final_weight →
  weight_added / first_loss = 2 := by
  sorry

end NUMINAMATH_CALUDE_weight_ratio_proof_l3236_323655


namespace NUMINAMATH_CALUDE_rice_yield_increase_l3236_323646

theorem rice_yield_increase : 
  let yield_changes : List Int := [50, -35, 10, -16, 27, -5, -20, 35]
  yield_changes.sum = 46 := by sorry

end NUMINAMATH_CALUDE_rice_yield_increase_l3236_323646


namespace NUMINAMATH_CALUDE_units_digit_of_powers_l3236_323614

theorem units_digit_of_powers : 
  (31^2020 % 10 = 1) ∧ (37^2020 % 10 = 1) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_powers_l3236_323614


namespace NUMINAMATH_CALUDE_box_fits_cubes_l3236_323664

/-- A rectangular box with given dimensions -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- A cube with a given volume -/
structure Cube where
  volume : ℝ

/-- The number of cubes that can fit in the box -/
def cubes_fit (b : Box) (c : Cube) : ℕ := 24

theorem box_fits_cubes (b : Box) (c : Cube) :
  b.length = 9 ∧ b.width = 8 ∧ b.height = 12 ∧ c.volume = 27 →
  cubes_fit b c = 24 := by
  sorry

end NUMINAMATH_CALUDE_box_fits_cubes_l3236_323664


namespace NUMINAMATH_CALUDE_correct_number_proof_l3236_323685

theorem correct_number_proof (n : ℕ) (initial_avg correct_avg : ℚ) 
  (first_error second_error : ℚ) (correct_second : ℚ) : 
  n = 10 → 
  initial_avg = 40.2 → 
  correct_avg = 40.3 → 
  first_error = 19 → 
  second_error = 13 → 
  (n : ℚ) * initial_avg - first_error - second_error + correct_second = (n : ℚ) * correct_avg → 
  correct_second = 33 := by
  sorry

end NUMINAMATH_CALUDE_correct_number_proof_l3236_323685


namespace NUMINAMATH_CALUDE_sum_square_plus_sqrt_sum_squares_l3236_323658

theorem sum_square_plus_sqrt_sum_squares :
  (5 + 9)^2 + Real.sqrt (5^2 + 9^2) = 196 + Real.sqrt 106 := by
  sorry

end NUMINAMATH_CALUDE_sum_square_plus_sqrt_sum_squares_l3236_323658


namespace NUMINAMATH_CALUDE_triangle_problem_l3236_323626

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- The dot product of two vectors -/
def dotProduct (v w : ℝ × ℝ) : ℝ := sorry

theorem triangle_problem (t : Triangle) 
  (h_area : area t = 30)
  (h_cos : Real.cos t.A = 12/13) : 
  ∃ (ab ac : ℝ × ℝ), 
    dotProduct ab ac = 144 ∧ 
    (t.c - t.b = 1 → t.a = 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3236_323626


namespace NUMINAMATH_CALUDE_sequence_constraint_l3236_323611

/-- An arithmetic sequence of four real numbers -/
structure ArithmeticSequence (x a₁ a₂ y : ℝ) : Prop where
  diff₁ : a₁ - x = a₂ - a₁
  diff₂ : a₂ - a₁ = y - a₂

/-- A geometric sequence of four real numbers -/
structure GeometricSequence (x b₁ b₂ y : ℝ) : Prop where
  ratio₁ : x ≠ 0
  ratio₂ : b₁ / x = b₂ / b₁
  ratio₃ : b₂ / b₁ = y / b₂

theorem sequence_constraint (x a₁ a₂ y b₁ b₂ : ℝ) 
  (h₁ : ArithmeticSequence x a₁ a₂ y) (h₂ : GeometricSequence x b₁ b₂ y) : 
  x ≥ 4 := by sorry

end NUMINAMATH_CALUDE_sequence_constraint_l3236_323611


namespace NUMINAMATH_CALUDE_cost_of_horse_l3236_323627

/-- Given Albert's purchase and sale of horses and cows, prove the cost of a horse -/
theorem cost_of_horse (total_cost : ℝ) (num_horses : ℕ) (num_cows : ℕ) 
  (horse_profit_rate : ℝ) (cow_profit_rate : ℝ) (total_profit : ℝ) :
  total_cost = 13400 ∧ 
  num_horses = 4 ∧ 
  num_cows = 9 ∧
  horse_profit_rate = 0.1 ∧
  cow_profit_rate = 0.2 ∧
  total_profit = 1880 →
  ∃ (horse_cost cow_cost : ℝ),
    num_horses * horse_cost + num_cows * cow_cost = total_cost ∧
    num_horses * horse_cost * horse_profit_rate + num_cows * cow_cost * cow_profit_rate = total_profit ∧
    horse_cost = 2000 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_horse_l3236_323627


namespace NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l3236_323660

theorem consecutive_integers_cube_sum (n : ℤ) : 
  (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2 = 2106 →
  (n - 1)^3 + n^3 + (n + 1)^3 + (n + 2)^3 = 45900 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_cube_sum_l3236_323660


namespace NUMINAMATH_CALUDE_polygon_distance_inequality_l3236_323692

-- Define a polygon type
structure Polygon :=
  (vertices : List (ℝ × ℝ))

-- Define the perimeter of a polygon
def perimeter (p : Polygon) : ℝ := sorry

-- Define the sum of distances from a point to vertices
def sum_distances_to_vertices (o : ℝ × ℝ) (p : Polygon) : ℝ := sorry

-- Define the sum of distances from a point to sides
def sum_distances_to_sides (o : ℝ × ℝ) (p : Polygon) : ℝ := sorry

-- State the theorem
theorem polygon_distance_inequality (o : ℝ × ℝ) (m : Polygon) :
  let ρ := perimeter m
  let d := sum_distances_to_vertices o m
  let h := sum_distances_to_sides o m
  d^2 - h^2 ≥ ρ^2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_polygon_distance_inequality_l3236_323692


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l3236_323680

theorem max_y_coordinate_sin_3theta :
  let r : ℝ → ℝ := fun θ => Real.sin (3 * θ)
  let y : ℝ → ℝ := fun θ => r θ * Real.sin θ
  ∃ (max_y : ℝ), (∀ θ, y θ ≤ max_y) ∧ (∃ θ, y θ = max_y) ∧ max_y = 1 := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l3236_323680


namespace NUMINAMATH_CALUDE_squares_in_50th_ring_l3236_323623

/-- The number of squares in the nth ring of a square pattern -/
def squares_in_ring (n : ℕ) : ℕ := 4 * n + 4

/-- The number of squares in the 50th ring is 204 -/
theorem squares_in_50th_ring : squares_in_ring 50 = 204 := by
  sorry

end NUMINAMATH_CALUDE_squares_in_50th_ring_l3236_323623


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_equation_l3236_323696

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x, P x) ↔ (∀ x, ¬ P x) :=
by sorry

theorem negation_of_exponential_equation :
  (¬ ∃ x : ℝ, Real.exp x = x - 1) ↔ (∀ x : ℝ, Real.exp x ≠ x - 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_exponential_equation_l3236_323696


namespace NUMINAMATH_CALUDE_stock_worth_l3236_323634

-- Define the total worth of the stock
variable (X : ℝ)

-- Define the profit percentage on 20% of stock
def profit_percent : ℝ := 0.10

-- Define the loss percentage on 80% of stock
def loss_percent : ℝ := 0.05

-- Define the overall loss
def overall_loss : ℝ := 200

-- Theorem statement
theorem stock_worth :
  (0.20 * X * (1 + profit_percent) + 0.80 * X * (1 - loss_percent) = X - overall_loss) →
  X = 10000 := by
sorry

end NUMINAMATH_CALUDE_stock_worth_l3236_323634


namespace NUMINAMATH_CALUDE_train_length_approximation_l3236_323648

/-- The length of a train given its speed and time to cross a fixed point -/
def trainLength (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Theorem: A train crossing a telegraph post in 13 seconds at 58.15384615384615 m/s has a length of approximately 756 meters -/
theorem train_length_approximation :
  let speed : ℝ := 58.15384615384615
  let time : ℝ := 13
  let length := trainLength speed time
  ∃ ε > 0, |length - 756| < ε :=
by sorry

end NUMINAMATH_CALUDE_train_length_approximation_l3236_323648


namespace NUMINAMATH_CALUDE_next_year_day_l3236_323607

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a year -/
structure Year where
  numDays : Nat
  firstDay : DayOfWeek
  numSaturdays : Nat

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem next_year_day (y : Year) (h1 : y.numDays = 366) (h2 : y.numSaturdays = 53) :
  nextDay y.firstDay = DayOfWeek.Monday := by
  sorry

end NUMINAMATH_CALUDE_next_year_day_l3236_323607


namespace NUMINAMATH_CALUDE_base_seven_subtraction_l3236_323671

/-- Represents a number in base 7 --/
def BaseSevenNumber := List Nat

/-- Converts a base 7 number to its decimal representation --/
def to_decimal (n : BaseSevenNumber) : Nat :=
  n.enum.foldl (fun acc (i, digit) => acc + digit * (7 ^ i)) 0

/-- Subtracts two base 7 numbers --/
def base_seven_subtract (a b : BaseSevenNumber) : BaseSevenNumber :=
  sorry

theorem base_seven_subtraction :
  let a : BaseSevenNumber := [4, 1, 2, 3]  -- 3214 in base 7
  let b : BaseSevenNumber := [4, 3, 2, 1]  -- 1234 in base 7
  let result : BaseSevenNumber := [0, 5, 6, 2]  -- 2650 in base 7
  base_seven_subtract a b = result := by sorry

end NUMINAMATH_CALUDE_base_seven_subtraction_l3236_323671


namespace NUMINAMATH_CALUDE_rectangle_fit_count_l3236_323662

/-- A rectangle with integer coordinates -/
structure Rectangle where
  x : ℤ
  y : ℤ

/-- The region defined by the problem -/
def inRegion (r : Rectangle) : Prop :=
  r.y ≤ 2 * r.x ∧ r.y ≥ -2 ∧ r.x ≤ 10 ∧ r.x ≥ 0

/-- A valid 2x1 rectangle within the region -/
def validRectangle (r : Rectangle) : Prop :=
  inRegion r ∧ inRegion ⟨r.x + 2, r.y⟩

/-- The count of valid rectangles -/
def rectangleCount : ℕ := sorry

theorem rectangle_fit_count : rectangleCount = 34 := by sorry

end NUMINAMATH_CALUDE_rectangle_fit_count_l3236_323662


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3236_323677

theorem complex_fraction_simplification :
  (3 + 4 * Complex.I) / (5 - 2 * Complex.I) = 7/29 + 26/29 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3236_323677


namespace NUMINAMATH_CALUDE_inequality_proof_l3236_323645

theorem inequality_proof (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a + b < 2 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3236_323645


namespace NUMINAMATH_CALUDE_birds_on_fence_l3236_323635

/-- Given a number of initial birds, additional birds, and additional storks,
    calculate the total number of birds on the fence. -/
def total_birds (initial : ℕ) (additional : ℕ) (storks : ℕ) : ℕ :=
  initial + additional + storks

/-- Theorem stating that with 6 initial birds, 4 additional birds, and 8 storks,
    the total number of birds on the fence is 18. -/
theorem birds_on_fence :
  total_birds 6 4 8 = 18 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l3236_323635


namespace NUMINAMATH_CALUDE_max_value_of_a_l3236_323687

theorem max_value_of_a : 
  (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → 
  ∃ a_max : ℝ, a_max = 1 ∧ ∀ a : ℝ, (∀ x : ℝ, |x - 2| + |x - a| ≥ a) → a ≤ a_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l3236_323687


namespace NUMINAMATH_CALUDE_electronics_store_purchase_l3236_323686

theorem electronics_store_purchase (total people_tv people_computer people_both : ℕ) 
  (h1 : total = 15)
  (h2 : people_tv = 9)
  (h3 : people_computer = 7)
  (h4 : people_both = 3)
  : total - (people_tv + people_computer - people_both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_electronics_store_purchase_l3236_323686


namespace NUMINAMATH_CALUDE_triangle_max_area_l3236_323624

theorem triangle_max_area (x y : ℝ) (h : x + y = 418) :
  ⌊(1/2 : ℝ) * x * y⌋ ≤ 21840 ∧ ∃ (x' y' : ℝ), x' + y' = 418 ∧ ⌊(1/2 : ℝ) * x' * y'⌋ = 21840 :=
sorry

end NUMINAMATH_CALUDE_triangle_max_area_l3236_323624


namespace NUMINAMATH_CALUDE_inequality_proof_l3236_323691

theorem inequality_proof (a b : ℝ) (ha : a < 0) (hb : 0 < b) (hb1 : b < 1) :
  ab^2 > ab ∧ ab > a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3236_323691


namespace NUMINAMATH_CALUDE_a_square_property_l3236_323604

def a : ℕ → ℤ
  | 0 => 1
  | 1 => 1
  | (n + 2) => 14 * a (n + 1) - a n

theorem a_square_property : ∃ k : ℕ → ℤ, ∀ n : ℕ, 2 * a n - 1 = k n ^ 2 := by
  sorry

end NUMINAMATH_CALUDE_a_square_property_l3236_323604


namespace NUMINAMATH_CALUDE_remainder_nine_power_2023_mod_50_l3236_323698

theorem remainder_nine_power_2023_mod_50 : 9^2023 % 50 = 41 := by
  sorry

end NUMINAMATH_CALUDE_remainder_nine_power_2023_mod_50_l3236_323698


namespace NUMINAMATH_CALUDE_three_digit_number_divided_by_11_l3236_323695

theorem three_digit_number_divided_by_11 : 
  ∀ n : ℕ, 
  100 ≤ n ∧ n < 1000 → 
  (n / 11 = (n / 100)^2 + ((n / 10) % 10)^2 + (n % 10)^2) ↔ 
  (n = 550 ∨ n = 803) := by
sorry

end NUMINAMATH_CALUDE_three_digit_number_divided_by_11_l3236_323695


namespace NUMINAMATH_CALUDE_cartesian_oval_properties_l3236_323637

-- Define the Cartesian oval
def cartesian_oval (x y : ℝ) : Prop := x^3 + y^3 - 3*x*y = 0

theorem cartesian_oval_properties :
  -- 1. The curve does not pass through the third quadrant
  (∀ x y : ℝ, cartesian_oval x y → ¬(x < 0 ∧ y < 0)) ∧
  -- 2. The curve is symmetric about the line y = x
  (∀ x y : ℝ, cartesian_oval x y ↔ cartesian_oval y x) ∧
  -- 3. The curve has no common point with the line x + y = -1
  (∀ x y : ℝ, cartesian_oval x y → x + y ≠ -1) :=
by sorry

end NUMINAMATH_CALUDE_cartesian_oval_properties_l3236_323637


namespace NUMINAMATH_CALUDE_quadratic_shift_theorem_l3236_323683

/-- Represents a quadratic function of the form y = ax^2 + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Applies a horizontal and vertical shift to a quadratic function -/
def shift_quadratic (f : QuadraticFunction) (h_shift v_shift : ℝ) : QuadraticFunction :=
  { a := f.a
  , b := -2 * f.a * h_shift
  , c := f.a * h_shift^2 + f.c - v_shift }

theorem quadratic_shift_theorem (f : QuadraticFunction) 
  (h : f.a = -2 ∧ f.b = 0 ∧ f.c = 1) : 
  shift_quadratic f 3 2 = { a := -2, b := 12, c := -1 } := by
  sorry

#check quadratic_shift_theorem

end NUMINAMATH_CALUDE_quadratic_shift_theorem_l3236_323683


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3236_323633

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x - Real.sqrt 2)^3 * (x + Real.sqrt 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3236_323633


namespace NUMINAMATH_CALUDE_probability_even_sum_four_primes_l3236_323697

-- Define the set of first twelve prime numbers
def first_twelve_primes : Finset Nat := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}

-- Define a function to check if a number is even
def is_even (n : Nat) : Bool := n % 2 = 0

-- Define a function to calculate the sum of a list of numbers
def sum_list (l : List Nat) : Nat := l.foldl (·+·) 0

-- Theorem statement
theorem probability_even_sum_four_primes :
  let all_selections := Finset.powerset first_twelve_primes
  let valid_selections := all_selections.filter (fun s => s.card = 4)
  let even_sum_selections := valid_selections.filter (fun s => is_even (sum_list s.toList))
  (even_sum_selections.card : Rat) / valid_selections.card = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_probability_even_sum_four_primes_l3236_323697


namespace NUMINAMATH_CALUDE_combination_sum_equals_55_l3236_323613

-- Define the combination function
def combination (n r : ℕ) : ℕ :=
  if r ≤ n then
    Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))
  else
    0

-- State the theorem
theorem combination_sum_equals_55 :
  combination 10 9 + combination 10 8 = 55 :=
sorry

end NUMINAMATH_CALUDE_combination_sum_equals_55_l3236_323613


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3236_323657

/-- A quadratic function f(x) = ax^2 + bx satisfying given conditions -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b : ℝ, ∀ x, f x = a * x^2 + b * x

theorem quadratic_function_range (f : ℝ → ℝ) 
  (hf : QuadraticFunction f)
  (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 2 ≤ f 1 ∧ f 1 ≤ 4) :
  5 ≤ f (-2) ∧ f (-2) ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3236_323657
