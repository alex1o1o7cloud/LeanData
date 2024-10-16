import Mathlib

namespace NUMINAMATH_CALUDE_equation_solution_l2546_254663

theorem equation_solution (x a b : ℝ) (h : b / x^3 + 1 / x^2 + 1 / x + 1 = 0) :
  x^3 + x^2 + x + a = a - b := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2546_254663


namespace NUMINAMATH_CALUDE_product_quotient_l2546_254669

theorem product_quotient (a b c d e f : ℚ) 
  (h1 : a * b * c = 130)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 750)
  (h4 : d * e * f = 250)
  : (a * f) / (c * d) = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_product_quotient_l2546_254669


namespace NUMINAMATH_CALUDE_coin_flip_probability_l2546_254643

theorem coin_flip_probability (n : ℕ) : 
  (n.choose 2 : ℚ) / 2^n = 1/8 → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l2546_254643


namespace NUMINAMATH_CALUDE_sibling_age_sum_l2546_254633

/-- Given the ages of three siblings, proves that the sum of two siblings' ages is correct. -/
theorem sibling_age_sum (juliet maggie ralph : ℕ) : 
  juliet = maggie + 3 →
  juliet + 2 = ralph →
  juliet = 10 →
  maggie + ralph = 19 := by
sorry

end NUMINAMATH_CALUDE_sibling_age_sum_l2546_254633


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l2546_254649

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- The original rectangle -/
def original : Rectangle := { width := 5, height := 7 }

/-- The rectangle after shortening one side by 2 inches -/
def shortened_by_2 : Rectangle := { width := 3, height := 7 }

/-- The rectangle after shortening the other side by 1 inch -/
def shortened_by_1 : Rectangle := { width := 5, height := 6 }

theorem rectangle_area_problem :
  shortened_by_2.area = 21 → shortened_by_1.area = 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l2546_254649


namespace NUMINAMATH_CALUDE_max_regions_formula_l2546_254699

/-- The maximum number of regions formed by n lines in R^2 -/
def max_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Theorem: The maximum number of regions formed by n lines in R^2 is (n(n+1)/2) + 1 -/
theorem max_regions_formula (n : ℕ) : 
  max_regions n = n * (n + 1) / 2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_formula_l2546_254699


namespace NUMINAMATH_CALUDE_shortest_distance_specific_rectangle_l2546_254667

/-- A rectangle on a cube face with given dimensions -/
structure RectangleOnCube where
  pq : ℝ
  qr : ℝ
  is_vertex_q : Bool
  is_vertex_s : Bool
  on_adjacent_faces : Bool

/-- The shortest distance between two points through a cube -/
def shortest_distance_through_cube (r : RectangleOnCube) : ℝ :=
  sorry

/-- Theorem stating the shortest distance for the given rectangle -/
theorem shortest_distance_specific_rectangle :
  let r : RectangleOnCube := {
    pq := 20,
    qr := 15,
    is_vertex_q := true,
    is_vertex_s := true,
    on_adjacent_faces := true
  }
  shortest_distance_through_cube r = Real.sqrt 337 := by
  sorry

end NUMINAMATH_CALUDE_shortest_distance_specific_rectangle_l2546_254667


namespace NUMINAMATH_CALUDE_horner_method_proof_l2546_254683

/-- Horner's method for evaluating a polynomial -/
def horner_eval (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => acc * x + a) 0

/-- The polynomial f(x) = 2x^4 + 3x^3 + 5x + 4 -/
def f : ℝ → ℝ := fun x => 2 * x^4 + 3 * x^3 + 5 * x + 4

theorem horner_method_proof :
  f 2 = horner_eval [2, 3, 0, 5, 4] 2 ∧ horner_eval [2, 3, 0, 5, 4] 2 = 70 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_proof_l2546_254683


namespace NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l2546_254600

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: The number of diagonals in a convex polygon with 25 sides is 275 -/
theorem diagonals_25_sided_polygon :
  num_diagonals 25 = 275 := by sorry

end NUMINAMATH_CALUDE_diagonals_25_sided_polygon_l2546_254600


namespace NUMINAMATH_CALUDE_angle_between_perpendicular_lines_in_dihedral_l2546_254675

-- Define the dihedral angle
def dihedral_angle (α l β : Line3) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line3) (α : Plane3) : Prop := sorry

-- Define the angle between two lines
def angle_between_lines (m n : Line3) : ℝ := sorry

-- Main theorem
theorem angle_between_perpendicular_lines_in_dihedral 
  (α l β : Line3) (m n : Line3) :
  dihedral_angle α l β = 60 →
  m ≠ n →
  perpendicular m α →
  perpendicular n β →
  angle_between_lines m n = 60 :=
sorry

end NUMINAMATH_CALUDE_angle_between_perpendicular_lines_in_dihedral_l2546_254675


namespace NUMINAMATH_CALUDE_line_intersection_theorem_l2546_254697

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the properties and relations
variable (skew : Line → Line → Prop)
variable (contains : Plane → Line → Prop)
variable (intersects : Line → Line → Prop)
variable (plane_intersection : Plane → Plane → Line)

-- State the theorem
theorem line_intersection_theorem 
  (m n l : Line) (α β : Plane) 
  (h1 : skew m n)
  (h2 : contains α m)
  (h3 : contains β n)
  (h4 : plane_intersection α β = l) :
  intersects l m ∨ intersects l n :=
sorry

end NUMINAMATH_CALUDE_line_intersection_theorem_l2546_254697


namespace NUMINAMATH_CALUDE_quadratic_one_positive_root_l2546_254677

theorem quadratic_one_positive_root (a : ℝ) : 
  (∃! x : ℝ, x > 0 ∧ x^2 - a*x + a - 2 = 0) → a ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_one_positive_root_l2546_254677


namespace NUMINAMATH_CALUDE_quadratic_inequality_l2546_254614

theorem quadratic_inequality (x : ℝ) : x^2 - x - 12 < 0 ↔ -3 < x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l2546_254614


namespace NUMINAMATH_CALUDE_pigeonhole_friends_l2546_254626

/-- Represents a class of students -/
structure ClassOfStudents where
  n : ℕ  -- number of students
  h : n > 0  -- ensures the class is not empty

/-- Represents the number of friends each student has -/
def FriendCount (c : ClassOfStudents) := Fin c.n → ℕ

/-- The property that if a student has 0 friends, no one has n-1 friends -/
def ValidFriendCount (c : ClassOfStudents) (f : FriendCount c) : Prop :=
  (∃ i, f i = 0) → ∀ j, f j < c.n - 1

theorem pigeonhole_friends (c : ClassOfStudents) (f : FriendCount c) 
    (hf : ValidFriendCount c f) : 
    ∃ i j, i ≠ j ∧ f i = f j :=
  sorry


end NUMINAMATH_CALUDE_pigeonhole_friends_l2546_254626


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_product_l2546_254658

theorem unique_solution_quadratic_product (k : ℝ) : 
  (∃! x : ℝ, k * x^2 + (k + 5) * x + 5 = 0) → k = 5 :=
by sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_product_l2546_254658


namespace NUMINAMATH_CALUDE_inscribed_squares_ratio_l2546_254624

theorem inscribed_squares_ratio (x y : ℝ) : 
  x > 0 ∧ y > 0 ∧
  (8 - x) / x = 8 / 6 ∧
  (6 - y) / y = 8 / 10 →
  x / y = 36 / 35 := by sorry

end NUMINAMATH_CALUDE_inscribed_squares_ratio_l2546_254624


namespace NUMINAMATH_CALUDE_square_difference_formula_expression_equivalence_l2546_254674

/-- The square difference formula -/
theorem square_difference_formula (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by sorry

/-- Proof that (x+y)(-x+y) is equivalent to y^2 - x^2 -/
theorem expression_equivalence (x y : ℝ) : (x + y) * (-x + y) = y^2 - x^2 := by sorry

end NUMINAMATH_CALUDE_square_difference_formula_expression_equivalence_l2546_254674


namespace NUMINAMATH_CALUDE_area_not_covered_by_circles_l2546_254648

/-- The area of a square not covered by four inscribed circles -/
theorem area_not_covered_by_circles (square_side : ℝ) (circle_radius : ℝ) 
  (h1 : square_side = 10)
  (h2 : circle_radius = 5)
  (h3 : circle_radius * 2 = square_side) :
  square_side ^ 2 - 4 * Real.pi * circle_radius ^ 2 + 4 * Real.pi * circle_radius ^ 2 / 2 = 100 - 50 * Real.pi := by
  sorry

#check area_not_covered_by_circles

end NUMINAMATH_CALUDE_area_not_covered_by_circles_l2546_254648


namespace NUMINAMATH_CALUDE_prob_qualified_volleyball_expected_net_profit_l2546_254661

-- Define the supply percentages and qualification rates
def supply_A : ℝ := 0.4
def supply_B : ℝ := 0.3
def supply_C : ℝ := 0.3
def qual_rate_A : ℝ := 0.95
def qual_rate_B : ℝ := 0.92
def qual_rate_C : ℝ := 0.96

-- Define profit and loss for each factory
def profit_A : ℝ := 10
def loss_A : ℝ := 5
def profit_C : ℝ := 8
def loss_C : ℝ := 6

-- Theorem 1: Probability of purchasing a qualified volleyball
theorem prob_qualified_volleyball :
  supply_A * qual_rate_A + supply_B * qual_rate_B + supply_C * qual_rate_C = 0.944 :=
sorry

-- Theorem 2: Expected net profit from purchasing one volleyball from Factory A and one from Factory C
theorem expected_net_profit :
  qual_rate_A * qual_rate_C * (profit_A + profit_C) +
  qual_rate_A * (1 - qual_rate_C) * (profit_A - loss_C) +
  (1 - qual_rate_A) * qual_rate_C * (profit_C - loss_A) +
  (1 - qual_rate_A) * (1 - qual_rate_C) * (-loss_A - loss_C) = 16.69 :=
sorry

end NUMINAMATH_CALUDE_prob_qualified_volleyball_expected_net_profit_l2546_254661


namespace NUMINAMATH_CALUDE_orange_cost_12kg_l2546_254606

/-- The cost of oranges given a rate and a quantity -/
def orange_cost (rate_price : ℚ) (rate_quantity : ℚ) (quantity : ℚ) : ℚ :=
  (quantity / rate_quantity) * rate_price

/-- Theorem: The cost of 12 kg of oranges at a rate of $5 per 3 kg is $20 -/
theorem orange_cost_12kg (rate_price : ℚ) (rate_quantity : ℚ) (quantity : ℚ)
    (h1 : rate_price = 5)
    (h2 : rate_quantity = 3)
    (h3 : quantity = 12) :
    orange_cost rate_price rate_quantity quantity = 20 := by
  sorry

end NUMINAMATH_CALUDE_orange_cost_12kg_l2546_254606


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2546_254642

theorem quadratic_equation_solution :
  ∃ (a b : ℕ+),
    (∃ (x : ℝ), x^2 + 8*x = 48 ∧ x > 0 ∧ x = Real.sqrt a - b) ∧
    a + b = 68 := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2546_254642


namespace NUMINAMATH_CALUDE_square_rectangle_area_multiplier_l2546_254639

theorem square_rectangle_area_multiplier :
  let square_perimeter : ℝ := 800
  let rectangle_length : ℝ := 125
  let rectangle_width : ℝ := 64
  let square_side : ℝ := square_perimeter / 4
  let square_area : ℝ := square_side * square_side
  let rectangle_area : ℝ := rectangle_length * rectangle_width
  square_area / rectangle_area = 5 := by
  sorry

end NUMINAMATH_CALUDE_square_rectangle_area_multiplier_l2546_254639


namespace NUMINAMATH_CALUDE_quadratic_prime_roots_unique_l2546_254644

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem quadratic_prime_roots_unique :
  ∃! k : ℤ, ∃ p q : ℕ,
    is_prime p ∧ is_prime q ∧
    p ≠ q ∧
    ∀ x : ℤ, x^2 - 74*x + k = 0 ↔ x = p ∨ x = q :=
sorry

end NUMINAMATH_CALUDE_quadratic_prime_roots_unique_l2546_254644


namespace NUMINAMATH_CALUDE_triangle_theorem_l2546_254605

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem about the triangle ABC -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b^2 + t.c^2 = t.a^2 + t.b * t.c)
  (h2 : Real.sin t.B = Real.sqrt 3 / 3)
  (h3 : t.b = 2) :
  t.A = π / 3 ∧ 
  (1/2 * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 2 + Real.sqrt 3) / 2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_theorem_l2546_254605


namespace NUMINAMATH_CALUDE_inequality_range_l2546_254641

theorem inequality_range (a b x : ℝ) (ha : a ≠ 0) :
  (|2*a + b| + |2*a - b| ≥ |a| * (|2 + x| + |2 - x|)) → x ∈ Set.Icc (-2) 2 :=
sorry

end NUMINAMATH_CALUDE_inequality_range_l2546_254641


namespace NUMINAMATH_CALUDE_yield_fertilization_correlation_l2546_254671

/-- Represents the yield of crops -/
def CropYield : Type := ℝ

/-- Represents the amount of fertilization -/
def Fertilization : Type := ℝ

/-- Defines the relationship between crop yield and fertilization -/
def dependsOn (y : CropYield) (f : Fertilization) : Prop := ∃ (g : Fertilization → CropYield), y = g f

/-- Defines correlation between two variables -/
def correlated (X Y : Type) : Prop := ∃ (f : X → Y), Function.Injective f ∨ Function.Surjective f

/-- Theorem stating that if crop yield depends on fertilization, then they are correlated -/
theorem yield_fertilization_correlation :
  (∀ y : CropYield, ∀ f : Fertilization, dependsOn y f) →
  correlated Fertilization CropYield :=
by sorry

end NUMINAMATH_CALUDE_yield_fertilization_correlation_l2546_254671


namespace NUMINAMATH_CALUDE_binomial_26_6_l2546_254662

theorem binomial_26_6 (h1 : Nat.choose 25 5 = 53130) (h2 : Nat.choose 25 6 = 177100) :
  Nat.choose 26 6 = 230230 := by
  sorry

end NUMINAMATH_CALUDE_binomial_26_6_l2546_254662


namespace NUMINAMATH_CALUDE_expression_proof_l2546_254659

theorem expression_proof (x : ℝ) (E : ℝ) : 
  ((x + 3)^2 / E = 3) ∧ 
  (∃ x₁ x₂ : ℝ, x₁ - x₂ = 12 ∧ 
    ((x₁ + 3)^2 / E = 3) ∧ 
    ((x₂ + 3)^2 / E = 3)) → 
  (E = (x + 3)^2 / 3 ∧ E = 12) := by
sorry

end NUMINAMATH_CALUDE_expression_proof_l2546_254659


namespace NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l2546_254695

def arithmetic_sum (a₁ : ℚ) (d : ℚ) (aₙ : ℚ) : ℚ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequences_ratio :
  let seq1_sum := arithmetic_sum 4 4 68
  let seq2_sum := arithmetic_sum 5 5 85
  seq1_sum / seq2_sum = 4 / 5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequences_ratio_l2546_254695


namespace NUMINAMATH_CALUDE_school_comparison_l2546_254672

theorem school_comparison (students_A : ℝ) (qualified_A : ℝ) (students_B : ℝ) (qualified_B : ℝ)
  (h1 : qualified_A = 0.7 * students_A)
  (h2 : qualified_B = 1.5 * qualified_A)
  (h3 : qualified_B = 0.875 * students_B) :
  (students_B - students_A) / students_A = 0.2 := by
  sorry

end NUMINAMATH_CALUDE_school_comparison_l2546_254672


namespace NUMINAMATH_CALUDE_find_C_l2546_254698

theorem find_C (A B C : ℕ) : A = 348 → B = A + 173 → C = B + 299 → C = 820 := by
  sorry

end NUMINAMATH_CALUDE_find_C_l2546_254698


namespace NUMINAMATH_CALUDE_mari_buttons_l2546_254652

theorem mari_buttons (kendra_buttons : ℕ) (mari_buttons : ℕ) : 
  kendra_buttons = 15 →
  mari_buttons = 5 * kendra_buttons + 4 →
  mari_buttons = 79 := by
  sorry

end NUMINAMATH_CALUDE_mari_buttons_l2546_254652


namespace NUMINAMATH_CALUDE_problem_statement_l2546_254608

theorem problem_statement (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  (a + b + 2*c ≤ 3) ∧ (b = 2*c → 1/a + 1/c ≥ 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2546_254608


namespace NUMINAMATH_CALUDE_library_books_pages_l2546_254657

theorem library_books_pages (num_books : ℕ) (total_pages : ℕ) (h1 : num_books = 8) (h2 : total_pages = 3824) :
  total_pages / num_books = 478 := by
sorry

end NUMINAMATH_CALUDE_library_books_pages_l2546_254657


namespace NUMINAMATH_CALUDE_product_sign_l2546_254647

theorem product_sign (a b c d e : ℝ) : ab^2*c^3*d^4*e^5 < 0 → ab^2*c*d^4*e < 0 := by
  sorry

end NUMINAMATH_CALUDE_product_sign_l2546_254647


namespace NUMINAMATH_CALUDE_hotel_rate_proof_l2546_254655

/-- The flat rate for the first night in a hotel. -/
def flat_rate : ℝ := 80

/-- The additional fee for each subsequent night. -/
def additional_fee : ℝ := 40

/-- The cost for a stay of n nights. -/
def cost (n : ℕ) : ℝ := flat_rate + additional_fee * (n - 1)

theorem hotel_rate_proof :
  (cost 4 = 200) ∧ (cost 7 = 320) → flat_rate = 80 := by
  sorry

end NUMINAMATH_CALUDE_hotel_rate_proof_l2546_254655


namespace NUMINAMATH_CALUDE_childrens_ticket_cost_l2546_254691

theorem childrens_ticket_cost :
  let adult_ticket_cost : ℚ := 25
  let total_receipts : ℚ := 7200
  let total_attendance : ℕ := 400
  let adult_attendance : ℕ := 280
  let child_attendance : ℕ := 120
  let child_ticket_cost : ℚ := (total_receipts - (adult_ticket_cost * adult_attendance)) / child_attendance
  child_ticket_cost = 5/3 := by sorry

end NUMINAMATH_CALUDE_childrens_ticket_cost_l2546_254691


namespace NUMINAMATH_CALUDE_petya_win_probability_l2546_254665

/-- Represents the number of stones a player can take in one turn -/
inductive StonesPerTurn
  | one
  | two
  | three
  | four

/-- Represents a player in the game -/
inductive Player
  | petya
  | computer

/-- Represents the state of the game -/
structure GameState where
  stones : Nat
  turn : Player

/-- The initial state of the game -/
def initialState : GameState :=
  { stones := 16, turn := Player.petya }

/-- Represents the strategy of a player -/
def Strategy := GameState → StonesPerTurn

/-- Petya's random strategy -/
def petyaStrategy : Strategy :=
  fun _ => sorry -- Randomly choose between 1 and 4 stones

/-- Computer's optimal strategy -/
def computerStrategy : Strategy :=
  fun _ => sorry -- Always choose the optimal number of stones

/-- The probability of Petya winning the game -/
def petyaWinProbability : ℚ :=
  1 / 256

/-- Theorem stating that Petya's win probability is 1/256 -/
theorem petya_win_probability :
  petyaWinProbability = 1 / 256 := by sorry


end NUMINAMATH_CALUDE_petya_win_probability_l2546_254665


namespace NUMINAMATH_CALUDE_cylinder_min_circumscribed_sphere_l2546_254666

/-- For a cylinder with surface area 16π and base radius r, 
    the surface area of its circumscribed sphere is minimized when r² = 8√5/5 -/
theorem cylinder_min_circumscribed_sphere (r : ℝ) : 
  (2 * π * r^2 + 2 * π * r * ((8 : ℝ) / r - r) = 16 * π) →
  (∃ (R : ℝ), R^2 = r^2 + ((8 : ℝ) / r - r)^2 / 4 ∧ 
    ∀ (R' : ℝ), R'^2 = r^2 + ((8 : ℝ) / r' - r')^2 / 4 → R'^2 ≥ R^2) →
  r^2 = 8 * Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_cylinder_min_circumscribed_sphere_l2546_254666


namespace NUMINAMATH_CALUDE_min_value_expression_l2546_254632

theorem min_value_expression (x : ℝ) (h : x > 2) :
  (x^2 + 8) / Real.sqrt (x - 2) ≥ 22 ∧
  ∃ x₀ > 2, (x₀^2 + 8) / Real.sqrt (x₀ - 2) = 22 :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l2546_254632


namespace NUMINAMATH_CALUDE_triangle_area_l2546_254689

theorem triangle_area (A B C : ℝ) (a b c : ℝ) :
  B = 60 * π / 180 →
  c = 3 →
  b = Real.sqrt 7 →
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4) ∨
  (1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_area_l2546_254689


namespace NUMINAMATH_CALUDE_symmetry_example_l2546_254615

/-- A point in 3D space is represented by its x, y, and z coordinates. -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Two points are symmetric with respect to the x-axis if their x-coordinates are the same,
    and their y and z coordinates are negatives of each other. -/
def symmetric_wrt_x_axis (p q : Point3D) : Prop :=
  p.x = q.x ∧ p.y = -q.y ∧ p.z = -q.z

/-- The theorem states that the point (-2, -1, -4) is symmetric to the point (-2, 1, 4)
    with respect to the x-axis. -/
theorem symmetry_example : 
  symmetric_wrt_x_axis (Point3D.mk (-2) 1 4) (Point3D.mk (-2) (-1) (-4)) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_example_l2546_254615


namespace NUMINAMATH_CALUDE_unique_solution_implies_negative_a_l2546_254664

theorem unique_solution_implies_negative_a :
  ∀ a : ℝ,
  (∃! x : ℝ, |x^2 - 1| = a * |x - 1|) →
  a < 0 := by
sorry

end NUMINAMATH_CALUDE_unique_solution_implies_negative_a_l2546_254664


namespace NUMINAMATH_CALUDE_second_shop_expense_l2546_254631

theorem second_shop_expense (first_shop_books : ℕ) (second_shop_books : ℕ) 
  (first_shop_cost : ℕ) (average_price : ℕ) (total_books : ℕ)
  (h1 : first_shop_books = 65)
  (h2 : second_shop_books = 35)
  (h3 : first_shop_cost = 6500)
  (h4 : average_price = 85)
  (h5 : total_books = first_shop_books + second_shop_books) :
  (average_price * total_books) - first_shop_cost = 2000 := by
  sorry

end NUMINAMATH_CALUDE_second_shop_expense_l2546_254631


namespace NUMINAMATH_CALUDE_ticket_price_is_eight_l2546_254612

/-- Represents a movie theater with its capacity, tickets sold, and lost revenue --/
structure MovieTheater where
  capacity : ℕ
  ticketsSold : ℕ
  lostRevenue : ℚ

/-- Calculates the ticket price given the theater's information --/
def calculateTicketPrice (theater : MovieTheater) : ℚ :=
  theater.lostRevenue / (theater.capacity - theater.ticketsSold)

/-- Theorem stating that the ticket price is $8 for the given conditions --/
theorem ticket_price_is_eight :
  let theater : MovieTheater := { capacity := 50, ticketsSold := 24, lostRevenue := 208 }
  calculateTicketPrice theater = 8 := by sorry

end NUMINAMATH_CALUDE_ticket_price_is_eight_l2546_254612


namespace NUMINAMATH_CALUDE_equal_squares_count_l2546_254618

/-- Represents a square grid -/
structure Grid (n : ℕ) where
  cells : Fin n → Fin n → Bool

/-- Counts squares with equal black and white cells in a 5x5 grid -/
def count_equal_squares (g : Grid 5) : ℕ :=
  let valid_2x2 : ℕ := 14  -- 16 total - 2 invalid
  let valid_4x4 : ℕ := 2
  valid_2x2 + valid_4x4

/-- Theorem: The number of squares with equal black and white cells is 16 -/
theorem equal_squares_count (g : Grid 5) : count_equal_squares g = 16 := by
  sorry

end NUMINAMATH_CALUDE_equal_squares_count_l2546_254618


namespace NUMINAMATH_CALUDE_stock_worth_l2546_254619

def total_modules : ℕ := 11
def cheap_modules : ℕ := 10
def expensive_modules : ℕ := total_modules - cheap_modules
def cheap_cost : ℚ := 3.5
def expensive_cost : ℚ := 10

def total_worth : ℚ := cheap_modules * cheap_cost + expensive_modules * expensive_cost

theorem stock_worth : total_worth = 45 := by
  sorry

end NUMINAMATH_CALUDE_stock_worth_l2546_254619


namespace NUMINAMATH_CALUDE_opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l2546_254681

theorem opposite_of_negative_fraction (n : ℕ) (n_pos : n > 0) :
  ((-1 : ℚ) / n) + (1 : ℚ) / n = 0 :=
by sorry

theorem opposite_of_negative_one_over_2023 :
  ((-1 : ℚ) / 2023) + (1 : ℚ) / 2023 = 0 :=
by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_fraction_opposite_of_negative_one_over_2023_l2546_254681


namespace NUMINAMATH_CALUDE_beach_population_l2546_254616

theorem beach_population (initial_group : ℕ) (joined : ℕ) (left : ℕ) : 
  initial_group = 3 → joined = 100 → left = 40 → 
  initial_group + joined - left = 63 := by
sorry

end NUMINAMATH_CALUDE_beach_population_l2546_254616


namespace NUMINAMATH_CALUDE_find_b_l2546_254637

-- Define the inverse relationship between a² and √b
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^2 * Real.sqrt b = k

-- Define the conditions
def condition1 : ℝ := 3
def condition2 : ℝ := 36

-- Define the target equation
def target_equation (a b : ℝ) : Prop := a * b = 54

-- Theorem statement
theorem find_b :
  ∀ a b : ℝ,
  inverse_relation a b →
  inverse_relation condition1 condition2 →
  target_equation a b →
  b = 18 * (4^(1/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_find_b_l2546_254637


namespace NUMINAMATH_CALUDE_only_B_is_quadratic_l2546_254670

-- Define the structure of a general function
structure GeneralFunction where
  f : ℝ → ℝ

-- Define what it means for a function to be quadratic
def is_quadratic (f : GeneralFunction) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f.f x = a * x^2 + b * x + c

-- Define the given functions
def function_A : GeneralFunction :=
  { f := λ x => 2 * x + 1 }

def function_B : GeneralFunction :=
  { f := λ x => -5 * x^2 - 3 }

def function_C (a b c : ℝ) : GeneralFunction :=
  { f := λ x => a * x^2 + b * x + c }

def function_D : GeneralFunction :=
  { f := λ x => x^3 + x + 1 }

-- State the theorem
theorem only_B_is_quadratic :
  ¬ is_quadratic function_A ∧
  is_quadratic function_B ∧
  (∃ a b c, ¬ is_quadratic (function_C a b c)) ∧
  ¬ is_quadratic function_D :=
sorry

end NUMINAMATH_CALUDE_only_B_is_quadratic_l2546_254670


namespace NUMINAMATH_CALUDE_sqrt_y_fourth_power_l2546_254610

theorem sqrt_y_fourth_power (y : ℝ) : (Real.sqrt y)^4 = 256 → y = 16 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_y_fourth_power_l2546_254610


namespace NUMINAMATH_CALUDE_min_grid_sum_l2546_254650

/-- Represents a 2x2 grid of positive integers -/
structure Grid :=
  (a b c d : ℕ+)

/-- Calculates the sum of the grid numbers and their row/column products -/
def totalSum (g : Grid) : ℕ :=
  g.a.val + g.b.val + g.c.val + g.d.val +
  (g.a.val * g.b.val) + (g.c.val * g.d.val) +
  (g.a.val * g.c.val) + (g.b.val * g.d.val)

/-- The minimum sum of grid numbers given the total sum constraint -/
theorem min_grid_sum :
  ∃ (g : Grid), totalSum g = 2015 ∧
  ∀ (h : Grid), totalSum h = 2015 →
  g.a.val + g.b.val + g.c.val + g.d.val ≤ h.a.val + h.b.val + h.c.val + h.d.val ∧
  g.a.val + g.b.val + g.c.val + g.d.val = 88 := by
  sorry

end NUMINAMATH_CALUDE_min_grid_sum_l2546_254650


namespace NUMINAMATH_CALUDE_expression_evaluation_l2546_254679

theorem expression_evaluation : (3^3 + 2)^2 - (3^3 - 2)^2 = 216 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2546_254679


namespace NUMINAMATH_CALUDE_expanded_ohara_triple_solution_l2546_254685

/-- An Expanded O'Hara triple is a tuple of four positive integers (a, b, c, x) 
    such that √a + √b + √c = x -/
def IsExpandedOHaraTriple (a b c x : ℕ) : Prop :=
  Real.sqrt a + Real.sqrt b + Real.sqrt c = x

theorem expanded_ohara_triple_solution :
  IsExpandedOHaraTriple 49 64 16 19 := by sorry

end NUMINAMATH_CALUDE_expanded_ohara_triple_solution_l2546_254685


namespace NUMINAMATH_CALUDE_sam_win_probability_l2546_254684

/-- The probability of hitting the target with one shot -/
def hit_prob : ℚ := 2/5

/-- The probability of missing the target with one shot -/
def miss_prob : ℚ := 3/5

/-- The probability that Sam wins the game -/
def win_prob : ℚ := 5/8

theorem sam_win_probability :
  (hit_prob + miss_prob * miss_prob * win_prob = win_prob) →
  win_prob = 5/8 := by sorry

end NUMINAMATH_CALUDE_sam_win_probability_l2546_254684


namespace NUMINAMATH_CALUDE_like_terms_imply_sum_l2546_254617

/-- Two terms are like terms if they have the same variables raised to the same powers. -/
def like_terms (term1 term2 : ℕ → ℕ → ℚ) : Prop :=
  ∀ x y, ∃ c, term1 x y = c * term2 x y

/-- The first term in our problem -/
def term1 (m : ℕ) (x y : ℕ) : ℚ := 3 * x^(2*m) * y^m

/-- The second term in our problem -/
def term2 (n : ℕ) (x y : ℕ) : ℚ := x^(4-n) * y^(n-1)

theorem like_terms_imply_sum (m n : ℕ) : 
  like_terms (term1 m) (term2 n) → m + n = 3 := by
sorry

end NUMINAMATH_CALUDE_like_terms_imply_sum_l2546_254617


namespace NUMINAMATH_CALUDE_max_value_of_s_l2546_254656

theorem max_value_of_s (p q r s : ℝ) 
  (sum_eq : p + q + r + s = 12)
  (sum_prod_eq : p*q + p*r + p*s + q*r + q*s + r*s = 24) :
  s ≤ 3 + 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_s_l2546_254656


namespace NUMINAMATH_CALUDE_virus_spread_l2546_254676

/-- Given that one infected computer leads to 121 infected computers after two rounds of infection,
    prove that the average number of computers infected by one computer in each round is 10. -/
theorem virus_spread (x : ℝ) : (1 + x + x * x = 121) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_virus_spread_l2546_254676


namespace NUMINAMATH_CALUDE_marcella_shoes_l2546_254623

/-- Given a number of initial shoe pairs and lost individual shoes, 
    calculate the maximum number of complete pairs remaining. -/
def max_remaining_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) : ℕ :=
  initial_pairs - lost_shoes

/-- Theorem stating that with 24 initial pairs and 9 lost shoes, 
    the maximum number of complete pairs remaining is 15. -/
theorem marcella_shoes : max_remaining_pairs 24 9 = 15 := by
  sorry

end NUMINAMATH_CALUDE_marcella_shoes_l2546_254623


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2546_254640

theorem complex_fraction_simplification :
  (1 * 3 * 5 * 7 * 9) * (10 * 12 * 14 * 16 * 18) / (5 * 6 * 7 * 8 * 9)^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2546_254640


namespace NUMINAMATH_CALUDE_expression_evaluation_l2546_254628

theorem expression_evaluation :
  3000 * (3000 ^ 2500) * 2 = 2 * 3000 ^ 2501 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2546_254628


namespace NUMINAMATH_CALUDE_unique_function_existence_l2546_254635

def is_valid_function (f : ℕ → ℝ) : Prop :=
  (∀ x : ℕ, f x > 0) ∧
  (∀ a b : ℕ, f (a + b) = f a * f b) ∧
  (f 2 = 4)

theorem unique_function_existence : 
  ∃! f : ℕ → ℝ, is_valid_function f ∧ ∀ x : ℕ, f x = 2^x :=
sorry

end NUMINAMATH_CALUDE_unique_function_existence_l2546_254635


namespace NUMINAMATH_CALUDE_special_numbers_characterization_l2546_254646

/-- A function that returns true if a natural number has all distinct digits -/
def has_distinct_digits (n : ℕ) : Bool :=
  sorry

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  sorry

/-- A function that returns the product of digits of a natural number -/
def product_of_digits (n : ℕ) : ℕ :=
  sorry

/-- The set of numbers that satisfy the conditions -/
def special_numbers : Finset ℕ :=
  {123, 132, 213, 231, 312, 321}

theorem special_numbers_characterization :
  ∀ n : ℕ, n ∈ special_numbers ↔
    n > 9 ∧
    has_distinct_digits n ∧
    sum_of_digits n = product_of_digits n :=
by sorry

end NUMINAMATH_CALUDE_special_numbers_characterization_l2546_254646


namespace NUMINAMATH_CALUDE_cracker_ratio_is_one_l2546_254668

/-- The number of crackers Marcus has -/
def marcus_crackers : ℕ := 27

/-- The number of crackers Mona has -/
def mona_crackers : ℕ := marcus_crackers

/-- The ratio of Marcus's crackers to Mona's crackers -/
def cracker_ratio : ℚ := marcus_crackers / mona_crackers

theorem cracker_ratio_is_one : cracker_ratio = 1 := by
  sorry

end NUMINAMATH_CALUDE_cracker_ratio_is_one_l2546_254668


namespace NUMINAMATH_CALUDE_quadratic_root_sums_l2546_254673

theorem quadratic_root_sums (a b c x₁ x₂ : ℝ) (h : a ≠ 0) : 
  (∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂) →
  (x₁^2 + x₂^2 = (b^2 - 2*a*c) / a^2 ∧ 
   x₁^3 + x₂^3 = (3*a*b*c - b^3) / a^3) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sums_l2546_254673


namespace NUMINAMATH_CALUDE_gift_cost_proof_l2546_254636

theorem gift_cost_proof (initial_friends : Nat) (dropped_out : Nat) (share_increase : Int) :
  initial_friends = 10 →
  dropped_out = 4 →
  share_increase = 8 →
  ∃ (cost : Int),
    cost > 0 ∧
    (cost / (initial_friends - dropped_out : Int) = cost / initial_friends + share_increase) ∧
    cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_proof_l2546_254636


namespace NUMINAMATH_CALUDE_rectangular_field_diagonal_l2546_254680

/-- Proves that a rectangular field with one side of 15 m and an area of 120 m² has a diagonal of 17 m -/
theorem rectangular_field_diagonal (side : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side = 15 → area = 120 → diagonal = 17 → 
  area = side * (area / side) ∧ diagonal^2 = side^2 + (area / side)^2 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_diagonal_l2546_254680


namespace NUMINAMATH_CALUDE_cube_dimension_ratio_l2546_254602

theorem cube_dimension_ratio (v1 v2 : ℝ) (h1 : v1 = 64) (h2 : v2 = 512) :
  (v2 / v1) ^ (1/3 : ℝ) = 2 := by
  sorry

end NUMINAMATH_CALUDE_cube_dimension_ratio_l2546_254602


namespace NUMINAMATH_CALUDE_taxi_fare_distance_l2546_254651

/-- Represents the taxi fare structure and proves the distance for each fare segment -/
theorem taxi_fare_distance (initial_fare : ℝ) (subsequent_fare : ℝ) (total_distance : ℝ) (total_fare : ℝ) :
  initial_fare = 8 →
  subsequent_fare = 0.8 →
  total_distance = 8 →
  total_fare = 39.2 →
  ∃ (d : ℝ), d > 0 ∧ d = 1/5 ∧
    total_fare = initial_fare + subsequent_fare * ((total_distance - d) / d) :=
by
  sorry


end NUMINAMATH_CALUDE_taxi_fare_distance_l2546_254651


namespace NUMINAMATH_CALUDE_division_with_remainder_l2546_254634

theorem division_with_remainder (m k : ℤ) (h : m ≠ 0) : 
  ∃ (q r : ℤ), mk + 1 = m * q + r ∧ q = k ∧ r = 1 :=
sorry

end NUMINAMATH_CALUDE_division_with_remainder_l2546_254634


namespace NUMINAMATH_CALUDE_problem_statement_l2546_254660

theorem problem_statement (a b : ℝ) : 
  |a + 2| + (b - 1)^2 = 0 → (a + b)^2005 = -1 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2546_254660


namespace NUMINAMATH_CALUDE_binomial_1500_1_l2546_254630

theorem binomial_1500_1 : Nat.choose 1500 1 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_binomial_1500_1_l2546_254630


namespace NUMINAMATH_CALUDE_period_of_tan_transformed_l2546_254693

open Real

/-- The period of the tangent function with a transformed argument -/
theorem period_of_tan_transformed (f : ℝ → ℝ) :
  (∀ x, f x = tan (2 * x / 3)) →
  (∃ p > 0, ∀ x, f (x + p) = f x) ∧
  (∀ q > 0, (∀ x, f (x + q) = f x) → q ≥ 3 * π / 2) :=
sorry

end NUMINAMATH_CALUDE_period_of_tan_transformed_l2546_254693


namespace NUMINAMATH_CALUDE_parallelogram_acute_angle_cosine_l2546_254678

/-- Given a parallelogram with sides a and b where a ≠ b, if perpendicular lines drawn from
    vertices of obtuse angles form a similar parallelogram, then the cosine of the acute angle α
    is (2ab) / (a² + b²) -/
theorem parallelogram_acute_angle_cosine (a b : ℝ) (h : a ≠ b) :
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧
  (∃ (similar : Bool), similar = true →
    Real.cos α = (2 * a * b) / (a^2 + b^2)) :=
sorry

end NUMINAMATH_CALUDE_parallelogram_acute_angle_cosine_l2546_254678


namespace NUMINAMATH_CALUDE_triangle_inequality_l2546_254601

theorem triangle_inequality (a b c a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) :
  a > 0 → b > 0 → c > 0 →
  a₁ ≥ 0 → b₁ ≥ 0 → c₁ ≥ 0 →
  a₂ ≥ 0 → b₂ ≥ 0 → c₂ ≥ 0 →
  a + b > c → b + c > a → c + a > b →
  a * a₁ * a₂ + b * b₁ * b₂ + c * c₁ * c₂ ≥ a * b * c :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_l2546_254601


namespace NUMINAMATH_CALUDE_painting_time_theorem_l2546_254627

def grace_rate : ℚ := 1 / 6
def henry_rate : ℚ := 1 / 8
def julia_rate : ℚ := 1 / 12
def grace_break : ℚ := 1
def henry_break : ℚ := 1
def julia_break : ℚ := 2

theorem painting_time_theorem :
  ∃ t : ℚ, t > 0 ∧ (grace_rate + henry_rate + julia_rate) * (t - 2) = 1 ∧ t = 14 / 3 := by
  sorry

end NUMINAMATH_CALUDE_painting_time_theorem_l2546_254627


namespace NUMINAMATH_CALUDE_circle_intersections_l2546_254686

-- Define the circle based on the given diameter endpoints
def circle_from_diameter (p1 p2 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  let center := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
  let radius := ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2).sqrt / 2
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the given circle
def given_circle : Set (ℝ × ℝ) := circle_from_diameter (2, 10) (14, 2)

-- Theorem statement
theorem circle_intersections :
  -- The x-coordinates of the intersections with the x-axis are 4 and 12
  (∃ (x : ℝ), (x, 0) ∈ given_circle ↔ x = 4 ∨ x = 12) ∧
  -- There are no intersections with the y-axis
  (∀ (y : ℝ), (0, y) ∉ given_circle) := by
  sorry

end NUMINAMATH_CALUDE_circle_intersections_l2546_254686


namespace NUMINAMATH_CALUDE_shirt_production_l2546_254653

theorem shirt_production (machines1 machines2 : ℕ) 
  (production1 production2 : ℕ) (time1 time2 : ℕ) : 
  machines1 = 12 → 
  machines2 = 20 → 
  production1 = 24 → 
  production2 = 45 → 
  time1 = 18 → 
  time2 = 22 → 
  production1 * time1 + production2 * time2 = 1422 := by
sorry

end NUMINAMATH_CALUDE_shirt_production_l2546_254653


namespace NUMINAMATH_CALUDE_product_of_fifth_and_eighth_l2546_254638

/-- A geometric progression with terms a_n -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), ∀ n : ℕ, a n = a₁ * r^(n - 1)

/-- The 3rd and 10th terms are roots of x^2 - 3x - 5 = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  (a 3)^2 - 3*(a 3) - 5 = 0 ∧ (a 10)^2 - 3*(a 10) - 5 = 0

theorem product_of_fifth_and_eighth (a : ℕ → ℝ) 
  (h1 : geometric_progression a) (h2 : roots_condition a) : 
  a 5 * a 8 = -5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fifth_and_eighth_l2546_254638


namespace NUMINAMATH_CALUDE_rebeccas_income_l2546_254694

/-- Rebecca's annual income problem -/
theorem rebeccas_income (R : ℚ) : 
  (∃ (J : ℚ), J = 18000 ∧ R + 3000 = 0.5 * (R + 3000 + J)) → R = 15000 := by
  sorry

end NUMINAMATH_CALUDE_rebeccas_income_l2546_254694


namespace NUMINAMATH_CALUDE_min_value_of_f_l2546_254692

/-- The quadratic function f(x) = x^2 + 10x + 21 -/
def f (x : ℝ) : ℝ := x^2 + 10*x + 21

/-- Theorem: The minimum value of f(x) = x^2 + 10x + 21 is -4 -/
theorem min_value_of_f : 
  ∀ x : ℝ, f x ≥ -4 ∧ ∃ x₀ : ℝ, f x₀ = -4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2546_254692


namespace NUMINAMATH_CALUDE_eldest_boy_age_l2546_254613

theorem eldest_boy_age (boys : Fin 3 → ℕ) 
  (avg_age : (boys 0 + boys 1 + boys 2) / 3 = 15)
  (proportion : ∃ (x : ℕ), boys 0 = 3 * x ∧ boys 1 = 5 * x ∧ boys 2 = 7 * x) :
  boys 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_eldest_boy_age_l2546_254613


namespace NUMINAMATH_CALUDE_wire_service_reporters_l2546_254621

/-- The percentage of reporters who cover local politics in country x -/
def local_politics_percentage : ℝ := 35

/-- The percentage of reporters who cover politics but not local politics in country x -/
def non_local_politics_percentage : ℝ := 30

/-- The percentage of reporters who do not cover politics -/
def non_politics_percentage : ℝ := 50

theorem wire_service_reporters :
  local_politics_percentage = 35 →
  non_local_politics_percentage = 30 →
  non_politics_percentage = 50 := by
  sorry

end NUMINAMATH_CALUDE_wire_service_reporters_l2546_254621


namespace NUMINAMATH_CALUDE_damage_cost_is_5445_l2546_254603

/-- Calculates the total cost of damages caused by Jack --/
def total_damage_cost (tire_costs : List ℕ) (window_costs : List ℕ) 
  (paint_job_cost : ℕ) (fence_plank_cost : ℕ) (fence_plank_count : ℕ) 
  (fence_labor_cost : ℕ) : ℕ :=
  let tire_total := tire_costs.sum
  let window_total := window_costs.sum
  let fence_total := fence_plank_cost * fence_plank_count + fence_labor_cost
  tire_total + window_total + paint_job_cost + fence_total

/-- Theorem stating that the total damage cost is $5445 --/
theorem damage_cost_is_5445 : 
  total_damage_cost [460, 500, 560] [700, 800, 900] 1200 35 5 150 = 5445 := by
  sorry

end NUMINAMATH_CALUDE_damage_cost_is_5445_l2546_254603


namespace NUMINAMATH_CALUDE_project_hours_difference_l2546_254622

theorem project_hours_difference (total : ℕ) (k p m : ℕ) : 
  total = k + p + m →
  p = 2 * k →
  3 * p = m →
  total = 153 →
  m - k = 85 := by sorry

end NUMINAMATH_CALUDE_project_hours_difference_l2546_254622


namespace NUMINAMATH_CALUDE_cut_prism_edge_count_l2546_254611

/-- A rectangular prism with cut corners -/
structure CutPrism where
  /-- The number of vertices in the original rectangular prism -/
  original_vertices : Nat
  /-- The number of edges in the original rectangular prism -/
  original_edges : Nat
  /-- The number of new edges created by each cut -/
  new_edges_per_cut : Nat
  /-- The planes cutting the prism do not intersect within the prism -/
  non_intersecting_cuts : Prop

/-- The number of edges in the new figure after cutting the corners -/
def new_edge_count (p : CutPrism) : Nat :=
  p.original_edges + p.original_vertices * p.new_edges_per_cut

/-- Theorem stating that a rectangular prism with cut corners has 36 edges -/
theorem cut_prism_edge_count :
  ∀ (p : CutPrism),
  p.original_vertices = 8 →
  p.original_edges = 12 →
  p.new_edges_per_cut = 3 →
  p.non_intersecting_cuts →
  new_edge_count p = 36 := by
  sorry

end NUMINAMATH_CALUDE_cut_prism_edge_count_l2546_254611


namespace NUMINAMATH_CALUDE_instrument_players_fraction_l2546_254696

theorem instrument_players_fraction (total : ℕ) (two_or_more : ℕ) (prob_one : ℚ) :
  total = 800 →
  two_or_more = 32 →
  prob_one = 1/10 + 3/50 →
  (((prob_one * total) + two_or_more) : ℚ) / total = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_instrument_players_fraction_l2546_254696


namespace NUMINAMATH_CALUDE_fourth_fifth_sum_geometric_l2546_254604

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem fourth_fifth_sum_geometric (a : ℕ → ℝ) :
  geometric_sequence a 2 →
  a 1 + a 2 = 3 →
  a 4 + a 5 = 24 := by sorry

end NUMINAMATH_CALUDE_fourth_fifth_sum_geometric_l2546_254604


namespace NUMINAMATH_CALUDE_sqrt_five_position_l2546_254609

/-- Given a sequence where the square of the n-th term is 3n - 1, 
    prove that 2√5 is the 7th term of this sequence. -/
theorem sqrt_five_position (n : ℕ) (a : ℕ → ℝ) 
  (h : ∀ n, a n ^ 2 = 3 * n - 1) : 
  a 7 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_five_position_l2546_254609


namespace NUMINAMATH_CALUDE_line_not_in_third_quadrant_implies_k_nonnegative_l2546_254690

/-- A line is defined by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Predicate to check if a point (x, y) is in the third quadrant -/
def in_third_quadrant (x y : ℝ) : Prop := x < 0 ∧ y < 0

/-- Predicate to check if a line passes through the third quadrant -/
def passes_through_third_quadrant (l : Line) : Prop :=
  ∃ x y : ℝ, y = l.slope * x + l.intercept ∧ in_third_quadrant x y

/-- Theorem: If a line with slope -3 and y-intercept k does not pass through the third quadrant, then k ≥ 0 -/
theorem line_not_in_third_quadrant_implies_k_nonnegative :
  ∀ k : ℝ, ¬passes_through_third_quadrant ⟨-3, k⟩ → k ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_line_not_in_third_quadrant_implies_k_nonnegative_l2546_254690


namespace NUMINAMATH_CALUDE_sum_of_ages_l2546_254620

/-- Given that Rachel is 19 years old and 4 years older than Leah, 
    prove that the sum of their ages is 34. -/
theorem sum_of_ages (rachel_age : ℕ) (leah_age : ℕ) : 
  rachel_age = 19 → rachel_age = leah_age + 4 → rachel_age + leah_age = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_l2546_254620


namespace NUMINAMATH_CALUDE_complex_radical_equality_l2546_254629

theorem complex_radical_equality (a b : ℝ) (ha : a ≥ 0) (hb : b > 0) :
  2.355 * |a^(1/4) - b^(1/6)| = 
  Real.sqrt ((a - 8 * (a^3 * b^2)^(1/6) + 4 * b^(2/3)) / 
             (a^(1/2) - 2 * b^(1/3) + 2 * (a^3 * b^2)^(1/12)) + 3 * b^(1/3)) := by
  sorry

end NUMINAMATH_CALUDE_complex_radical_equality_l2546_254629


namespace NUMINAMATH_CALUDE_complex_subtraction_l2546_254625

theorem complex_subtraction (a b : ℂ) (h1 : a = 4 - 2*I) (h2 : b = 3 + 2*I) :
  a - 2*b = -2 - 6*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l2546_254625


namespace NUMINAMATH_CALUDE_common_chord_triangle_area_l2546_254607

/-- Circle type representing x^2 + y^2 + ax + by + c = 0 --/
structure Circle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Line type representing ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to find the common chord of two circles --/
def commonChord (c1 c2 : Circle) : Line := sorry

/-- Function to find the intersection points of a line with the coordinate axes --/
def axisIntersections (l : Line) : ℝ × ℝ := sorry

/-- Function to calculate the area of a triangle given two side lengths --/
def triangleArea (base height : ℝ) : ℝ := sorry

theorem common_chord_triangle_area :
  let c1 : Circle := { a := 0, b := 0, c := -1 }
  let c2 : Circle := { a := -2, b := 2, c := 0 }
  let commonChordLine := commonChord c1 c2
  let (xIntercept, yIntercept) := axisIntersections commonChordLine
  triangleArea xIntercept yIntercept = 1/8 := by sorry

end NUMINAMATH_CALUDE_common_chord_triangle_area_l2546_254607


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2546_254688

/-- Calculates the molecular weight of a compound given the number of atoms and their atomic weights -/
def molecular_weight (carbon_atoms hydrogen_atoms oxygen_atoms : ℕ) 
  (carbon_weight hydrogen_weight oxygen_weight : ℝ) : ℝ :=
  (carbon_atoms : ℝ) * carbon_weight + 
  (hydrogen_atoms : ℝ) * hydrogen_weight + 
  (oxygen_atoms : ℝ) * oxygen_weight

/-- The molecular weight of a compound with 7 Carbon, 6 Hydrogen, and 2 Oxygen atoms is approximately 122.118 g/mol -/
theorem compound_molecular_weight : 
  ∀ (ε : ℝ), ε > 0 → 
  |molecular_weight 7 6 2 12.01 1.008 16.00 - 122.118| < ε :=
by sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2546_254688


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l2546_254645

theorem difference_of_squares_factorization (y : ℝ) : 64 - 16 * y^2 = 16 * (2 - y) * (2 + y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l2546_254645


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2546_254654

theorem simplify_trig_expression :
  let cos45 := Real.sqrt 2 / 2
  let sin45 := Real.sqrt 2 / 2
  (cos45^3 + sin45^3) / (cos45 + sin45) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2546_254654


namespace NUMINAMATH_CALUDE_marbles_left_after_giving_away_bag_d_l2546_254687

/-- Represents the number of marbles in each bag -/
structure BagContents where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  e : Nat
  f : Nat

/-- The problem setup for James' marbles -/
def jamesMarbles : BagContents := {
  a := 4
  b := 6
  c := 2
  d := 8
  e := 4
  f := 4
}

/-- The total number of marbles James initially had -/
def initialMarbles : Nat := 28

/-- Theorem stating that James has 20 marbles left after giving away Bag D -/
theorem marbles_left_after_giving_away_bag_d :
  initialMarbles - jamesMarbles.d = 20 := by
  sorry

end NUMINAMATH_CALUDE_marbles_left_after_giving_away_bag_d_l2546_254687


namespace NUMINAMATH_CALUDE_exam_mean_score_l2546_254682

theorem exam_mean_score (q σ : ℝ) 
  (h1 : 58 = q - 2 * σ) 
  (h2 : 98 = q + 3 * σ) : 
  q = 74 := by sorry

end NUMINAMATH_CALUDE_exam_mean_score_l2546_254682
