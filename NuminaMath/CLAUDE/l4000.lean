import Mathlib

namespace NUMINAMATH_CALUDE_linear_func_not_in_M_exp_func_in_M_sin_func_in_M_iff_l4000_400022

-- Define the property for a function to be in set M
def in_set_M (f : ℝ → ℝ) : Prop :=
  ∃ T : ℝ, T ≠ 0 ∧ ∀ x : ℝ, f (x + T) = T * f x

-- Part 1
theorem linear_func_not_in_M : ¬ in_set_M (λ x : ℝ ↦ x) := by sorry

-- Part 2
theorem exp_func_in_M (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  (∃ T : ℝ, T > 0 ∧ a^T = T) → in_set_M (λ x : ℝ ↦ a^x) := by sorry

-- Part 3
theorem sin_func_in_M_iff (k : ℝ) :
  in_set_M (λ x : ℝ ↦ Real.sin (k * x)) ↔ ∃ m : ℤ, k = m * Real.pi := by sorry

end NUMINAMATH_CALUDE_linear_func_not_in_M_exp_func_in_M_sin_func_in_M_iff_l4000_400022


namespace NUMINAMATH_CALUDE_arithmetic_simplification_l4000_400020

theorem arithmetic_simplification : 4 * (8 - 3) - 6 / 3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_simplification_l4000_400020


namespace NUMINAMATH_CALUDE_ball_attendees_l4000_400089

theorem ball_attendees :
  ∀ (ladies gentlemen : ℕ),
  ladies + gentlemen < 50 →
  (3 * ladies) / 4 = (5 * gentlemen) / 7 →
  ladies + gentlemen = 41 :=
by
  sorry

end NUMINAMATH_CALUDE_ball_attendees_l4000_400089


namespace NUMINAMATH_CALUDE_quadratic_equal_roots_l4000_400008

theorem quadratic_equal_roots (k : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - k * x - 2 * x + 12 = 0 ∧ 
   ∀ y : ℝ, 3 * y^2 - k * y - 2 * y + 12 = 0 → y = x) ↔ 
  (k = 10 ∨ k = -14) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equal_roots_l4000_400008


namespace NUMINAMATH_CALUDE_base_equivalence_l4000_400065

/-- Converts a base 6 number to base 10 -/
def base6ToBase10 (x y : Nat) : Nat :=
  x * 6 + y

/-- Converts a number in base b to base 10 -/
def baseBToBase10 (b x y z : Nat) : Nat :=
  x * b^2 + y * b + z

theorem base_equivalence :
  ∃! (b : Nat), b > 0 ∧ base6ToBase10 5 3 = baseBToBase10 b 1 1 3 :=
by sorry

end NUMINAMATH_CALUDE_base_equivalence_l4000_400065


namespace NUMINAMATH_CALUDE_leon_order_total_l4000_400074

/-- Calculates the total amount Leon paid for his order, including discounts and delivery fee. -/
def total_paid (toy_organizer_price : ℚ) (toy_organizer_count : ℕ) 
                (gaming_chair_price : ℚ) (gaming_chair_count : ℕ)
                (desk_price : ℚ) (bookshelf_price : ℚ)
                (toy_organizer_discount : ℚ) (gaming_chair_discount : ℚ)
                (delivery_fee_rate : ℚ → ℚ) : ℚ :=
  let toy_organizer_total := toy_organizer_price * toy_organizer_count * (1 - toy_organizer_discount)
  let gaming_chair_total := gaming_chair_price * gaming_chair_count * (1 - gaming_chair_discount)
  let subtotal := toy_organizer_total + gaming_chair_total + desk_price + bookshelf_price
  let total_items := toy_organizer_count + gaming_chair_count + 2
  let delivery_fee := subtotal * delivery_fee_rate total_items
  subtotal + delivery_fee

/-- The statement to be proved -/
theorem leon_order_total :
  let toy_organizer_price : ℚ := 78
  let toy_organizer_count : ℕ := 3
  let gaming_chair_price : ℚ := 83
  let gaming_chair_count : ℕ := 2
  let desk_price : ℚ := 120
  let bookshelf_price : ℚ := 95
  let toy_organizer_discount : ℚ := 0.1
  let gaming_chair_discount : ℚ := 0.05
  let delivery_fee_rate (items : ℚ) : ℚ :=
    if items ≤ 3 then 0.04
    else if items ≤ 5 then 0.06
    else 0.08
  total_paid toy_organizer_price toy_organizer_count 
             gaming_chair_price gaming_chair_count
             desk_price bookshelf_price
             toy_organizer_discount gaming_chair_discount
             delivery_fee_rate = 629.96 := by
  sorry

end NUMINAMATH_CALUDE_leon_order_total_l4000_400074


namespace NUMINAMATH_CALUDE_triangle_properties_l4000_400005

-- Define the triangle ABC
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (-1, -1)

-- Define the equation of angle bisector CD
def angle_bisector_eq (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the equation of the perpendicular bisector of AB
def perp_bisector_eq (x y : ℝ) : Prop := 4*x + 6*y - 3 = 0

-- Define vertex C
def C : ℝ × ℝ := (-1, 2)

theorem triangle_properties :
  (∀ x y : ℝ, angle_bisector_eq x y ↔ x + y - 1 = 0) ∧
  (∀ x y : ℝ, perp_bisector_eq x y ↔ 4*x + 6*y - 3 = 0) ∧
  C = (-1, 2) :=
sorry

end NUMINAMATH_CALUDE_triangle_properties_l4000_400005


namespace NUMINAMATH_CALUDE_jigsaw_puzzle_pieces_l4000_400075

/-- The number of pieces in Luke's jigsaw puzzle -/
def P : ℕ := sorry

/-- The fraction of pieces remaining after each day -/
def remaining_pieces (day : ℕ) : ℚ :=
  match day with
  | 0 => 1
  | 1 => 0.9
  | 2 => 0.72
  | 3 => 0.504
  | _ => 0

theorem jigsaw_puzzle_pieces :
  P = 1000 ∧
  remaining_pieces 1 = 0.9 ∧
  remaining_pieces 2 = 0.72 ∧
  remaining_pieces 3 = 0.504 ∧
  (remaining_pieces 3 * P : ℚ) = 504 :=
by sorry

end NUMINAMATH_CALUDE_jigsaw_puzzle_pieces_l4000_400075


namespace NUMINAMATH_CALUDE_line_circle_intersection_l4000_400061

/-- If a line mx + ny = 0 intersects the circle (x+3)² + (y+1)² = 1 with a chord length of 2, then m/n = -1/3 -/
theorem line_circle_intersection (m n : ℝ) (h : m ≠ 0 ∧ n ≠ 0) :
  (∀ x y : ℝ, m * x + n * y = 0 →
    ((x + 3)^2 + (y + 1)^2 = 1 →
      ∃ x₁ y₁ x₂ y₂ : ℝ,
        m * x₁ + n * y₁ = 0 ∧
        (x₁ + 3)^2 + (y₁ + 1)^2 = 1 ∧
        m * x₂ + n * y₂ = 0 ∧
        (x₂ + 3)^2 + (y₂ + 1)^2 = 1 ∧
        (x₁ - x₂)^2 + (y₁ - y₂)^2 = 4)) →
  m / n = -1/3 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l4000_400061


namespace NUMINAMATH_CALUDE_sum_of_roots_l4000_400080

theorem sum_of_roots (a b : ℝ) 
  (ha : a^3 - 12*a^2 + 47*a - 60 = 0)
  (hb : 8*b^3 - 48*b^2 + 18*b + 162 = 0) : 
  a + b = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l4000_400080


namespace NUMINAMATH_CALUDE_checkerboard_theorem_l4000_400018

def board_size : Nat := 9
def num_lines : Nat := 10

/-- The number of rectangles on the checkerboard -/
def num_rectangles : Nat := (num_lines.choose 2) * (num_lines.choose 2)

/-- The number of squares on the checkerboard -/
def num_squares : Nat := (board_size * (board_size + 1) * (2 * board_size + 1)) / 6

/-- The ratio of squares to rectangles -/
def ratio : Rat := num_squares / num_rectangles

theorem checkerboard_theorem :
  num_rectangles = 2025 ∧
  num_squares = 285 ∧
  ratio = 19 / 135 ∧
  19 + 135 = 154 := by sorry

end NUMINAMATH_CALUDE_checkerboard_theorem_l4000_400018


namespace NUMINAMATH_CALUDE_equation_solution_range_l4000_400063

-- Define the equation
def equation (a m x : ℝ) : Prop :=
  a^(2*x) + (1 + 1/m)*a^x + 1 = 0

-- Define the conditions
def conditions (a m : ℝ) : Prop :=
  a > 0 ∧ a ≠ 1

-- Define the range of m
def m_range (m : ℝ) : Prop :=
  -1/3 ≤ m ∧ m < 0

-- Theorem statement
theorem equation_solution_range (a m : ℝ) :
  conditions a m →
  (∃ x : ℝ, equation a m x ∧ a^x > 0) ↔
  m_range m :=
sorry

end NUMINAMATH_CALUDE_equation_solution_range_l4000_400063


namespace NUMINAMATH_CALUDE_domain_condition_implies_m_range_l4000_400032

theorem domain_condition_implies_m_range (m : ℝ) :
  (∀ x : ℝ, mx^2 + mx + 1 ≥ 0) ↔ 0 ≤ m ∧ m ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_domain_condition_implies_m_range_l4000_400032


namespace NUMINAMATH_CALUDE_intersecting_segments_l4000_400079

/-- Given two intersecting line segments PQ and RS, prove that x + y = 145 -/
theorem intersecting_segments (x y : ℝ) : 
  (60 + (y + 5) = 180) →  -- Linear pair on PQ
  (4 * x = y + 5) →       -- Vertically opposite angles
  x + y = 145 := by sorry

end NUMINAMATH_CALUDE_intersecting_segments_l4000_400079


namespace NUMINAMATH_CALUDE_sum_of_coordinates_of_symmetric_points_l4000_400071

/-- Two points are symmetric with respect to the origin if their coordinates are negatives of each other -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- Given that point P(x,1) is symmetric to point Q(-3,y) with respect to the origin, prove that x + y = 2 -/
theorem sum_of_coordinates_of_symmetric_points :
  ∀ x y : ℝ, symmetric_wrt_origin (x, 1) (-3, y) → x + y = 2 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_coordinates_of_symmetric_points_l4000_400071


namespace NUMINAMATH_CALUDE_percent_swap_l4000_400066

theorem percent_swap (x : ℝ) (h : 0.30 * 0.15 * x = 27) : 0.15 * 0.30 * x = 27 := by
  sorry

end NUMINAMATH_CALUDE_percent_swap_l4000_400066


namespace NUMINAMATH_CALUDE_inequality_condition_l4000_400004

theorem inequality_condition (n : ℕ) (hn : n > 0) :
  (2 * n - 1 : ℝ) * Real.log (1 + Real.log 2023 / Real.log 2) > 
  Real.log 2023 / Real.log 2 * (Real.log 2 + Real.log n) ↔ n ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_inequality_condition_l4000_400004


namespace NUMINAMATH_CALUDE_triangle_arctan_sum_l4000_400040

theorem triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) :
  let angle_C := 2 * Real.pi / 3
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = Real.pi / 4 :=
by sorry

end NUMINAMATH_CALUDE_triangle_arctan_sum_l4000_400040


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l4000_400010

theorem polynomial_identity_sum (a₁ a₂ a₃ d₁ d₂ d₃ : ℝ) :
  (∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 
    (x^2 + a₁ * x + d₁) * (x^2 + a₂ * x + d₂) * (x^2 + a₃ * x + d₃)) →
  a₁ * d₁ + a₂ * d₂ + a₃ * d₃ = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l4000_400010


namespace NUMINAMATH_CALUDE_function_inequality_l4000_400048

/-- Given a function f(x) = x^2 - (a + 1/a)x + 1, if for any x in (1, 3),
    f(x) + (1/a)x > -3 always holds, then a < 4. -/
theorem function_inequality (a : ℝ) (h : a > 0) : 
  (∀ x ∈ Set.Ioo 1 3, x^2 - (a + 1/a)*x + 1 + (1/a)*x > -3) → a < 4 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l4000_400048


namespace NUMINAMATH_CALUDE_inverse_proportion_problem_l4000_400046

/-- Two numbers are inversely proportional if their product is constant -/
def InverselyProportional (x y : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ x * y = k

theorem inverse_proportion_problem (x y : ℝ) :
  InverselyProportional x y →
  x + y = 40 →
  x - y = 8 →
  (∃ y' : ℝ, InverselyProportional 7 y' ∧ y' = 384 / 7) :=
by sorry

end NUMINAMATH_CALUDE_inverse_proportion_problem_l4000_400046


namespace NUMINAMATH_CALUDE_basketball_lineup_combinations_l4000_400057

def total_players : ℕ := 16
def num_quadruplets : ℕ := 4
def num_starters : ℕ := 6
def num_quadruplets_in_lineup : ℕ := 2

theorem basketball_lineup_combinations :
  (Nat.choose num_quadruplets num_quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets + num_quadruplets_in_lineup)
              (num_starters - num_quadruplets_in_lineup)) = 6006 := by
  sorry

end NUMINAMATH_CALUDE_basketball_lineup_combinations_l4000_400057


namespace NUMINAMATH_CALUDE_larger_number_in_ratio_l4000_400011

theorem larger_number_in_ratio (x y : ℝ) : 
  x > 0 → y > 0 → x + y = 40 → x / y = 3 → x = 30 :=
by sorry

end NUMINAMATH_CALUDE_larger_number_in_ratio_l4000_400011


namespace NUMINAMATH_CALUDE_order_of_abc_l4000_400091

noncomputable def a : ℝ := 2 * Real.log 1.01
noncomputable def b : ℝ := Real.log 1.02
noncomputable def c : ℝ := Real.sqrt 1.04 - 1

theorem order_of_abc : a > c ∧ c > b := by sorry

end NUMINAMATH_CALUDE_order_of_abc_l4000_400091


namespace NUMINAMATH_CALUDE_essay_pages_filled_l4000_400033

theorem essay_pages_filled (johnny_words madeline_words timothy_words words_per_page : ℕ) 
  (h1 : johnny_words = 150)
  (h2 : madeline_words = 2 * johnny_words)
  (h3 : timothy_words = madeline_words + 30)
  (h4 : words_per_page = 260) : 
  (johnny_words + madeline_words + timothy_words) / words_per_page = 3 := by
  sorry

end NUMINAMATH_CALUDE_essay_pages_filled_l4000_400033


namespace NUMINAMATH_CALUDE_train_distance_difference_l4000_400050

theorem train_distance_difference (v1 v2 d : ℝ) (hv1 : v1 = 20) (hv2 : v2 = 25) (hd : d = 495) :
  let t := d / (v1 + v2)
  let d1 := v1 * t
  let d2 := v2 * t
  d2 - d1 = 55 := by sorry

end NUMINAMATH_CALUDE_train_distance_difference_l4000_400050


namespace NUMINAMATH_CALUDE_balls_after_2017_steps_l4000_400056

/-- Converts a natural number to its base-5 representation -/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec go (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else go (m / 5) ((m % 5) :: acc)
    go n []

/-- Sums the digits in a list of natural numbers -/
def sumDigits (digits : List ℕ) : ℕ :=
  digits.foldl (· + ·) 0

/-- The sum of digits in the base-5 representation of 2017 equals 9 -/
theorem balls_after_2017_steps : sumDigits (toBase5 2017) = 9 := by
  sorry


end NUMINAMATH_CALUDE_balls_after_2017_steps_l4000_400056


namespace NUMINAMATH_CALUDE_total_weight_N2O3_l4000_400054

-- Define atomic weights
def atomic_weight_N : ℝ := 14.01
def atomic_weight_O : ℝ := 16.00

-- Define the molecular formula of Dinitrogen trioxide
def N_atoms_in_N2O3 : ℕ := 2
def O_atoms_in_N2O3 : ℕ := 3

-- Define the total molecular weight
def total_molecular_weight : ℝ := 228

-- Define the molecular weight of a single molecule of N2O3
def molecular_weight_N2O3 : ℝ := 
  N_atoms_in_N2O3 * atomic_weight_N * O_atoms_in_N2O3 * atomic_weight_O

-- Theorem: The total molecular weight of some moles of N2O3 is 228 g
theorem total_weight_N2O3 : 
  ∃ (n : ℝ), n * molecular_weight_N2O3 = total_molecular_weight :=
sorry

end NUMINAMATH_CALUDE_total_weight_N2O3_l4000_400054


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l4000_400047

theorem ratio_a_to_b (a b : ℝ) (h : (3 * a + 2 * b) / (3 * a - 2 * b) = 3) : a / b = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l4000_400047


namespace NUMINAMATH_CALUDE_inverse_proportion_point_l4000_400037

/-- Given an inverse proportion function y = 14/x passing through the point (a, 7), prove that a = 2 -/
theorem inverse_proportion_point (a : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x, f x = 14 / x) ∧ f a = 7) → a = 2 := by
sorry

end NUMINAMATH_CALUDE_inverse_proportion_point_l4000_400037


namespace NUMINAMATH_CALUDE_spheres_in_base_of_pyramid_l4000_400069

/-- The number of spheres in a regular triangular pyramid with n levels -/
def triangular_pyramid (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem spheres_in_base_of_pyramid (total_spheres : ℕ) (h : total_spheres = 165) :
  ∃ n : ℕ, triangular_pyramid n = total_spheres ∧ triangular_number n = 45 := by
  sorry

end NUMINAMATH_CALUDE_spheres_in_base_of_pyramid_l4000_400069


namespace NUMINAMATH_CALUDE_quarters_for_mowing_lawns_l4000_400030

def penny_value : ℚ := 1 / 100
def quarter_value : ℚ := 25 / 100

def pennies : ℕ := 9
def total_amount : ℚ := 184 / 100

theorem quarters_for_mowing_lawns :
  (total_amount - pennies * penny_value) / quarter_value = 7 := by sorry

end NUMINAMATH_CALUDE_quarters_for_mowing_lawns_l4000_400030


namespace NUMINAMATH_CALUDE_triangle_problem_l4000_400012

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  side_angle_correspondence : True -- This is a placeholder for the correspondence between sides and angles

/-- The theorem statement for the given triangle problem -/
theorem triangle_problem (t : Triangle) :
  (3 * t.a = 2 * t.b) →
  ((t.B = Real.pi / 3 → Real.sin t.C = (Real.sqrt 3 + 3 * Real.sqrt 2) / 6) ∧
   (t.b - t.c = (1 / 3) * t.a → Real.cos t.C = 17 / 27)) := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l4000_400012


namespace NUMINAMATH_CALUDE_pursuer_catches_pursued_l4000_400036

/-- Represents a point on an infinite straight line -/
structure Point where
  position : ℝ

/-- Represents a moving object on the line -/
structure MovingObject where
  initialPosition : Point
  speed : ℝ
  direction : Bool  -- True for positive direction, False for negative

/-- The pursuer (new police car) -/
def pursuer : MovingObject := {
  initialPosition := { position := 0 },
  speed := 1,  -- Normalized to 1
  direction := true  -- Arbitrary initial direction
}

/-- The pursued (stolen police car) -/
def pursued : MovingObject := {
  initialPosition := { position := 0 },  -- Arbitrary initial position
  speed := 0.9,  -- 90% of pursuer's speed
  direction := true  -- Arbitrary initial direction
}

/-- Theorem stating that the pursuer can always catch the pursued -/
theorem pursuer_catches_pursued :
  ∃ (t : ℝ), t ≥ 0 ∧ 
  pursuer.initialPosition.position + t * pursuer.speed = 
  pursued.initialPosition.position + t * pursued.speed :=
sorry

end NUMINAMATH_CALUDE_pursuer_catches_pursued_l4000_400036


namespace NUMINAMATH_CALUDE_multiply_725143_by_999999_l4000_400043

theorem multiply_725143_by_999999 : 725143 * 999999 = 725142274857 := by
  sorry

end NUMINAMATH_CALUDE_multiply_725143_by_999999_l4000_400043


namespace NUMINAMATH_CALUDE_circle_regions_theorem_l4000_400077

/-- The number of regions created by radii and concentric circles inside a circle -/
def num_regions (num_radii : ℕ) (num_concentric_circles : ℕ) : ℕ :=
  (num_concentric_circles + 1) * num_radii

/-- Theorem: Given a circle with 16 radii and 10 concentric circles,
    the number of regions created is 176 -/
theorem circle_regions_theorem :
  num_regions 16 10 = 176 := by
  sorry

#eval num_regions 16 10  -- Should output 176

end NUMINAMATH_CALUDE_circle_regions_theorem_l4000_400077


namespace NUMINAMATH_CALUDE_problem_statement_l4000_400031

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the sequence x_n
def x : ℕ → ℝ := sorry

-- State the theorem
theorem problem_statement :
  (∀ a b : ℝ, a < b → f a < f b) →  -- f is monotonically increasing
  (∀ a : ℝ, f (-a) = -f a) →  -- f is odd
  (∀ n : ℕ, x (n + 1) = x n + 2) →  -- x_n is arithmetic with common difference 2
  (f (x 8) + f (x 9) + f (x 10) + f (x 11) = 0) →  -- given condition
  x 2012 = 4005 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4000_400031


namespace NUMINAMATH_CALUDE_second_group_cost_l4000_400053

/-- The cost of a hotdog in dollars -/
def hotdog_cost : ℚ := 1/2

/-- The cost of a soft drink in dollars -/
def soft_drink_cost : ℚ := 1/2

/-- The number of hotdogs purchased by the first group -/
def first_group_hotdogs : ℕ := 10

/-- The number of soft drinks purchased by the first group -/
def first_group_drinks : ℕ := 5

/-- The total cost of the first group's purchase in dollars -/
def first_group_total : ℚ := 25/2

/-- The number of hotdogs purchased by the second group -/
def second_group_hotdogs : ℕ := 7

/-- The number of soft drinks purchased by the second group -/
def second_group_drinks : ℕ := 4

/-- Theorem stating that the cost of the second group's purchase is $5.50 -/
theorem second_group_cost : 
  (second_group_hotdogs : ℚ) * hotdog_cost + (second_group_drinks : ℚ) * soft_drink_cost = 11/2 := by
  sorry

end NUMINAMATH_CALUDE_second_group_cost_l4000_400053


namespace NUMINAMATH_CALUDE_prob_at_least_one_even_is_five_ninths_l4000_400021

/-- A set of cards labeled 1, 2, and 3 -/
def cards : Finset ℕ := {1, 2, 3}

/-- The event of drawing an even number -/
def is_even (n : ℕ) : Bool := n % 2 = 0

/-- The sample space of two draws with replacement -/
def sample_space : Finset (ℕ × ℕ) :=
  (cards.product cards)

/-- The favorable outcomes (at least one even number) -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  sample_space.filter (λ p => is_even p.1 ∨ is_even p.2)

/-- The probability of drawing at least one even number in two draws -/
def prob_at_least_one_even : ℚ :=
  (favorable_outcomes.card : ℚ) / (sample_space.card : ℚ)

theorem prob_at_least_one_even_is_five_ninths :
  prob_at_least_one_even = 5 / 9 := by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_even_is_five_ninths_l4000_400021


namespace NUMINAMATH_CALUDE_rhombus_symmetry_proposition_l4000_400095

-- Define the set of all rhombuses
variable (Rhombus : Type)

-- Define the property of having central symmetry
variable (has_central_symmetry : Rhombus → Prop)

-- Define the universal quantifier proposition
def universal_proposition : Prop := ∀ r : Rhombus, has_central_symmetry r

-- Define the negation of the proposition
def negation_proposition : Prop := ∃ r : Rhombus, ¬has_central_symmetry r

-- Theorem stating that the original proposition is a universal quantifier
-- and its negation is an existential quantifier with negated property
theorem rhombus_symmetry_proposition :
  (universal_proposition Rhombus has_central_symmetry) ∧
  (negation_proposition Rhombus has_central_symmetry) :=
sorry

end NUMINAMATH_CALUDE_rhombus_symmetry_proposition_l4000_400095


namespace NUMINAMATH_CALUDE_vlads_height_in_feet_l4000_400014

/-- Represents a height in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ
  h_inches_lt_12 : inches < 12

/-- Converts a height to total inches -/
def Height.to_inches (h : Height) : ℕ :=
  h.feet * 12 + h.inches

/-- The height of Vlad's sister -/
def sister_height : Height :=
  { feet := 2, inches := 10, h_inches_lt_12 := by sorry }

/-- The difference in height between Vlad and his sister in inches -/
def height_difference : ℕ := 41

/-- Theorem: Vlad's height in feet is 6 -/
theorem vlads_height_in_feet :
  (Height.to_inches sister_height + height_difference) / 12 = 6 := by sorry

end NUMINAMATH_CALUDE_vlads_height_in_feet_l4000_400014


namespace NUMINAMATH_CALUDE_interest_rate_equation_l4000_400019

/-- Given a principal that doubles in 10 years with semiannual compounding,
    this theorem states the equation that the annual interest rate must satisfy. -/
theorem interest_rate_equation (r : ℝ) : 
  (∀ P : ℝ, P > 0 → 2 * P = P * (1 + r / 2) ^ 20) ↔ 2 = (1 + r / 2) ^ 20 :=
sorry

end NUMINAMATH_CALUDE_interest_rate_equation_l4000_400019


namespace NUMINAMATH_CALUDE_optimal_scheme_is_best_l4000_400052

/-- Represents a horticultural design scheme -/
structure Scheme where
  a : ℕ  -- number of A type designs
  b : ℕ  -- number of B type designs

/-- Checks if a scheme is feasible given the constraints -/
def is_feasible (s : Scheme) : Prop :=
  s.a + s.b = 50 ∧
  80 * s.a + 50 * s.b ≤ 3490 ∧
  40 * s.a + 90 * s.b ≤ 2950

/-- Calculates the cost of a scheme -/
def cost (s : Scheme) : ℕ :=
  800 * s.a + 960 * s.b

/-- The optimal scheme -/
def optimal_scheme : Scheme :=
  ⟨33, 17⟩

theorem optimal_scheme_is_best :
  is_feasible optimal_scheme ∧
  ∀ s : Scheme, is_feasible s → cost s ≥ cost optimal_scheme :=
sorry

end NUMINAMATH_CALUDE_optimal_scheme_is_best_l4000_400052


namespace NUMINAMATH_CALUDE_line_equation_from_intercept_and_angle_l4000_400002

/-- The equation of a line with given x-intercept and inclination angle -/
theorem line_equation_from_intercept_and_angle (x_intercept : ℝ) (angle : ℝ) :
  x_intercept = 2 ∧ angle = 135 * π / 180 →
  ∀ x y : ℝ, (x + y - 2 = 0) ↔ (y = (x - x_intercept) * Real.tan angle) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_from_intercept_and_angle_l4000_400002


namespace NUMINAMATH_CALUDE_james_printing_problem_l4000_400016

/-- Calculates the minimum number of sheets required for printing books -/
def sheets_required (num_books : ℕ) (pages_per_book : ℕ) (sides_per_sheet : ℕ) (pages_per_side : ℕ) : ℕ :=
  let total_pages := num_books * pages_per_book
  let pages_per_sheet := sides_per_sheet * pages_per_side
  (total_pages + pages_per_sheet - 1) / pages_per_sheet

theorem james_printing_problem :
  sheets_required 5 800 3 6 = 223 := by
  sorry

end NUMINAMATH_CALUDE_james_printing_problem_l4000_400016


namespace NUMINAMATH_CALUDE_total_items_for_40_notebooks_l4000_400041

/-- Given the number of notebooks, calculate the total number of items (notebooks, pens, and pencils) -/
def total_items (notebooks : ℕ) : ℕ :=
  notebooks + (notebooks + 80) + (notebooks + 45)

/-- Theorem stating that given 40 notebooks, the total number of items is 245 -/
theorem total_items_for_40_notebooks : total_items 40 = 245 := by
  sorry

end NUMINAMATH_CALUDE_total_items_for_40_notebooks_l4000_400041


namespace NUMINAMATH_CALUDE_subtract_fractions_l4000_400023

theorem subtract_fractions : (5 : ℚ) / 9 - (1 : ℚ) / 6 = (7 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_subtract_fractions_l4000_400023


namespace NUMINAMATH_CALUDE_f_monotone_implies_a_range_l4000_400073

/-- A piecewise function f depending on parameter a -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then -x^2 + 2*(a-1)*x else (8-a)*x + 4

/-- f is monotonically increasing on ℝ -/
def monotone_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem stating that if f is monotonically increasing, then 2 ≤ a ≤ 5 -/
theorem f_monotone_implies_a_range (a : ℝ) :
  monotone_increasing (f a) → 2 ≤ a ∧ a ≤ 5 :=
by
  sorry

#check f_monotone_implies_a_range

end NUMINAMATH_CALUDE_f_monotone_implies_a_range_l4000_400073


namespace NUMINAMATH_CALUDE_polynomial_roots_l4000_400013

theorem polynomial_roots : ∃ (x₁ x₂ x₃ x₄ : ℝ), 
  (x₁ = 0 ∧ x₂ = 1/3 ∧ x₃ = 2 ∧ x₄ = -5) ∧
  (∀ x : ℝ, 3*x^4 + 11*x^3 - 28*x^2 + 10*x = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l4000_400013


namespace NUMINAMATH_CALUDE_sector_arc_length_l4000_400082

/-- Given a circular sector with a central angle of 150° and a radius of 6 cm,
    the arc length is 5π cm. -/
theorem sector_arc_length :
  let θ : ℝ := 150  -- Central angle in degrees
  let r : ℝ := 6    -- Radius in cm
  let L : ℝ := (θ / 360) * (2 * Real.pi * r)  -- Arc length formula
  L = 5 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_l4000_400082


namespace NUMINAMATH_CALUDE_color_film_fraction_l4000_400068

theorem color_film_fraction (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) : 
  let total_bw := 30 * x
  let total_color := 6 * y
  let bw_selected_percent := y / x
  let bw_selected := bw_selected_percent * total_bw / 100
  let color_selected := total_color
  let total_selected := bw_selected + color_selected
  color_selected / total_selected = 20 / 21 := by
sorry

end NUMINAMATH_CALUDE_color_film_fraction_l4000_400068


namespace NUMINAMATH_CALUDE_fraction_problem_l4000_400038

theorem fraction_problem (n d : ℚ) : 
  d = 2 * n - 1 → 
  (n + 1) / (d + 1) = 3 / 5 → 
  n / d = 5 / 9 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l4000_400038


namespace NUMINAMATH_CALUDE_constant_term_zero_implies_a_equals_six_l4000_400087

theorem constant_term_zero_implies_a_equals_six (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (a + 2) * x^2 + b * x + (a - 6) = 0) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_zero_implies_a_equals_six_l4000_400087


namespace NUMINAMATH_CALUDE_sculpture_exposed_area_l4000_400076

/-- Represents a layer in the sculpture -/
structure Layer where
  cubes : Nat
  exposed_top : Nat
  exposed_side : Nat

/-- Represents the sculpture -/
def Sculpture : List Layer := [
  { cubes := 1, exposed_top := 1, exposed_side := 4 },
  { cubes := 4, exposed_top := 4, exposed_side := 12 },
  { cubes := 9, exposed_top := 9, exposed_side := 6 },
  { cubes := 6, exposed_top := 6, exposed_side := 0 }
]

/-- Calculates the exposed surface area of a layer -/
def exposedAreaLayer (layer : Layer) : Nat :=
  layer.exposed_top + layer.exposed_side

/-- Calculates the total exposed surface area of the sculpture -/
def totalExposedArea (sculpture : List Layer) : Nat :=
  List.sum (List.map exposedAreaLayer sculpture)

/-- Theorem: The total exposed surface area of the sculpture is 42 square meters -/
theorem sculpture_exposed_area :
  totalExposedArea Sculpture = 42 := by
  sorry

end NUMINAMATH_CALUDE_sculpture_exposed_area_l4000_400076


namespace NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_equals_four_l4000_400096

theorem tan_alpha_two_implies_fraction_equals_four (α : Real) 
  (h : Real.tan α = 2) : 
  (Real.sin α + 2 * Real.cos α) / (Real.sin α - Real.cos α) = 4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_two_implies_fraction_equals_four_l4000_400096


namespace NUMINAMATH_CALUDE_average_physics_math_l4000_400088

/-- Given the scores of three subjects, prove the average of two specific subjects -/
theorem average_physics_math (total_average : ℝ) (physics_chem_average : ℝ) (physics_score : ℝ) : 
  total_average = 60 →
  physics_chem_average = 70 →
  physics_score = 140 →
  (physics_score + (3 * total_average - physics_score - 
    (2 * physics_chem_average - physics_score))) / 2 = 90 := by
  sorry


end NUMINAMATH_CALUDE_average_physics_math_l4000_400088


namespace NUMINAMATH_CALUDE_pencil_total_length_l4000_400003

/-- The total length of a pencil with specified colored sections -/
def pencil_length (purple_length black_length blue_length : ℝ) : ℝ :=
  purple_length + black_length + blue_length

/-- Theorem stating that a pencil with 3 cm purple, 2 cm black, and 1 cm blue sections is 6 cm long -/
theorem pencil_total_length : 
  pencil_length 3 2 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_pencil_total_length_l4000_400003


namespace NUMINAMATH_CALUDE_volume_of_special_prism_l4000_400000

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cube with given edge length -/
structure Cube where
  edgeLength : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  base : Set Point3D
  height : ℝ

/-- Given a cube, returns the midpoints of edges AB, AD, and AA₁ -/
def getMidpoints (c : Cube) : Set Point3D :=
  { Point3D.mk (c.edgeLength / 2) 0 0,
    Point3D.mk 0 (c.edgeLength / 2) 0,
    Point3D.mk 0 0 (c.edgeLength / 2) }

/-- Constructs a triangular prism from given midpoints -/
def constructPrism (midpoints : Set Point3D) (c : Cube) : TriangularPrism :=
  sorry

/-- Calculates the volume of a triangular prism -/
def prismVolume (p : TriangularPrism) : ℝ :=
  sorry

theorem volume_of_special_prism (c : Cube) :
  c.edgeLength = 1 →
  let midpoints := getMidpoints c
  let prism := constructPrism midpoints c
  prismVolume prism = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_volume_of_special_prism_l4000_400000


namespace NUMINAMATH_CALUDE_M_equals_N_l4000_400086

def M : Set ℝ := {x : ℝ | ∃ y : ℝ, y = Real.sqrt x}
def N : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}

theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l4000_400086


namespace NUMINAMATH_CALUDE_first_class_rate_total_l4000_400078

-- Define the pass rate
def pass_rate : ℝ := 0.95

-- Define the rate of first-class products among qualified products
def first_class_rate_qualified : ℝ := 0.20

-- Theorem statement
theorem first_class_rate_total (pass_rate : ℝ) (first_class_rate_qualified : ℝ) :
  pass_rate * first_class_rate_qualified = 0.19 := by
  sorry

end NUMINAMATH_CALUDE_first_class_rate_total_l4000_400078


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l4000_400094

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a₁ + (n - 1 : ℝ) * d

/-- The 10th term of the specific arithmetic sequence -/
theorem tenth_term_of_sequence : 
  arithmeticSequenceTerm 3 6 10 = 57 := by
sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l4000_400094


namespace NUMINAMATH_CALUDE_model_a_sample_size_l4000_400017

/-- Calculates the number of items to select in stratified sampling -/
def stratified_sample_size (total_production : ℕ) (model_production : ℕ) (sample_size : ℕ) : ℕ :=
  (model_production * sample_size) / total_production

/-- Proves that the stratified sample size for Model A is 6 -/
theorem model_a_sample_size :
  stratified_sample_size 9200 1200 46 = 6 := by
sorry

end NUMINAMATH_CALUDE_model_a_sample_size_l4000_400017


namespace NUMINAMATH_CALUDE_emilia_valentin_numbers_l4000_400015

theorem emilia_valentin_numbers (x : ℝ) : 
  (5 + 9) / 2 = 7 ∧ 
  (5 + x) / 2 = 10 ∧ 
  (x + 9) / 2 = 12 → 
  x = 15 := by
sorry

end NUMINAMATH_CALUDE_emilia_valentin_numbers_l4000_400015


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l4000_400001

/-- Represents the dimensions of a rectangle -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the maximum number of tiles that can fit on a floor -/
def maxTiles (floor : Dimensions) (tile : Dimensions) : ℕ :=
  max
    ((floor.length / tile.length) * (floor.width / tile.width))
    ((floor.length / tile.width) * (floor.width / tile.length))

/-- Theorem stating the maximum number of tiles that can be placed on the given floor -/
theorem max_tiles_on_floor :
  let floor := Dimensions.mk 560 240
  let tile := Dimensions.mk 60 56
  maxTiles floor tile = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l4000_400001


namespace NUMINAMATH_CALUDE_percentage_of_female_students_l4000_400027

theorem percentage_of_female_students 
  (total_students : ℕ) 
  (female_percentage : ℝ) 
  (brunette_percentage : ℝ) 
  (under_5ft_percentage : ℝ) 
  (under_5ft_count : ℕ) :
  total_students = 200 →
  brunette_percentage = 50 →
  under_5ft_percentage = 50 →
  under_5ft_count = 30 →
  (female_percentage / 100) * (brunette_percentage / 100) * (under_5ft_percentage / 100) * total_students = under_5ft_count →
  female_percentage = 60 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_female_students_l4000_400027


namespace NUMINAMATH_CALUDE_certain_number_proof_l4000_400093

/-- Given that g is the smallest positive integer such that n * g is a perfect square, 
    and g = 14, prove that n = 14 -/
theorem certain_number_proof (n : ℕ) (g : ℕ) (h1 : g = 14) 
  (h2 : ∃ m : ℕ, n * g = m^2)
  (h3 : ∀ k < g, ¬∃ m : ℕ, n * k = m^2) : n = 14 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l4000_400093


namespace NUMINAMATH_CALUDE_sum_of_consecutive_integers_l4000_400081

theorem sum_of_consecutive_integers (n : ℤ) : n + (n + 1) + (n + 2) + (n + 3) = 22 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_integers_l4000_400081


namespace NUMINAMATH_CALUDE_mark_sprint_distance_l4000_400009

/-- The distance traveled by Mark given his sprint duration and speed -/
theorem mark_sprint_distance (duration : ℝ) (speed : ℝ) (h1 : duration = 24.0) (h2 : speed = 6.0) :
  duration * speed = 144.0 := by
  sorry

end NUMINAMATH_CALUDE_mark_sprint_distance_l4000_400009


namespace NUMINAMATH_CALUDE_polynomial_identity_sum_l4000_400026

theorem polynomial_identity_sum (d1 d2 d3 e1 e2 e3 : ℝ) : 
  (∀ x : ℝ, x^7 - x^6 + x^4 - x^3 + x^2 - x + 1 = 
    (x^2 + d1*x + e1) * (x^2 + d2*x + e2) * (x^2 + d3*x + e3)) →
  d1*e1 + d2*e2 + d3*e3 = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_identity_sum_l4000_400026


namespace NUMINAMATH_CALUDE_average_carnations_example_l4000_400062

/-- The average number of carnations in three bouquets -/
def average_carnations (b1 b2 b3 : ℕ) : ℚ :=
  (b1 + b2 + b3 : ℚ) / 3

/-- Theorem: The average number of carnations in three bouquets containing 9, 14, and 13 carnations respectively is 12 -/
theorem average_carnations_example : average_carnations 9 14 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_average_carnations_example_l4000_400062


namespace NUMINAMATH_CALUDE_division_example_exists_l4000_400006

theorem division_example_exists : ∃ (D d q : ℕ+), 
  (D : ℚ) / (d : ℚ) = q ∧ 
  (q : ℚ) = (D : ℚ) / 5 ∧ 
  (q : ℚ) = 7 * (d : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_division_example_exists_l4000_400006


namespace NUMINAMATH_CALUDE_total_divisors_xyz_l4000_400035

-- Define the variables and their properties
variable (p q r : ℕ) -- Natural numbers for primes
variable (hp : Prime p) -- p is prime
variable (hq : Prime q) -- q is prime
variable (hr : Prime r) -- r is prime
variable (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) -- p, q, and r are distinct

-- Define x, y, and z
def x : ℕ := p^2
def y : ℕ := q^2
def z : ℕ := r^4

-- State the theorem
theorem total_divisors_xyz (p q r : ℕ) (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (hdistinct : p ≠ q ∧ p ≠ r ∧ q ≠ r) :
  (Finset.card (Nat.divisors ((x p)^3 * (y q)^4 * (z r)^2))) = 567 := by
  sorry

end NUMINAMATH_CALUDE_total_divisors_xyz_l4000_400035


namespace NUMINAMATH_CALUDE_basket_weight_l4000_400055

/-- Proves that the weight of an empty basket is 1.40 kg given specific conditions -/
theorem basket_weight (total_weight : Real) (remaining_weight : Real) 
  (h1 : total_weight = 11.48)
  (h2 : remaining_weight = 8.12) : 
  ∃ (basket_weight : Real) (apple_weight : Real),
    basket_weight = 1.40 ∧ 
    apple_weight > 0 ∧
    total_weight = basket_weight + 12 * apple_weight ∧
    remaining_weight = basket_weight + 8 * apple_weight :=
by
  sorry

end NUMINAMATH_CALUDE_basket_weight_l4000_400055


namespace NUMINAMATH_CALUDE_steak_price_per_pound_l4000_400029

theorem steak_price_per_pound (steak_price : ℚ) : 
  4.5 * steak_price + 1.5 * 8 = 42 → steak_price = 20 / 3 := by
  sorry

end NUMINAMATH_CALUDE_steak_price_per_pound_l4000_400029


namespace NUMINAMATH_CALUDE_conjunction_is_false_l4000_400072

theorem conjunction_is_false :
  let p := ∀ x : ℝ, x < 1 → x < 2
  let q := ∃ x : ℝ, x^2 + 1 = 0
  ¬(p ∧ q) := by sorry

end NUMINAMATH_CALUDE_conjunction_is_false_l4000_400072


namespace NUMINAMATH_CALUDE_rounding_317500_equals_31_8_ten_thousand_l4000_400051

-- Define rounding to the nearest thousand
def roundToThousand (n : ℕ) : ℕ :=
  (n + 500) / 1000 * 1000

-- Define representation in ten thousands
def toTenThousand (n : ℕ) : ℚ :=
  n / 10000

-- Theorem statement
theorem rounding_317500_equals_31_8_ten_thousand :
  toTenThousand (roundToThousand 317500) = 31.8 := by
  sorry

end NUMINAMATH_CALUDE_rounding_317500_equals_31_8_ten_thousand_l4000_400051


namespace NUMINAMATH_CALUDE_matrix_equation_l4000_400060

-- Define the matrices
def A : Matrix (Fin 2) (Fin 2) ℚ := !![2, -5; 4, -3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![-20, -8; 12, 5]
def N : Matrix (Fin 2) (Fin 2) ℚ := !![46/7, -58/7; -28/7, 35/7]

-- State the theorem
theorem matrix_equation : N * A = B := by sorry

end NUMINAMATH_CALUDE_matrix_equation_l4000_400060


namespace NUMINAMATH_CALUDE_profit_ratio_from_investment_l4000_400058

/-- The profit ratio of two partners given their investment ratio and investment durations -/
theorem profit_ratio_from_investment 
  (p_investment q_investment : ℕ) 
  (p_duration q_duration : ℚ) 
  (h_investment_ratio : p_investment * 5 = q_investment * 7)
  (h_p_duration : p_duration = 5)
  (h_q_duration : q_duration = 11) :
  p_investment * p_duration * 11 = q_investment * q_duration * 7 :=
by sorry

end NUMINAMATH_CALUDE_profit_ratio_from_investment_l4000_400058


namespace NUMINAMATH_CALUDE_quadratic_root_condition_l4000_400034

theorem quadratic_root_condition (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ + m - 1 = 0 ∧ 
                x₂^2 - 4*x₂ + m - 1 = 0 ∧ 
                3*x₁*x₂ - x₁ - x₂ > 2) →
  3 < m ∧ m ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_condition_l4000_400034


namespace NUMINAMATH_CALUDE_line_slope_l4000_400067

-- Define the parametric equations of the line
def x (t : ℝ) : ℝ := 3 + 4 * t
def y (t : ℝ) : ℝ := 4 - 5 * t

-- State the theorem
theorem line_slope :
  ∃ m : ℝ, ∀ t₁ t₂ : ℝ, t₁ ≠ t₂ → (y t₂ - y t₁) / (x t₂ - x t₁) = m ∧ m = -5/4 :=
sorry

end NUMINAMATH_CALUDE_line_slope_l4000_400067


namespace NUMINAMATH_CALUDE_parabola_x_axis_intersection_l4000_400042

theorem parabola_x_axis_intersection :
  let f (x : ℝ) := x^2 - 2*x - 3
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 ∧ ∀ x, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_parabola_x_axis_intersection_l4000_400042


namespace NUMINAMATH_CALUDE_bird_speed_indeterminate_l4000_400025

/-- A structure representing the problem scenario -/
structure ScenarioData where
  train_speed : ℝ
  bird_distance : ℝ

/-- A function that attempts to calculate the bird's speed -/
def calculate_bird_speed (data : ScenarioData) : Option ℝ :=
  none

/-- Theorem stating that the bird's speed cannot be uniquely determined -/
theorem bird_speed_indeterminate (data : ScenarioData) 
  (h1 : data.train_speed = 60)
  (h2 : data.bird_distance = 120) :
  ∀ (s : ℝ), s > 0 → ∃ (t : ℝ), t > 0 ∧ s * t = data.bird_distance :=
sorry

#check bird_speed_indeterminate

end NUMINAMATH_CALUDE_bird_speed_indeterminate_l4000_400025


namespace NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l4000_400028

theorem negation_of_cube_odd_is_odd :
  (¬ ∀ n : ℤ, Odd n → Odd (n^3)) ↔ (∃ n : ℤ, Odd n ∧ Even (n^3)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_cube_odd_is_odd_l4000_400028


namespace NUMINAMATH_CALUDE_folded_paper_distance_l4000_400045

theorem folded_paper_distance (sheet_area : ℝ) (folded_leg : ℝ) : 
  sheet_area = 6 →
  folded_leg ^ 2 / 2 = sheet_area - folded_leg ^ 2 →
  Real.sqrt (2 * folded_leg ^ 2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_distance_l4000_400045


namespace NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l4000_400049

theorem no_real_solution_for_sqrt_equation :
  ¬∃ (x : ℝ), Real.sqrt (3 - Real.sqrt x) = 2 := by
sorry

end NUMINAMATH_CALUDE_no_real_solution_for_sqrt_equation_l4000_400049


namespace NUMINAMATH_CALUDE_g_of_one_eq_neg_25_l4000_400064

/-- g is a rational function satisfying the given equation for all non-zero x -/
def g_equation (g : ℚ → ℚ) : Prop :=
  ∀ x : ℚ, x ≠ 0 → 4 * g (2 / x) + 3 * g x / x = 2 * x^3 - x

/-- Theorem: If g satisfies the equation, then g(1) = -25 -/
theorem g_of_one_eq_neg_25 (g : ℚ → ℚ) (h : g_equation g) : g 1 = -25 := by
  sorry

end NUMINAMATH_CALUDE_g_of_one_eq_neg_25_l4000_400064


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4000_400090

/-- An isosceles triangle with two sides of lengths 1 and 2 has a perimeter of 5 -/
theorem isosceles_triangle_perimeter : ∀ a b c : ℝ,
  a = 1 ∧ b = 2 ∧ c = 2 →  -- Two sides are 1 and 2, the third side must be 2 to form an isosceles triangle
  a + b + c = 5 :=         -- The perimeter is the sum of all sides
by
  sorry


end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l4000_400090


namespace NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l4000_400039

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | x^2 - 4*x - 5 < 0}

-- Define set N
def N : Set ℝ := {x | x ≥ 1}

-- State the theorem
theorem intersection_of_M_and_complement_of_N :
  M ∩ (Set.compl N) = {x | -1 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_complement_of_N_l4000_400039


namespace NUMINAMATH_CALUDE_lemons_for_drinks_l4000_400097

/-- The number of lemons needed to make a certain amount of lemonade and lemon tea -/
def lemons_needed (lemonade_ratio : ℚ) (tea_ratio : ℚ) (lemonade_gallons : ℚ) (tea_gallons : ℚ) : ℚ :=
  lemonade_ratio * lemonade_gallons + tea_ratio * tea_gallons

/-- Theorem stating the number of lemons needed for 6 gallons of lemonade and 5 gallons of lemon tea -/
theorem lemons_for_drinks : 
  let lemonade_ratio : ℚ := 36 / 48
  let tea_ratio : ℚ := 20 / 10
  lemons_needed lemonade_ratio tea_ratio 6 5 = 29/2 := by
  sorry

#eval (29 : ℚ) / 2  -- To verify that 29/2 is indeed equal to 14.5

end NUMINAMATH_CALUDE_lemons_for_drinks_l4000_400097


namespace NUMINAMATH_CALUDE_larger_number_proof_l4000_400059

theorem larger_number_proof (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l4000_400059


namespace NUMINAMATH_CALUDE_min_distance_is_8_l4000_400070

-- Define the condition function
def condition (a b c d : ℝ) : Prop :=
  (a - 2 * Real.exp a) / b = (1 - c) / (d - 1) ∧ (a - 2 * Real.exp a) / b = 1

-- Define the distance function
def distance (a b c d : ℝ) : ℝ :=
  (a - c)^2 + (b - d)^2

-- Theorem statement
theorem min_distance_is_8 :
  ∀ a b c d : ℝ, condition a b c d → 
  ∀ x y z w : ℝ, condition x y z w →
  distance a b c d ≥ 8 ∧ (∃ a₀ b₀ c₀ d₀ : ℝ, condition a₀ b₀ c₀ d₀ ∧ distance a₀ b₀ c₀ d₀ = 8) :=
sorry

end NUMINAMATH_CALUDE_min_distance_is_8_l4000_400070


namespace NUMINAMATH_CALUDE_area_polygon1_area_polygon2_area_polygon3_l4000_400085

-- Define the polygons
def polygon1 := {(x, y) : ℝ × ℝ | |x| ≤ 1 ∧ |y| ≤ 1}
def polygon2 := {(x, y) : ℝ × ℝ | |x| + |y| ≤ 10}
def polygon3 := {(x, y) : ℝ × ℝ | |x| + |y| + |x+y| ≤ 2020}

-- Define the areas
def area (s : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statements
theorem area_polygon1 : area polygon1 = 4 := by sorry

theorem area_polygon2 : area polygon2 = 200 := by sorry

theorem area_polygon3 : area polygon3 = 3060300 := by sorry

end NUMINAMATH_CALUDE_area_polygon1_area_polygon2_area_polygon3_l4000_400085


namespace NUMINAMATH_CALUDE_will_toy_cost_l4000_400007

def toy_cost (initial_money : ℕ) (game_cost : ℕ) (num_toys : ℕ) : ℚ :=
  (initial_money - game_cost) / num_toys

theorem will_toy_cost :
  toy_cost 57 27 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_will_toy_cost_l4000_400007


namespace NUMINAMATH_CALUDE_f_max_value_l4000_400099

/-- The quadratic function f(x) = -x^2 + 2x + 3 -/
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

/-- Theorem: The maximum value of f(x) = -x^2 + 2x + 3 is 4 -/
theorem f_max_value : ∃ (M : ℝ), M = 4 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end NUMINAMATH_CALUDE_f_max_value_l4000_400099


namespace NUMINAMATH_CALUDE_trigonometric_identity_l4000_400024

open Real

theorem trigonometric_identity (α β : ℝ) 
  (h1 : sin (π - α) - 2 * sin ((π / 2) + α) = 0) 
  (h2 : tan (α + β) = -1) : 
  (sin α * cos α + sin α ^ 2 = 6 / 5) ∧ 
  (tan β = 3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l4000_400024


namespace NUMINAMATH_CALUDE_parabola_values_l4000_400044

/-- A parabola passing through (1, 1) with a specific tangent line -/
def Parabola (a b : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x - 7

theorem parabola_values (a b : ℝ) :
  (Parabola a b 1 = 1) ∧ 
  (4 * 1 - Parabola a b 1 - 3 = 0) ∧
  (2 * a * 1 + b = 4) →
  a = -4 ∧ b = 12 := by sorry

end NUMINAMATH_CALUDE_parabola_values_l4000_400044


namespace NUMINAMATH_CALUDE_negative_three_inverse_l4000_400098

theorem negative_three_inverse : (-3 : ℚ)⁻¹ = -1/3 := by sorry

end NUMINAMATH_CALUDE_negative_three_inverse_l4000_400098


namespace NUMINAMATH_CALUDE_x_yz_equals_12_l4000_400092

theorem x_yz_equals_12 (x y z : ℝ) (h : x * (x + y + z) = x^2 + 12) : x * (y + z) = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_yz_equals_12_l4000_400092


namespace NUMINAMATH_CALUDE_function_properties_l4000_400083

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sqrt 3 * Real.cos (ω * x) * Real.sin (ω * x) - 2 * (Real.cos (ω * x))^2 + 1

theorem function_properties (ω : ℝ) (h_ω : ω > 0) 
  (h_period : ∀ x, f ω (x + π) = f ω x) :
  (∀ x, f ω x = 2 * Real.sin (2 * x - π / 6)) ∧
  (∃ x_min ∈ Set.Icc 0 (π / 2), ∀ x ∈ Set.Icc 0 (π / 2), f ω x_min ≤ f ω x) ∧
  (∃ x_max ∈ Set.Icc 0 (π / 2), ∀ x ∈ Set.Icc 0 (π / 2), f ω x ≤ f ω x_max) ∧
  (f ω 0 = -1) ∧
  (f ω (π / 3) = 2) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), -1 ≤ f ω x ∧ f ω x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l4000_400083


namespace NUMINAMATH_CALUDE_signal_arrangements_l4000_400084

def num_red_flags : ℕ := 3
def num_white_flags : ℕ := 2
def total_flags : ℕ := num_red_flags + num_white_flags

theorem signal_arrangements : (total_flags.choose num_red_flags) = 10 := by
  sorry

end NUMINAMATH_CALUDE_signal_arrangements_l4000_400084
