import Mathlib

namespace NUMINAMATH_CALUDE_power_sum_of_i_l4048_404825

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^23 + i^67 + i^101 = -i := by sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l4048_404825


namespace NUMINAMATH_CALUDE_two_books_different_genres_l4048_404811

/-- The number of ways to choose two books of different genres -/
def choose_two_books (mystery fantasy biography : ℕ) : ℕ :=
  mystery * fantasy + mystery * biography + fantasy * biography

/-- Theorem: Given 4 mystery novels, 3 fantasy novels, and 3 biographies,
    the number of ways to choose two books of different genres is 33 -/
theorem two_books_different_genres :
  choose_two_books 4 3 3 = 33 := by
  sorry

end NUMINAMATH_CALUDE_two_books_different_genres_l4048_404811


namespace NUMINAMATH_CALUDE_decimal_arithmetic_l4048_404846

theorem decimal_arithmetic : 
  (∃ x : ℝ, x = 3.92 + 0.4 ∧ x = 3.96) ∧
  (∃ y : ℝ, y = 4.93 - 1.5 ∧ y = 3.43) := by
  sorry

end NUMINAMATH_CALUDE_decimal_arithmetic_l4048_404846


namespace NUMINAMATH_CALUDE_bridgette_guest_count_l4048_404869

/-- The number of guests Bridgette is inviting -/
def bridgette_guests : ℕ := 84

/-- The number of guests Alex is inviting -/
def alex_guests : ℕ := (2 * bridgette_guests) / 3

/-- The number of extra plates the caterer makes -/
def extra_plates : ℕ := 10

/-- The number of asparagus spears per plate -/
def spears_per_plate : ℕ := 8

/-- The total number of asparagus spears needed -/
def total_spears : ℕ := 1200

theorem bridgette_guest_count : 
  spears_per_plate * (bridgette_guests + alex_guests + extra_plates) = total_spears :=
by sorry

end NUMINAMATH_CALUDE_bridgette_guest_count_l4048_404869


namespace NUMINAMATH_CALUDE_student_group_assignments_non_empty_coin_subsets_l4048_404870

/-- The number of students --/
def num_students : ℕ := 5

/-- The number of groups --/
def num_groups : ℕ := 2

/-- The number of coins --/
def num_coins : ℕ := 7

/-- Theorem for the number of ways to assign students to groups --/
theorem student_group_assignments :
  (num_groups : ℕ) ^ num_students = 32 := by sorry

/-- Theorem for the number of non-empty subsets of coins --/
theorem non_empty_coin_subsets :
  2 ^ num_coins - 1 = 127 := by sorry

end NUMINAMATH_CALUDE_student_group_assignments_non_empty_coin_subsets_l4048_404870


namespace NUMINAMATH_CALUDE_bryan_has_more_candies_l4048_404848

/-- Given that Bryan has 50 skittles and Ben has 20 M&M's, 
    prove that Bryan has 30 more candies than Ben. -/
theorem bryan_has_more_candies : 
  ∀ (bryan_skittles ben_mms : ℕ), 
    bryan_skittles = 50 → 
    ben_mms = 20 → 
    bryan_skittles - ben_mms = 30 := by
  sorry

end NUMINAMATH_CALUDE_bryan_has_more_candies_l4048_404848


namespace NUMINAMATH_CALUDE_wax_sculpture_theorem_l4048_404890

/-- Proves that the total number of wax sticks used is 20 --/
theorem wax_sculpture_theorem (large_sticks small_sticks : ℕ) 
  (h1 : large_sticks = 4)
  (h2 : small_sticks = 2)
  (small_animals large_animals : ℕ)
  (h3 : small_animals = 3 * large_animals)
  (total_small_sticks : ℕ)
  (h4 : total_small_sticks = 12)
  (h5 : total_small_sticks = small_animals * small_sticks) :
  total_small_sticks + large_animals * large_sticks = 20 := by
sorry

end NUMINAMATH_CALUDE_wax_sculpture_theorem_l4048_404890


namespace NUMINAMATH_CALUDE_complex_multiplication_l4048_404888

theorem complex_multiplication (z : ℂ) (h : z = 1 + 2 * I) : I * z = -2 + I := by sorry

end NUMINAMATH_CALUDE_complex_multiplication_l4048_404888


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l4048_404823

theorem complex_fraction_simplification :
  let z₁ : ℂ := 4 + 6 * Complex.I
  let z₂ : ℂ := 4 - 6 * Complex.I
  (z₁ / z₂) + (z₂ / z₁) = (-10 : ℚ) / 13 :=
by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l4048_404823


namespace NUMINAMATH_CALUDE_eliminate_y_condition_l4048_404821

/-- Represents a system of two linear equations in two variables -/
structure LinearSystem (α : Type*) [Field α] :=
  (a₁ b₁ c₁ : α)
  (a₂ b₂ c₂ : α)

/-- Checks if y can be directly eliminated when subtracting the second equation from the first -/
def canEliminateY {α : Type*} [Field α] (sys : LinearSystem α) : Prop :=
  sys.b₁ + sys.b₂ = 0

/-- The specific linear system from the problem -/
def problemSystem (α : Type*) [Field α] (m n : α) : LinearSystem α :=
  { a₁ := 6, b₁ := m, c₁ := 3,
    a₂ := 2, b₂ := -n, c₂ := -6 }

theorem eliminate_y_condition (α : Type*) [Field α] (m n : α) :
  canEliminateY (problemSystem α m n) ↔ m + n = 0 :=
sorry

end NUMINAMATH_CALUDE_eliminate_y_condition_l4048_404821


namespace NUMINAMATH_CALUDE_unique_solution_l4048_404824

def is_valid_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

def satisfies_equation (Θ : ℕ) : Prop :=
  is_valid_digit Θ ∧ 
  (198 : ℚ) / Θ = (40 : ℚ) + 2 * Θ

theorem unique_solution : 
  ∃! Θ : ℕ, satisfies_equation Θ ∧ Θ = 4 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l4048_404824


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4048_404814

/-- In an arithmetic sequence {aₙ}, if a₄ + a₆ + a₈ + a₁₀ = 28, then a₇ = 7 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  (a 4 + a 6 + a 8 + a 10 = 28) →                   -- given condition
  a 7 = 7 :=                                        -- conclusion to prove
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4048_404814


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l4048_404895

theorem contrapositive_equivalence (a b : ℝ) :
  (¬(a - 8 > b - 8) → ¬(a > b)) ↔ ((a - 8 ≤ b - 8) → (a ≤ b)) := by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l4048_404895


namespace NUMINAMATH_CALUDE_intersection_M_N_l4048_404834

-- Define the sets M and N
def M : Set ℝ := {s | |s| < 4}
def N : Set ℝ := {x | 3 * x ≥ -1}

-- State the theorem
theorem intersection_M_N : M ∩ N = {x | -1/3 ≤ x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l4048_404834


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l4048_404880

def A (a : ℝ) : Set ℝ := {a^2, a+1, -3}
def B (a : ℝ) : Set ℝ := {a-3, 2*a-1, a^2+1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-3} → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l4048_404880


namespace NUMINAMATH_CALUDE_trig_identities_l4048_404836

/-- Given that sin(3π + α) = 2sin(3π/2 + α), prove two trigonometric identities. -/
theorem trig_identities (α : ℝ) 
  (h : Real.sin (3 * Real.pi + α) = 2 * Real.sin ((3 * Real.pi) / 2 + α)) : 
  (((2 * Real.sin α - 3 * Real.cos α) / (4 * Real.sin α - 9 * Real.cos α)) = 7 / 17) ∧ 
  ((Real.sin α)^2 + Real.sin (2 * α) = 0) := by
  sorry

end NUMINAMATH_CALUDE_trig_identities_l4048_404836


namespace NUMINAMATH_CALUDE_beanie_babies_per_stocking_l4048_404852

theorem beanie_babies_per_stocking : 
  ∀ (candy_canes_per_stocking : ℕ) 
    (books_per_stocking : ℕ) 
    (num_stockings : ℕ) 
    (total_stuffers : ℕ),
  candy_canes_per_stocking = 4 →
  books_per_stocking = 1 →
  num_stockings = 3 →
  total_stuffers = 21 →
  (total_stuffers - (candy_canes_per_stocking + books_per_stocking) * num_stockings) / num_stockings = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_beanie_babies_per_stocking_l4048_404852


namespace NUMINAMATH_CALUDE_sector_arc_length_l4048_404898

/-- Given a circular sector with area 2 cm² and central angle 4 radians,
    the length of the arc of the sector is 6 cm. -/
theorem sector_arc_length (area : ℝ) (angle : ℝ) (arc_length : ℝ) :
  area = 2 →
  angle = 4 →
  arc_length = 6 :=
by sorry

end NUMINAMATH_CALUDE_sector_arc_length_l4048_404898


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l4048_404831

/-- An ellipse with given properties -/
structure Ellipse where
  -- Point P on the ellipse
  p : ℝ × ℝ
  -- Focus F₁
  f1 : ℝ × ℝ
  -- Focus F₂
  f2 : ℝ × ℝ

/-- The eccentricity of an ellipse -/
def eccentricity (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the eccentricity of the specific ellipse -/
theorem ellipse_eccentricity :
  let e : Ellipse := {
    p := (2, 3)
    f1 := (-2, 0)
    f2 := (2, 0)
  }
  eccentricity e = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l4048_404831


namespace NUMINAMATH_CALUDE_train_length_l4048_404855

/-- The length of a train given its speed, time to cross a bridge, and the bridge length -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) (bridge_length : ℝ) :
  train_speed = 60 →
  crossing_time = 29.997600191984642 →
  bridge_length = 390 →
  ∃ (train_length : ℝ), abs (train_length - 110) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l4048_404855


namespace NUMINAMATH_CALUDE_parking_probability_theorem_l4048_404800

/-- Represents the parking fee structure and probabilities for a business district parking lot. -/
structure ParkingLot where
  base_fee : ℕ := 6  -- Base fee for first hour
  hourly_fee : ℕ := 8  -- Fee for each additional hour
  max_hours : ℕ := 4  -- Maximum parking duration
  prob_A_1to2 : ℚ := 1/3  -- Probability A parks between 1-2 hours
  prob_A_over14 : ℚ := 5/12  -- Probability A pays over 14 yuan

/-- Calculates the probability of various parking scenarios. -/
def parking_probabilities (lot : ParkingLot) : ℚ × ℚ :=
  let prob_A_6yuan := 1 - (lot.prob_A_1to2 + lot.prob_A_over14)
  let prob_total_36yuan := 1/4  -- Given equal probability for each time interval
  (prob_A_6yuan, prob_total_36yuan)

/-- Theorem stating the probabilities of specific parking scenarios. -/
theorem parking_probability_theorem (lot : ParkingLot) :
  parking_probabilities lot = (1/4, 1/4) := by sorry

/-- Verifies that the calculated probabilities match the expected values. -/
example (lot : ParkingLot) : 
  parking_probabilities lot = (1/4, 1/4) := by sorry

end NUMINAMATH_CALUDE_parking_probability_theorem_l4048_404800


namespace NUMINAMATH_CALUDE_negation_equivalence_l4048_404849

theorem negation_equivalence : 
  (¬ ∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ (∀ x : ℝ, -1 < x ∧ x < 2) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l4048_404849


namespace NUMINAMATH_CALUDE_smaller_number_problem_l4048_404882

theorem smaller_number_problem (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 45) : 
  min x y = 3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_number_problem_l4048_404882


namespace NUMINAMATH_CALUDE_polynomial_roots_and_constant_term_l4048_404847

def polynomial (a b c d : ℤ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

theorem polynomial_roots_and_constant_term 
  (a b c d : ℤ) 
  (h1 : ∀ x : ℝ, polynomial a b c d x = 0 → (∃ n : ℕ, x = -↑n))
  (h2 : a + b + c + d = 2009) :
  d = 528 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_and_constant_term_l4048_404847


namespace NUMINAMATH_CALUDE_no_natural_solutions_l4048_404863

theorem no_natural_solutions : ¬∃ (x y : ℕ), x^4 - 2*y^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_solutions_l4048_404863


namespace NUMINAMATH_CALUDE_triangle_perimeters_l4048_404810

/-- The possible side lengths of the triangle -/
def triangle_sides : Set ℝ := {3, 6}

/-- Check if three numbers can form a triangle -/
def is_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

/-- The set of possible perimeters of the triangle -/
def possible_perimeters : Set ℝ := {9, 15, 18}

/-- Theorem stating that the possible perimeters are 9, 15, or 18 -/
theorem triangle_perimeters :
  ∀ a b c : ℝ,
  a ∈ triangle_sides → b ∈ triangle_sides → c ∈ triangle_sides →
  is_triangle a b c →
  a + b + c ∈ possible_perimeters :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeters_l4048_404810


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l4048_404815

theorem arithmetic_mean_difference (a b c : ℝ) :
  (a + b) / 2 = (a + b + c) / 3 + 5 →
  (a + c) / 2 = (a + b + c) / 3 - 8 →
  (b + c) / 2 = (a + b + c) / 3 + 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l4048_404815


namespace NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l4048_404839

theorem square_sum_given_sum_and_product (a b : ℝ) 
  (h1 : a + b = 12) (h2 : a * b = 20) : a^2 + b^2 = 104 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_given_sum_and_product_l4048_404839


namespace NUMINAMATH_CALUDE_intersection_point_l4048_404891

/-- Curve C₁ is defined by y = √x for x ≥ 0 -/
def C₁ (x y : ℝ) : Prop := y = Real.sqrt x ∧ x ≥ 0

/-- Curve C₂ is defined by x² + y² = 2 -/
def C₂ (x y : ℝ) : Prop := x^2 + y^2 = 2

/-- The point (1, 1) is the unique intersection point of curves C₁ and C₂ -/
theorem intersection_point : 
  (∃! p : ℝ × ℝ, C₁ p.1 p.2 ∧ C₂ p.1 p.2) ∧ 
  (C₁ 1 1 ∧ C₂ 1 1) := by
  sorry

#check intersection_point

end NUMINAMATH_CALUDE_intersection_point_l4048_404891


namespace NUMINAMATH_CALUDE_third_side_length_l4048_404804

theorem third_side_length (a b : ℝ) (h1 : a = 3.14) (h2 : b = 0.67) : 
  ∃ m : ℤ, (m : ℝ) > |a - b| ∧ (m : ℝ) < a + b ∧ m = 3 :=
by sorry

end NUMINAMATH_CALUDE_third_side_length_l4048_404804


namespace NUMINAMATH_CALUDE_max_sum_of_complex_product_l4048_404803

/-- The maximum sum of real and imaginary parts of the product of two specific complex functions of θ -/
theorem max_sum_of_complex_product :
  let z1 (θ : ℝ) := (8 + Complex.I) * Real.sin θ + (7 + 4 * Complex.I) * Real.cos θ
  let z2 (θ : ℝ) := (1 + 8 * Complex.I) * Real.sin θ + (4 + 7 * Complex.I) * Real.cos θ
  ∃ (θ : ℝ), ∀ (φ : ℝ), (z1 θ * z2 θ).re + (z1 θ * z2 θ).im ≥ (z1 φ * z2 φ).re + (z1 φ * z2 φ).im ∧
  (z1 θ * z2 θ).re + (z1 θ * z2 θ).im = 125 :=
by
  sorry


end NUMINAMATH_CALUDE_max_sum_of_complex_product_l4048_404803


namespace NUMINAMATH_CALUDE_exactly_two_balls_distribution_l4048_404816

-- Define the number of balls and boxes
def num_balls : ℕ := 5
def num_boxes : ℕ := 3

-- Define the function to calculate the number of ways to distribute balls
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  (n.choose 2) * k * (k ^ (n - 2))

-- Theorem statement
theorem exactly_two_balls_distribution :
  distribute_balls num_balls num_boxes = 810 :=
sorry

end NUMINAMATH_CALUDE_exactly_two_balls_distribution_l4048_404816


namespace NUMINAMATH_CALUDE_tyrones_money_is_thirteen_l4048_404827

/-- Calculates the total amount of money Tyrone has -/
def tyrones_money (one_dollar_bills : ℕ) (five_dollar_bills : ℕ) (quarters : ℕ) (dimes : ℕ) (nickels : ℕ) (pennies : ℕ) : ℚ :=
  one_dollar_bills + 5 * five_dollar_bills + (1/4) * quarters + (1/10) * dimes + (1/20) * nickels + (1/100) * pennies

theorem tyrones_money_is_thirteen :
  tyrones_money 2 1 13 20 8 35 = 13 := by
  sorry

end NUMINAMATH_CALUDE_tyrones_money_is_thirteen_l4048_404827


namespace NUMINAMATH_CALUDE_sum_of_roots_is_12_l4048_404885

-- Define the function g
variable (g : ℝ → ℝ)

-- Define the symmetry property of g
def symmetric_about_3 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (3 + x) = g (3 - x)

-- Define a proposition that g has exactly four distinct real roots
def has_four_distinct_roots (g : ℝ → ℝ) : Prop :=
  ∃ a b c d : ℝ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (∀ x : ℝ, g x = 0 → (x = a ∨ x = b ∨ x = c ∨ x = d))

-- The theorem statement
theorem sum_of_roots_is_12 (g : ℝ → ℝ) 
    (h1 : symmetric_about_3 g) 
    (h2 : has_four_distinct_roots g) : 
  ∃ a b c d : ℝ, (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) ∧
    (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧
    (a + b + c + d = 12) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_is_12_l4048_404885


namespace NUMINAMATH_CALUDE_base4_multiplication_division_l4048_404844

/-- Converts a base 4 number to base 10 --/
def base4ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- Converts a base 10 number to base 4 --/
def base10ToBase4 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 4) ((m % 4) :: acc)
    aux n []

/-- Theorem stating that 132₄ × 21₄ ÷ 3₄ = 1122₄ --/
theorem base4_multiplication_division :
  let a := base4ToBase10 [2, 3, 1]  -- 132₄
  let b := base4ToBase10 [1, 2]     -- 21₄
  let c := base4ToBase10 [3]        -- 3₄
  let result := base10ToBase4 ((a * b) / c)
  result = [2, 2, 1, 1] := by sorry

end NUMINAMATH_CALUDE_base4_multiplication_division_l4048_404844


namespace NUMINAMATH_CALUDE_marble_weight_problem_l4048_404856

theorem marble_weight_problem (weight_piece1 weight_piece2 total_weight : ℝ) 
  (h1 : weight_piece1 = 0.33)
  (h2 : weight_piece2 = 0.33)
  (h3 : total_weight = 0.75) :
  total_weight - (weight_piece1 + weight_piece2) = 0.09 := by
  sorry

end NUMINAMATH_CALUDE_marble_weight_problem_l4048_404856


namespace NUMINAMATH_CALUDE_oil_drop_probability_l4048_404865

/-- The probability of an oil drop falling into a square hole in a circular coin -/
theorem oil_drop_probability (coin_diameter : Real) (hole_side : Real) : 
  coin_diameter = 2 → hole_side = 0.5 → 
  (hole_side^2) / (π * (coin_diameter/2)^2) = 1 / (4 * π) := by
sorry

end NUMINAMATH_CALUDE_oil_drop_probability_l4048_404865


namespace NUMINAMATH_CALUDE_distance_between_points_l4048_404899

/-- The distance between two points when two people walk towards each other --/
theorem distance_between_points (speed_a speed_b : ℝ) (midpoint_offset : ℝ) : 
  speed_a = 70 →
  speed_b = 60 →
  midpoint_offset = 80 →
  (speed_a - speed_b) * ((2 * midpoint_offset) / (speed_a - speed_b)) = speed_a + speed_b →
  (speed_a + speed_b) * ((2 * midpoint_offset) / (speed_a - speed_b)) = 2080 :=
by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_distance_between_points_l4048_404899


namespace NUMINAMATH_CALUDE_conjunction_implies_left_prop_l4048_404868

theorem conjunction_implies_left_prop (p q : Prop) : (p ∧ q) → p := by
  sorry

end NUMINAMATH_CALUDE_conjunction_implies_left_prop_l4048_404868


namespace NUMINAMATH_CALUDE_ellipse_properties_l4048_404850

-- Define the ellipse C
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- Define the left focus
def LeftFocus : ℝ × ℝ := (-1, 0)

-- Define the line l
def Line (x y : ℝ) : Prop :=
  y = x + 1

-- Theorem statement
theorem ellipse_properties :
  -- Given conditions
  let C := Ellipse
  let e : ℝ := 1/2
  let max_distance : ℝ := 3
  let l := Line

  -- Prove
  (∀ x y, C x y → x^2 / 4 + y^2 / 3 = 1) ∧
  (∃ A B : ℝ × ℝ,
    C A.1 A.2 ∧ C B.1 B.2 ∧
    l A.1 A.2 ∧ l B.1 B.2 ∧
    ((A.1 - B.1)^2 + (A.2 - B.2)^2)^(1/2) = 24/7) :=
by
  sorry

end NUMINAMATH_CALUDE_ellipse_properties_l4048_404850


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l4048_404829

/-- Given a geometric sequence with first term a and common ratio r, 
    such that the sum of the first 1500 terms is 300 and 
    the sum of the first 3000 terms is 570,
    prove that the sum of the first 4500 terms is 813. -/
theorem geometric_sequence_sum (a r : ℝ) 
  (h1 : a * (1 - r^1500) / (1 - r) = 300) 
  (h2 : a * (1 - r^3000) / (1 - r) = 570) : 
  a * (1 - r^4500) / (1 - r) = 813 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l4048_404829


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocal_l4048_404875

theorem quadratic_roots_sum_reciprocal (a b : ℝ) : 
  a^2 - 6*a - 5 = 0 → 
  b^2 - 6*b - 5 = 0 → 
  a ≠ 0 → 
  b ≠ 0 → 
  1/a + 1/b = -6/5 := by sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_reciprocal_l4048_404875


namespace NUMINAMATH_CALUDE_toms_seashells_l4048_404813

theorem toms_seashells (sally_shells : ℕ) (jessica_shells : ℕ) (total_shells : ℕ) 
  (h1 : sally_shells = 9)
  (h2 : jessica_shells = 5)
  (h3 : total_shells = 21) :
  total_shells - (sally_shells + jessica_shells) = 7 := by
  sorry

end NUMINAMATH_CALUDE_toms_seashells_l4048_404813


namespace NUMINAMATH_CALUDE_second_discount_percentage_l4048_404802

theorem second_discount_percentage (original_price : ℝ) (first_discount : ℝ) (final_price : ℝ) : 
  original_price = 10000 →
  first_discount = 20 →
  final_price = 6840 →
  ∃ (second_discount : ℝ),
    final_price = original_price * (1 - first_discount / 100) * (1 - second_discount / 100) ∧
    second_discount = 14.5 := by
  sorry

end NUMINAMATH_CALUDE_second_discount_percentage_l4048_404802


namespace NUMINAMATH_CALUDE_xyz_equals_seven_cubed_l4048_404873

theorem xyz_equals_seven_cubed 
  (x y z : ℝ) 
  (h1 : x^2 * y * z^3 = 7^4) 
  (h2 : x * y^2 = 7^5) : 
  x * y * z = 7^3 := by
  sorry

end NUMINAMATH_CALUDE_xyz_equals_seven_cubed_l4048_404873


namespace NUMINAMATH_CALUDE_parking_solution_is_correct_l4048_404842

/-- Represents the parking lot problem. -/
structure ParkingLot where
  total_cars : ℕ
  total_fee : ℕ
  medium_fee : ℕ
  small_fee : ℕ

/-- Represents the solution to the parking lot problem. -/
structure ParkingSolution where
  medium_cars : ℕ
  small_cars : ℕ

/-- Checks if a given solution satisfies the parking lot conditions. -/
def is_valid_solution (p : ParkingLot) (s : ParkingSolution) : Prop :=
  s.medium_cars + s.small_cars = p.total_cars ∧
  s.medium_cars * p.medium_fee + s.small_cars * p.small_fee = p.total_fee

/-- The parking lot problem instance. -/
def parking_problem : ParkingLot :=
  { total_cars := 30
  , total_fee := 324
  , medium_fee := 15
  , small_fee := 8 }

/-- The proposed solution to the parking lot problem. -/
def parking_solution : ParkingSolution :=
  { medium_cars := 12
  , small_cars := 18 }

/-- Theorem stating that the proposed solution is correct for the given problem. -/
theorem parking_solution_is_correct :
  is_valid_solution parking_problem parking_solution := by
  sorry

end NUMINAMATH_CALUDE_parking_solution_is_correct_l4048_404842


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l4048_404805

/-- The point corresponding to the complex number (a^2 - 4a + 5) + (-b^2 + 2b - 6)i 
    is in the fourth quadrant for all real a and b. -/
theorem complex_number_in_fourth_quadrant (a b : ℝ) : 
  (a^2 - 4*a + 5 > 0) ∧ (-b^2 + 2*b - 6 < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l4048_404805


namespace NUMINAMATH_CALUDE_rabbit_count_l4048_404866

theorem rabbit_count (white_rabbits black_rabbits female_rabbits : ℕ) : 
  white_rabbits = 12 → black_rabbits = 9 → female_rabbits = 8 → 
  white_rabbits + black_rabbits - female_rabbits = 13 := by
sorry

end NUMINAMATH_CALUDE_rabbit_count_l4048_404866


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_57_l4048_404845

theorem smallest_n_divisible_by_57 :
  ∃ (n : ℕ), n > 0 ∧ 57 ∣ (7^n + 2*n) ∧ ∀ (m : ℕ), m > 0 ∧ 57 ∣ (7^m + 2*m) → n ≤ m :=
by
  use 25
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_57_l4048_404845


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l4048_404887

theorem product_of_three_numbers (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 32) (hac : a * c = 48) (hbc : b * c = 80) :
  a * b * c = 64 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l4048_404887


namespace NUMINAMATH_CALUDE_circle_equation_l4048_404876

/-- The equation of a circle passing through (0,0), (4,0), and (-1,1) -/
theorem circle_equation (x y : ℝ) : 
  (x^2 + y^2 - 4*x - 6*y = 0) ↔ 
  ((x = 0 ∧ y = 0) ∨ (x = 4 ∧ y = 0) ∨ (x = -1 ∧ y = 1)) := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_l4048_404876


namespace NUMINAMATH_CALUDE_binomial_variance_l4048_404812

/-- A binomial distribution with parameter p -/
structure BinomialDistribution (p : ℝ) where
  (h1 : 0 < p)
  (h2 : p < 1)

/-- The variance of a binomial distribution -/
def variance (p : ℝ) (X : BinomialDistribution p) : ℝ := sorry

theorem binomial_variance (p : ℝ) (X : BinomialDistribution p) :
  variance p X = p * (1 - p) := by sorry

end NUMINAMATH_CALUDE_binomial_variance_l4048_404812


namespace NUMINAMATH_CALUDE_more_students_than_rabbits_l4048_404830

theorem more_students_than_rabbits : 
  let num_classrooms : ℕ := 5
  let students_per_classroom : ℕ := 22
  let rabbits_per_classroom : ℕ := 2
  let total_students : ℕ := num_classrooms * students_per_classroom
  let total_rabbits : ℕ := num_classrooms * rabbits_per_classroom
  total_students - total_rabbits = 100 := by
  sorry

end NUMINAMATH_CALUDE_more_students_than_rabbits_l4048_404830


namespace NUMINAMATH_CALUDE_quirky_triangle_characterization_l4048_404867

/-- A triangle is quirky if there exist integers r₁, r₂, r₃, not all zero, 
    such that r₁θ₁ + r₂θ₂ + r₃θ₃ = 0, where θ₁, θ₂, θ₃ are the measures of the triangle's angles. -/
def IsQuirky (θ₁ θ₂ θ₃ : ℝ) : Prop :=
  ∃ r₁ r₂ r₃ : ℤ, (r₁ ≠ 0 ∨ r₂ ≠ 0 ∨ r₃ ≠ 0) ∧ r₁ * θ₁ + r₂ * θ₂ + r₃ * θ₃ = 0

/-- The angles of a triangle with side lengths n-1, n, n+1 -/
def TriangleAngles (n : ℕ) : (ℝ × ℝ × ℝ) :=
  sorry

theorem quirky_triangle_characterization (n : ℕ) (h : n ≥ 3) :
  let (θ₁, θ₂, θ₃) := TriangleAngles n
  IsQuirky θ₁ θ₂ θ₃ ↔ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 7 :=
sorry

end NUMINAMATH_CALUDE_quirky_triangle_characterization_l4048_404867


namespace NUMINAMATH_CALUDE_trig_identity_l4048_404862

theorem trig_identity (α : Real) 
  (h1 : α ∈ Set.Ioo 0 (π / 2))
  (h2 : 2 * (Real.sin α)^2 - Real.sin α * Real.cos α - 3 * (Real.cos α)^2 = 0) :
  Real.sin (α + π / 4) / (Real.sin (2 * α) + Real.cos (2 * α) + 1) = Real.sqrt 26 / 8 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4048_404862


namespace NUMINAMATH_CALUDE_sum_of_squares_ratio_l4048_404837

theorem sum_of_squares_ratio (x : ℚ) : 
  x + 2*x + 4*x = 15 → 
  x^2 + (2*x)^2 + (4*x)^2 = 4725 / 49 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_ratio_l4048_404837


namespace NUMINAMATH_CALUDE_smallest_factor_l4048_404878

def is_divisible (a b : ℕ) : Prop := ∃ k, a = b * k

theorem smallest_factor (w n : ℕ) : 
  w > 0 → 
  n > 0 → 
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (2^5)) →
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (3^3)) →
  (∀ w' : ℕ, w' ≥ w → is_divisible (w' * n) (10^2)) →
  w = 120 →
  n ≥ 180 :=
sorry

end NUMINAMATH_CALUDE_smallest_factor_l4048_404878


namespace NUMINAMATH_CALUDE_quadratic_sum_zero_l4048_404892

-- Define the quadratic function P(x)
def P (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_sum_zero 
  (a b c : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_Pa : P a b c a = 2021 * b * c)
  (h_Pb : P a b c b = 2021 * c * a)
  (h_Pc : P a b c c = 2021 * a * b) :
  a + 2021 * b + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_zero_l4048_404892


namespace NUMINAMATH_CALUDE_division_remainder_l4048_404822

theorem division_remainder (dividend : Nat) (divisor : Nat) (quotient : Nat) (remainder : Nat) :
  dividend = divisor * quotient + remainder →
  dividend = 13 →
  divisor = 7 →
  quotient = 1 →
  remainder = 6 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l4048_404822


namespace NUMINAMATH_CALUDE_coloring_books_sale_result_l4048_404864

/-- The number of coloring books gotten rid of in a store sale -/
def books_gotten_rid_of (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : ℕ :=
  initial_stock - (shelves * books_per_shelf)

/-- Theorem stating that the number of coloring books gotten rid of is 39 -/
theorem coloring_books_sale_result : 
  books_gotten_rid_of 120 9 9 = 39 := by
  sorry

end NUMINAMATH_CALUDE_coloring_books_sale_result_l4048_404864


namespace NUMINAMATH_CALUDE_range_of_a_minus_b_l4048_404883

theorem range_of_a_minus_b (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : -2 < b ∧ b < 4) :
  ∃ (x : ℝ), -3 < x ∧ x < 6 ∧ ∃ (a' b' : ℝ), 1 < a' ∧ a' < 4 ∧ -2 < b' ∧ b' < 4 ∧ x = a' - b' :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_minus_b_l4048_404883


namespace NUMINAMATH_CALUDE_pencils_broken_l4048_404872

theorem pencils_broken (initial bought found misplaced final : ℕ) : 
  initial = 20 → 
  bought = 2 → 
  found = 4 → 
  misplaced = 7 → 
  final = 16 → 
  initial + bought + found - misplaced - final = 3 := by
  sorry

end NUMINAMATH_CALUDE_pencils_broken_l4048_404872


namespace NUMINAMATH_CALUDE_sin_cos_equation_solution_l4048_404877

theorem sin_cos_equation_solution (x : Real) 
  (h1 : x ∈ Set.Icc 0 Real.pi) 
  (h2 : Real.sin (x + Real.sin x) = Real.cos (x - Real.cos x)) : 
  x = Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_cos_equation_solution_l4048_404877


namespace NUMINAMATH_CALUDE_house_sale_tax_percentage_l4048_404889

theorem house_sale_tax_percentage (market_value : ℝ) (over_market_percentage : ℝ) 
  (num_people : ℕ) (amount_per_person : ℝ) :
  market_value = 500000 →
  over_market_percentage = 0.20 →
  num_people = 4 →
  amount_per_person = 135000 →
  (market_value * (1 + over_market_percentage) - num_people * amount_per_person) / 
    (market_value * (1 + over_market_percentage)) = 0.10 := by
  sorry

end NUMINAMATH_CALUDE_house_sale_tax_percentage_l4048_404889


namespace NUMINAMATH_CALUDE_same_grade_percentage_is_50_l4048_404894

/-- Represents the number of students who got the same grade on both tests for each grade -/
structure GradeCount where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ

/-- Calculates the percentage of students who got the same grade on both tests -/
def sameGradePercentage (totalStudents : ℕ) (gradeCount : GradeCount) : ℚ :=
  (gradeCount.a + gradeCount.b + gradeCount.c + gradeCount.d : ℚ) / totalStudents * 100

/-- The main theorem stating that 50% of students received the same grade on both tests -/
theorem same_grade_percentage_is_50 :
  let totalStudents : ℕ := 40
  let gradeCount : GradeCount := { a := 3, b := 6, c := 7, d := 4 }
  sameGradePercentage totalStudents gradeCount = 50 := by
  sorry


end NUMINAMATH_CALUDE_same_grade_percentage_is_50_l4048_404894


namespace NUMINAMATH_CALUDE_circle_diameter_from_inscribed_triangles_l4048_404843

theorem circle_diameter_from_inscribed_triangles
  (triangle_a_side1 triangle_a_side2 triangle_a_hypotenuse : ℝ)
  (triangle_b_side1 triangle_b_side2 triangle_b_hypotenuse : ℝ)
  (h1 : triangle_a_side1 = 7)
  (h2 : triangle_a_side2 = 24)
  (h3 : triangle_a_hypotenuse = 39)
  (h4 : triangle_b_side1 = 15)
  (h5 : triangle_b_side2 = 36)
  (h6 : triangle_b_hypotenuse = 39)
  (h7 : triangle_a_side1^2 + triangle_a_side2^2 = triangle_a_hypotenuse^2)
  (h8 : triangle_b_side1^2 + triangle_b_side2^2 = triangle_b_hypotenuse^2)
  (h9 : triangle_a_hypotenuse = triangle_b_hypotenuse) :
  39 = triangle_a_hypotenuse ∧ 39 = triangle_b_hypotenuse := by
  sorry

#check circle_diameter_from_inscribed_triangles

end NUMINAMATH_CALUDE_circle_diameter_from_inscribed_triangles_l4048_404843


namespace NUMINAMATH_CALUDE_work_ratio_man_to_boy_l4048_404809

theorem work_ratio_man_to_boy :
  ∀ (m b : ℝ),
  m > 0 → b > 0 →
  7 * m + 2 * b = 6 * (m + b) →
  m / b = 4 := by
sorry

end NUMINAMATH_CALUDE_work_ratio_man_to_boy_l4048_404809


namespace NUMINAMATH_CALUDE_distributive_property_negative_l4048_404857

theorem distributive_property_negative (a b : ℝ) : -3 * (a - b) = -3 * a + 3 * b := by
  sorry

end NUMINAMATH_CALUDE_distributive_property_negative_l4048_404857


namespace NUMINAMATH_CALUDE_candy_sales_average_l4048_404818

/-- The average of candy sales for five months -/
def average_candy_sales (jan feb mar apr may : ℕ) : ℚ :=
  (jan + feb + mar + apr + may) / 5

/-- Theorem stating that the average candy sales is 96 dollars -/
theorem candy_sales_average :
  average_candy_sales 110 80 70 130 90 = 96 := by sorry

end NUMINAMATH_CALUDE_candy_sales_average_l4048_404818


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_shift_l4048_404819

/-- For any constant a > 0 and a ≠ 1, the function f(x) = a^(x-1) - 1 passes through the point (1, 0) -/
theorem fixed_point_of_exponential_shift (a : ℝ) (ha : a > 0) (ha_neq : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1) - 1
  f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_shift_l4048_404819


namespace NUMINAMATH_CALUDE_ipads_sold_l4048_404879

/-- Proves that the number of iPads sold is 20 given the conditions of the problem -/
theorem ipads_sold (iphones : ℕ) (ipads : ℕ) (apple_tvs : ℕ) 
  (iphone_cost : ℝ) (ipad_cost : ℝ) (apple_tv_cost : ℝ) (average_cost : ℝ) :
  iphones = 100 →
  apple_tvs = 80 →
  iphone_cost = 1000 →
  ipad_cost = 900 →
  apple_tv_cost = 200 →
  average_cost = 670 →
  (iphones * iphone_cost + ipads * ipad_cost + apple_tvs * apple_tv_cost) / 
    (iphones + ipads + apple_tvs : ℝ) = average_cost →
  ipads = 20 := by
  sorry

#check ipads_sold

end NUMINAMATH_CALUDE_ipads_sold_l4048_404879


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l4048_404833

/-- For an arithmetic sequence {a_n}, if a_3 + a_11 = 22, then a_7 = 11 -/
theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m) →  -- arithmetic sequence property
  a 3 + a 11 = 22 →                                -- given condition
  a 7 = 11                                         -- conclusion to prove
:= by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l4048_404833


namespace NUMINAMATH_CALUDE_closest_perfect_square_to_314_l4048_404841

/-- The perfect-square integer closest to 314 is 324. -/
theorem closest_perfect_square_to_314 : 
  ∀ n : ℕ, n ≠ 324 → n * n ≠ 0 → |314 - (324 : ℤ)| ≤ |314 - (n * n : ℤ)| := by
  sorry

end NUMINAMATH_CALUDE_closest_perfect_square_to_314_l4048_404841


namespace NUMINAMATH_CALUDE_polynomial_factorization_l4048_404851

theorem polynomial_factorization (x : ℝ) : 
  x^6 - 4*x^4 + 6*x^2 - 4 = (x^2 - 2)^3 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l4048_404851


namespace NUMINAMATH_CALUDE_annual_mischief_convention_handshakes_l4048_404881

/-- The number of handshakes at the Annual Mischief Convention -/
def total_handshakes (num_gremlins num_imps num_friendly_imps : ℕ) : ℕ :=
  let gremlin_handshakes := num_gremlins * (num_gremlins - 1) / 2
  let imp_gremlin_handshakes := num_imps * num_gremlins
  let friendly_imp_handshakes := num_friendly_imps * (num_friendly_imps - 1) / 2
  gremlin_handshakes + imp_gremlin_handshakes + friendly_imp_handshakes

/-- Theorem stating the total number of handshakes at the Annual Mischief Convention -/
theorem annual_mischief_convention_handshakes :
  total_handshakes 30 20 5 = 1045 := by
  sorry

end NUMINAMATH_CALUDE_annual_mischief_convention_handshakes_l4048_404881


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l4048_404820

-- Define the quadratic function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 3

-- State the theorem
theorem quadratic_function_properties (a : ℝ) :
  (∀ x : ℝ, f a x = f a (4 - x)) →
  (a = 4) ∧
  (Set.Icc 0 3).image (f a) = Set.Icc (-1) 3 ∧
  ∃ (g : ℝ → ℝ), (∀ x : ℝ, f a x = (x - 2)^2 + 1) :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_function_properties_l4048_404820


namespace NUMINAMATH_CALUDE_two_points_with_area_three_l4048_404826

/-- The ellipse equation -/
def ellipse (x y : ℝ) : Prop := x^2/16 + y^2/9 = 1

/-- The line equation -/
def line (x y : ℝ) : Prop := x/4 + y/3 = 1

/-- Point on the ellipse -/
structure PointOnEllipse where
  x : ℝ
  y : ℝ
  on_ellipse : ellipse x y

/-- Intersection points of the line and ellipse -/
structure IntersectionPoints where
  A : PointOnEllipse
  B : PointOnEllipse
  on_line_A : line A.x A.y
  on_line_B : line B.x B.y

/-- Area of a triangle given three points -/
noncomputable def triangleArea (P Q R : ℝ × ℝ) : ℝ := sorry

/-- The main theorem -/
theorem two_points_with_area_three (intersections : IntersectionPoints) :
  ∃! (points : Finset PointOnEllipse),
    points.card = 2 ∧
    ∀ P ∈ points,
      triangleArea (P.x, P.y) (intersections.A.x, intersections.A.y) (intersections.B.x, intersections.B.y) = 3 :=
sorry

end NUMINAMATH_CALUDE_two_points_with_area_three_l4048_404826


namespace NUMINAMATH_CALUDE_range_of_z_l4048_404893

theorem range_of_z (x y : ℝ) (hx : -1 ≤ x ∧ x ≤ 2) (hy : 0 ≤ y ∧ y ≤ 1) :
  let z := 2 * x - 3 * y
  ∃ (a b : ℝ), a = -5 ∧ b = 4 ∧ ∀ w, w ∈ Set.Icc a b ↔ ∃ (x' y' : ℝ), 
    -1 ≤ x' ∧ x' ≤ 2 ∧ 0 ≤ y' ∧ y' ≤ 1 ∧ w = 2 * x' - 3 * y' :=
by sorry

end NUMINAMATH_CALUDE_range_of_z_l4048_404893


namespace NUMINAMATH_CALUDE_point_difference_l4048_404853

-- Define the value of a touchdown
def touchdown_value : ℕ := 7

-- Define the number of touchdowns for each team
def brayden_gavin_touchdowns : ℕ := 7
def cole_freddy_touchdowns : ℕ := 9

-- Calculate the points for each team
def brayden_gavin_points : ℕ := brayden_gavin_touchdowns * touchdown_value
def cole_freddy_points : ℕ := cole_freddy_touchdowns * touchdown_value

-- State the theorem
theorem point_difference : cole_freddy_points - brayden_gavin_points = 14 := by
  sorry

end NUMINAMATH_CALUDE_point_difference_l4048_404853


namespace NUMINAMATH_CALUDE_zero_of_f_l4048_404854

/-- The function f(x) = 4x - 2 -/
def f (x : ℝ) : ℝ := 4 * x - 2

/-- Theorem: The zero of the function f(x) = 4x - 2 is 1/2 -/
theorem zero_of_f : ∃ x : ℝ, f x = 0 ∧ x = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_zero_of_f_l4048_404854


namespace NUMINAMATH_CALUDE_perfect_square_condition_l4048_404835

theorem perfect_square_condition (n : ℕ) : 
  ∃ k : ℕ, n^2 + n + 1 = k^2 ↔ n = 0 :=
by sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l4048_404835


namespace NUMINAMATH_CALUDE_division_of_fractions_l4048_404817

theorem division_of_fractions : (4 : ℚ) / (8 / 13) = 13 / 2 := by
  sorry

end NUMINAMATH_CALUDE_division_of_fractions_l4048_404817


namespace NUMINAMATH_CALUDE_saree_sale_price_l4048_404840

/-- The sale price of a saree after successive discounts -/
theorem saree_sale_price (original_price : ℝ) (discount1 discount2 discount3 discount4 : ℝ) :
  original_price = 400 →
  discount1 = 0.20 →
  discount2 = 0.05 →
  discount3 = 0.10 →
  discount4 = 0.15 →
  original_price * (1 - discount1) * (1 - discount2) * (1 - discount3) * (1 - discount4) = 232.56 := by
  sorry

end NUMINAMATH_CALUDE_saree_sale_price_l4048_404840


namespace NUMINAMATH_CALUDE_jerry_action_figures_l4048_404884

/-- Given an initial count of action figures, a number removed, and a final count,
    this function calculates how many action figures were added. -/
def actionFiguresAdded (initial final removed : ℕ) : ℕ :=
  final + removed - initial

/-- Theorem stating that given the specific conditions in the problem,
    the number of action figures added must be 11. -/
theorem jerry_action_figures :
  actionFiguresAdded 7 8 10 = 11 := by
  sorry

end NUMINAMATH_CALUDE_jerry_action_figures_l4048_404884


namespace NUMINAMATH_CALUDE_log21_not_calculable_l4048_404801

-- Define the given logarithm values
def log5 : ℝ := 0.6990
def log7 : ℝ := 0.8451

-- Define a function to represent the ability to calculate a logarithm
def can_calculate (x : ℝ) : Prop := ∃ (a b : ℝ), x = a * log5 + b * log7

-- Theorem stating that log 21 cannot be calculated directly
theorem log21_not_calculable : ¬(can_calculate (Real.log 21)) :=
sorry

end NUMINAMATH_CALUDE_log21_not_calculable_l4048_404801


namespace NUMINAMATH_CALUDE_unique_grid_solution_l4048_404832

-- Define the grid type
def Grid := Fin 3 → Fin 3 → ℕ

-- Define adjacency
def adjacent (i j k l : Fin 3) : Prop :=
  (i = k ∧ j.val + 1 = l.val) ∨ 
  (i = k ∧ l.val + 1 = j.val) ∨ 
  (j = l ∧ i.val + 1 = k.val) ∨ 
  (j = l ∧ k.val + 1 = i.val)

-- Define the property of sum of adjacent cells being less than 12
def valid_sum (g : Grid) : Prop :=
  ∀ i j k l, adjacent i j k l → g i j + g k l < 12

-- Define the given partial grid
def partial_grid (g : Grid) : Prop :=
  g 0 1 = 1 ∧ g 0 2 = 9 ∧ g 1 0 = 3 ∧ g 1 1 = 5 ∧ g 2 2 = 7

-- Define the property that all numbers from 1 to 9 are used
def all_numbers_used (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → ∃ i j, g i j = n

-- The main theorem
theorem unique_grid_solution :
  ∀ g : Grid, 
    valid_sum g → 
    partial_grid g → 
    all_numbers_used g → 
    g 0 0 = 8 ∧ g 2 0 = 6 ∧ g 2 1 = 4 ∧ g 1 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_grid_solution_l4048_404832


namespace NUMINAMATH_CALUDE_accidental_addition_correction_l4048_404874

theorem accidental_addition_correction (x : ℤ) (h : x + 5 = 43) : 5 * x = 190 := by
  sorry

end NUMINAMATH_CALUDE_accidental_addition_correction_l4048_404874


namespace NUMINAMATH_CALUDE_max_value_quadratic_l4048_404861

theorem max_value_quadratic (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 - 2*x*y + 3*y^2 = 12) : 
  x^2 + 2*x*y + 3*y^2 ≤ 24 + 12*Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_max_value_quadratic_l4048_404861


namespace NUMINAMATH_CALUDE_existence_of_abc_l4048_404838

theorem existence_of_abc (p : ℕ) (hp : p.Prime) (hp_gt_2011 : p > 2011) :
  ∃ (a b c : ℕ+), (¬(p ∣ a) ∨ ¬(p ∣ b) ∨ ¬(p ∣ c)) ∧
    ∀ (n : ℕ+), p ∣ (n^4 - 2*n^2 + 9) → p ∣ (24*a*n^2 + 5*b*n + 2011*c) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_abc_l4048_404838


namespace NUMINAMATH_CALUDE_not_right_triangle_l4048_404859

theorem not_right_triangle : ∀ a b c : ℕ,
  (a = 3 ∧ b = 4 ∧ c = 5) ∨ 
  (a = 5 ∧ b = 12 ∧ c = 13) ∨ 
  (a = 6 ∧ b = 8 ∧ c = 10) ∨ 
  (a = 7 ∧ b = 8 ∧ c = 13) →
  (a^2 + b^2 ≠ c^2) ↔ (a = 7 ∧ b = 8 ∧ c = 13) :=
by sorry

end NUMINAMATH_CALUDE_not_right_triangle_l4048_404859


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l4048_404886

theorem modular_congruence_solution : ∃! n : ℕ, 1 ≤ n ∧ n ≤ 10 ∧ n ≡ 123456 [ZMOD 11] ∧ n = 3 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l4048_404886


namespace NUMINAMATH_CALUDE_multiplication_result_l4048_404896

theorem multiplication_result : (935421 * 625) = 584638125 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_result_l4048_404896


namespace NUMINAMATH_CALUDE_area_DEF_value_l4048_404806

/-- Triangle ABC with sides 5, 12, and 13 -/
structure Triangle :=
  (A B C : ℝ × ℝ)
  (side_a : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 5)
  (side_b : Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 12)
  (side_c : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 13)

/-- Parabola with focus F and directrix L -/
structure Parabola :=
  (F : ℝ × ℝ)
  (L : Set (ℝ × ℝ))

/-- Intersection points of parabolas with triangle sides -/
structure Intersections (t : Triangle) :=
  (A1 A2 B1 B2 C1 C2 : ℝ × ℝ)
  (on_parabola_A : Parabola → Prop)
  (on_parabola_B : Parabola → Prop)
  (on_parabola_C : Parabola → Prop)

/-- The area of triangle DEF formed by A1C2, B1A2, and C1B2 -/
def area_DEF (t : Triangle) (i : Intersections t) : ℝ := sorry

/-- Main theorem: The area of triangle DEF is 6728/3375 -/
theorem area_DEF_value (t : Triangle) (i : Intersections t) : 
  area_DEF t i = 6728 / 3375 := by sorry

end NUMINAMATH_CALUDE_area_DEF_value_l4048_404806


namespace NUMINAMATH_CALUDE_a_range_l4048_404897

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ a ≤ Real.exp x

def q (a : ℝ) : Prop := ∀ x : ℝ, x^2 + x + a > 0

-- State the theorem
theorem a_range (a : ℝ) (h : p a ∧ q a) : 1/4 < a ∧ a ≤ Real.exp 1 := by
  sorry


end NUMINAMATH_CALUDE_a_range_l4048_404897


namespace NUMINAMATH_CALUDE_parabola_directrix_coefficient_l4048_404828

/-- For a parabola y = ax^2 with directrix y = 2, prove that a = -1/8 -/
theorem parabola_directrix_coefficient : 
  ∀ (a : ℝ), (∀ x y : ℝ, y = a * x^2) → 
  (∃ k : ℝ, k = 2 ∧ ∀ x : ℝ, k = -(1 / (4 * a))) → 
  a = -1/8 := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_coefficient_l4048_404828


namespace NUMINAMATH_CALUDE_caleb_hamburger_cost_l4048_404808

def total_burgers : ℕ := 50
def single_burger_cost : ℚ := 1
def double_burger_cost : ℚ := 1.5
def double_burgers_bought : ℕ := 29

theorem caleb_hamburger_cost :
  let single_burgers := total_burgers - double_burgers_bought
  let total_cost := (single_burgers : ℚ) * single_burger_cost +
                    (double_burgers_bought : ℚ) * double_burger_cost
  total_cost = 64.5 := by sorry

end NUMINAMATH_CALUDE_caleb_hamburger_cost_l4048_404808


namespace NUMINAMATH_CALUDE_angle_A_is_30_degrees_max_area_is_3_l4048_404860

namespace TriangleProof

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.b = 2 ∧ Real.cos t.B = 4/5

-- Theorem 1: Angle A is 30° when a = 5/3
theorem angle_A_is_30_degrees (t : Triangle) 
  (h : triangle_conditions t) (ha : t.a = 5/3) : 
  t.A = Real.pi / 6 := by sorry

-- Theorem 2: Maximum area is 3
theorem max_area_is_3 (t : Triangle) 
  (h : triangle_conditions t) : 
  (∃ (max_area : ℝ), ∀ (s : Triangle), 
    triangle_conditions s → 
    (1/2 * s.a * s.c * Real.sin s.B) ≤ max_area ∧ 
    max_area = 3) := by sorry

end TriangleProof

end NUMINAMATH_CALUDE_angle_A_is_30_degrees_max_area_is_3_l4048_404860


namespace NUMINAMATH_CALUDE_train_bus_cost_l4048_404807

theorem train_bus_cost (bus_cost : ℝ) (train_extra_cost : ℝ) : 
  bus_cost = 1.40 →
  train_extra_cost = 6.85 →
  bus_cost + (bus_cost + train_extra_cost) = 9.65 := by
sorry

end NUMINAMATH_CALUDE_train_bus_cost_l4048_404807


namespace NUMINAMATH_CALUDE_smallest_x_and_corresponding_yzw_l4048_404858

theorem smallest_x_and_corresponding_yzw :
  ∀ (x y z w : ℝ),
  x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0 ∧ w ≥ 0 →
  y = x - 2003 →
  z = 2*y - 2003 →
  w = 3*z - 2003 →
  (x ≥ 10015/3 ∧ 
   (x = 10015/3 → y = 4006/3 ∧ z = 2003/3 ∧ w = 0)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_and_corresponding_yzw_l4048_404858


namespace NUMINAMATH_CALUDE_lauras_average_speed_l4048_404871

def first_distance : ℝ := 420
def first_time : ℝ := 6.5
def second_distance : ℝ := 480
def second_time : ℝ := 8.25

def total_distance : ℝ := first_distance + second_distance
def total_time : ℝ := first_time + second_time

theorem lauras_average_speed :
  total_distance / total_time = 900 / 14.75 := by sorry

end NUMINAMATH_CALUDE_lauras_average_speed_l4048_404871
