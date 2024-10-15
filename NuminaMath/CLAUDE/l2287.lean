import Mathlib

namespace NUMINAMATH_CALUDE_product_integers_exist_l2287_228712

theorem product_integers_exist : ∃ (a b c : ℝ), 
  (¬ ∃ (n : ℤ), a = n) ∧ 
  (¬ ∃ (n : ℤ), b = n) ∧ 
  (¬ ∃ (n : ℤ), c = n) ∧ 
  (∃ (n : ℤ), a * b = n) ∧ 
  (∃ (n : ℤ), b * c = n) ∧ 
  (∃ (n : ℤ), c * a = n) ∧ 
  (∃ (n : ℤ), a * b * c = n) := by
sorry

end NUMINAMATH_CALUDE_product_integers_exist_l2287_228712


namespace NUMINAMATH_CALUDE_problem_solution_l2287_228743

noncomputable section

-- Define the function f
def f (a k : ℝ) (x : ℝ) : ℝ := a^x - (k-1) * a^(-x)

-- Define the function g
def g (a m : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 2*m*(f a 2 x)

theorem problem_solution (a : ℝ) (h_a : a > 0 ∧ a ≠ 1) :
  -- Part 1: k = 2 if f is an odd function
  (∀ x, f a 2 x = -(f a 2 (-x))) →
  -- Part 2: If f(1) < 0, then f is decreasing and the inequality holds iff -3 < t < 5
  (f a 2 1 < 0 →
    (∀ x y, x < y → f a 2 x > f a 2 y) ∧
    (∀ t, (∀ x, f a 2 (x^2 + t*x) + f a 2 (4-x) < 0) ↔ -3 < t ∧ t < 5)) ∧
  -- Part 3: If f(1) = 3/2 and g has min value -2 on [1,+∞), then m = 2
  (f a 2 1 = 3/2 →
    (∃ m, (∀ x ≥ 1, g a m x ≥ -2) ∧
          (∃ x ≥ 1, g a m x = -2)) →
    m = 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2287_228743


namespace NUMINAMATH_CALUDE_total_scoops_l2287_228735

def flour_cups : ℚ := 3
def sugar_cups : ℚ := 2
def flour_scoop : ℚ := 1/4
def sugar_scoop : ℚ := 1/3

theorem total_scoops : 
  (flour_cups / flour_scoop + sugar_cups / sugar_scoop : ℚ) = 18 := by
  sorry

end NUMINAMATH_CALUDE_total_scoops_l2287_228735


namespace NUMINAMATH_CALUDE_min_value_product_l2287_228794

theorem min_value_product (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h_pos : θ₁ > 0 ∧ θ₂ > 0 ∧ θ₃ > 0 ∧ θ₄ > 0) 
  (h_sum : θ₁ + θ₂ + θ₃ + θ₄ = π) : 
  (2 * Real.sin θ₁ ^ 2 + 1 / Real.sin θ₁ ^ 2) *
  (2 * Real.sin θ₂ ^ 2 + 1 / Real.sin θ₂ ^ 2) *
  (2 * Real.sin θ₃ ^ 2 + 1 / Real.sin θ₃ ^ 2) *
  (2 * Real.sin θ₄ ^ 2 + 1 / Real.sin θ₄ ^ 2) ≥ 81 := by
  sorry

end NUMINAMATH_CALUDE_min_value_product_l2287_228794


namespace NUMINAMATH_CALUDE_committee_selection_problem_l2287_228776

/-- The number of ways to select a committee under specific constraints -/
def committeeSelections (n : ℕ) (k : ℕ) (pairTogether : Fin n → Fin n → Prop) (pairApart : Fin n → Fin n → Prop) : ℕ :=
  sorry

/-- The specific problem setup -/
theorem committee_selection_problem :
  let n : ℕ := 9
  let k : ℕ := 5
  let a : Fin n := 0
  let b : Fin n := 1
  let c : Fin n := 2
  let d : Fin n := 3
  let pairTogether (i j : Fin n) := (i = a ∧ j = b) ∨ (i = b ∧ j = a)
  let pairApart (i j : Fin n) := (i = c ∧ j = d) ∨ (i = d ∧ j = c)
  committeeSelections n k pairTogether pairApart = 41 :=
sorry

end NUMINAMATH_CALUDE_committee_selection_problem_l2287_228776


namespace NUMINAMATH_CALUDE_star_shape_perimeter_star_shape_perimeter_is_4pi_l2287_228734

/-- The perimeter of a star-like shape formed by arcs of six unit circles arranged in a regular hexagon configuration --/
theorem star_shape_perimeter : ℝ :=
  let n : ℕ := 6  -- number of coins
  let r : ℝ := 1  -- radius of each coin
  let angle_sum : ℝ := 2 * Real.pi  -- sum of internal angles of a hexagon
  4 * Real.pi

/-- Proof that the perimeter of the star-like shape is 4π --/
theorem star_shape_perimeter_is_4pi : star_shape_perimeter = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_star_shape_perimeter_star_shape_perimeter_is_4pi_l2287_228734


namespace NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l2287_228763

/-- A function that checks if a fraction is a terminating decimal -/
def isTerminatingDecimal (numerator : ℕ) (denominator : ℕ) : Prop :=
  ∃ (a b : ℕ), denominator = 2^a * 5^b

/-- The smallest positive integer n such that n/(n+150) is a terminating decimal -/
theorem smallest_n_for_terminating_decimal : 
  (∀ n : ℕ, n > 0 → n < 50 → ¬(isTerminatingDecimal n (n + 150))) ∧ 
  (isTerminatingDecimal 50 200) := by
  sorry

#check smallest_n_for_terminating_decimal

end NUMINAMATH_CALUDE_smallest_n_for_terminating_decimal_l2287_228763


namespace NUMINAMATH_CALUDE_second_printer_theorem_l2287_228740

/-- The time (in minutes) it takes for the second printer to print 800 flyers -/
def second_printer_time (first_printer_time second_printer_time combined_time : ℚ) : ℚ :=
  30 / 7

/-- Given the specifications of two printers, proves that the second printer
    takes 30/7 minutes to print 800 flyers -/
theorem second_printer_theorem (first_printer_time combined_time : ℚ) 
  (h1 : first_printer_time = 10)
  (h2 : combined_time = 3) :
  second_printer_time first_printer_time (second_printer_time first_printer_time (30/7) combined_time) combined_time = 30 / 7 := by
  sorry

#check second_printer_theorem

end NUMINAMATH_CALUDE_second_printer_theorem_l2287_228740


namespace NUMINAMATH_CALUDE_parallelogram_vertex_sum_l2287_228750

structure Parallelogram where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

def is_valid_parallelogram (p : Parallelogram) : Prop :=
  p.A.1 = 2 ∧ p.A.2 = -3 ∧
  p.B.1 = 7 ∧ p.B.2 = 0 ∧
  p.D.1 = -2 ∧ p.D.2 = 5 ∧
  (p.A.1 + p.D.1) / 2 = (p.B.1 + p.C.1) / 2 ∧
  (p.A.2 + p.D.2) / 2 = (p.B.2 + p.C.2) / 2

theorem parallelogram_vertex_sum (p : Parallelogram) 
  (h : is_valid_parallelogram p) : p.C.1 + p.C.2 = -5 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_vertex_sum_l2287_228750


namespace NUMINAMATH_CALUDE_surface_area_comparison_l2287_228707

/-- Given a cube, cylinder, and sphere with equal volumes, their surface areas satisfy S₃ < S₂ < S₁ -/
theorem surface_area_comparison 
  (V : ℝ) 
  (h_V_pos : V > 0) 
  (S₁ : ℝ) 
  (S₂ : ℝ) 
  (S₃ : ℝ) 
  (h_S₁ : S₁ = Real.rpow (216 * V^2) (1/3))
  (h_S₂ : S₂ = Real.rpow (54 * π * V^2) (1/3))
  (h_S₃ : S₃ = Real.rpow (36 * π * V^2) (1/3)) :
  S₃ < S₂ ∧ S₂ < S₁ :=
by sorry

end NUMINAMATH_CALUDE_surface_area_comparison_l2287_228707


namespace NUMINAMATH_CALUDE_sqrt_equation_roots_l2287_228718

theorem sqrt_equation_roots (p : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ Real.sqrt (x - p) = x ∧ Real.sqrt (y - p) = y) ↔ 0 ≤ p ∧ p < (1/4 : ℝ) :=
sorry

end NUMINAMATH_CALUDE_sqrt_equation_roots_l2287_228718


namespace NUMINAMATH_CALUDE_gcd_228_1995_l2287_228746

theorem gcd_228_1995 : Nat.gcd 228 1995 = 57 := by
  sorry

end NUMINAMATH_CALUDE_gcd_228_1995_l2287_228746


namespace NUMINAMATH_CALUDE_no_equal_notebooks_l2287_228758

theorem no_equal_notebooks : ¬∃ (x : ℝ), x > 0 ∧ 12 / x = 21 / (x + 1.2) := by
  sorry

end NUMINAMATH_CALUDE_no_equal_notebooks_l2287_228758


namespace NUMINAMATH_CALUDE_profit_maximizing_price_profit_function_increase_current_state_verification_cost_price_verification_l2287_228775

/-- Represents the profit function for a product with given pricing and demand characteristics. -/
def profit_function (x : ℝ) : ℝ :=
  (60 + x - 40) * (300 - 10 * x)

/-- Theorem stating that the profit-maximizing price is 65 yuan. -/
theorem profit_maximizing_price :
  ∃ (max_profit : ℝ), 
    (∀ (x : ℝ), profit_function x ≤ profit_function 5) ∧ 
    (profit_function 5 = max_profit) ∧
    (60 + 5 = 65) := by
  sorry

/-- Verifies that the profit function behaves as expected for price increases. -/
theorem profit_function_increase (x : ℝ) :
  profit_function x = -10 * x^2 + 100 * x + 6000 := by
  sorry

/-- Verifies that the current price and sales volume are consistent with the problem statement. -/
theorem current_state_verification :
  profit_function 0 = (60 - 40) * 300 := by
  sorry

/-- Ensures that the cost price is correctly represented in the profit function. -/
theorem cost_price_verification (x : ℝ) :
  (60 + x - 40) = (profit_function x) / (300 - 10 * x) := by
  sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_profit_function_increase_current_state_verification_cost_price_verification_l2287_228775


namespace NUMINAMATH_CALUDE_price_after_two_reductions_l2287_228785

-- Define the price reductions
def first_reduction : ℝ := 0.1  -- 10%
def second_reduction : ℝ := 0.14  -- 14%

-- Define the theorem
theorem price_after_two_reductions :
  let original_price : ℝ := 100
  let price_after_first_reduction := original_price * (1 - first_reduction)
  let final_price := price_after_first_reduction * (1 - second_reduction)
  final_price / original_price = 0.774 :=
by sorry

end NUMINAMATH_CALUDE_price_after_two_reductions_l2287_228785


namespace NUMINAMATH_CALUDE_base10_157_equals_base12_B21_l2287_228737

/-- Converts a base 12 number to base 10 -/
def base12ToBase10 (digits : List Nat) : Nat :=
  digits.foldl (fun acc d => acc * 12 + d) 0

/-- Represents 'B' as 11 in base 12 -/
def baseB : Nat := 11

theorem base10_157_equals_base12_B21 :
  157 = base12ToBase10 [baseB, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_base10_157_equals_base12_B21_l2287_228737


namespace NUMINAMATH_CALUDE_inequality_proof_l2287_228725

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 1) : 
  Real.sqrt (1 - a^2) + Real.sqrt (1 - b^2) + Real.sqrt (1 - c^2) + a + b + c > 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2287_228725


namespace NUMINAMATH_CALUDE_z_in_second_quadrant_l2287_228732

-- Define the complex number z
def z : ℂ := sorry

-- Define the condition z / (1 + i) = 2i
axiom z_condition : z / (1 + Complex.I) = 2 * Complex.I

-- Define the second quadrant
def second_quadrant (w : ℂ) : Prop :=
  w.re < 0 ∧ w.im > 0

-- Theorem statement
theorem z_in_second_quadrant : second_quadrant z := by
  sorry

end NUMINAMATH_CALUDE_z_in_second_quadrant_l2287_228732


namespace NUMINAMATH_CALUDE_probability_letter_in_mathematics_l2287_228792

def alphabet_size : ℕ := 26
def unique_letters_in_mathematics : ℕ := 8

theorem probability_letter_in_mathematics :
  (unique_letters_in_mathematics : ℚ) / (alphabet_size : ℚ) = 4 / 13 := by
  sorry

end NUMINAMATH_CALUDE_probability_letter_in_mathematics_l2287_228792


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l2287_228770

theorem fractional_equation_solution :
  ∃! x : ℝ, x ≠ 0 ∧ x ≠ -1 ∧ (2 / x = 1 / (x + 1)) ∧ x = -2 := by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l2287_228770


namespace NUMINAMATH_CALUDE_subtraction_division_equality_l2287_228768

theorem subtraction_division_equality : 6000 - (105 / 21.0) = 5995 := by
  sorry

end NUMINAMATH_CALUDE_subtraction_division_equality_l2287_228768


namespace NUMINAMATH_CALUDE_area_of_side_face_l2287_228741

/-- Represents a rectangular box with length, width, and height -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Theorem: Area of side face of a rectangular box -/
theorem area_of_side_face (b : Box) 
  (h1 : b.width * b.height = 0.5 * (b.length * b.width))
  (h2 : b.length * b.width = 1.5 * (b.length * b.height))
  (h3 : b.length * b.width * b.height = 5184) :
  b.length * b.height = 288 := by
  sorry

end NUMINAMATH_CALUDE_area_of_side_face_l2287_228741


namespace NUMINAMATH_CALUDE_kittens_remaining_l2287_228738

theorem kittens_remaining (initial_kittens given_away : ℕ) : 
  initial_kittens = 8 → given_away = 2 → initial_kittens - given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_kittens_remaining_l2287_228738


namespace NUMINAMATH_CALUDE_area_of_region_is_24_l2287_228769

/-- The region in the plane defined by the given inequality -/
def Region : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (|p.1| + |3 * p.2| - 6) * (|3 * p.1| + |p.2| - 6) ≤ 0}

/-- The area of the region -/
def AreaOfRegion : ℝ := sorry

/-- Theorem stating that the area of the region is 24 -/
theorem area_of_region_is_24 : AreaOfRegion = 24 := by sorry

end NUMINAMATH_CALUDE_area_of_region_is_24_l2287_228769


namespace NUMINAMATH_CALUDE_log_5_125000_bounds_l2287_228706

theorem log_5_125000_bounds : ∃ (a b : ℤ), 
  (a : ℝ) < Real.log 125000 / Real.log 5 ∧ 
  Real.log 125000 / Real.log 5 < (b : ℝ) ∧ 
  a = 6 ∧ 
  b = 7 ∧ 
  a + b = 13 := by
sorry

end NUMINAMATH_CALUDE_log_5_125000_bounds_l2287_228706


namespace NUMINAMATH_CALUDE_susan_board_game_movement_l2287_228705

theorem susan_board_game_movement (total_spaces : ℕ) (first_move : ℕ) (third_move : ℕ) (sent_back : ℕ) (remaining_spaces : ℕ) : 
  total_spaces = 48 →
  first_move = 8 →
  third_move = 6 →
  sent_back = 5 →
  remaining_spaces = 37 →
  first_move + third_move + remaining_spaces + sent_back = total_spaces →
  ∃ (second_move : ℕ), second_move = 28 := by
sorry

end NUMINAMATH_CALUDE_susan_board_game_movement_l2287_228705


namespace NUMINAMATH_CALUDE_halloween_bags_cost_l2287_228700

/-- Calculates the minimum cost to buy a given number of items, 
    where items can be bought in packs of 5 or individually --/
def minCost (numItems : ℕ) (packPrice packSize : ℕ) (individualPrice : ℕ) : ℕ :=
  let numPacks := numItems / packSize
  let numIndividuals := numItems % packSize
  numPacks * packPrice + numIndividuals * individualPrice

theorem halloween_bags_cost : 
  let totalStudents : ℕ := 25
  let vampireRequests : ℕ := 11
  let pumpkinRequests : ℕ := 14
  let packPrice : ℕ := 3
  let packSize : ℕ := 5
  let individualPrice : ℕ := 1
  
  vampireRequests + pumpkinRequests = totalStudents →
  
  minCost vampireRequests packPrice packSize individualPrice + 
  minCost pumpkinRequests packPrice packSize individualPrice = 17 := by
  sorry

end NUMINAMATH_CALUDE_halloween_bags_cost_l2287_228700


namespace NUMINAMATH_CALUDE_final_position_of_A_l2287_228709

-- Define the initial position of point A
def initial_position : ℝ := -3

-- Define the movement in the positive direction
def movement : ℝ := 4.5

-- Theorem to prove the final position of point A
theorem final_position_of_A : initial_position + movement = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_final_position_of_A_l2287_228709


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l2287_228739

theorem greatest_integer_inequality :
  ∀ x : ℤ, x ≤ 0 ↔ 5 * x - 4 < 3 - 2 * x :=
by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l2287_228739


namespace NUMINAMATH_CALUDE_domain_implies_a_eq_3_odd_function_implies_a_eq_1_odd_function_solution_set_l2287_228754

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log ((2 / (x - 1)) + a)

-- Define the domain condition
def domain_condition (a : ℝ) : Prop :=
  ∀ x, f a x ≠ 0 ↔ (x < 1/3 ∨ x > 1)

-- Define the odd function condition
def odd_function (a : ℝ) : Prop :=
  ∀ x, f a (-x) = -(f a x)

-- State the theorems
theorem domain_implies_a_eq_3 :
  ∃ a, domain_condition a → a = 3 :=
sorry

theorem odd_function_implies_a_eq_1 :
  ∃ a, odd_function a → a = 1 :=
sorry

theorem odd_function_solution_set :
  ∀ a, odd_function a →
    (∀ x, f a x > 0 ↔ x > 1) :=
sorry

end NUMINAMATH_CALUDE_domain_implies_a_eq_3_odd_function_implies_a_eq_1_odd_function_solution_set_l2287_228754


namespace NUMINAMATH_CALUDE_total_miles_ridden_l2287_228767

-- Define the given conditions
def miles_to_school : ℕ := 6
def miles_from_school : ℕ := 7
def trips_per_week : ℕ := 5

-- Define the theorem to prove
theorem total_miles_ridden : 
  miles_to_school * trips_per_week + miles_from_school * trips_per_week = 65 := by
  sorry

end NUMINAMATH_CALUDE_total_miles_ridden_l2287_228767


namespace NUMINAMATH_CALUDE_kirill_height_difference_l2287_228752

theorem kirill_height_difference (combined_height kirill_height : ℕ) 
  (h1 : combined_height = 112)
  (h2 : kirill_height = 49) :
  combined_height - kirill_height - kirill_height = 14 := by
  sorry

end NUMINAMATH_CALUDE_kirill_height_difference_l2287_228752


namespace NUMINAMATH_CALUDE_discount_clinic_savings_l2287_228766

theorem discount_clinic_savings (normal_fee : ℝ) : 
  (normal_fee - 2 * (0.3 * normal_fee) = 80) → normal_fee = 200 := by
  sorry

end NUMINAMATH_CALUDE_discount_clinic_savings_l2287_228766


namespace NUMINAMATH_CALUDE_octagon_area_theorem_l2287_228726

-- Define the rectangle BDEF
structure Rectangle :=
  (B D E F : ℝ × ℝ)

-- Define the octagon
structure Octagon :=
  (vertices : Fin 8 → ℝ × ℝ)

-- Define the condition AB = BC = 2
def side_length : ℝ := 2

-- Define the function to calculate the area of the octagon
noncomputable def octagon_area (rect : Rectangle) (side : ℝ) : ℝ :=
  sorry

-- Theorem statement
theorem octagon_area_theorem (rect : Rectangle) :
  octagon_area rect side_length = 16 + 8 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_octagon_area_theorem_l2287_228726


namespace NUMINAMATH_CALUDE_equation_solution_l2287_228728

theorem equation_solution : 
  ∃ x : ℚ, (x ≠ -2) ∧ (8 * x / (x + 2) - 5 / (x + 2) = 2 / (x + 2)) ∧ (x = 7 / 8) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2287_228728


namespace NUMINAMATH_CALUDE_area_ratio_is_459_625_l2287_228780

/-- Triangle XYZ with points P and Q -/
structure TriangleXYZ where
  /-- Side length XY -/
  xy : ℝ
  /-- Side length YZ -/
  yz : ℝ
  /-- Side length XZ -/
  xz : ℝ
  /-- Length XP -/
  xp : ℝ
  /-- Length XQ -/
  xq : ℝ
  /-- xy is positive -/
  xy_pos : 0 < xy
  /-- yz is positive -/
  yz_pos : 0 < yz
  /-- xz is positive -/
  xz_pos : 0 < xz
  /-- xp is positive and less than xy -/
  xp_bounds : 0 < xp ∧ xp < xy
  /-- xq is positive and less than xz -/
  xq_bounds : 0 < xq ∧ xq < xz

/-- The ratio of areas in the triangle -/
def areaRatio (t : TriangleXYZ) : ℝ := sorry

/-- Theorem stating the ratio of areas -/
theorem area_ratio_is_459_625 (t : TriangleXYZ) 
  (h1 : t.xy = 30) (h2 : t.yz = 45) (h3 : t.xz = 51) 
  (h4 : t.xp = 18) (h5 : t.xq = 15) : 
  areaRatio t = 459 / 625 := by sorry

end NUMINAMATH_CALUDE_area_ratio_is_459_625_l2287_228780


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2287_228793

/-- A geometric sequence with common ratio q < 0 -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  q < 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ) :
  geometric_sequence a q →
  a 2 = 1 - a 1 →
  a 4 = 4 - a 3 →
  a 4 + a 5 = -8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2287_228793


namespace NUMINAMATH_CALUDE_even_function_solution_set_l2287_228757

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | x * f x < 0}

theorem even_function_solution_set
  (f : ℝ → ℝ)
  (h_even : ∀ x, f (-x) = f x)
  (h_zero : f (-4) = 0 ∧ f 2 = 0)
  (h_decreasing : ∀ x ∈ Set.Icc 0 3, ∀ y ∈ Set.Icc 0 3, x < y → f x > f y)
  (h_increasing : ∀ x ∈ Set.Ici 3, ∀ y ∈ Set.Ici 3, x < y → f x < f y) :
  solution_set f = Set.union (Set.union (Set.Iio (-4)) (Set.Ioo (-2) 0)) (Set.Ioo 2 4) :=
sorry

end NUMINAMATH_CALUDE_even_function_solution_set_l2287_228757


namespace NUMINAMATH_CALUDE_triangle_special_angle_l2287_228745

theorem triangle_special_angle (a b c : ℝ) (h : (a + 2*b + c)*(a + b - c - 2) = 4*a*b) :
  let C := Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))
  C = π / 3 := by sorry

end NUMINAMATH_CALUDE_triangle_special_angle_l2287_228745


namespace NUMINAMATH_CALUDE_min_decimal_digits_l2287_228774

theorem min_decimal_digits (n : ℕ) (d : ℕ) : 
  n = 987654321 ∧ d = 2^30 * 5^6 → 
  (∃ (k : ℕ), k = 30 ∧ 
    ∀ (m : ℕ), (∃ (q r : ℚ), q * 10^m = n / d ∧ r = 0) → m ≥ k) ∧
  (∀ (l : ℕ), l < 30 → 
    ∃ (q r : ℚ), q * 10^l = n / d ∧ r ≠ 0) :=
sorry

end NUMINAMATH_CALUDE_min_decimal_digits_l2287_228774


namespace NUMINAMATH_CALUDE_homework_students_l2287_228778

theorem homework_students (total : ℕ) (reading : ℕ) (games : ℕ) (homework : ℕ) : 
  total = 24 ∧ 
  reading = total / 2 ∧ 
  games = total / 3 ∧ 
  homework = total - (reading + games) →
  homework = 4 := by
sorry

end NUMINAMATH_CALUDE_homework_students_l2287_228778


namespace NUMINAMATH_CALUDE_optimal_tax_and_revenue_l2287_228796

-- Define the market supply function
def supply_function (P : ℝ) : ℝ := 6 * P - 312

-- Define the market demand function
def demand_function (P : ℝ) (a : ℝ) : ℝ := a - 4 * P

-- Define the tax revenue function
def tax_revenue (t : ℝ) : ℝ := 288 * t - 2.4 * t^2

-- State the theorem
theorem optimal_tax_and_revenue 
  (elasticity_ratio : ℝ) 
  (consumer_price_after_tax : ℝ) 
  (initial_tax_rate : ℝ) :
  elasticity_ratio = 1.5 →
  consumer_price_after_tax = 118 →
  initial_tax_rate = 30 →
  ∃ (optimal_tax : ℝ) (max_revenue : ℝ),
    optimal_tax = 60 ∧
    max_revenue = 8640 ∧
    ∀ (t : ℝ), tax_revenue t ≤ max_revenue :=
by sorry

end NUMINAMATH_CALUDE_optimal_tax_and_revenue_l2287_228796


namespace NUMINAMATH_CALUDE_percent_profit_calculation_l2287_228703

/-- Given that the cost price of 75 articles after a 5% discount
    equals the selling price of 60 articles before a 12% sales tax,
    prove that the percent profit is 25%. -/
theorem percent_profit_calculation (CP : ℝ) (SP : ℝ) :
  75 * CP * (1 - 0.05) = 60 * SP →
  (SP - CP * (1 - 0.05)) / (CP * (1 - 0.05)) * 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_percent_profit_calculation_l2287_228703


namespace NUMINAMATH_CALUDE_line_AB_intersects_S₂_and_S_l2287_228710

-- Define the circles and points
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def S₁ : Circle := { center := (0, 0), radius := 1 }
def S₂ : Circle := { center := (2, 0), radius := 1 }
def S  : Circle := { center := (1, 1), radius := 2 }
def A  : ℝ × ℝ := (1, 0)
def B  : ℝ × ℝ := (1, 2)
def O  : ℝ × ℝ := S.center

-- Define the conditions
axiom S₁_S₂_tangent : S₁.center.fst + S₁.radius = S₂.center.fst - S₂.radius
axiom O_on_S₁ : (O.fst - S₁.center.fst)^2 + (O.snd - S₁.center.snd)^2 = S₁.radius^2
axiom S₁_S_tangent_at_B : (B.fst - S₁.center.fst)^2 + (B.snd - S₁.center.snd)^2 = S₁.radius^2 ∧
                          (B.fst - S.center.fst)^2 + (B.snd - S.center.snd)^2 = S.radius^2

-- Theorem to prove
theorem line_AB_intersects_S₂_and_S :
  ∃ (P : ℝ × ℝ), P ≠ A ∧ P ≠ B ∧
  (∃ (t : ℝ), P = (A.fst + t * (B.fst - A.fst), A.snd + t * (B.snd - A.snd))) ∧
  (P.fst - S₂.center.fst)^2 + (P.snd - S₂.center.snd)^2 = S₂.radius^2 ∧
  (P.fst - S.center.fst)^2 + (P.snd - S.center.snd)^2 = S.radius^2 :=
sorry

end NUMINAMATH_CALUDE_line_AB_intersects_S₂_and_S_l2287_228710


namespace NUMINAMATH_CALUDE_problem_statement_l2287_228744

theorem problem_statement (a : ℤ) 
  (h1 : 0 ≤ a) (h2 : a < 13) 
  (h3 : (51^2018 + a) % 13 = 0) : 
  a = 12 := by sorry

end NUMINAMATH_CALUDE_problem_statement_l2287_228744


namespace NUMINAMATH_CALUDE_count_complementary_sets_l2287_228761

/-- Represents a card in the deck -/
structure Card where
  shape : Fin 4
  color : Fin 4
  shade : Fin 4

/-- The deck of cards -/
def Deck : Finset Card := sorry

/-- A set of three cards -/
def ThreeCardSet : Type := Finset Card

/-- Checks if a set of three cards is complementary -/
def isComplementary (set : ThreeCardSet) : Prop := sorry

/-- The set of all complementary three-card sets -/
def ComplementarySets : Finset ThreeCardSet := sorry

theorem count_complementary_sets :
  Finset.card ComplementarySets = 360 := by sorry

end NUMINAMATH_CALUDE_count_complementary_sets_l2287_228761


namespace NUMINAMATH_CALUDE_ellipse_k_range_l2287_228719

/-- Represents an ellipse equation in the form x^2 + ky^2 = 2 --/
structure EllipseEquation where
  k : ℝ

/-- Predicate to check if the equation represents a valid ellipse with foci on the y-axis --/
def is_valid_ellipse (e : EllipseEquation) : Prop :=
  0 < e.k ∧ e.k < 1

/-- Theorem stating the range of k for a valid ellipse with foci on the y-axis --/
theorem ellipse_k_range (e : EllipseEquation) : 
  (∃ (x y : ℝ), x^2 + e.k * y^2 = 2) ∧ 
  (∃ (c : ℝ), c ≠ 0 ∧ ∀ (x y : ℝ), x^2 + e.k * y^2 = 2 → x^2 + (y - c)^2 = x^2 + (y + c)^2) 
  ↔ is_valid_ellipse e :=
sorry

end NUMINAMATH_CALUDE_ellipse_k_range_l2287_228719


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2287_228781

theorem cubic_roots_sum (p q r : ℝ) : 
  (3 * p^3 + 4 * p^2 - 5 * p - 6 = 0) →
  (3 * q^3 + 4 * q^2 - 5 * q - 6 = 0) →
  (3 * r^3 + 4 * r^2 - 5 * r - 6 = 0) →
  (p + q + 1)^3 + (q + r + 1)^3 + (r + p + 1)^3 = 2/3 := by
sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2287_228781


namespace NUMINAMATH_CALUDE_min_distance_line_circle_l2287_228777

/-- The minimum distance between a point on the line x - y + 1 = 0 
    and a point on the circle (x - 1)² + y² = 1 is √2 - 1 -/
theorem min_distance_line_circle :
  let line := {(x, y) : ℝ × ℝ | x - y + 1 = 0}
  let circle := {(x, y) : ℝ × ℝ | (x - 1)^2 + y^2 = 1}
  ∃ (d : ℝ), d = Real.sqrt 2 - 1 ∧ 
    (∀ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line → q ∈ circle → 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d) ∧
    (∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧ 
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = d) :=
by
  sorry


end NUMINAMATH_CALUDE_min_distance_line_circle_l2287_228777


namespace NUMINAMATH_CALUDE_power_greater_than_linear_l2287_228701

theorem power_greater_than_linear (n : ℕ) (h : n ≥ 3) : 2^n > 2*n + 1 := by
  sorry

end NUMINAMATH_CALUDE_power_greater_than_linear_l2287_228701


namespace NUMINAMATH_CALUDE_f_monotone_increasing_interval_l2287_228749

/-- The function f(x) = x^2 + 2x + 1 -/
def f (x : ℝ) := x^2 + 2*x + 1

/-- The monotonically increasing interval of f(x) is [-1, +∞) -/
theorem f_monotone_increasing_interval :
  ∀ x y : ℝ, x ≥ -1 → y ≥ -1 → x < y → f x < f y :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_interval_l2287_228749


namespace NUMINAMATH_CALUDE_scientific_notation_of_70_62_million_l2287_228731

/-- Proves that 70.62 million is equal to 7.062 × 10^7 in scientific notation -/
theorem scientific_notation_of_70_62_million :
  (70.62 * 1000000 : ℝ) = 7.062 * (10 ^ 7) := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_70_62_million_l2287_228731


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l2287_228773

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

/-- Theorem: A convex dodecagon has 54 diagonals -/
theorem dodecagon_diagonals : num_diagonals dodecagon_sides = 54 := by
  sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l2287_228773


namespace NUMINAMATH_CALUDE_planning_committee_selection_l2287_228723

theorem planning_committee_selection (n : ℕ) : 
  (n.choose 2 = 21) → (n.choose 4 = 35) := by
  sorry

end NUMINAMATH_CALUDE_planning_committee_selection_l2287_228723


namespace NUMINAMATH_CALUDE_jinyoung_has_fewest_l2287_228711

/-- Represents the number of marbles each person has -/
structure Marbles where
  seonho : ℕ
  minjeong : ℕ
  jinyoung : ℕ
  joohwan : ℕ

/-- The conditions of the marble problem -/
def marble_conditions (m : Marbles) : Prop :=
  m.seonho = m.minjeong + 1 ∧
  m.jinyoung = m.joohwan - 3 ∧
  m.minjeong = 6 ∧
  m.joohwan = 7

/-- Jinyoung has the fewest marbles -/
theorem jinyoung_has_fewest (m : Marbles) (h : marble_conditions m) :
  m.jinyoung ≤ m.seonho ∧ m.jinyoung ≤ m.minjeong ∧ m.jinyoung ≤ m.joohwan :=
by sorry

end NUMINAMATH_CALUDE_jinyoung_has_fewest_l2287_228711


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l2287_228784

theorem solution_set_of_inequality (x : ℝ) :
  Set.Ioo (-1 : ℝ) 2 = {x | |x^2 - x| < 2} := by sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l2287_228784


namespace NUMINAMATH_CALUDE_right_triangle_set_l2287_228708

/-- Checks if three numbers can form a right triangle --/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The given sets of numbers --/
def set_A : Fin 3 → ℕ := ![4, 5, 6]
def set_B : Fin 3 → ℕ := ![12, 16, 20]
def set_C : Fin 3 → ℕ := ![5, 10, 13]
def set_D : Fin 3 → ℕ := ![8, 40, 41]

/-- The main theorem --/
theorem right_triangle_set :
  (¬ is_right_triangle (set_A 0) (set_A 1) (set_A 2)) ∧
  (is_right_triangle (set_B 0) (set_B 1) (set_B 2)) ∧
  (¬ is_right_triangle (set_C 0) (set_C 1) (set_C 2)) ∧
  (¬ is_right_triangle (set_D 0) (set_D 1) (set_D 2)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_set_l2287_228708


namespace NUMINAMATH_CALUDE_shelby_poster_purchase_l2287_228716

/-- Calculates the number of posters Shelby can buy given the problem conditions --/
def calculate_posters (initial_amount coupon_value tax_rate : ℚ)
  (book1_cost book2_cost bookmark_cost pencils_cost notebook_cost poster_cost : ℚ)
  (discount_rate1 discount_rate2 : ℚ)
  (discount_threshold1 discount_threshold2 : ℚ) : ℕ :=
  sorry

/-- Theorem stating that Shelby can buy exactly 4 posters --/
theorem shelby_poster_purchase :
  let initial_amount : ℚ := 60
  let book1_cost : ℚ := 15
  let book2_cost : ℚ := 9
  let bookmark_cost : ℚ := 3.5
  let pencils_cost : ℚ := 4.8
  let notebook_cost : ℚ := 6.2
  let poster_cost : ℚ := 6
  let discount_rate1 : ℚ := 0.15
  let discount_rate2 : ℚ := 0.10
  let discount_threshold1 : ℚ := 40
  let discount_threshold2 : ℚ := 25
  let coupon_value : ℚ := 5
  let tax_rate : ℚ := 0.08
  calculate_posters initial_amount coupon_value tax_rate
    book1_cost book2_cost bookmark_cost pencils_cost notebook_cost poster_cost
    discount_rate1 discount_rate2 discount_threshold1 discount_threshold2 = 4 :=
by sorry

end NUMINAMATH_CALUDE_shelby_poster_purchase_l2287_228716


namespace NUMINAMATH_CALUDE_quadratic_function_properties_range_of_g_l2287_228717

-- Define the quadratic function f
def f (x : ℝ) : ℝ := -2 * x^2 - 4 * x

-- State the theorem
theorem quadratic_function_properties :
  -- The vertex of f is (-1, 2)
  (f (-1) = 2 ∧ ∀ x, f x ≤ f (-1)) ∧
  -- f passes through the origin
  f 0 = 0 ∧
  -- The range of f(2x) is (-∞, 0)
  (∀ y, (∃ x, f (2*x) = y) ↔ y < 0) := by
sorry

-- Define g as f(2x)
def g (x : ℝ) : ℝ := f (2*x)

-- Additional theorem for the range of g
theorem range_of_g :
  (∀ y, (∃ x, g x = y) ↔ y < 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_range_of_g_l2287_228717


namespace NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2287_228760

-- Define the function g
noncomputable def g (x : ℝ) : ℤ :=
  if x > -3 then Int.ceil (1 / (x + 3))
  else if x < -3 then Int.floor (1 / (x + 3))
  else 0  -- This value won't be used, it's just to make the function total

-- Theorem statement
theorem zero_not_in_range_of_g :
  ∀ x : ℝ, x ≠ -3 → g x ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_zero_not_in_range_of_g_l2287_228760


namespace NUMINAMATH_CALUDE_water_tank_capacity_l2287_228789

theorem water_tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (added_volume : Rat) (total_capacity : Rat) : 
  initial_fraction = 1/3 →
  final_fraction = 2/5 →
  added_volume = 5 →
  initial_fraction * total_capacity + added_volume = final_fraction * total_capacity →
  total_capacity = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_water_tank_capacity_l2287_228789


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_q_l2287_228720

-- Define the propositions p and q
def p (m : ℝ) : Prop := m ≥ (1/4 : ℝ)
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x + m = 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- Theorem statement
theorem not_p_sufficient_not_necessary_q :
  sufficient_not_necessary (¬∀ m, p m) (∀ m, q m) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_q_l2287_228720


namespace NUMINAMATH_CALUDE_sum_of_a_and_b_l2287_228783

theorem sum_of_a_and_b (a b : ℝ) : (2*a + 2*b - 1) * (2*a + 2*b + 1) = 99 → a + b = 5 ∨ a + b = -5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_a_and_b_l2287_228783


namespace NUMINAMATH_CALUDE_product_of_recurring_decimal_and_seven_l2287_228759

theorem product_of_recurring_decimal_and_seven :
  ∃ (x : ℚ), (∃ (n : ℕ), x = (456 : ℚ) / (10^3 - 1)) ∧ 7 * x = 355 / 111 := by
  sorry

end NUMINAMATH_CALUDE_product_of_recurring_decimal_and_seven_l2287_228759


namespace NUMINAMATH_CALUDE_statement_a_statement_b_l2287_228724

-- Define rationality for real numbers
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Statement a
theorem statement_a : ∃ (x : ℝ), IsRational (x^7) ∧ IsRational (x^12) ∧ ¬IsRational x := by
  sorry

-- Statement b
theorem statement_b : ∀ (x : ℝ), IsRational (x^9) ∧ IsRational (x^12) → IsRational x := by
  sorry

end NUMINAMATH_CALUDE_statement_a_statement_b_l2287_228724


namespace NUMINAMATH_CALUDE_pasture_feeding_theorem_l2287_228786

/-- Represents a pasture with growing grass -/
structure Pasture where
  dailyGrowthRate : ℕ
  initialGrass : ℕ

/-- Calculates the number of days a pasture can feed a given number of cows -/
def feedingDays (p : Pasture) (cows : ℕ) : ℕ :=
  (p.initialGrass + p.dailyGrowthRate * cows) / cows

theorem pasture_feeding_theorem (p : Pasture) : 
  feedingDays p 10 = 20 → 
  feedingDays p 15 = 10 → 
  p.dailyGrowthRate = 5 ∧ 
  feedingDays p 30 = 4 := by
  sorry

#check pasture_feeding_theorem

end NUMINAMATH_CALUDE_pasture_feeding_theorem_l2287_228786


namespace NUMINAMATH_CALUDE_expression_evaluation_l2287_228788

theorem expression_evaluation (m : ℤ) : 
  m = -1 → (6 * m^2 - m + 3) + (-5 * m^2 + 2 * m + 1) = 4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2287_228788


namespace NUMINAMATH_CALUDE_chinese_character_multiplication_l2287_228790

theorem chinese_character_multiplication : ∃! (x y : Nat), 
  x ≠ y ∧ x ≠ 3 ∧ x ≠ 0 ∧ y ≠ 3 ∧ y ≠ 0 ∧
  (3000 + 100 * x + y) * (3000 + 100 * x + y) ≥ 10000000 ∧
  (3000 + 100 * x + y) * (3000 + 100 * x + y) < 100000000 :=
by sorry

#check chinese_character_multiplication

end NUMINAMATH_CALUDE_chinese_character_multiplication_l2287_228790


namespace NUMINAMATH_CALUDE_line_slope_angle_l2287_228736

theorem line_slope_angle (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y + 2 = 0) → -- line equation
  (Real.tan (45 * Real.pi / 180) = -1 / a) → -- slope angle is 45°
  a = -1 := by
sorry

end NUMINAMATH_CALUDE_line_slope_angle_l2287_228736


namespace NUMINAMATH_CALUDE_simplify_and_rationalize_l2287_228756

theorem simplify_and_rationalize (x : ℝ) (h : x^3 = 3) : 
  1 / (1 + 1 / (x + 1)) = 9 / 13 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_rationalize_l2287_228756


namespace NUMINAMATH_CALUDE_total_problems_l2287_228755

def math_pages : ℕ := 6
def reading_pages : ℕ := 4
def problems_per_page : ℕ := 3

theorem total_problems : math_pages + reading_pages * problems_per_page = 30 := by
  sorry

end NUMINAMATH_CALUDE_total_problems_l2287_228755


namespace NUMINAMATH_CALUDE_sofa_payment_difference_l2287_228729

/-- Given that Joan and Karl bought sofas with a total cost of $600,
    Joan paid $230, and twice Joan's payment is more than Karl's,
    prove that the difference between twice Joan's payment and Karl's is $90. -/
theorem sofa_payment_difference :
  ∀ (joan_payment karl_payment : ℕ),
  joan_payment + karl_payment = 600 →
  joan_payment = 230 →
  2 * joan_payment > karl_payment →
  2 * joan_payment - karl_payment = 90 := by
sorry

end NUMINAMATH_CALUDE_sofa_payment_difference_l2287_228729


namespace NUMINAMATH_CALUDE_expression_evaluation_l2287_228798

theorem expression_evaluation :
  let a : ℚ := -1/3
  let expr := (a / (a - 1)) / ((a + 1) / (a^2 - 1)) - (1 - 2*a)
  expr = -2 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2287_228798


namespace NUMINAMATH_CALUDE_assembly_line_theorem_l2287_228714

/-- Represents the assembly line production --/
structure AssemblyLine where
  initial_rate : ℕ
  initial_order : ℕ
  increased_rate : ℕ
  second_order : ℕ

/-- Calculates the overall average output of the assembly line --/
def average_output (line : AssemblyLine) : ℚ :=
  let total_cogs := line.initial_order + line.second_order
  let total_time := (line.initial_order : ℚ) / line.initial_rate + (line.second_order : ℚ) / line.increased_rate
  total_cogs / total_time

/-- Theorem stating that the average output for the given conditions is 40 cogs per hour --/
theorem assembly_line_theorem (line : AssemblyLine) 
    (h1 : line.initial_rate = 30)
    (h2 : line.initial_order = 60)
    (h3 : line.increased_rate = 60)
    (h4 : line.second_order = 60) :
  average_output line = 40 := by
  sorry

end NUMINAMATH_CALUDE_assembly_line_theorem_l2287_228714


namespace NUMINAMATH_CALUDE_linear_function_x_intercept_l2287_228779

/-- A linear function f(x) = -x + 2 -/
def f (x : ℝ) : ℝ := -x + 2

/-- The x-coordinate of the intersection point with the x-axis -/
def x_intercept : ℝ := 2

theorem linear_function_x_intercept :
  f x_intercept = 0 ∧ x_intercept = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_x_intercept_l2287_228779


namespace NUMINAMATH_CALUDE_penguin_fish_distribution_penguin_fish_distribution_proof_l2287_228727

theorem penguin_fish_distribution (total_penguins : ℕ) 
  (emperor_ratio adelie_ratio : ℕ) 
  (emperor_fish adelie_fish : ℚ) 
  (fish_constraint : ℕ) : Prop :=
  let emperor_count := (total_penguins * emperor_ratio) / (emperor_ratio + adelie_ratio)
  let adelie_count := (total_penguins * adelie_ratio) / (emperor_ratio + adelie_ratio)
  let total_fish_needed := (emperor_count : ℚ) * emperor_fish + (adelie_count : ℚ) * adelie_fish
  total_penguins = 48 ∧ 
  emperor_ratio = 3 ∧ 
  adelie_ratio = 5 ∧ 
  emperor_fish = 3/2 ∧ 
  adelie_fish = 2 ∧ 
  fish_constraint = 115 →
  total_fish_needed ≤ fish_constraint

-- Proof
theorem penguin_fish_distribution_proof : 
  penguin_fish_distribution 48 3 5 (3/2) 2 115 := by
  sorry

end NUMINAMATH_CALUDE_penguin_fish_distribution_penguin_fish_distribution_proof_l2287_228727


namespace NUMINAMATH_CALUDE_student_average_equals_actual_average_l2287_228753

theorem student_average_equals_actual_average 
  (w x y z : ℤ) (h : w < x ∧ x < y ∧ y < z) :
  (((w + x) / 2 + (y + z) / 2) / 2 : ℚ) = ((w + x + y + z) / 4 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_student_average_equals_actual_average_l2287_228753


namespace NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_two_to_one_l2287_228715

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square with side length 3 -/
structure Square where
  p : Point
  q : Point
  r : Point
  s : Point

/-- Represents the configuration of points and lines in the square -/
structure SquareConfiguration where
  square : Square
  t : Point
  u : Point
  v : Point
  w : Point

/-- The ratio of shaded to unshaded area in the square configuration -/
def shadedToUnshadedRatio (config : SquareConfiguration) : ℚ := 2

/-- Theorem stating that the ratio of shaded to unshaded area is 2:1 -/
theorem shaded_to_unshaded_ratio_is_two_to_one (config : SquareConfiguration) :
  shadedToUnshadedRatio config = 2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_to_unshaded_ratio_is_two_to_one_l2287_228715


namespace NUMINAMATH_CALUDE_shekar_average_marks_l2287_228791

/-- Represents Shekar's scores in different subjects -/
structure ShekarScores where
  mathematics : ℕ
  science : ℕ
  social_studies : ℕ
  english : ℕ
  biology : ℕ

/-- Calculates the average of Shekar's scores -/
def average_marks (scores : ShekarScores) : ℚ :=
  (scores.mathematics + scores.science + scores.social_studies + scores.english + scores.biology) / 5

/-- Theorem stating that Shekar's average marks are 77 -/
theorem shekar_average_marks :
  let scores : ShekarScores := {
    mathematics := 76,
    science := 65,
    social_studies := 82,
    english := 67,
    biology := 95
  }
  average_marks scores = 77 := by sorry

end NUMINAMATH_CALUDE_shekar_average_marks_l2287_228791


namespace NUMINAMATH_CALUDE_firecracker_sales_properties_l2287_228722

/-- Electronic firecracker sales model -/
structure FirecrackerSales where
  cost : ℝ
  demand : ℝ → ℝ
  price_range : Set ℝ

/-- Daily profit function -/
def daily_profit (model : FirecrackerSales) (x : ℝ) : ℝ :=
  (x - model.cost) * model.demand x

theorem firecracker_sales_properties (model : FirecrackerSales) 
  (h_cost : model.cost = 80)
  (h_demand : ∀ x, model.demand x = -2 * x + 320)
  (h_range : model.price_range = {x | 80 ≤ x ∧ x ≤ 160}) :
  (∀ x ∈ model.price_range, daily_profit model x = -2 * x^2 + 480 * x - 25600) ∧
  (∃ max_price ∈ model.price_range, 
    (∀ x ∈ model.price_range, daily_profit model x ≤ daily_profit model max_price) ∧
    daily_profit model max_price = 3200 ∧
    max_price = 120) ∧
  (∃ price ∈ model.price_range, daily_profit model price = 2400 ∧ price = 100) := by
  sorry

end NUMINAMATH_CALUDE_firecracker_sales_properties_l2287_228722


namespace NUMINAMATH_CALUDE_square_of_complex_is_pure_imaginary_l2287_228799

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem square_of_complex_is_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) ^ 2) → a = 1 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_square_of_complex_is_pure_imaginary_l2287_228799


namespace NUMINAMATH_CALUDE_equation_not_quadratic_l2287_228713

def is_quadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

def equation (y : ℝ) : ℝ := 3 * y * (y - 1) - y * (3 * y + 1)

theorem equation_not_quadratic : ¬ is_quadratic equation := by
  sorry

end NUMINAMATH_CALUDE_equation_not_quadratic_l2287_228713


namespace NUMINAMATH_CALUDE_books_second_shop_l2287_228721

def books_first_shop : ℕ := 65
def cost_first_shop : ℕ := 1080
def cost_second_shop : ℕ := 840
def average_price : ℕ := 16

theorem books_second_shop :
  (cost_first_shop + cost_second_shop) / average_price - books_first_shop = 55 := by
  sorry

end NUMINAMATH_CALUDE_books_second_shop_l2287_228721


namespace NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l2287_228704

/-- The cost per trip to an amusement park given the following conditions:
  - Two season passes are purchased
  - Each pass costs 100 units of currency
  - One person uses their pass 35 times
  - Another person uses their pass 15 times
-/
theorem amusement_park_cost_per_trip 
  (pass_cost : ℝ) 
  (trips_person1 : ℕ) 
  (trips_person2 : ℕ) : 
  pass_cost = 100 ∧ 
  trips_person1 = 35 ∧ 
  trips_person2 = 15 → 
  (2 * pass_cost) / (trips_person1 + trips_person2 : ℝ) = 4 := by
  sorry

#check amusement_park_cost_per_trip

end NUMINAMATH_CALUDE_amusement_park_cost_per_trip_l2287_228704


namespace NUMINAMATH_CALUDE_parallel_iff_no_common_points_l2287_228730

-- Define a structure for a 3D space
structure Space3D where
  Point : Type
  Line : Type
  Plane : Type

-- Define the concept of a line being parallel to a plane
def parallel (S : Space3D) (l : S.Line) (p : S.Plane) : Prop := sorry

-- Define the concept of a line having no common points with a plane
def no_common_points (S : Space3D) (l : S.Line) (p : S.Plane) : Prop := sorry

-- Theorem statement
theorem parallel_iff_no_common_points (S : Space3D) (a : S.Line) (M : S.Plane) :
  parallel S a M ↔ no_common_points S a M := by sorry

end NUMINAMATH_CALUDE_parallel_iff_no_common_points_l2287_228730


namespace NUMINAMATH_CALUDE_problem_1_l2287_228797

theorem problem_1 (x y : ℝ) (hx : x = Real.sqrt 3 + Real.sqrt 5) (hy : y = Real.sqrt 3 - Real.sqrt 5) :
  2 * x^2 - 4 * x * y + 2 * y^2 = 40 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l2287_228797


namespace NUMINAMATH_CALUDE_min_value_squared_sum_l2287_228762

theorem min_value_squared_sum (p q r s t u v w : ℝ) 
  (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 16) : 
  (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 ≥ 64 ∧ 
  ∃ p q r s t u v w : ℝ, p * q * r * s = 16 ∧ t * u * v * w = 16 ∧ 
    (p * t)^2 + (q * u)^2 + (r * v)^2 + (s * w)^2 = 64 := by
  sorry

end NUMINAMATH_CALUDE_min_value_squared_sum_l2287_228762


namespace NUMINAMATH_CALUDE_fold_sum_value_l2287_228765

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a fold on graph paper -/
structure Fold :=
  (p1 : Point)
  (p2 : Point)
  (q1 : Point)
  (q2 : Point)

/-- The sum of the coordinates of the unknown point in a fold -/
def fold_sum (f : Fold) : ℝ := f.q2.x + f.q2.y

/-- The theorem stating the sum of coordinates of the unknown point -/
theorem fold_sum_value (f : Fold) 
  (h1 : f.p1 = ⟨0, 2⟩)
  (h2 : f.p2 = ⟨4, 0⟩)
  (h3 : f.q1 = ⟨7, 3⟩) :
  fold_sum f = 6.8 := by
  sorry

end NUMINAMATH_CALUDE_fold_sum_value_l2287_228765


namespace NUMINAMATH_CALUDE_solution_pairs_l2287_228782

theorem solution_pairs : ∀ x y : ℝ,
  (x^2 + y^2 - 48*x - 29*y + 714 = 0 ∧
   2*x*y - 29*x - 48*y + 756 = 0) ↔
  ((x = 31.5 ∧ y = 10.5) ∨
   (x = 20 ∧ y = 22) ∨
   (x = 28 ∧ y = 7) ∨
   (x = 16.5 ∧ y = 18.5)) := by
sorry

end NUMINAMATH_CALUDE_solution_pairs_l2287_228782


namespace NUMINAMATH_CALUDE_parabola_intersection_right_angle_l2287_228787

-- Define the line equation
def line (x y : ℝ) : Prop := x - 2*y - 1 = 0

-- Define the parabola equation
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define a point on the parabola
def point_on_parabola (t : ℝ) : ℝ × ℝ := (t^2, 2*t)

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem parabola_intersection_right_angle :
  ∃ (A B C : ℝ × ℝ),
    A ≠ B ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    parabola A.1 A.2 ∧
    parabola B.1 B.2 ∧
    parabola C.1 C.2 ∧
    angle A C B = π/2 →
    C = (1, -2) ∨ C = (9, -6) :=
sorry

end NUMINAMATH_CALUDE_parabola_intersection_right_angle_l2287_228787


namespace NUMINAMATH_CALUDE_function_zero_range_l2287_228702

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x^2

def has_exactly_one_zero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0

theorem function_zero_range (a : ℝ) :
  has_exactly_one_zero (f a) 1 (Real.exp 2) →
  a ∈ Set.Iic (-(Real.exp 4) / 2) ∪ {-2 * Real.exp 1} :=
sorry

end NUMINAMATH_CALUDE_function_zero_range_l2287_228702


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l2287_228771

theorem sum_of_coefficients (b₀ b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) :
  (∀ x : ℝ, (5*x - 2)^6 = b₆*x^6 + b₅*x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀) →
  b₆ + b₅ + b₄ + b₃ + b₂ + b₁ + b₀ = 729 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l2287_228771


namespace NUMINAMATH_CALUDE_unique_function_satisfying_condition_l2287_228764

open Function Real

theorem unique_function_satisfying_condition :
  ∃! f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + f y) = y + (f x)^2 ∧ f = id := by
  sorry

end NUMINAMATH_CALUDE_unique_function_satisfying_condition_l2287_228764


namespace NUMINAMATH_CALUDE_guitar_sales_proof_l2287_228772

/-- Calculates the total amount earned from selling guitars -/
def total_guitar_sales (total_guitars : ℕ) (electric_guitars : ℕ) (electric_price : ℕ) (acoustic_price : ℕ) : ℕ :=
  let acoustic_guitars := total_guitars - electric_guitars
  electric_guitars * electric_price + acoustic_guitars * acoustic_price

/-- Proves that the total amount earned from selling 9 guitars, 
    consisting of 4 electric guitars at $479 each and 5 acoustic guitars at $339 each, is $3611 -/
theorem guitar_sales_proof : 
  total_guitar_sales 9 4 479 339 = 3611 := by
  sorry

#eval total_guitar_sales 9 4 479 339

end NUMINAMATH_CALUDE_guitar_sales_proof_l2287_228772


namespace NUMINAMATH_CALUDE_two_lines_forming_30_degrees_l2287_228742

/-- Represents a line in 3D space -/
structure Line3D where
  -- Define necessary properties for a line

/-- Represents a plane in 3D space -/
structure Plane3D where
  -- Define necessary properties for a plane

/-- Angle between a line and a plane -/
def angle_line_plane (l : Line3D) (p : Plane3D) : ℝ :=
  sorry

/-- Angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Checks if a line passes through a point -/
def line_passes_through (l : Line3D) (p : Point3D) : Prop :=
  sorry

/-- The main theorem -/
theorem two_lines_forming_30_degrees (a : Line3D) (α : Plane3D) (P : Point3D) :
  angle_line_plane a α = 30 →
  ∃! (s : Finset Line3D), 
    s.card = 2 ∧ 
    ∀ b ∈ s, line_passes_through b P ∧ 
              angle_between_lines a b = 30 ∧ 
              angle_line_plane b α = 30 :=
sorry

end NUMINAMATH_CALUDE_two_lines_forming_30_degrees_l2287_228742


namespace NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2287_228733

-- Define the repeating decimal 0.454545...
def repeating_decimal : ℚ := 45 / 99

-- State the theorem
theorem repeating_decimal_equals_fraction : repeating_decimal = 5 / 11 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_equals_fraction_l2287_228733


namespace NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l2287_228748

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2*a - 1 ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

-- Theorem 1: When a = -2, A ∩ B = {x | -5 ≤ x < -1}
theorem intersection_when_a_neg_two :
  A (-2) ∩ B = {x : ℝ | -5 ≤ x ∧ x < -1} :=
sorry

-- Theorem 2: A ⊆ B if and only if a ≤ -4 or a ≥ 3
theorem subset_condition (a : ℝ) :
  A a ⊆ B ↔ a ≤ -4 ∨ a ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_intersection_when_a_neg_two_subset_condition_l2287_228748


namespace NUMINAMATH_CALUDE_sum_division_problem_l2287_228751

/-- Proof of the total amount in a sum division problem -/
theorem sum_division_problem (x y z total : ℚ) : 
  y = 0.45 * x →  -- For each rupee x gets, y gets 45 paisa
  z = 0.5 * x →   -- For each rupee x gets, z gets 50 paisa
  y = 18 →        -- The share of y is Rs. 18
  total = x + y + z →  -- The total is the sum of all shares
  total = 78 := by  -- The total amount is Rs. 78
sorry


end NUMINAMATH_CALUDE_sum_division_problem_l2287_228751


namespace NUMINAMATH_CALUDE_classroom_average_l2287_228795

theorem classroom_average (class_size : ℕ) (class_avg : ℚ) (two_thirds_avg : ℚ) :
  class_size > 0 →
  class_avg = 55 →
  two_thirds_avg = 60 →
  ∃ (one_third_avg : ℚ),
    (1 : ℚ) / 3 * one_third_avg + (2 : ℚ) / 3 * two_thirds_avg = class_avg ∧
    one_third_avg = 45 :=
by sorry

end NUMINAMATH_CALUDE_classroom_average_l2287_228795


namespace NUMINAMATH_CALUDE_investment_rate_problem_l2287_228747

theorem investment_rate_problem (total_investment : ℝ) (first_investment : ℝ) (second_rate : ℝ) (total_interest : ℝ) :
  total_investment = 10000 →
  first_investment = 6000 →
  second_rate = 0.09 →
  total_interest = 840 →
  ∃ (r : ℝ),
    r * first_investment + second_rate * (total_investment - first_investment) = total_interest ∧
    r = 0.08 :=
by sorry

end NUMINAMATH_CALUDE_investment_rate_problem_l2287_228747
