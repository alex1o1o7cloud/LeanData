import Mathlib

namespace NUMINAMATH_CALUDE_lcm_problem_l3319_331951

theorem lcm_problem (a b c : ℕ) 
  (h1 : Nat.lcm a b = 60) 
  (h2 : Nat.lcm a c = 270) : 
  Nat.lcm b c = 540 := by
sorry

end NUMINAMATH_CALUDE_lcm_problem_l3319_331951


namespace NUMINAMATH_CALUDE_taco_castle_parking_lot_l3319_331981

/-- The number of Volkswagen Bugs in the parking lot of Taco Castle -/
def volkswagen_bugs (dodge ford toyota : ℕ) : ℕ :=
  toyota / 2

theorem taco_castle_parking_lot (dodge ford toyota : ℕ) 
  (h1 : ford = dodge / 3)
  (h2 : ford = toyota * 2)
  (h3 : dodge = 60) :
  volkswagen_bugs dodge ford toyota = 5 := by
sorry

end NUMINAMATH_CALUDE_taco_castle_parking_lot_l3319_331981


namespace NUMINAMATH_CALUDE_eunji_uncle_money_l3319_331982

/-- The amount of money Eunji received from her uncle -/
def uncle_money : ℕ := sorry

/-- The amount of money Eunji received from her mother -/
def mother_money : ℕ := 550

/-- The total amount of money Eunji has after receiving money from her mother -/
def total_money : ℕ := 1000

/-- Theorem stating that Eunji received 900 won from her uncle -/
theorem eunji_uncle_money :
  uncle_money = 900 ∧
  uncle_money / 2 + mother_money = total_money :=
sorry

end NUMINAMATH_CALUDE_eunji_uncle_money_l3319_331982


namespace NUMINAMATH_CALUDE_solution_set_equality_l3319_331990

theorem solution_set_equality : 
  {x : ℝ | 1 < |x + 2| ∧ |x + 2| < 5} = 
  {x : ℝ | -7 < x ∧ x < -3} ∪ {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_solution_set_equality_l3319_331990


namespace NUMINAMATH_CALUDE_probability_same_color_l3319_331970

def num_balls : ℕ := 6
def num_colors : ℕ := 3
def balls_per_color : ℕ := 2

def same_color_combinations : ℕ := num_colors

def total_combinations : ℕ := num_balls.choose 2

theorem probability_same_color :
  (same_color_combinations : ℚ) / total_combinations = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_same_color_l3319_331970


namespace NUMINAMATH_CALUDE_gathering_handshakes_l3319_331914

/-- The number of handshakes in a gathering of elves and dwarves -/
def total_handshakes (num_elves num_dwarves : ℕ) : ℕ :=
  let elf_handshakes := num_elves * (num_elves - 1) / 2
  let elf_dwarf_handshakes := num_elves * num_dwarves
  elf_handshakes + elf_dwarf_handshakes

/-- Theorem stating the total number of handshakes in the gathering -/
theorem gathering_handshakes :
  total_handshakes 25 18 = 750 := by
  sorry

#eval total_handshakes 25 18

end NUMINAMATH_CALUDE_gathering_handshakes_l3319_331914


namespace NUMINAMATH_CALUDE_ryan_english_hours_l3319_331915

/-- The number of hours Ryan spends on learning Chinese -/
def chinese_hours : ℕ := 2

/-- The number of hours Ryan spends on learning English -/
def english_hours : ℕ := chinese_hours + 4

/-- Theorem: Ryan spends 6 hours on learning English -/
theorem ryan_english_hours : english_hours = 6 := by
  sorry

end NUMINAMATH_CALUDE_ryan_english_hours_l3319_331915


namespace NUMINAMATH_CALUDE_imaginary_power_sum_l3319_331928

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- State the theorem
theorem imaginary_power_sum : i^13 + i^18 + i^23 + i^28 + i^33 = i :=
by
  sorry

end NUMINAMATH_CALUDE_imaginary_power_sum_l3319_331928


namespace NUMINAMATH_CALUDE_min_jumps_to_visit_all_l3319_331956

/-- Represents a jump on the circle -/
inductive Jump
| Two : Jump  -- Jump by 2 points
| Three : Jump  -- Jump by 3 points

/-- The number of points on the circle -/
def numPoints : Nat := 2016

/-- A sequence of jumps -/
def JumpSequence := List Jump

/-- Function to calculate the total distance covered by a sequence of jumps -/
def totalDistance (seq : JumpSequence) : Nat :=
  seq.foldl (fun acc jump => acc + match jump with
    | Jump.Two => 2
    | Jump.Three => 3) 0

/-- Predicate to check if a sequence of jumps visits all points and returns to start -/
def isValidSequence (seq : JumpSequence) : Prop :=
  totalDistance seq % numPoints = 0 ∧ seq.length ≥ numPoints

/-- The main theorem -/
theorem min_jumps_to_visit_all :
  ∃ (seq : JumpSequence), isValidSequence seq ∧ seq.length = 2017 ∧
  (∀ (other : JumpSequence), isValidSequence other → seq.length ≤ other.length) :=
sorry

end NUMINAMATH_CALUDE_min_jumps_to_visit_all_l3319_331956


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3319_331934

theorem sqrt_equation_solution :
  ∃ x : ℝ, (Real.sqrt (2 * x + 12) = 8) ∧ (x = 26) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3319_331934


namespace NUMINAMATH_CALUDE_car_cost_proof_l3319_331994

def down_payment : ℕ := 8000
def num_payments : ℕ := 48
def monthly_payment : ℕ := 525
def interest_rate : ℚ := 5 / 100

def total_car_cost : ℕ := 34460

theorem car_cost_proof :
  down_payment +
  num_payments * monthly_payment +
  num_payments * (interest_rate * monthly_payment).floor = total_car_cost := by
  sorry

end NUMINAMATH_CALUDE_car_cost_proof_l3319_331994


namespace NUMINAMATH_CALUDE_persons_age_l3319_331957

theorem persons_age : ∃ (age : ℕ), 
  (6 * (age + 6) - 6 * (age - 6) = age) ∧ (age = 72) := by
  sorry

end NUMINAMATH_CALUDE_persons_age_l3319_331957


namespace NUMINAMATH_CALUDE_root_product_l3319_331966

theorem root_product (d e : ℤ) : 
  (∀ x : ℝ, x^2 + x - 2 = 0 → x^7 - d*x^3 - e = 0) → 
  d * e = 70 := by
  sorry

end NUMINAMATH_CALUDE_root_product_l3319_331966


namespace NUMINAMATH_CALUDE_subset_condition_intersection_condition_l3319_331952

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x < 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | 2*m - 1 ≤ x ∧ x ≤ m + 1}

-- Theorem 1: B ⊆ A iff m ∈ [-1, +∞)
theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m ≥ -1 := by sorry

-- Theorem 2: ∃x ∈ A such that x ∈ B iff m ∈ [-4, 2]
theorem intersection_condition (m : ℝ) : (∃ x, x ∈ A ∧ x ∈ B m) ↔ -4 ≤ m ∧ m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_condition_l3319_331952


namespace NUMINAMATH_CALUDE_sin_2alpha_value_l3319_331918

theorem sin_2alpha_value (α : Real) 
  (h : Real.tan (α - π/4) = Real.sqrt 2 / 4) : 
  Real.sin (2 * α) = 7/9 := by
  sorry

end NUMINAMATH_CALUDE_sin_2alpha_value_l3319_331918


namespace NUMINAMATH_CALUDE_prism_volume_l3319_331933

-- Define a right rectangular prism
structure RectangularPrism where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the volume of a rectangular prism
def volume (p : RectangularPrism) : ℝ := p.a * p.b * p.c

-- Define the face areas of a rectangular prism
def sideFaceArea (p : RectangularPrism) : ℝ := p.a * p.b
def frontFaceArea (p : RectangularPrism) : ℝ := p.b * p.c
def bottomFaceArea (p : RectangularPrism) : ℝ := p.a * p.c

-- Theorem: The volume of the prism is 12 cubic inches
theorem prism_volume (p : RectangularPrism) 
  (h1 : sideFaceArea p = 18) 
  (h2 : frontFaceArea p = 12) 
  (h3 : bottomFaceArea p = 8) : 
  volume p = 12 := by
  sorry


end NUMINAMATH_CALUDE_prism_volume_l3319_331933


namespace NUMINAMATH_CALUDE_decagon_diagonals_l3319_331977

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A regular decagon has 35 diagonals -/
theorem decagon_diagonals : num_diagonals 10 = 35 := by
  sorry

end NUMINAMATH_CALUDE_decagon_diagonals_l3319_331977


namespace NUMINAMATH_CALUDE_congruence_problem_l3319_331998

theorem congruence_problem (y : ℤ) 
  (h1 : (2 + y) % (2^4) = 2^3 % (2^4))
  (h2 : (4 + y) % (4^3) = 4^2 % (4^3))
  (h3 : (6 + y) % (6^3) = 6^2 % (6^3)) :
  y % 48 = 44 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l3319_331998


namespace NUMINAMATH_CALUDE_initial_balloons_l3319_331905

theorem initial_balloons (initial : ℕ) : initial + 2 = 11 → initial = 9 := by
  sorry

end NUMINAMATH_CALUDE_initial_balloons_l3319_331905


namespace NUMINAMATH_CALUDE_square_side_length_l3319_331975

/-- Given a square with diagonal length 2√2, prove that its side length is 2. -/
theorem square_side_length (d : ℝ) (h : d = 2 * Real.sqrt 2) :
  ∃ (s : ℝ), s * s = (d * d) / 2 ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_l3319_331975


namespace NUMINAMATH_CALUDE_initial_water_percentage_l3319_331940

theorem initial_water_percentage
  (initial_volume : ℝ)
  (added_water : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_volume = 125)
  (h2 : added_water = 8.333333333333334)
  (h3 : final_water_percentage = 25)
  (h4 : (initial_volume * x + added_water) / (initial_volume + added_water) * 100 = final_water_percentage) :
  x * 100 = 20 :=
by
  sorry

#check initial_water_percentage

end NUMINAMATH_CALUDE_initial_water_percentage_l3319_331940


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3319_331991

open Set

def M : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 3}

theorem union_of_M_and_N :
  M ∪ N = {x : ℝ | -1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3319_331991


namespace NUMINAMATH_CALUDE_total_revenue_equals_4452_4_l3319_331949

def calculate_revenue (price : ℝ) (quantity : ℕ) (discount : ℝ) (tax : ℝ) (surcharge : ℝ) : ℝ :=
  let discounted_price := price * (1 - discount)
  let taxed_price := discounted_price * (1 + tax)
  let final_price := taxed_price * (1 + surcharge)
  final_price * quantity

def total_revenue : ℝ :=
  calculate_revenue 25 60 0.1 0.05 0 +
  calculate_revenue 25 10 0 0 0.03 +
  calculate_revenue 25 20 0.05 0.02 0 +
  calculate_revenue 25 44 0.15 0 0.04 +
  calculate_revenue 25 66 0.2 0 0

theorem total_revenue_equals_4452_4 :
  total_revenue = 4452.4 := by
  sorry

end NUMINAMATH_CALUDE_total_revenue_equals_4452_4_l3319_331949


namespace NUMINAMATH_CALUDE_remainder_theorem_l3319_331983

theorem remainder_theorem (n : ℕ) : (3^(2*n) + 8) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3319_331983


namespace NUMINAMATH_CALUDE_space_diagonals_of_Q_l3319_331963

/-- A convex polyhedron -/
structure ConvexPolyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  triangular_faces : ℕ
  quadrilateral_faces : ℕ

/-- The number of space diagonals in a convex polyhedron -/
def space_diagonals (Q : ConvexPolyhedron) : ℕ :=
  (Q.vertices.choose 2) - Q.edges - (2 * Q.quadrilateral_faces)

/-- Theorem: The number of space diagonals in the given polyhedron Q is 315 -/
theorem space_diagonals_of_Q :
  ∃ Q : ConvexPolyhedron,
    Q.vertices = 30 ∧
    Q.edges = 72 ∧
    Q.faces = 44 ∧
    Q.triangular_faces = 20 ∧
    Q.quadrilateral_faces = 24 ∧
    space_diagonals Q = 315 :=
sorry

end NUMINAMATH_CALUDE_space_diagonals_of_Q_l3319_331963


namespace NUMINAMATH_CALUDE_gcd_7200_13230_l3319_331904

theorem gcd_7200_13230 : Int.gcd 7200 13230 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcd_7200_13230_l3319_331904


namespace NUMINAMATH_CALUDE_sandwich_fraction_proof_l3319_331927

theorem sandwich_fraction_proof (total : ℚ) (ticket : ℚ) (book : ℚ) (leftover : ℚ) 
  (h_total : total = 150)
  (h_ticket : ticket = 1 / 6)
  (h_book : book = 1 / 2)
  (h_leftover : leftover = 20)
  (h_spent : total - leftover = ticket * total + book * total + (total - leftover - ticket * total - book * total)) :
  (total - leftover - ticket * total - book * total) / total = 1 / 5 := by
sorry

end NUMINAMATH_CALUDE_sandwich_fraction_proof_l3319_331927


namespace NUMINAMATH_CALUDE_cone_volume_l3319_331911

theorem cone_volume (slant_height height : ℝ) (h1 : slant_height = 15) (h2 : height = 8) :
  (1 / 3 : ℝ) * π * (slant_height ^ 2 - height ^ 2) * height = 429 * (1 / 3 : ℝ) * π :=
by sorry

end NUMINAMATH_CALUDE_cone_volume_l3319_331911


namespace NUMINAMATH_CALUDE_circle_plus_equality_l3319_331907

/-- Definition of the ⊕ operation -/
def circle_plus (a b : ℝ) : ℝ := a * b + a - b

/-- Theorem stating the equality to be proved -/
theorem circle_plus_equality (a b : ℝ) : 
  circle_plus a b + circle_plus (b - a) b = b^2 - b := by
  sorry

end NUMINAMATH_CALUDE_circle_plus_equality_l3319_331907


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3319_331954

theorem rectangle_diagonal (side : ℝ) (area : ℝ) (diagonal : ℝ) : 
  side = 15 → area = 120 → diagonal = 17 → 
  ∃ other_side : ℝ, 
    area = side * other_side ∧ 
    diagonal^2 = side^2 + other_side^2 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3319_331954


namespace NUMINAMATH_CALUDE_difference_of_variables_l3319_331967

theorem difference_of_variables (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (diff_squares_eq : x^2 - y^2 = 40) : 
  x - y = 4 := by sorry

end NUMINAMATH_CALUDE_difference_of_variables_l3319_331967


namespace NUMINAMATH_CALUDE_extreme_value_condition_l3319_331946

theorem extreme_value_condition (a : ℝ) :
  let f : ℝ → ℝ := λ x ↦ (a * x^2 - 1) * Real.exp x
  (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f x ≤ f 1 ∨ f x ≥ f 1) →
  a = 1/3 := by
sorry

end NUMINAMATH_CALUDE_extreme_value_condition_l3319_331946


namespace NUMINAMATH_CALUDE_sum_of_selected_elements_ge_one_l3319_331922

/-- Definition of the table element at position (i, j) -/
def table_element (i j : ℕ) : ℚ := 1 / (i + j - 1)

/-- A selection of n elements from an n × n table, where no two elements are in the same row or column -/
def valid_selection (n : ℕ) : Type := 
  { s : Finset (ℕ × ℕ) // s.card = n ∧ 
    (∀ (a b : ℕ × ℕ), a ∈ s → b ∈ s → a ≠ b → a.1 ≠ b.1 ∧ a.2 ≠ b.2) ∧
    (∀ (a : ℕ × ℕ), a ∈ s → a.1 ≤ n ∧ a.2 ≤ n) }

/-- The main theorem: The sum of selected elements is not less than 1 -/
theorem sum_of_selected_elements_ge_one (n : ℕ) (h : n > 0) :
  ∀ (s : valid_selection n), (s.val.sum (λ (x : ℕ × ℕ) => table_element x.1 x.2)) ≥ 1 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_selected_elements_ge_one_l3319_331922


namespace NUMINAMATH_CALUDE_backpack_price_calculation_l3319_331902

theorem backpack_price_calculation
  (num_backpacks : ℕ)
  (monogram_cost : ℚ)
  (total_cost : ℚ)
  (h1 : num_backpacks = 5)
  (h2 : monogram_cost = 12)
  (h3 : total_cost = 140) :
  (total_cost - num_backpacks * monogram_cost) / num_backpacks = 16 :=
by sorry

end NUMINAMATH_CALUDE_backpack_price_calculation_l3319_331902


namespace NUMINAMATH_CALUDE_sqrt_two_times_sqrt_six_equals_two_sqrt_three_l3319_331985

theorem sqrt_two_times_sqrt_six_equals_two_sqrt_three :
  Real.sqrt 2 * Real.sqrt 6 = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_times_sqrt_six_equals_two_sqrt_three_l3319_331985


namespace NUMINAMATH_CALUDE_f_value_at_2_l3319_331959

/-- Given a function f(x) = a*sin(x) + b*x*cos(x) - 2c*tan(x) + x^2 where f(-2) = 3,
    prove that f(2) = 5 -/
theorem f_value_at_2 (a b c : ℝ) (f : ℝ → ℝ)
  (h1 : ∀ x, f x = a * Real.sin x + b * x * Real.cos x - 2 * c * Real.tan x + x^2)
  (h2 : f (-2) = 3) :
  f 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_2_l3319_331959


namespace NUMINAMATH_CALUDE_line_segment_solution_l3319_331935

def line_segment_start : ℝ × ℝ := (2, 5)
def line_segment_end (x : ℝ) : ℝ × ℝ := (x, 10)

theorem line_segment_solution (x : ℝ) 
  (h1 : Real.sqrt ((x - 2)^2 + (10 - 5)^2) = 13)
  (h2 : x > 0) : 
  x = 14 := by sorry

end NUMINAMATH_CALUDE_line_segment_solution_l3319_331935


namespace NUMINAMATH_CALUDE_interval_length_implies_difference_l3319_331995

/-- Given an inequality a ≤ 3x + 5 ≤ b, where the length of the interval of solutions is 15, prove that b - a = 45 -/
theorem interval_length_implies_difference (a b : ℝ) : 
  (∃ x : ℝ, a ≤ 3*x + 5 ∧ 3*x + 5 ≤ b) → 
  ((b - 5) / 3 - (a - 5) / 3 = 15) → 
  b - a = 45 := by sorry

end NUMINAMATH_CALUDE_interval_length_implies_difference_l3319_331995


namespace NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3319_331953

/-- Given a cone where the total surface area is three times its base area,
    the central angle of the sector in the lateral surface development diagram is 180 degrees. -/
theorem cone_lateral_surface_angle (r l : ℝ) (h : r > 0) (h' : l > 0) : 
  (π * r^2 + π * r * l = 3 * π * r^2) → 
  (2 * π * r / l) * (180 / π) = 180 := by
  sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_angle_l3319_331953


namespace NUMINAMATH_CALUDE_sue_buttons_l3319_331930

theorem sue_buttons (mari kendra sue : ℕ) : 
  mari = 8 →
  kendra = 5 * mari + 4 →
  sue = kendra / 2 →
  sue = 22 := by
sorry

end NUMINAMATH_CALUDE_sue_buttons_l3319_331930


namespace NUMINAMATH_CALUDE_hybrid_car_percentage_l3319_331931

/-- Proves that the percentage of hybrid cars in a dealership is 60% -/
theorem hybrid_car_percentage
  (total_cars : ℕ)
  (hybrids_with_full_headlights : ℕ)
  (hybrid_one_headlight_percent : ℚ)
  (h1 : total_cars = 600)
  (h2 : hybrids_with_full_headlights = 216)
  (h3 : hybrid_one_headlight_percent = 40 / 100) :
  (hybrids_with_full_headlights / (1 - hybrid_one_headlight_percent) : ℚ) / total_cars = 60 / 100 :=
by sorry

end NUMINAMATH_CALUDE_hybrid_car_percentage_l3319_331931


namespace NUMINAMATH_CALUDE_positive_intervals_l3319_331948

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 2)

theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_positive_intervals_l3319_331948


namespace NUMINAMATH_CALUDE_tooth_permutations_l3319_331986

def word_length : ℕ := 5
def t_occurrences : ℕ := 3
def o_occurrences : ℕ := 2

theorem tooth_permutations : 
  (word_length.factorial) / (t_occurrences.factorial * o_occurrences.factorial) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tooth_permutations_l3319_331986


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l3319_331996

theorem binomial_coefficient_equality (n : ℕ+) :
  (Nat.choose 9 (n + 1) = Nat.choose 9 (2 * n - 1)) → (n = 2 ∨ n = 3) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l3319_331996


namespace NUMINAMATH_CALUDE_polynomial_range_l3319_331971

theorem polynomial_range (x : ℝ) :
  x^2 - 7*x + 12 < 0 →
  90 < x^3 + 5*x^2 + 6*x ∧ x^3 + 5*x^2 + 6*x < 168 := by
sorry

end NUMINAMATH_CALUDE_polynomial_range_l3319_331971


namespace NUMINAMATH_CALUDE_hour_hand_angle_after_one_hour_l3319_331979

/-- Represents the angle turned by the hour hand of a watch. -/
def angle_turned (hours : ℝ) : ℝ :=
  -30 * hours

/-- The theorem states that the angle turned by the hour hand after 1 hour is -30°. -/
theorem hour_hand_angle_after_one_hour :
  angle_turned 1 = -30 := by sorry

end NUMINAMATH_CALUDE_hour_hand_angle_after_one_hour_l3319_331979


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3319_331989

/-- A line y = kx + 3 intersects a circle (x - 3)^2 + (y - 2)^2 = 4 at two points M and N. 
    If the distance between M and N is at least 2, then k is outside the interval (3 - 2√2, 3 + 2√2). -/
theorem line_circle_intersection (k : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    (M.1 - 3)^2 + (M.2 - 2)^2 = 4 ∧
    (N.1 - 3)^2 + (N.2 - 2)^2 = 4 ∧
    M.2 = k * M.1 + 3 ∧
    N.2 = k * N.1 + 3 ∧
    (M.1 - N.1)^2 + (M.2 - N.2)^2 ≥ 4) →
  k < 3 - 2 * Real.sqrt 2 ∨ k > 3 + 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3319_331989


namespace NUMINAMATH_CALUDE_cube_split_theorem_l3319_331939

/-- The sum of consecutive integers from 2 to n -/
def consecutiveSum (n : ℕ) : ℕ := (n + 2) * (n - 1) / 2

/-- The nth odd number starting from 3 -/
def nthOddFrom3 (n : ℕ) : ℕ := 2 * n + 1

theorem cube_split_theorem (m : ℕ) (h1 : m > 1) :
  (∃ k, k ∈ Finset.range m ∧ nthOddFrom3 (consecutiveSum m - k) = 333) ↔ m = 18 := by
  sorry

end NUMINAMATH_CALUDE_cube_split_theorem_l3319_331939


namespace NUMINAMATH_CALUDE_function_symmetry_l3319_331901

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
axiom not_identically_zero : ∃ x, f x ≠ 0
axiom functional_equation : ∀ a b : ℝ, f (a + b) + f (a - b) = 2 * f a + 2 * f b

-- State the theorem
theorem function_symmetry : ∀ x : ℝ, f (-x) = f x := by sorry

end NUMINAMATH_CALUDE_function_symmetry_l3319_331901


namespace NUMINAMATH_CALUDE_equation_root_implies_z_value_l3319_331919

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (x : ℂ) (a : ℝ) : Prop :=
  x^2 + (4 + i) * x + (4 : ℂ) + a * i = 0

-- Define the complex number z
def z (a b : ℝ) : ℂ := a + b * i

-- Theorem statement
theorem equation_root_implies_z_value (a b : ℝ) :
  equation b a → z a b = 2 - 2 * i :=
by
  sorry

end NUMINAMATH_CALUDE_equation_root_implies_z_value_l3319_331919


namespace NUMINAMATH_CALUDE_arrangement_count_l3319_331926

/-- The number of distinct arrangements of 8 indistinguishable items and 2 other indistinguishable items in a row of 10 slots -/
def distinct_arrangements : ℕ := 45

/-- The total number of slots available -/
def total_slots : ℕ := 10

/-- The number of the first type of indistinguishable items -/
def first_item_count : ℕ := 8

/-- The number of the second type of indistinguishable items -/
def second_item_count : ℕ := 2

theorem arrangement_count :
  distinct_arrangements = (total_slots.choose second_item_count) :=
by sorry

end NUMINAMATH_CALUDE_arrangement_count_l3319_331926


namespace NUMINAMATH_CALUDE_speed_limit_correct_l3319_331917

/-- The fine rate for speeding in dollars per mph over the limit -/
def fine_rate : ℕ := 16

/-- The total fine Jed received in dollars -/
def total_fine : ℕ := 256

/-- Jed's speed in mph -/
def jed_speed : ℕ := 66

/-- The posted speed limit on the road -/
def speed_limit : ℕ := 50

/-- Theorem stating that the given speed limit satisfies the conditions of the problem -/
theorem speed_limit_correct : 
  fine_rate * (jed_speed - speed_limit) = total_fine :=
by
  sorry


end NUMINAMATH_CALUDE_speed_limit_correct_l3319_331917


namespace NUMINAMATH_CALUDE_cos_105_degrees_l3319_331916

theorem cos_105_degrees : Real.cos (105 * Real.pi / 180) = (Real.sqrt 2 - Real.sqrt 6) / 4 := by
  sorry

end NUMINAMATH_CALUDE_cos_105_degrees_l3319_331916


namespace NUMINAMATH_CALUDE_inequality_solution_l3319_331921

theorem inequality_solution (x : ℝ) : 
  x > 0 → |5 - 2*x| ≤ 8 → 0 ≤ x ∧ x ≤ 6.5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3319_331921


namespace NUMINAMATH_CALUDE_soccer_game_scoring_l3319_331960

theorem soccer_game_scoring (team_a_first_half : ℕ) : 
  (team_a_first_half : ℝ) + (team_a_first_half : ℝ) / 2 + 
  (team_a_first_half : ℝ) + (team_a_first_half : ℝ) - 2 = 26 →
  team_a_first_half = 8 := by
sorry

end NUMINAMATH_CALUDE_soccer_game_scoring_l3319_331960


namespace NUMINAMATH_CALUDE_no_solution_fractional_equation_l3319_331987

theorem no_solution_fractional_equation :
  ∀ x : ℝ, x ≠ 2 → ¬ (1 / (x - 2) = (1 - x) / (2 - x) - 3) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_fractional_equation_l3319_331987


namespace NUMINAMATH_CALUDE_scale_model_height_l3319_331993

/-- The scale ratio used for the model -/
def scale_ratio : ℚ := 1 / 25

/-- The actual height of the Eiffel Tower in feet -/
def actual_height : ℕ := 984

/-- The height of the scale model before rounding -/
def model_height : ℚ := actual_height / scale_ratio

/-- Function to round a rational number to the nearest integer -/
def round_to_nearest (x : ℚ) : ℤ :=
  ⌊x + 1/2⌋

theorem scale_model_height :
  round_to_nearest model_height = 39 := by
  sorry

end NUMINAMATH_CALUDE_scale_model_height_l3319_331993


namespace NUMINAMATH_CALUDE_probability_of_triangle_in_decagon_l3319_331942

/-- A regular decagon is a 10-sided polygon with all sides and angles equal -/
structure RegularDecagon where
  -- No specific properties needed for this problem

/-- The number of diagonals in a regular decagon -/
def num_diagonals : ℕ := 35

/-- The number of ways to choose 3 diagonals from the total number of diagonals -/
def total_diagonal_choices : ℕ := Nat.choose num_diagonals 3

/-- The number of ways to choose 4 points from 10 points -/
def four_point_choices : ℕ := Nat.choose 10 4

/-- The number of ways to choose 3 points out of 4 points -/
def three_out_of_four : ℕ := Nat.choose 4 3

/-- The number of triangle-forming sets of diagonals -/
def triangle_forming_sets : ℕ := four_point_choices * three_out_of_four

theorem probability_of_triangle_in_decagon (d : RegularDecagon) :
  (triangle_forming_sets : ℚ) / total_diagonal_choices = 840 / 6545 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_triangle_in_decagon_l3319_331942


namespace NUMINAMATH_CALUDE_range_of_a_l3319_331969

theorem range_of_a (p q : Prop) 
  (hp : p ↔ ∀ x : ℝ, x > 0 → x + 1/x > a)
  (hq : q ↔ ∃ x₀ : ℝ, x₀^2 - 2*a*x₀ + 1 ≤ 0)
  (hnq : ¬¬q)
  (hpq : ¬(p ∧ q)) :
  a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3319_331969


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3319_331913

def numbers : Finset ℕ := {12, 14, 16, 18}

def expression (A B C D : ℕ) : ℕ := A * B + B * C + B * D + C * D

theorem max_value_of_expression :
  ∃ (A B C D : ℕ), A ∈ numbers ∧ B ∈ numbers ∧ C ∈ numbers ∧ D ∈ numbers ∧
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  expression A B C D = 1116 ∧
  ∀ (A' B' C' D' : ℕ), A' ∈ numbers → B' ∈ numbers → C' ∈ numbers → D' ∈ numbers →
  A' ≠ B' → A' ≠ C' → A' ≠ D' → B' ≠ C' → B' ≠ D' → C' ≠ D' →
  expression A' B' C' D' ≤ 1116 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3319_331913


namespace NUMINAMATH_CALUDE_integer_pair_conditions_l3319_331936

theorem integer_pair_conditions (a b : ℤ) : 
  (a - b - 1) ∣ (a^2 + b^2) ∧ 
  (a^2 + b^2) / (2*a*b - 1) = 20/19 ↔ 
  ((a = 22 ∧ b = 16) ∨ 
   (a = -16 ∧ b = -22) ∨ 
   (a = 8 ∧ b = 6) ∨ 
   (a = -6 ∧ b = -8)) := by
sorry

end NUMINAMATH_CALUDE_integer_pair_conditions_l3319_331936


namespace NUMINAMATH_CALUDE_intersection_equality_implies_a_values_l3319_331955

def A : Set ℝ := {x | x^2 - x - 2 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem intersection_equality_implies_a_values (a : ℝ) :
  A ∩ B a = B a → a = 0 ∨ a = -1 ∨ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_implies_a_values_l3319_331955


namespace NUMINAMATH_CALUDE_min_value_expression_l3319_331912

theorem min_value_expression (a b m n : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_sum : a + b = 1) (h_prod : m * n = 2) :
  2 ≤ (a * m + b * n) * (b * m + a * n) :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l3319_331912


namespace NUMINAMATH_CALUDE_weight_qualification_l3319_331992

/-- A weight is qualified if it falls within the acceptable range -/
def is_qualified (weight : ℝ) : Prop :=
  24.75 ≤ weight ∧ weight ≤ 25.25

/-- The labeled weight of the flour -/
def labeled_weight : ℝ := 25

/-- The tolerance of the weight -/
def tolerance : ℝ := 0.25

theorem weight_qualification (weight : ℝ) :
  is_qualified weight ↔ labeled_weight - tolerance ≤ weight ∧ weight ≤ labeled_weight + tolerance :=
by sorry

end NUMINAMATH_CALUDE_weight_qualification_l3319_331992


namespace NUMINAMATH_CALUDE_min_pizzas_for_johns_van_l3319_331976

/-- The minimum whole number of pizzas needed to recover the van's cost -/
def min_pizzas (van_cost : ℕ) (earnings_per_pizza : ℕ) (gas_cost : ℕ) : ℕ :=
  (van_cost + (earnings_per_pizza - gas_cost - 1)) / (earnings_per_pizza - gas_cost)

theorem min_pizzas_for_johns_van :
  min_pizzas 8000 15 4 = 728 := by sorry

end NUMINAMATH_CALUDE_min_pizzas_for_johns_van_l3319_331976


namespace NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3319_331906

/-- Given two hyperbolas with equations (x²/9) - (y²/16) = 1 and (y²/25) - (x²/M) = 1
    that have the same asymptotes, M equals 225/16. -/
theorem hyperbolas_same_asymptotes (M : ℝ) : 
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) → M = 225 / 16 :=
by sorry

end NUMINAMATH_CALUDE_hyperbolas_same_asymptotes_l3319_331906


namespace NUMINAMATH_CALUDE_point_p_coordinates_l3319_331974

/-- A point on the x-axis with distance 3 from the origin -/
structure PointP where
  x : ℝ
  y : ℝ
  on_x_axis : y = 0
  distance_3 : x^2 + y^2 = 3^2

/-- The coordinates of point P are either (-3,0) or (3,0) -/
theorem point_p_coordinates (p : PointP) : (p.x = -3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 0) := by
  sorry

end NUMINAMATH_CALUDE_point_p_coordinates_l3319_331974


namespace NUMINAMATH_CALUDE_set_union_condition_implies_m_geq_two_l3319_331910

theorem set_union_condition_implies_m_geq_two (m : ℝ) :
  let A : Set ℝ := {x | x ≥ 2}
  let B : Set ℝ := {x | x ≥ m}
  A ∪ B = A → m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_set_union_condition_implies_m_geq_two_l3319_331910


namespace NUMINAMATH_CALUDE_smallest_k_coprime_subset_l3319_331938

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem smallest_k_coprime_subset : ∃ (k : ℕ),
  (k = 51) ∧ 
  (∀ (S : Finset ℕ), S ⊆ Finset.range 100 → S.card ≥ k → 
    ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ is_coprime a b) ∧
  (∀ (k' : ℕ), k' < k → 
    ∃ (S : Finset ℕ), S ⊆ Finset.range 100 ∧ S.card = k' ∧
      ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → ¬is_coprime a b) :=
by sorry

end NUMINAMATH_CALUDE_smallest_k_coprime_subset_l3319_331938


namespace NUMINAMATH_CALUDE_sandbox_width_l3319_331999

/-- The width of a rectangle given its length and area -/
theorem sandbox_width (length : ℝ) (area : ℝ) (h1 : length = 312) (h2 : area = 45552) :
  area / length = 146 := by
  sorry

end NUMINAMATH_CALUDE_sandbox_width_l3319_331999


namespace NUMINAMATH_CALUDE_range_of_a_l3319_331958

def p (x : ℝ) : Prop := |x + 1| > 3

def q (x a : ℝ) : Prop := x > a

theorem range_of_a (h1 : ∀ x, q x a → p x) 
                   (h2 : ∃ x, p x ∧ ¬q x a) : 
  a ≥ 2 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l3319_331958


namespace NUMINAMATH_CALUDE_new_average_weight_l3319_331950

/-- Given a group of students and a new student joining, calculate the new average weight -/
theorem new_average_weight 
  (initial_count : ℕ) 
  (initial_average : ℝ) 
  (new_student_weight : ℝ) : 
  initial_count = 29 → 
  initial_average = 28 → 
  new_student_weight = 7 → 
  let total_weight := initial_count * initial_average + new_student_weight
  let new_count := initial_count + 1
  (total_weight / new_count : ℝ) = 27.3 := by
  sorry

end NUMINAMATH_CALUDE_new_average_weight_l3319_331950


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3319_331997

theorem sum_of_squares_of_roots (x₁ x₂ : ℝ) : 
  (2 * x₁^2 + 3 * x₁ - 5 = 0) →
  (2 * x₂^2 + 3 * x₂ - 5 = 0) →
  (x₁ ≠ x₂) →
  (x₁^2 + x₂^2 = 29/4) := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3319_331997


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l3319_331924

/-- Given positive integers a, b, c with (a,b,c) = 1 and (a,b) = d, 
    if n > (ab/d) + cd - a - b - c, then there exist nonnegative integers x, y, z 
    such that ax + by + cz = n -/
theorem diophantine_equation_solution 
  (a b c d : ℕ+) (n : ℕ) 
  (h1 : Nat.gcd a.val (Nat.gcd b.val c.val) = 1)
  (h2 : Nat.gcd a.val b.val = d.val)
  (h3 : n > a.val * b.val / d.val + c.val * d.val - a.val - b.val - c.val) :
  ∃ x y z : ℕ, a.val * x + b.val * y + c.val * z = n :=
sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l3319_331924


namespace NUMINAMATH_CALUDE_total_paintable_area_is_1129_l3319_331943

def bedroom_area (length width height : ℕ) : ℕ :=
  2 * (length * height + width * height)

def paintable_area (total_area unpaintable_area : ℕ) : ℕ :=
  total_area - unpaintable_area

theorem total_paintable_area_is_1129 :
  let bedroom1_total := bedroom_area 14 12 9
  let bedroom2_total := bedroom_area 12 11 9
  let bedroom3_total := bedroom_area 13 12 9
  let bedroom1_paintable := paintable_area bedroom1_total 70
  let bedroom2_paintable := paintable_area bedroom2_total 65
  let bedroom3_paintable := paintable_area bedroom3_total 68
  bedroom1_paintable + bedroom2_paintable + bedroom3_paintable = 1129 := by
  sorry

end NUMINAMATH_CALUDE_total_paintable_area_is_1129_l3319_331943


namespace NUMINAMATH_CALUDE_min_split_links_for_all_weights_l3319_331980

/-- Represents a chain of links -/
structure Chain where
  totalLinks : Nat
  linkWeight : Nat

/-- Represents the result of splitting a chain -/
structure SplitChain where
  originalChain : Chain
  splitLinks : Nat

/-- Checks if all weights from 1 to the total weight can be assembled -/
def canAssembleAllWeights (sc : SplitChain) : Prop :=
  ∀ w : Nat, 1 ≤ w ∧ w ≤ sc.originalChain.totalLinks → 
    ∃ (subset : Finset Nat), subset.card ≤ sc.splitLinks + 1 ∧ 
      (subset.sum (λ i => sc.originalChain.linkWeight)) = w

/-- The main theorem -/
theorem min_split_links_for_all_weights 
  (c : Chain) 
  (h1 : c.totalLinks = 60) 
  (h2 : c.linkWeight = 1) :
  (∃ (k : Nat), k = 3 ∧ 
    canAssembleAllWeights ⟨c, k⟩ ∧
    (∀ (m : Nat), m < k → ¬canAssembleAllWeights ⟨c, m⟩)) :=
  sorry

end NUMINAMATH_CALUDE_min_split_links_for_all_weights_l3319_331980


namespace NUMINAMATH_CALUDE_trigonometric_identities_l3319_331944

theorem trigonometric_identities :
  (2 * Real.cos (30 * π / 180) - Real.tan (60 * π / 180) + Real.sin (45 * π / 180) * Real.cos (45 * π / 180) = 1/2) ∧
  ((-1)^2023 + 2 * Real.sin (45 * π / 180) - Real.cos (30 * π / 180) + Real.sin (60 * π / 180) + Real.tan (60 * π / 180)^2 = 2 + Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_identities_l3319_331944


namespace NUMINAMATH_CALUDE_base_of_second_exponent_l3319_331937

theorem base_of_second_exponent (a b : ℕ+) (some_number : ℕ) 
  (h1 : (18 ^ a.val) * 9 ^ (3 * a.val - 1) = (2 ^ 6) * (some_number ^ b.val))
  (h2 : a = 6) : 
  some_number = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_of_second_exponent_l3319_331937


namespace NUMINAMATH_CALUDE_secret_eggs_count_l3319_331909

/-- Given a jar with candy and secret eggs, calculate the number of secret eggs. -/
theorem secret_eggs_count (candy : ℝ) (total : ℕ) (h1 : candy = 3409.0) (h2 : total = 3554) :
  ↑total - candy = 145 :=
by sorry

end NUMINAMATH_CALUDE_secret_eggs_count_l3319_331909


namespace NUMINAMATH_CALUDE_glasses_cost_l3319_331973

theorem glasses_cost (frame_cost : ℝ) (coupon : ℝ) (insurance_coverage : ℝ) (total_cost : ℝ) :
  frame_cost = 200 →
  coupon = 50 →
  insurance_coverage = 0.8 →
  total_cost = 250 →
  ∃ (lens_cost : ℝ), lens_cost = 500 ∧
    total_cost = (frame_cost - coupon) + (1 - insurance_coverage) * lens_cost :=
by sorry

end NUMINAMATH_CALUDE_glasses_cost_l3319_331973


namespace NUMINAMATH_CALUDE_bookstore_profit_percentage_l3319_331978

/-- Given three textbooks with their cost and selling prices, prove that the total profit percentage
    based on the combined selling prices is approximately 20.94%. -/
theorem bookstore_profit_percentage
  (cost1 : ℝ) (sell1 : ℝ) (cost2 : ℝ) (sell2 : ℝ) (cost3 : ℝ) (sell3 : ℝ)
  (h1 : cost1 = 44)
  (h2 : sell1 = 55)
  (h3 : cost2 = 58)
  (h4 : sell2 = 72)
  (h5 : cost3 = 83)
  (h6 : sell3 = 107) :
  let total_profit := (sell1 - cost1) + (sell2 - cost2) + (sell3 - cost3)
  let total_selling_price := sell1 + sell2 + sell3
  let profit_percentage := (total_profit / total_selling_price) * 100
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |profit_percentage - 20.94| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_bookstore_profit_percentage_l3319_331978


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3319_331965

-- Equation 1
theorem equation_one_solution (x : ℚ) :
  x / (x - 1) = 3 / (2 * x - 2) - 2 ↔ x = 7 / 6 :=
sorry

-- Equation 2
theorem equation_two_no_solution :
  ¬∃ (x : ℚ), (5 * x + 2) / (x^2 + x) = 3 / (x + 1) :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_no_solution_l3319_331965


namespace NUMINAMATH_CALUDE_complex_modulus_l3319_331903

theorem complex_modulus (x y : ℝ) (z : ℂ) (h : z = x + y * I) 
  (eq : (1/2 * x - y) + (x + y) * I = 3 * I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_l3319_331903


namespace NUMINAMATH_CALUDE_butter_for_original_recipe_l3319_331945

/-- Given a recipe where 12 ounces of butter is used for 28 cups of flour
    in a 4x version of the original recipe, prove that the amount of butter
    needed for the original recipe is 3 ounces. -/
theorem butter_for_original_recipe
  (butter_4x : ℝ) -- Amount of butter for 4x recipe
  (flour_4x : ℝ) -- Amount of flour for 4x recipe
  (scale_factor : ℕ) -- Factor by which the original recipe is scaled
  (h1 : butter_4x = 12) -- 12 ounces of butter used in 4x recipe
  (h2 : flour_4x = 28) -- 28 cups of flour used in 4x recipe
  (h3 : scale_factor = 4) -- The recipe is scaled by a factor of 4
  : butter_4x / scale_factor = 3 := by
  sorry

end NUMINAMATH_CALUDE_butter_for_original_recipe_l3319_331945


namespace NUMINAMATH_CALUDE_probability_most_expensive_chosen_l3319_331925

def num_computers : ℕ := 10
def num_display : ℕ := 3

theorem probability_most_expensive_chosen :
  (Nat.choose (num_computers - 2) (num_display - 2)) / (Nat.choose num_computers num_display) = 1 / 15 := by
sorry

end NUMINAMATH_CALUDE_probability_most_expensive_chosen_l3319_331925


namespace NUMINAMATH_CALUDE_range_of_b_length_of_AB_l3319_331920

-- Define the line and ellipse
def line (x b : ℝ) : ℝ := x + b
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the intersection condition
def intersects (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
  ellipse x₁ (line x₁ b) ∧ 
  ellipse x₂ (line x₂ b)

-- Theorem for the range of b
theorem range_of_b :
  ∀ b : ℝ, intersects b ↔ -Real.sqrt 3 < b ∧ b < Real.sqrt 3 :=
sorry

-- Theorem for the length of AB when b = 1
theorem length_of_AB :
  intersects 1 →
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse x₁ y₁ ∧ 
    ellipse x₂ y₂ ∧
    y₁ = line x₁ 1 ∧
    y₂ = line x₂ 1 ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = 4 * Real.sqrt 2 / 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_b_length_of_AB_l3319_331920


namespace NUMINAMATH_CALUDE_jordyn_zrinka_age_ratio_l3319_331900

/-- Given the ages of Mehki, Jordyn, and Zrinka, prove the ratio of Jordyn's to Zrinka's age -/
theorem jordyn_zrinka_age_ratio :
  ∀ (mehki_age jordyn_age zrinka_age : ℕ),
  mehki_age = 22 →
  zrinka_age = 6 →
  mehki_age = jordyn_age + 10 →
  (jordyn_age : ℚ) / (zrinka_age : ℚ) = 2 := by
sorry

end NUMINAMATH_CALUDE_jordyn_zrinka_age_ratio_l3319_331900


namespace NUMINAMATH_CALUDE_expected_value_of_sum_l3319_331961

def marbles : Finset ℕ := {1, 2, 3, 4, 5, 6}

def sum_of_pairs (s : Finset ℕ) : ℕ :=
  (s.powerset.filter (fun t => t.card = 2)).sum (fun t => t.sum id)

def number_of_pairs (s : Finset ℕ) : ℕ :=
  (s.powerset.filter (fun t => t.card = 2)).card

theorem expected_value_of_sum (s : Finset ℕ) :
  s = marbles →
  (sum_of_pairs s : ℚ) / (number_of_pairs s : ℚ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_of_sum_l3319_331961


namespace NUMINAMATH_CALUDE_chess_club_officers_l3319_331988

/-- The number of members in the Chess Club -/
def total_members : ℕ := 25

/-- The number of officers to be selected -/
def num_officers : ℕ := 3

/-- Function to calculate the number of ways to select officers -/
def select_officers (total : ℕ) (officers : ℕ) : ℕ :=
  let case1 := (total - 2) * (total - 3) * (total - 4)  -- Neither Alice nor Bob
  let case2 := 3 * 2 * (total - 3)  -- Both Alice and Bob
  case1 + case2

/-- Theorem stating the number of ways to select officers -/
theorem chess_club_officers :
  select_officers total_members num_officers = 10758 := by
  sorry

end NUMINAMATH_CALUDE_chess_club_officers_l3319_331988


namespace NUMINAMATH_CALUDE_perfect_square_divisibility_l3319_331947

theorem perfect_square_divisibility (a b : ℕ) (h : (a^2 + b^2 + a) % (a * b) = 0) :
  ∃ k : ℕ, a = k^2 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_divisibility_l3319_331947


namespace NUMINAMATH_CALUDE_min_xy_value_l3319_331972

theorem min_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2*y = x*y) :
  x * y ≥ 8 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ + 2*y₀ = x₀*y₀ ∧ x₀ * y₀ = 8 :=
by sorry

end NUMINAMATH_CALUDE_min_xy_value_l3319_331972


namespace NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_l3319_331908

theorem sum_of_real_and_imag_parts : ∃ (z : ℂ), z = (Complex.I / (1 + Complex.I)) - (1 / (2 * Complex.I)) ∧ z.re + z.im = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_real_and_imag_parts_l3319_331908


namespace NUMINAMATH_CALUDE_tim_appetizers_l3319_331923

theorem tim_appetizers (total_spent : ℚ) (entree_percentage : ℚ) (appetizer_cost : ℚ) : 
  total_spent = 50 →
  entree_percentage = 80 / 100 →
  appetizer_cost = 5 →
  (total_spent * (1 - entree_percentage)) / appetizer_cost = 2 := by
sorry

end NUMINAMATH_CALUDE_tim_appetizers_l3319_331923


namespace NUMINAMATH_CALUDE_gcd_52800_35275_l3319_331984

theorem gcd_52800_35275 : Nat.gcd 52800 35275 = 25 := by
  sorry

end NUMINAMATH_CALUDE_gcd_52800_35275_l3319_331984


namespace NUMINAMATH_CALUDE_real_part_of_complex_product_l3319_331964

def complex_mul (a b c d : ℝ) : ℂ := Complex.mk (a*c - b*d) (a*d + b*c)

theorem real_part_of_complex_product : 
  (complex_mul 1 1 1 (-2)).re = 3 := by sorry

end NUMINAMATH_CALUDE_real_part_of_complex_product_l3319_331964


namespace NUMINAMATH_CALUDE_equation_solutions_sum_of_squares_complex_equation_l3319_331962

theorem equation_solutions (x : ℝ) :
  (x^2 + 2) / x = 5 + 2/5 → x = 5 ∨ x = 2/5 := by sorry

theorem sum_of_squares (a b : ℝ) :
  a + 3/a = 7 ∧ b + 3/b = 7 → a^2 + b^2 = 43 := by sorry

theorem complex_equation (t k : ℝ) :
  (∃ x₁ x₂ : ℝ, 6/(x₁ - 1) = k - x₁ ∧ 6/(x₂ - 1) = k - x₂ ∧ x₁ = t + 1 ∧ x₂ = t^2 + 2) →
  k^2 - 4*k + 4*t^3 = 32 := by sorry

end NUMINAMATH_CALUDE_equation_solutions_sum_of_squares_complex_equation_l3319_331962


namespace NUMINAMATH_CALUDE_dodecahedron_edge_probability_l3319_331929

/-- A regular dodecahedron -/
structure RegularDodecahedron where
  vertices : Finset (Fin 20)
  edges : Finset (Fin 20 × Fin 20)
  vertex_count : vertices.card = 20
  edge_count : edges.card = 30
  vertex_degree : ∀ v ∈ vertices, (edges.filter (λ e => e.1 = v ∨ e.2 = v)).card = 3

/-- The probability of selecting two vertices that form an edge in a regular dodecahedron -/
def edge_selection_probability (d : RegularDodecahedron) : ℚ :=
  3 / 19

/-- Theorem: The probability of randomly selecting two vertices that form an edge in a regular dodecahedron is 3/19 -/
theorem dodecahedron_edge_probability (d : RegularDodecahedron) :
  edge_selection_probability d = 3 / 19 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_edge_probability_l3319_331929


namespace NUMINAMATH_CALUDE_points_are_coplanar_l3319_331932

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the points
variable (A B C D : V)

-- State the theorem
theorem points_are_coplanar
  (h_nonzero : e₁ ≠ 0 ∧ e₂ ≠ 0)
  (h_not_collinear : ¬ ∃ (k : ℝ), e₁ = k • e₂)
  (h_AB : B - A = e₁ + e₂)
  (h_AC : C - A = -3 • e₁ + 7 • e₂)
  (h_AD : D - A = 2 • e₁ - 3 • e₂) :
  ∃ (a b c d : ℝ), a • (B - A) + b • (C - A) + c • (D - A) = d • (0 : V) :=
sorry

end NUMINAMATH_CALUDE_points_are_coplanar_l3319_331932


namespace NUMINAMATH_CALUDE_friday_temperature_l3319_331968

/-- Given the average temperatures for two sets of four days and the temperature on Monday,
    prove that the temperature on Friday is 36 degrees. -/
theorem friday_temperature
  (avg_mon_to_thu : (mon + tue + wed + thu) / 4 = 48)
  (avg_tue_to_fri : (tue + wed + thu + fri) / 4 = 46)
  (monday_temp : mon = 44)
  : fri = 36 := by
  sorry

end NUMINAMATH_CALUDE_friday_temperature_l3319_331968


namespace NUMINAMATH_CALUDE_arithmetic_sequence_index_l3319_331941

/-- Given an arithmetic sequence {a_n} with first term a₁ = 1 and common difference d = 5,
    prove that if a_n = 2016, then n = 404. -/
theorem arithmetic_sequence_index (a : ℕ → ℝ) (n : ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ k : ℕ, a (k + 1) - a k = 5)
  (h3 : a n = 2016) :
  n = 404 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_index_l3319_331941
