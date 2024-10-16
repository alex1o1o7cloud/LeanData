import Mathlib

namespace NUMINAMATH_CALUDE_square_difference_minus_product_l3855_385587

theorem square_difference_minus_product (a b : ℝ) : (a - b)^2 - b * (b - 2*a) = a^2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_minus_product_l3855_385587


namespace NUMINAMATH_CALUDE_solutions_sum_greater_than_two_l3855_385535

noncomputable section

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^2 + 2/x

-- State the theorem
theorem solutions_sum_greater_than_two 
  (t : ℝ) 
  (h_t : t > 3) 
  (x₁ x₂ : ℝ) 
  (h_x₁ : x₁ > 0) 
  (h_x₂ : x₂ > 0) 
  (h_x₁_neq_x₂ : x₁ ≠ x₂) 
  (h_f_x₁ : f x₁ = t) 
  (h_f_x₂ : f x₂ = t) : 
  x₁ + x₂ > 2 := by
  sorry

end

end NUMINAMATH_CALUDE_solutions_sum_greater_than_two_l3855_385535


namespace NUMINAMATH_CALUDE_square_area_ratio_l3855_385508

theorem square_area_ratio (x : ℝ) (hx : x > 0) : 
  (2 * x)^2 / (6 * x)^2 = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_square_area_ratio_l3855_385508


namespace NUMINAMATH_CALUDE_trigonometric_expression_equality_l3855_385513

theorem trigonometric_expression_equality : 
  1 / Real.cos (70 * π / 180) - Real.sqrt 2 / Real.sin (70 * π / 180) = 4 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equality_l3855_385513


namespace NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_17_gt_200_l3855_385583

def digit_sum (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + digit_sum (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

theorem smallest_prime_with_digit_sum_17_gt_200 :
  ∃ p : ℕ,
    is_prime p ∧
    digit_sum p = 17 ∧
    p > 200 ∧
    (∀ q : ℕ, is_prime q → digit_sum q = 17 → q > 200 → p ≤ q) ∧
    p = 197 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_with_digit_sum_17_gt_200_l3855_385583


namespace NUMINAMATH_CALUDE_factor_expression_l3855_385553

theorem factor_expression (x : ℝ) :
  (10 * x^3 + 50 * x^2 - 5) - (-5 * x^3 + 15 * x^2 - 5) = 5 * x^2 * (3 * x + 7) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3855_385553


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3855_385515

theorem imaginary_part_of_complex_fraction (z : ℂ) : z = (1 + 2*I) / (2 - I) → z.im = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3855_385515


namespace NUMINAMATH_CALUDE_museum_ticket_cost_l3855_385589

theorem museum_ticket_cost (num_students : ℕ) (num_teachers : ℕ) 
  (student_ticket_price : ℕ) (teacher_ticket_price : ℕ) : 
  num_students = 12 → 
  num_teachers = 4 → 
  student_ticket_price = 1 → 
  teacher_ticket_price = 3 → 
  num_students * student_ticket_price + num_teachers * teacher_ticket_price = 24 := by
sorry

end NUMINAMATH_CALUDE_museum_ticket_cost_l3855_385589


namespace NUMINAMATH_CALUDE_fixed_monthly_costs_l3855_385524

/-- The fixed monthly costs for a computer manufacturer producing electronic components --/
theorem fixed_monthly_costs 
  (production_cost : ℝ) 
  (shipping_cost : ℝ) 
  (units_sold : ℕ) 
  (selling_price : ℝ) 
  (h1 : production_cost = 80)
  (h2 : shipping_cost = 3)
  (h3 : units_sold = 150)
  (h4 : selling_price = 191.67)
  (h5 : selling_price * (units_sold : ℝ) = (production_cost + shipping_cost) * (units_sold : ℝ) + fixed_costs) :
  fixed_costs = 16300.50 := by
  sorry

#check fixed_monthly_costs

end NUMINAMATH_CALUDE_fixed_monthly_costs_l3855_385524


namespace NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3855_385542

/-- The surface area of a cube inscribed in a sphere, which is itself inscribed in another cube --/
theorem inscribed_cube_surface_area (outer_cube_area : ℝ) : 
  outer_cube_area = 150 →
  ∃ (inner_cube_area : ℝ), inner_cube_area = 50 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_surface_area_l3855_385542


namespace NUMINAMATH_CALUDE_book_reading_competition_l3855_385503

/-- Represents the number of pages read by each girl -/
structure PageCount where
  ivana : ℕ
  majka : ℕ
  lucka : ℕ
  sasa : ℕ
  zuzka : ℕ

/-- Checks if all values in the PageCount are distinct -/
def allDistinct (p : PageCount) : Prop :=
  p.ivana ≠ p.majka ∧ p.ivana ≠ p.lucka ∧ p.ivana ≠ p.sasa ∧ p.ivana ≠ p.zuzka ∧
  p.majka ≠ p.lucka ∧ p.majka ≠ p.sasa ∧ p.majka ≠ p.zuzka ∧
  p.lucka ≠ p.sasa ∧ p.lucka ≠ p.zuzka ∧
  p.sasa ≠ p.zuzka

/-- The theorem representing the book reading competition -/
theorem book_reading_competition :
  ∃! (p : PageCount),
    p.lucka = 32 ∧
    p.lucka = (p.sasa + p.zuzka) / 2 ∧
    p.ivana = p.zuzka + 5 ∧
    p.majka = p.sasa - 8 ∧
    allDistinct p ∧
    (∀ x ∈ [p.ivana, p.majka, p.lucka, p.sasa, p.zuzka], x ≥ 27) ∧
    p.ivana = 34 ∧ p.majka = 27 ∧ p.lucka = 32 ∧ p.sasa = 35 ∧ p.zuzka = 29 :=
by sorry

end NUMINAMATH_CALUDE_book_reading_competition_l3855_385503


namespace NUMINAMATH_CALUDE_probability_three_fives_in_eight_rolls_l3855_385569

/-- A fair die has 6 sides -/
def die_sides : ℕ := 6

/-- The number of times the die is rolled -/
def num_rolls : ℕ := 8

/-- The number of times we want the specific outcome (5 in this case) to appear -/
def target_occurrences : ℕ := 3

/-- The probability of rolling exactly 3 fives in 8 rolls of a fair die -/
theorem probability_three_fives_in_eight_rolls :
  (Nat.choose num_rolls target_occurrences : ℚ) / (die_sides ^ num_rolls) = 56 / 1679616 := by
  sorry

end NUMINAMATH_CALUDE_probability_three_fives_in_eight_rolls_l3855_385569


namespace NUMINAMATH_CALUDE_old_cars_less_than_half_after_three_years_l3855_385533

/-- Represents the state of the car fleet after a certain number of years -/
structure FleetState where
  years : ℕ
  oldCars : ℕ
  newCars : ℕ

/-- Updates the fleet state for one year -/
def updateFleet (state : FleetState) : FleetState :=
  { years := state.years + 1,
    oldCars := max (state.oldCars - 5) 0,
    newCars := state.newCars + 6 }

/-- Calculates the fleet state after a given number of years -/
def fleetAfterYears (years : ℕ) : FleetState :=
  (List.range years).foldl (fun state _ => updateFleet state) { years := 0, oldCars := 20, newCars := 0 }

/-- Theorem: After 3 years, the number of old cars is less than 50% of the total fleet -/
theorem old_cars_less_than_half_after_three_years :
  let state := fleetAfterYears 3
  state.oldCars < (state.oldCars + state.newCars) / 2 := by
  sorry


end NUMINAMATH_CALUDE_old_cars_less_than_half_after_three_years_l3855_385533


namespace NUMINAMATH_CALUDE_last_row_value_l3855_385527

/-- Represents a triangular table with the given properties -/
def TriangularTable (n : ℕ) : Type :=
  Fin n → Fin n → ℕ

/-- The first row of the table contains the first n positive integers -/
def FirstRowProperty (t : TriangularTable 100) : Prop :=
  ∀ i : Fin 100, t 0 i = i.val + 1

/-- Each element (from the second row onwards) is the sum of the two elements directly above it -/
def SumProperty (t : TriangularTable 100) : Prop :=
  ∀ i j : Fin 100, i > 0 → j < i → t i j = t (i-1) j + t (i-1) (j+1)

/-- The last row contains only one element -/
def LastRowProperty (t : TriangularTable 100) : Prop :=
  t 99 0 = t 99 0  -- This is always true, but it ensures the element exists

/-- The main theorem: the value in the last row is 101 × 2^98 -/
theorem last_row_value (t : TriangularTable 100) 
  (h1 : FirstRowProperty t) 
  (h2 : SumProperty t) 
  (h3 : LastRowProperty t) : 
  t 99 0 = 101 * 2^98 := by
  sorry

end NUMINAMATH_CALUDE_last_row_value_l3855_385527


namespace NUMINAMATH_CALUDE_jane_sequin_count_l3855_385592

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a₁ n d : ℕ) : ℕ := n * (2 * a₁ + (n - 1) * d) / 2

/-- Represents the sequin count problem for Jane's costume -/
def sequinCount : Prop :=
  let blueStars := 10 * 12
  let purpleSquares := 8 * 15
  let greenHexagons := 14 * 20
  let redCircles := arithmeticSum 10 5 5
  blueStars + purpleSquares + greenHexagons + redCircles = 620

theorem jane_sequin_count : sequinCount := by
  sorry

end NUMINAMATH_CALUDE_jane_sequin_count_l3855_385592


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3855_385591

theorem binomial_coefficient_sum (n : ℕ) : 
  let m := (4 : ℕ) ^ n
  let k := (2 : ℕ) ^ n
  m + k = 1056 → n = 5 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3855_385591


namespace NUMINAMATH_CALUDE_units_digit_of_33_power_l3855_385578

theorem units_digit_of_33_power (n : ℕ) : (33 ^ (33 * (22 ^ 22))) % 10 = 1 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_33_power_l3855_385578


namespace NUMINAMATH_CALUDE_fifteen_by_fifteen_grid_toothpicks_l3855_385512

/-- Represents a rectangular grid of toothpicks with diagonal lines -/
structure ToothpickGrid where
  height : ℕ
  width : ℕ
  has_diagonals : Bool

/-- Calculates the total number of toothpicks in the grid -/
def total_toothpicks (grid : ToothpickGrid) : ℕ :=
  let horizontal := (grid.height + 1) * grid.width
  let vertical := (grid.width + 1) * grid.height
  let diagonal := if grid.has_diagonals then 2 * grid.height else 0
  horizontal + vertical + diagonal

/-- The theorem stating that a 15x15 grid with diagonals has 510 toothpicks -/
theorem fifteen_by_fifteen_grid_toothpicks :
  total_toothpicks { height := 15, width := 15, has_diagonals := true } = 510 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_by_fifteen_grid_toothpicks_l3855_385512


namespace NUMINAMATH_CALUDE_sqrt_two_minus_one_to_zero_l3855_385575

theorem sqrt_two_minus_one_to_zero : (Real.sqrt 2 - 1) ^ (0 : ℕ) = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_two_minus_one_to_zero_l3855_385575


namespace NUMINAMATH_CALUDE_lease_problem_l3855_385547

theorem lease_problem (elapsed_time : ℝ) : 
  elapsed_time > 0 ∧ 
  elapsed_time < 99 ∧
  (2 / 3) * elapsed_time = (4 / 5) * (99 - elapsed_time) →
  elapsed_time = 54 := by
sorry

end NUMINAMATH_CALUDE_lease_problem_l3855_385547


namespace NUMINAMATH_CALUDE_combined_weight_of_boxes_l3855_385561

theorem combined_weight_of_boxes (box1 box2 box3 : ℕ) 
  (h1 : box1 = 2) 
  (h2 : box2 = 11) 
  (h3 : box3 = 5) : 
  box1 + box2 + box3 = 18 := by
  sorry

end NUMINAMATH_CALUDE_combined_weight_of_boxes_l3855_385561


namespace NUMINAMATH_CALUDE_total_heads_l3855_385545

theorem total_heads (num_hens : ℕ) (total_feet : ℕ) : num_hens = 24 → total_feet = 136 → ∃ (num_cows : ℕ), num_hens + num_cows = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_heads_l3855_385545


namespace NUMINAMATH_CALUDE_bee_count_l3855_385597

theorem bee_count (initial_bees new_bees : ℕ) : 
  initial_bees = 16 → new_bees = 7 → initial_bees + new_bees = 23 := by
  sorry

end NUMINAMATH_CALUDE_bee_count_l3855_385597


namespace NUMINAMATH_CALUDE_number_problem_l3855_385551

theorem number_problem : 
  let x : ℝ := 25
  80 / 100 * 60 - (4 / 5 * x) = 28 := by sorry

end NUMINAMATH_CALUDE_number_problem_l3855_385551


namespace NUMINAMATH_CALUDE_min_value_theorem_l3855_385585

theorem min_value_theorem (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0)
  (sum_squares : x^2 + y^2 + z^2 = 1) :
  x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2) ≥ 3 * Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3855_385585


namespace NUMINAMATH_CALUDE_vector_relations_l3855_385546

/-- Two-dimensional vector -/
structure Vec2D where
  x : ℝ
  y : ℝ

/-- Parallel vectors -/
def parallel (v w : Vec2D) : Prop :=
  v.x * w.y = v.y * w.x

/-- Perpendicular vectors -/
def perpendicular (v w : Vec2D) : Prop :=
  v.x * w.x + v.y * w.y = 0

theorem vector_relations (m : ℝ) :
  let a : Vec2D := ⟨1, 2⟩
  let b : Vec2D := ⟨-2, m⟩
  (parallel a b → m = -4) ∧
  (perpendicular a b → m = 1) := by
  sorry

end NUMINAMATH_CALUDE_vector_relations_l3855_385546


namespace NUMINAMATH_CALUDE_square_sum_from_means_l3855_385536

theorem square_sum_from_means (x y : ℝ) 
  (h_arithmetic : (x + y) / 2 = 20) 
  (h_geometric : Real.sqrt (x * y) = Real.sqrt 110) : 
  x^2 + y^2 = 1380 := by
sorry

end NUMINAMATH_CALUDE_square_sum_from_means_l3855_385536


namespace NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3855_385510

def f (x : ℝ) := x^3 - 15*x^2 - 33*x + 6

theorem monotonic_decreasing_interval :
  {x : ℝ | ∀ y, x < y → f x > f y} = {x | -1 < x ∧ x < 11} := by sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_interval_l3855_385510


namespace NUMINAMATH_CALUDE_shaded_area_of_carpet_l3855_385501

theorem shaded_area_of_carpet (S T : ℝ) : 
  12 / S = 4 →
  S / T = 4 →
  (8 * T^2 + S^2) = 27/2 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_of_carpet_l3855_385501


namespace NUMINAMATH_CALUDE_abs_three_minus_a_l3855_385514

theorem abs_three_minus_a (a : ℝ) (h : |1 - a| = 1 + |a|) : |3 - a| = 3 - a := by
  sorry

end NUMINAMATH_CALUDE_abs_three_minus_a_l3855_385514


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_power_l3855_385500

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem simplify_complex_fraction_power :
  ((1 + i) / (1 - i)) ^ 1002 = -1 :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_power_l3855_385500


namespace NUMINAMATH_CALUDE_digit_sum_problem_l3855_385526

theorem digit_sum_problem (A B C D E : ℕ) : 
  (A < 10) → (B < 10) → (C < 10) → (D < 10) → (E < 10) →
  (10 * E + A) + (10 * E + C) = 10 * D + A →
  (10 * E + A) - (10 * E + C) = A →
  D = 8 := by
sorry

end NUMINAMATH_CALUDE_digit_sum_problem_l3855_385526


namespace NUMINAMATH_CALUDE_sum_of_cube_difference_l3855_385594

theorem sum_of_cube_difference (a b c : ℕ+) :
  (a + b + c : ℕ+)^3 - a^3 - b^3 - c^3 = 210 →
  (a : ℕ) + b + c = 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cube_difference_l3855_385594


namespace NUMINAMATH_CALUDE_cone_radius_l3855_385511

/-- Given a cone with surface area 6π and whose lateral surface unfolds into a semicircle,
    the radius of the base of the cone is √2. -/
theorem cone_radius (r : Real) (l : Real) : 
  (π * r * r + π * r * l = 6 * π) →  -- Surface area of cone is 6π
  (2 * π * r = π * l) →              -- Lateral surface unfolds into a semicircle
  r = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_cone_radius_l3855_385511


namespace NUMINAMATH_CALUDE_hexagon_enclosure_octagon_enclosure_l3855_385559

-- Define the shapes
def Square (sideLength : ℝ) : Type := Unit
def RegularHexagon (sideLength : ℝ) : Type := Unit
def Circle (diameter : ℝ) : Type := Unit

-- Define the derived shapes
def Hexagon (s : Square 1) : Type := Unit
def Octagon (h : RegularHexagon (Real.sqrt 3 / 3)) : Type := Unit

-- Define the enclosure property
def CanEnclose (shape : Type) (figure : Circle 1) : Prop := sorry

-- State the theorems
theorem hexagon_enclosure (s : Square 1) (f : Circle 1) :
  CanEnclose (Hexagon s) f := sorry

theorem octagon_enclosure (h : RegularHexagon (Real.sqrt 3 / 3)) (f : Circle 1) :
  CanEnclose (Octagon h) f := sorry

end NUMINAMATH_CALUDE_hexagon_enclosure_octagon_enclosure_l3855_385559


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l3855_385509

-- Define the complex number z
variable (z : ℂ)

-- Define the condition
def condition (z : ℂ) : Prop := 1 + z = 2 + 3 * Complex.I

-- Theorem statement
theorem imaginary_part_of_z (h : condition z) : z.im = 3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l3855_385509


namespace NUMINAMATH_CALUDE_rectangular_plot_length_l3855_385550

theorem rectangular_plot_length (breadth : ℝ) (length : ℝ) (perimeter : ℝ) : 
  length = breadth + 24 →
  perimeter = 2 * (length + breadth) →
  26.50 * perimeter = 5300 →
  length = 62 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_length_l3855_385550


namespace NUMINAMATH_CALUDE_program1_output_program2_output_l3855_385574

-- Define a type to represent the state of the program
structure ProgramState where
  a : Int
  b : Int
  c : Int

-- Function to simulate the first program
def program1 (initial : ProgramState) : ProgramState :=
  { a := initial.b
  , b := initial.c
  , c := initial.c }

-- Function to simulate the second program
def program2 (initial : ProgramState) : ProgramState :=
  { a := initial.b
  , b := initial.c
  , c := initial.b }

-- Theorem for the first program
theorem program1_output :
  let initial := ProgramState.mk 3 (-5) 8
  let final := program1 initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = 8 := by sorry

-- Theorem for the second program
theorem program2_output :
  let initial := ProgramState.mk 3 (-5) 8
  let final := program2 initial
  final.a = -5 ∧ final.b = 8 ∧ final.c = -5 := by sorry

end NUMINAMATH_CALUDE_program1_output_program2_output_l3855_385574


namespace NUMINAMATH_CALUDE_max_envelopes_proof_l3855_385525

def number_of_bus_tickets : ℕ := 18
def number_of_subway_tickets : ℕ := 12

def max_envelopes : ℕ := Nat.gcd number_of_bus_tickets number_of_subway_tickets

theorem max_envelopes_proof :
  (∀ k : ℕ, k ∣ number_of_bus_tickets ∧ k ∣ number_of_subway_tickets → k ≤ max_envelopes) ∧
  (max_envelopes ∣ number_of_bus_tickets) ∧
  (max_envelopes ∣ number_of_subway_tickets) :=
sorry

end NUMINAMATH_CALUDE_max_envelopes_proof_l3855_385525


namespace NUMINAMATH_CALUDE_negation_of_existential_proposition_l3855_385590

theorem negation_of_existential_proposition :
  (¬ ∃ x : ℝ, x^2 + x - 1 ≥ 0) ↔ (∀ x : ℝ, x^2 + x - 1 < 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existential_proposition_l3855_385590


namespace NUMINAMATH_CALUDE_min_diff_composites_sum_96_l3855_385529

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem min_diff_composites_sum_96 : 
  ∃ (a b : ℕ), is_composite a ∧ is_composite b ∧ a + b = 96 ∧
  ∀ (c d : ℕ), is_composite c → is_composite d → c + d = 96 → 
  (c : ℤ) - (d : ℤ) ≥ 2 ∨ (d : ℤ) - (c : ℤ) ≥ 2 :=
sorry

end NUMINAMATH_CALUDE_min_diff_composites_sum_96_l3855_385529


namespace NUMINAMATH_CALUDE_one_count_greater_than_zero_count_l3855_385570

/-- Represents the sequence of concatenated decimal representations of numbers from 1 to n -/
def concatenatedSequence (n : ℕ) : List ℕ := sorry

/-- Counts the occurrences of a specific digit in the concatenated sequence -/
def digitCount (d : ℕ) (n : ℕ) : ℕ := sorry

/-- Theorem stating that the count of '1' is always greater than the count of '0' in the sequence -/
theorem one_count_greater_than_zero_count (n : ℕ) : digitCount 1 n > digitCount 0 n := by sorry

end NUMINAMATH_CALUDE_one_count_greater_than_zero_count_l3855_385570


namespace NUMINAMATH_CALUDE_max_value_x2_y2_z3_max_value_achieved_l3855_385558

theorem max_value_x2_y2_z3 (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : x + 2*y + 3*z = 1) : 
  x^2 + y^2 + z^3 ≤ 1 :=
by sorry

theorem max_value_achieved (x y z : ℝ) 
  (h_nonneg : x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0) 
  (h_constraint : x + 2*y + 3*z = 1) : 
  ∃ (a b c : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ a + 2*b + 3*c = 1 ∧ a^2 + b^2 + c^3 = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_x2_y2_z3_max_value_achieved_l3855_385558


namespace NUMINAMATH_CALUDE_sqrt_nine_factorial_over_ninety_l3855_385506

theorem sqrt_nine_factorial_over_ninety : 
  Real.sqrt (Nat.factorial 9 / 90) = 4 * Real.sqrt 42 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_nine_factorial_over_ninety_l3855_385506


namespace NUMINAMATH_CALUDE_local_min_implies_b_in_open_unit_interval_l3855_385548

/-- If f(x) = x^3 - 3bx + b has a local minimum in (0, 1), then b ∈ (0, 1) -/
theorem local_min_implies_b_in_open_unit_interval (b : ℝ) : 
  (∃ c ∈ Set.Ioo 0 1, IsLocalMin (fun x => x^3 - 3*b*x + b) c) → 
  b ∈ Set.Ioo 0 1 := by
sorry

end NUMINAMATH_CALUDE_local_min_implies_b_in_open_unit_interval_l3855_385548


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3855_385502

theorem quadratic_inequality_solution (a b : ℝ) (h : Set ℝ) : 
  (∀ x, x ∈ h ↔ (a * x^2 - 3*x + 6 > 4 ∧ (x < 1 ∨ x > b))) →
  (a = 1 ∧ b = 2) ∧
  (∀ c, 
    (c > 2 → {x | 2 < x ∧ x < c} = {x | x^2 - (2 + c)*x + 2*c < 0}) ∧
    (c < 2 → {x | c < x ∧ x < 2} = {x | x^2 - (2 + c)*x + 2*c < 0}) ∧
    (c = 2 → (∅ : Set ℝ) = {x | x^2 - (2 + c)*x + 2*c < 0})) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3855_385502


namespace NUMINAMATH_CALUDE_parabola_vertex_l3855_385538

/-- The vertex of a parabola given by the equation y^2 + 8y + 4x + 9 = 0 is (7/4, -4) -/
theorem parabola_vertex (x y : ℝ) : 
  y^2 + 8*y + 4*x + 9 = 0 → (x, y) = (7/4, -4) ∨ ∃ t : ℝ, (x, y) = (7/4 - t^2, -4 + t) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_l3855_385538


namespace NUMINAMATH_CALUDE_simplify_expression_l3855_385505

theorem simplify_expression (x : ℝ) : x^3 * x^2 * x + (x^3)^2 + (-2*x^2)^3 = -6*x^6 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3855_385505


namespace NUMINAMATH_CALUDE_harmonic_mean_of_2_3_6_l3855_385531

theorem harmonic_mean_of_2_3_6 (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 6) :
  3 / (1/a + 1/b + 1/c) = 3 := by
  sorry

end NUMINAMATH_CALUDE_harmonic_mean_of_2_3_6_l3855_385531


namespace NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l3855_385577

/-- The surface area of a sphere inscribed in a cube with edge length 2 is 4π. -/
theorem inscribed_sphere_surface_area (cube_edge : ℝ) (h : cube_edge = 2) :
  let sphere_radius := cube_edge / 2
  4 * Real.pi * sphere_radius ^ 2 = 4 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_inscribed_sphere_surface_area_l3855_385577


namespace NUMINAMATH_CALUDE_reduced_price_per_dozen_l3855_385563

/-- Represents the price reduction percentage -/
def price_reduction : ℝ := 0.40

/-- Represents the additional number of apples that can be bought after the price reduction -/
def additional_apples : ℕ := 64

/-- Represents the fixed amount of money spent on apples -/
def fixed_amount : ℝ := 40

/-- Represents the number of apples in a dozen -/
def apples_per_dozen : ℕ := 12

/-- Theorem stating that given the conditions, the reduced price per dozen apples is Rs. 3 -/
theorem reduced_price_per_dozen (original_price : ℝ) (h : original_price > 0) :
  let reduced_price := original_price * (1 - price_reduction)
  let original_quantity := fixed_amount / original_price
  let new_quantity := original_quantity + additional_apples
  (new_quantity : ℝ) * reduced_price = fixed_amount →
  apples_per_dozen * (fixed_amount / new_quantity) = 3 :=
by sorry

end NUMINAMATH_CALUDE_reduced_price_per_dozen_l3855_385563


namespace NUMINAMATH_CALUDE_b_profit_is_4000_l3855_385586

/-- Represents the profit distribution in a joint business venture -/
structure JointBusiness where
  total_profit : ℕ
  a_investment_ratio : ℕ
  a_time_ratio : ℕ

/-- Calculates B's profit in the joint business venture -/
def calculate_b_profit (jb : JointBusiness) : ℕ :=
  jb.total_profit / (jb.a_investment_ratio * jb.a_time_ratio + 1)

/-- Theorem stating that B's profit is 4000 given the specified conditions -/
theorem b_profit_is_4000 (jb : JointBusiness) 
  (h1 : jb.total_profit = 28000)
  (h2 : jb.a_investment_ratio = 3)
  (h3 : jb.a_time_ratio = 2) : 
  calculate_b_profit jb = 4000 := by
  sorry

#eval calculate_b_profit { total_profit := 28000, a_investment_ratio := 3, a_time_ratio := 2 }

end NUMINAMATH_CALUDE_b_profit_is_4000_l3855_385586


namespace NUMINAMATH_CALUDE_exist_four_cells_l3855_385522

/-- Represents a cell in the grid -/
structure Cell :=
  (x : Fin 17)
  (y : Fin 17)
  (value : Fin 70)

/-- The type of the grid -/
def Grid := Fin 17 → Fin 17 → Fin 70

/-- Predicate to check if all numbers from 1 to 70 appear exactly once in the grid -/
def valid_grid (g : Grid) : Prop :=
  ∀ n : Fin 70, ∃! (x y : Fin 17), g x y = n

/-- Distance between two cells -/
def distance (a b : Cell) : ℕ :=
  (a.x - b.x) ^ 2 + (a.y - b.y) ^ 2

/-- Sum of values in two cells -/
def sum_values (a b : Cell) : ℕ :=
  a.value.val + b.value.val

/-- Main theorem -/
theorem exist_four_cells (g : Grid) (h : valid_grid g) :
  ∃ (a b c d : Cell),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧
    distance a b = distance c d ∧
    distance a d = distance b c ∧
    sum_values a c = sum_values b d :=
  sorry

end NUMINAMATH_CALUDE_exist_four_cells_l3855_385522


namespace NUMINAMATH_CALUDE_prime_product_divisors_l3855_385530

theorem prime_product_divisors (p q : Nat) (x : Nat) :
  Prime p →
  Prime q →
  (Nat.divisors (p^x * q^5)).card = 30 →
  x = 4 := by
sorry

end NUMINAMATH_CALUDE_prime_product_divisors_l3855_385530


namespace NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3855_385540

theorem ceiling_sum_sqrt : ⌈Real.sqrt 2⌉ + ⌈Real.sqrt 22⌉ + ⌈Real.sqrt 222⌉ = 22 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_sum_sqrt_l3855_385540


namespace NUMINAMATH_CALUDE_composition_result_l3855_385593

/-- Given two functions f and g, prove that f(g(2)) = 169 -/
theorem composition_result (f g : ℝ → ℝ) 
  (hf : ∀ x, f x = x^2)
  (hg : ∀ x, g x = 2*x^2 + x + 3) : 
  f (g 2) = 169 := by
  sorry

end NUMINAMATH_CALUDE_composition_result_l3855_385593


namespace NUMINAMATH_CALUDE_jacob_test_score_l3855_385576

theorem jacob_test_score (x : ℝ) : 
  (x + 79 + 92 + 84 + 85) / 5 = 85 → x = 85 := by
sorry

end NUMINAMATH_CALUDE_jacob_test_score_l3855_385576


namespace NUMINAMATH_CALUDE_return_trip_times_l3855_385565

/-- Represents the flight scenario between two cities -/
structure FlightScenario where
  d : ℝ  -- distance between cities
  p : ℝ  -- speed of plane in still air
  w : ℝ  -- wind speed
  time_against_wind : ℝ  -- time flying against the wind

/-- Conditions of the flight scenario -/
def flight_conditions (scenario : FlightScenario) : Prop :=
  scenario.time_against_wind = 1.4 ∧  -- 1.4 hours against the wind
  scenario.d = 1.4 * (scenario.p - scenario.w) ∧  -- distance equation
  scenario.d / (scenario.p + scenario.w) = scenario.d / scenario.p - 0.25  -- return trip equation

/-- Theorem stating the possible return trip times -/
theorem return_trip_times (scenario : FlightScenario) 
  (h : flight_conditions scenario) : 
  (scenario.d / (scenario.p + scenario.w) = 12 / 60) ∨ 
  (scenario.d / (scenario.p + scenario.w) = 69 / 60) := by
  sorry


end NUMINAMATH_CALUDE_return_trip_times_l3855_385565


namespace NUMINAMATH_CALUDE_cube_volume_from_diagonal_edge_distance_l3855_385588

/-- The volume of a cube, given the distance from its space diagonal to a non-intersecting edge. -/
theorem cube_volume_from_diagonal_edge_distance (d : ℝ) (d_pos : 0 < d) : 
  ∃ (V : ℝ), V = 2 * d^3 * Real.sqrt 2 ∧ 
  (∃ (a : ℝ), a > 0 ∧ a = d * Real.sqrt 2 ∧ V = a^3) :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_diagonal_edge_distance_l3855_385588


namespace NUMINAMATH_CALUDE_first_term_is_5_5_l3855_385581

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  /-- The first term of the sequence -/
  a : ℝ
  /-- The common difference of the sequence -/
  d : ℝ
  /-- The sum of the first 30 terms is 600 -/
  sum_30 : (30 / 2 : ℝ) * (2 * a + 29 * d) = 600
  /-- The sum of the next 70 terms (31st to 100th) is 4900 -/
  sum_70 : (70 / 2 : ℝ) * (2 * (a + 30 * d) + 69 * d) = 4900

/-- The first term of the arithmetic sequence with the given properties is 5.5 -/
theorem first_term_is_5_5 (seq : ArithmeticSequence) : seq.a = 5.5 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_5_5_l3855_385581


namespace NUMINAMATH_CALUDE_min_value_when_a_is_one_a_range_when_f_2_gt_5_l3855_385582

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x + a| + |x - a|

-- Part I
theorem min_value_when_a_is_one :
  ∀ x : ℝ, f 1 x ≥ 2 :=
sorry

-- Part II
theorem a_range_when_f_2_gt_5 :
  ∀ a : ℝ, f a 2 > 5 → a < -5/2 ∨ a > 5/2 :=
sorry

end NUMINAMATH_CALUDE_min_value_when_a_is_one_a_range_when_f_2_gt_5_l3855_385582


namespace NUMINAMATH_CALUDE_seed_germination_percentage_l3855_385516

theorem seed_germination_percentage (seeds_plot1 seeds_plot2 : ℕ)
  (germination_rate_plot1 total_germination_rate : ℚ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  germination_rate_plot1 = 30 / 100 →
  total_germination_rate = 32 / 100 →
  (germination_rate_plot1 * seeds_plot1 + (35 / 100) * seeds_plot2) / (seeds_plot1 + seeds_plot2) = total_germination_rate :=
by sorry

end NUMINAMATH_CALUDE_seed_germination_percentage_l3855_385516


namespace NUMINAMATH_CALUDE_gcd_of_B_is_two_l3855_385532

def B : Set ℕ := {n : ℕ | ∃ x : ℕ, x > 0 ∧ n = (x - 1) + x + (x + 1) + (x + 2)}

theorem gcd_of_B_is_two :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, m > 0 → (∀ n ∈ B, m ∣ n) → m ≤ d) ∧ d = 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_two_l3855_385532


namespace NUMINAMATH_CALUDE_power_of_power_of_three_l3855_385541

theorem power_of_power_of_three : (3^3)^3 = 19683 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_of_three_l3855_385541


namespace NUMINAMATH_CALUDE_prob_roll_less_than_4_l3855_385573

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset (Fin 8) := Finset.univ

/-- The event of rolling a number less than 4 -/
def roll_less_than_4 : Finset (Fin 8) := Finset.filter (λ x => x.val < 4) fair_8_sided_die

/-- The probability of an event occurring when rolling a fair 8-sided die -/
def prob (event : Finset (Fin 8)) : ℚ :=
  (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

theorem prob_roll_less_than_4 : 
  prob roll_less_than_4 = 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_prob_roll_less_than_4_l3855_385573


namespace NUMINAMATH_CALUDE_sierra_crest_trail_length_l3855_385507

/-- Represents the Sierra Crest Trail hike -/
structure HikeData where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- The Sierra Crest Trail hike theorem -/
theorem sierra_crest_trail_length (h : HikeData) : 
  h.day1 + h.day2 + h.day3 = 36 →
  (h.day2 + h.day4) / 2 = 15 →
  h.day4 + h.day5 = 38 →
  h.day1 + h.day4 = 32 →
  h.day1 + h.day2 + h.day3 + h.day4 + h.day5 = 74 := by
  sorry


end NUMINAMATH_CALUDE_sierra_crest_trail_length_l3855_385507


namespace NUMINAMATH_CALUDE_factorization_equality_l3855_385584

theorem factorization_equality (x : ℝ) : 
  75 * x^19 + 165 * x^38 = 15 * x^19 * (5 + 11 * x^19) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l3855_385584


namespace NUMINAMATH_CALUDE_prove_additional_cans_l3855_385549

/-- The number of additional cans Alyssa and Abigail need to collect. -/
def additional_cans_needed (total_needed alyssa_collected abigail_collected : ℕ) : ℕ :=
  total_needed - (alyssa_collected + abigail_collected)

/-- Theorem: Given the conditions, the additional cans needed is 27. -/
theorem prove_additional_cans : additional_cans_needed 100 30 43 = 27 := by
  sorry

end NUMINAMATH_CALUDE_prove_additional_cans_l3855_385549


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3855_385596

theorem imaginary_part_of_complex_fraction : 
  let i : ℂ := Complex.I
  let z : ℂ := (7 + i) / (3 + 4 * i)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3855_385596


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l3855_385521

/-- The motion equation of an object -/
def s (t : ℝ) : ℝ := 7 * t^2 - 13 * t + 8

/-- The instantaneous velocity (derivative of s with respect to t) -/
def v (t : ℝ) : ℝ := 14 * t - 13

/-- Theorem: If the instantaneous velocity at t₀ is 1, then t₀ = 1 -/
theorem instantaneous_velocity_at_one (t₀ : ℝ) : v t₀ = 1 → t₀ = 1 := by
  sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_one_l3855_385521


namespace NUMINAMATH_CALUDE_quarter_circles_sum_limit_l3855_385568

theorem quarter_circles_sum_limit (D : ℝ) (h : D > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N,
    |2 * n * (π * D / (8 * n)) - (π * D / 4)| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circles_sum_limit_l3855_385568


namespace NUMINAMATH_CALUDE_total_wheels_eq_160_l3855_385537

/-- The number of bicycles -/
def num_bicycles : ℕ := 50

/-- The number of tricycles -/
def num_tricycles : ℕ := 20

/-- The number of wheels on a bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of wheels on a tricycle -/
def wheels_per_tricycle : ℕ := 3

/-- The total number of wheels on all bicycles and tricycles -/
def total_wheels : ℕ := num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle

theorem total_wheels_eq_160 : total_wheels = 160 := by
  sorry

end NUMINAMATH_CALUDE_total_wheels_eq_160_l3855_385537


namespace NUMINAMATH_CALUDE_opposite_expressions_solution_l3855_385517

theorem opposite_expressions_solution (x : ℚ) : (8*x - 7 = -(6 - 2*x)) → x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_expressions_solution_l3855_385517


namespace NUMINAMATH_CALUDE_terms_difference_l3855_385552

theorem terms_difference (k : ℕ) (h : k > 1) :
  (Finset.range (2^(k+1) - 1)).card - (Finset.range (2^k - 1)).card = 2^k :=
sorry

end NUMINAMATH_CALUDE_terms_difference_l3855_385552


namespace NUMINAMATH_CALUDE_function_property_l3855_385595

theorem function_property (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) 
  (h2 : f (-3) = a) : 
  f 12 = -4 * a := by
  sorry

end NUMINAMATH_CALUDE_function_property_l3855_385595


namespace NUMINAMATH_CALUDE_original_ratio_first_term_l3855_385534

/-- Given a ratio where the second term is 11, and adding 5 to both terms results
    in a ratio of 3:4, prove that the first term of the original ratio is 7. -/
theorem original_ratio_first_term :
  ∀ x : ℚ,
  (x + 5) / (11 + 5) = 3 / 4 →
  x = 7 := by
sorry

end NUMINAMATH_CALUDE_original_ratio_first_term_l3855_385534


namespace NUMINAMATH_CALUDE_folded_paper_distance_l3855_385523

/-- Given a square sheet of paper with area 18 cm², when folded so that a corner point A
    rests on the diagonal making the visible black area equal to the visible white area,
    the distance from A to its original position is 2√6 cm. -/
theorem folded_paper_distance (s : ℝ) (x : ℝ) :
  s^2 = 18 →
  (1/2) * x^2 = 18 - x^2 →
  Real.sqrt (2 * x^2) = 2 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_folded_paper_distance_l3855_385523


namespace NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_inradius_l3855_385557

theorem right_triangle_arithmetic_progression_inradius (a b c d : ℝ) :
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive side lengths
  a < b ∧ b < c →  -- Ordered side lengths
  a^2 + b^2 = c^2 →  -- Right triangle (Pythagorean theorem)
  b = a + d ∧ c = a + 2*d →  -- Arithmetic progression
  d = (a*b*c) / (a + b + c)  -- d equals inradius
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_arithmetic_progression_inradius_l3855_385557


namespace NUMINAMATH_CALUDE_class_composition_l3855_385564

theorem class_composition (N : ℕ) : 
  (∃ k : ℕ, N = 8 + k) →
  (∀ i : ℕ, i < N → (i < 8 → (8 : ℚ) / (N - 1 : ℚ) < 1/3) ∧ 
                    (i ≥ 8 → (8 : ℚ) / (N - 1 : ℚ) ≥ 1/3)) →
  (N = 23 ∨ N = 24 ∨ N = 25) := by
sorry

end NUMINAMATH_CALUDE_class_composition_l3855_385564


namespace NUMINAMATH_CALUDE_triangle_inequalities_l3855_385518

/-- Theorem about triangle inequalities involving area, side lengths, altitudes, and excircle radii -/
theorem triangle_inequalities (a b c : ℝ) (S : ℝ) (h_a h_b h_c : ℝ) (r_a r_b r_c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_area : S > 0)
  (h_altitudes : h_a > 0 ∧ h_b > 0 ∧ c > 0)
  (h_radii : r_a > 0 ∧ r_b > 0 ∧ r_c > 0) :
  S^3 ≤ (Real.sqrt 3 / 4)^3 * (a * b * c)^2 ∧ 
  3 * h_a * h_b * h_c ≤ 3 * Real.sqrt 3 * S ∧
  3 * Real.sqrt 3 * S ≤ 3 * r_a * r_b * r_c := by
  sorry


end NUMINAMATH_CALUDE_triangle_inequalities_l3855_385518


namespace NUMINAMATH_CALUDE_problem_statement_l3855_385580

theorem problem_statement (x₁ x₂ : ℝ) (h1 : |x₁ - 2| < 1) (h2 : |x₂ - 2| < 1) (h3 : x₁ ≠ x₂) : 
  let f := fun x => x^2 - x + 1
  2 < x₁ + x₂ ∧ x₁ + x₂ < 6 ∧ 
  |x₁ - x₂| < 2 ∧
  |x₁ - x₂| < |f x₁ - f x₂| ∧ |f x₁ - f x₂| < 5 * |x₁ - x₂| := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l3855_385580


namespace NUMINAMATH_CALUDE_complex_number_magnitude_l3855_385599

theorem complex_number_magnitude (a : ℝ) (i : ℂ) (z : ℂ) : 
  a < 0 → 
  i * i = -1 → 
  z = a * i / (1 - 2 * i) → 
  Complex.abs z = Real.sqrt 5 → 
  a = -5 := by sorry

end NUMINAMATH_CALUDE_complex_number_magnitude_l3855_385599


namespace NUMINAMATH_CALUDE_line_curve_intersection_m_bound_l3855_385566

/-- Given a straight line and a curve with a common point, prove that m ≥ 3 -/
theorem line_curve_intersection_m_bound (k : ℝ) (m : ℝ) :
  (∃ x y : ℝ, y = k * x - k + 1 ∧ x^2 + 2 * y^2 = m) →
  m ≥ 3 :=
by sorry

end NUMINAMATH_CALUDE_line_curve_intersection_m_bound_l3855_385566


namespace NUMINAMATH_CALUDE_problems_left_to_grade_l3855_385572

theorem problems_left_to_grade (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) :
  total_worksheets = 14 →
  problems_per_worksheet = 2 →
  graded_worksheets = 7 →
  (total_worksheets - graded_worksheets) * problems_per_worksheet = 14 :=
by sorry

end NUMINAMATH_CALUDE_problems_left_to_grade_l3855_385572


namespace NUMINAMATH_CALUDE_tetrahedron_volume_from_pentagon_tetrahedron_volume_proof_l3855_385554

/-- The volume of a tetrahedron formed from a regular pentagon -/
theorem tetrahedron_volume_from_pentagon (side_length : ℝ) 
  (h_side : side_length = 1) : ℝ :=
let diagonal_length := (1 + Real.sqrt 5) / 2
let base_area := Real.sqrt 3 / 4 * side_length ^ 2
let height := Real.sqrt ((5 + 2 * Real.sqrt 5) / 4)
let volume := (1 / 3) * base_area * height
(1 + Real.sqrt 5) / 24

/-- The theorem statement -/
theorem tetrahedron_volume_proof : 
  ∃ (v : ℝ), tetrahedron_volume_from_pentagon 1 rfl = v ∧ v = (1 + Real.sqrt 5) / 24 := by
sorry

end NUMINAMATH_CALUDE_tetrahedron_volume_from_pentagon_tetrahedron_volume_proof_l3855_385554


namespace NUMINAMATH_CALUDE_identical_solutions_l3855_385520

/-- Two equations have identical solutions when k = 0 -/
theorem identical_solutions (x y k : ℝ) : 
  (y = x^2 ∧ y = 3*x^2 + k) ↔ k = 0 :=
by sorry

end NUMINAMATH_CALUDE_identical_solutions_l3855_385520


namespace NUMINAMATH_CALUDE_pizza_toppings_theorem_l3855_385567

/-- Given a number of pizza flavors and total pizza varieties (including pizzas with and without additional toppings), 
    calculate the number of possible additional toppings. -/
def calculate_toppings (flavors : ℕ) (total_varieties : ℕ) : ℕ :=
  (total_varieties / flavors) - 1

/-- Theorem stating that with 4 pizza flavors and 16 total pizza varieties, 
    there are 3 possible additional toppings. -/
theorem pizza_toppings_theorem :
  calculate_toppings 4 16 = 3 := by
  sorry

#eval calculate_toppings 4 16

end NUMINAMATH_CALUDE_pizza_toppings_theorem_l3855_385567


namespace NUMINAMATH_CALUDE_fraction_problem_l3855_385560

theorem fraction_problem (p q x y : ℚ) :
  p / q = 4 / 5 →
  x / y + (2 * q - p) / (2 * q + p) = 3 →
  x / y = 18 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3855_385560


namespace NUMINAMATH_CALUDE_alex_not_reading_probability_l3855_385528

theorem alex_not_reading_probability :
  let p_reading : ℚ := 5/9
  let p_not_reading : ℚ := 1 - p_reading
  p_not_reading = 4/9 :=
by sorry

end NUMINAMATH_CALUDE_alex_not_reading_probability_l3855_385528


namespace NUMINAMATH_CALUDE_two_enchiladas_five_tacos_cost_l3855_385598

/-- The price of an enchilada in dollars -/
def enchilada_price : ℝ := sorry

/-- The price of a taco in dollars -/
def taco_price : ℝ := sorry

/-- The condition that one enchilada and four tacos cost $3.50 -/
axiom condition1 : enchilada_price + 4 * taco_price = 3.50

/-- The condition that four enchiladas and one taco cost $4.20 -/
axiom condition2 : 4 * enchilada_price + taco_price = 4.20

/-- The theorem stating that two enchiladas and five tacos cost $5.04 -/
theorem two_enchiladas_five_tacos_cost : 
  2 * enchilada_price + 5 * taco_price = 5.04 := by sorry

end NUMINAMATH_CALUDE_two_enchiladas_five_tacos_cost_l3855_385598


namespace NUMINAMATH_CALUDE_rotation_direction_undetermined_l3855_385519

-- Define a type for rotation direction
inductive RotationDirection
| Clockwise
| Counterclockwise

-- Define a type for a quadrilateral
structure Quadrilateral where
  -- We don't need to specify the exact structure of a quadrilateral for this problem
  mk :: 

-- Define a point Z
def Z : ℝ × ℝ := sorry

-- Define the rotation transformation
def rotate (q : Quadrilateral) (center : ℝ × ℝ) (angle : ℝ) : Quadrilateral := sorry

-- State the theorem
theorem rotation_direction_undetermined 
  (q1 q2 : Quadrilateral) 
  (h1 : rotate q1 Z (270 : ℝ) = q2) : 
  ¬ ∃ (d : RotationDirection), d = RotationDirection.Clockwise ∨ d = RotationDirection.Counterclockwise := 
sorry

end NUMINAMATH_CALUDE_rotation_direction_undetermined_l3855_385519


namespace NUMINAMATH_CALUDE_f_three_equals_three_l3855_385579

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_three_equals_three :
  (∀ x, f (2 * x - 1) = x + 1) → f 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_f_three_equals_three_l3855_385579


namespace NUMINAMATH_CALUDE_sector_properties_l3855_385504

/-- Represents a circular sector --/
structure Sector where
  α : Real  -- Central angle in radians
  r : Real  -- Radius
  h_r_pos : r > 0

/-- Calculates the arc length of a sector --/
def arcLength (s : Sector) : Real :=
  s.α * s.r

/-- Calculates the perimeter of a sector --/
def perimeter (s : Sector) : Real :=
  s.r * (s.α + 2)

/-- Calculates the area of a sector --/
def area (s : Sector) : Real :=
  0.5 * s.α * s.r^2

theorem sector_properties :
  ∃ (s1 s2 : Sector),
    s1.α = 2 * Real.pi / 3 ∧
    s1.r = 6 ∧
    arcLength s1 = 4 * Real.pi ∧
    perimeter s2 = 24 ∧
    s2.α = 2 ∧
    area s2 = 36 ∧
    ∀ (s : Sector), perimeter s = 24 → area s ≤ area s2 := by
  sorry

end NUMINAMATH_CALUDE_sector_properties_l3855_385504


namespace NUMINAMATH_CALUDE_bird_bath_frequency_l3855_385543

theorem bird_bath_frequency 
  (num_dogs : ℕ) (num_cats : ℕ) (num_birds : ℕ)
  (dog_baths_per_month : ℕ) (cat_baths_per_month : ℕ)
  (total_baths_per_year : ℕ) :
  num_dogs = 2 →
  num_cats = 3 →
  num_birds = 4 →
  dog_baths_per_month = 2 →
  cat_baths_per_month = 1 →
  total_baths_per_year = 96 →
  (total_baths_per_year - (num_dogs * dog_baths_per_month * 12 + num_cats * cat_baths_per_month * 12)) / num_birds = 3 := by
  sorry

#check bird_bath_frequency

end NUMINAMATH_CALUDE_bird_bath_frequency_l3855_385543


namespace NUMINAMATH_CALUDE_boat_travel_theorem_l3855_385539

/-- Represents the distance traveled by a boat in one hour -/
structure BoatTravel where
  alongStream : ℝ
  againstStream : ℝ

/-- Calculates the stream speed given the boat's still water speed and against-stream distance -/
def streamSpeed (boatSpeed : ℝ) (againstStreamDist : ℝ) : ℝ :=
  boatSpeed - againstStreamDist

/-- Calculates the distance traveled along the stream in one hour -/
def alongStreamDistance (boatSpeed : ℝ) (streamSpeed : ℝ) : ℝ :=
  boatSpeed + streamSpeed

/-- Theorem stating the relationship between boat speed, against-stream distance, and along-stream distance -/
theorem boat_travel_theorem (boatSpeed : ℝ) (travel : BoatTravel) 
    (h1 : boatSpeed = 8)
    (h2 : travel.againstStream = 5) :
    travel.alongStream = 11 := by
  sorry

end NUMINAMATH_CALUDE_boat_travel_theorem_l3855_385539


namespace NUMINAMATH_CALUDE_decreasing_function_inequality_l3855_385571

theorem decreasing_function_inequality (f : ℝ → ℝ) (a : ℝ) :
  (∀ x y, 0 < x ∧ x < y ∧ y < 4 → f x > f y) →  -- f is decreasing on (0,4)
  (0 < a^2 - a ∧ a^2 - a < 4) →                 -- domain condition
  f (a^2 - a) > f 2 →                           -- given inequality
  (-1 < a ∧ a < 0) ∨ (1 < a ∧ a < 2) :=         -- conclusion
by sorry

end NUMINAMATH_CALUDE_decreasing_function_inequality_l3855_385571


namespace NUMINAMATH_CALUDE_half_power_five_decimal_l3855_385556

theorem half_power_five_decimal : (1/2)^5 = 0.03125 := by
  sorry

end NUMINAMATH_CALUDE_half_power_five_decimal_l3855_385556


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_counterexample_l3855_385562

/-- A line in 3D space -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A plane in 3D space -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Two lines intersect -/
def intersect (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to another line -/
def perp_line (l1 l2 : Line3D) : Prop := sorry

/-- A line is perpendicular to a plane -/
def perp_plane (l : Line3D) (p : Plane) : Prop := sorry

/-- A line is within a plane -/
def line_in_plane (l : Line3D) (p : Plane) : Prop := sorry

theorem perpendicular_line_plane_counterexample :
  ∃ (l : Line3D) (p : Plane) (l1 l2 : Line3D),
    line_in_plane l1 p ∧
    line_in_plane l2 p ∧
    intersect l1 l2 ∧
    perp_line l l1 ∧
    perp_line l l2 ∧
    ¬(perp_plane l p) := by sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_counterexample_l3855_385562


namespace NUMINAMATH_CALUDE_b_share_is_2200_l3855_385555

/- Define the investments and a's share -/
def investment_a : ℕ := 7000
def investment_b : ℕ := 11000
def investment_c : ℕ := 18000
def share_a : ℕ := 1400

/- Define the function to calculate b's share -/
def calculate_b_share (inv_a inv_b inv_c share_a : ℕ) : ℕ :=
  let total_ratio := inv_a + inv_b + inv_c
  let total_profit := share_a * total_ratio / inv_a
  inv_b * total_profit / total_ratio

/- Theorem statement -/
theorem b_share_is_2200 : 
  calculate_b_share investment_a investment_b investment_c share_a = 2200 := by
  sorry


end NUMINAMATH_CALUDE_b_share_is_2200_l3855_385555


namespace NUMINAMATH_CALUDE_at_least_one_non_defective_l3855_385544

def probability_non_defective (p_defective : ℝ) : ℝ :=
  1 - p_defective^3

theorem at_least_one_non_defective :
  probability_non_defective 0.3 = 0.973 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_non_defective_l3855_385544
