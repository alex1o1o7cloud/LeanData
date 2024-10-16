import Mathlib

namespace NUMINAMATH_CALUDE_crayons_erasers_difference_l2236_223683

/-- Given the initial numbers of crayons and erasers, and the final number of crayons,
    prove that the difference between the number of crayons left and the number of erasers is 66. -/
theorem crayons_erasers_difference 
  (initial_crayons : ℕ) 
  (initial_erasers : ℕ) 
  (final_crayons : ℕ) 
  (h1 : initial_crayons = 617) 
  (h2 : initial_erasers = 457) 
  (h3 : final_crayons = 523) : 
  final_crayons - initial_erasers = 66 := by
  sorry

#check crayons_erasers_difference

end NUMINAMATH_CALUDE_crayons_erasers_difference_l2236_223683


namespace NUMINAMATH_CALUDE_down_payment_equals_108000_l2236_223638

/-- The amount of money needed for a down payment on a house -/
def down_payment (richard_monthly_savings : ℕ) (sarah_monthly_savings : ℕ) (years : ℕ) : ℕ :=
  (richard_monthly_savings + sarah_monthly_savings) * years * 12

/-- Theorem stating that Richard and Sarah's savings over 3 years equal $108,000 -/
theorem down_payment_equals_108000 :
  down_payment 1500 1500 3 = 108000 := by
  sorry

end NUMINAMATH_CALUDE_down_payment_equals_108000_l2236_223638


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2236_223669

-- Define the surface area of the cube
def surface_area : ℝ := 150

-- Theorem stating that a cube with surface area 150 has volume 125
theorem cube_volume_from_surface_area :
  ∃ (s : ℝ), s > 0 ∧ 6 * s^2 = surface_area ∧ s^3 = 125 :=
by sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2236_223669


namespace NUMINAMATH_CALUDE_triangle_min_ab_value_l2236_223670

theorem triangle_min_ab_value (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  (2 * c * Real.cos B = 2 * a + b) →
  (1 / 2 * c = 1 / 2 * a * b * Real.sin C) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 → a' * b' ≥ a * b) →
  a * b = 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_min_ab_value_l2236_223670


namespace NUMINAMATH_CALUDE_sum_of_four_numbers_l2236_223615

theorem sum_of_four_numbers : 1234 + 2341 + 3412 + 4123 = 11110 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_numbers_l2236_223615


namespace NUMINAMATH_CALUDE_right_triangle_with_inscribed_circle_l2236_223653

theorem right_triangle_with_inscribed_circle (d : ℝ) (h : d > 0) :
  ∃ (a b c : ℝ),
    b = a + d ∧
    c = b + d ∧
    a^2 + b^2 = c^2 ∧
    (a + b - c) / 2 = d :=
by
  sorry

#check right_triangle_with_inscribed_circle

end NUMINAMATH_CALUDE_right_triangle_with_inscribed_circle_l2236_223653


namespace NUMINAMATH_CALUDE_woodworker_extra_parts_l2236_223695

/-- A woodworker's production scenario -/
structure WoodworkerProduction where
  normal_days : ℕ
  normal_parts : ℕ
  productivity_increase : ℕ
  new_days : ℕ

/-- Calculate the extra parts made by the woodworker -/
def extra_parts (w : WoodworkerProduction) : ℕ :=
  let normal_daily := w.normal_parts / w.normal_days
  let new_daily := normal_daily + w.productivity_increase
  new_daily * w.new_days - w.normal_parts

/-- Theorem stating the extra parts made by the woodworker -/
theorem woodworker_extra_parts :
  ∀ (w : WoodworkerProduction),
    w.normal_days = 24 ∧
    w.normal_parts = 360 ∧
    w.productivity_increase = 5 ∧
    w.new_days = 22 →
    extra_parts w = 80 := by
  sorry

end NUMINAMATH_CALUDE_woodworker_extra_parts_l2236_223695


namespace NUMINAMATH_CALUDE_pascal_row_10_sum_l2236_223657

/-- The sum of numbers in a row of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^n

/-- Theorem: The sum of numbers in Row 10 of Pascal's Triangle is 1024 -/
theorem pascal_row_10_sum : pascal_row_sum 10 = 1024 := by
  sorry

end NUMINAMATH_CALUDE_pascal_row_10_sum_l2236_223657


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2236_223676

/-- An equilateral triangle with perimeter 69 cm has sides of length 23 cm -/
theorem equilateral_triangle_side_length (triangle : Set ℝ) (perimeter : ℝ) :
  perimeter = 69 →
  ∃ (side_length : ℝ), side_length * 3 = perimeter ∧ side_length = 23 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l2236_223676


namespace NUMINAMATH_CALUDE_particle_probability_l2236_223624

/-- The probability of a particle reaching (0,0) from (x,y) -/
def P (x y : ℕ) : ℚ :=
  if x = 0 ∧ y = 0 then 1
  else if x = 0 ∨ y = 0 then 0
  else (1/3) * P (x-1) y + (1/3) * P x (y-1) + (1/3) * P (x-1) (y-1)

theorem particle_probability :
  P 5 5 = 793 / 6561 :=
sorry

end NUMINAMATH_CALUDE_particle_probability_l2236_223624


namespace NUMINAMATH_CALUDE_max_square_plots_l2236_223688

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def availableFence : ℕ := 2400

/-- Calculates the number of square plots along the field's width -/
def numPlotsWidth (field : FieldDimensions) : ℕ :=
  20

/-- Calculates the number of square plots along the field's length -/
def numPlotsLength (field : FieldDimensions) : ℕ :=
  30

/-- Calculates the total number of square plots -/
def totalPlots (field : FieldDimensions) : ℕ :=
  numPlotsWidth field * numPlotsLength field

/-- Calculates the length of internal fencing used -/
def usedFence (field : FieldDimensions) : ℕ :=
  field.width * (numPlotsLength field - 1) + field.length * (numPlotsWidth field - 1)

/-- Theorem stating that 600 is the maximum number of square plots -/
theorem max_square_plots (field : FieldDimensions) 
    (h1 : field.width = 40) 
    (h2 : field.length = 60) : 
    totalPlots field = 600 ∧ 
    usedFence field ≤ availableFence ∧ 
    ∀ n m : ℕ, n * m > 600 → 
      field.width * (m - 1) + field.length * (n - 1) > availableFence :=
  sorry

#check max_square_plots

end NUMINAMATH_CALUDE_max_square_plots_l2236_223688


namespace NUMINAMATH_CALUDE_intersection_M_N_l2236_223655

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 + x = 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2236_223655


namespace NUMINAMATH_CALUDE_correct_calculation_l2236_223632

theorem correct_calculation (x : ℝ) : (x / 7 = 49) → (x * 6 = 2058) := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l2236_223632


namespace NUMINAMATH_CALUDE_appointment_schemes_count_l2236_223607

def total_students : ℕ := 9
def male_students : ℕ := 5
def female_students : ℕ := 4
def students_to_select : ℕ := 3

def permutations (n : ℕ) (r : ℕ) : ℕ :=
  Nat.factorial n / Nat.factorial (n - r)

theorem appointment_schemes_count :
  permutations total_students students_to_select -
  (permutations male_students students_to_select +
   permutations female_students students_to_select) = 420 := by
  sorry

end NUMINAMATH_CALUDE_appointment_schemes_count_l2236_223607


namespace NUMINAMATH_CALUDE_M_divisible_by_49_l2236_223684

/-- M is the concatenated number formed by writing integers from 1 to 48 in order -/
def M : ℕ := sorry

/-- Theorem stating that M is divisible by 49 -/
theorem M_divisible_by_49 : 49 ∣ M := by sorry

end NUMINAMATH_CALUDE_M_divisible_by_49_l2236_223684


namespace NUMINAMATH_CALUDE_negative_fractions_comparison_l2236_223681

theorem negative_fractions_comparison : -2/3 > -3/4 := by
  sorry

end NUMINAMATH_CALUDE_negative_fractions_comparison_l2236_223681


namespace NUMINAMATH_CALUDE_range_of_a_over_b_l2236_223611

def ellipse (x y : ℝ) : Prop := x^2 / 6 + y^2 / 2 = 1

def line (a b x y : ℝ) : Prop := a * x + b * y = 2

theorem range_of_a_over_b (a b : ℝ) :
  a^2 + b^2 = 1 →
  b ≠ 0 →
  (∃ x y : ℝ, ellipse x y ∧ line a b x y) →
  (a / b < -1 ∨ a / b = -1 ∨ a / b = 1 ∨ a / b > 1) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_over_b_l2236_223611


namespace NUMINAMATH_CALUDE_valid_numbers_divisible_by_36_l2236_223692

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ n = 52000 + a * 100 + 20 + b

def is_divisible_by_36 (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 36 * k

theorem valid_numbers_divisible_by_36 :
  ∀ n : ℕ, is_valid_number n ∧ is_divisible_by_36 n ↔ 
    n = 52524 ∨ n = 52128 ∨ n = 52020 ∨ n = 52920 :=
by sorry

end NUMINAMATH_CALUDE_valid_numbers_divisible_by_36_l2236_223692


namespace NUMINAMATH_CALUDE_marco_new_cards_l2236_223661

/-- Given a total number of cards, calculate the number of new cards obtained by trading
    one-fifth of the duplicate cards, where duplicates are one-fourth of the total. -/
def new_cards (total : ℕ) : ℕ :=
  let duplicates := total / 4
  duplicates / 5

theorem marco_new_cards :
  new_cards 500 = 25 := by sorry

end NUMINAMATH_CALUDE_marco_new_cards_l2236_223661


namespace NUMINAMATH_CALUDE_binomial_coefficient_third_term_l2236_223621

theorem binomial_coefficient_third_term (a b : ℝ) : 
  Nat.choose 6 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_third_term_l2236_223621


namespace NUMINAMATH_CALUDE_rectangle_perimeter_theorem_l2236_223675

theorem rectangle_perimeter_theorem (a b : ℝ) (h : a > 0 ∧ b > 0) :
  a * b > 2 * (a + b) → 2 * (a + b) > 16 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_theorem_l2236_223675


namespace NUMINAMATH_CALUDE_equation_solution_l2236_223600

theorem equation_solution : 
  ∃ x : ℝ, (2 * x / (x - 2) + 3 / (2 - x) = 1) ∧ (x = 1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2236_223600


namespace NUMINAMATH_CALUDE_frog_corner_probability_l2236_223671

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Represents a direction of hop -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- The grid on which the frog hops -/
def Grid := Fin 4 → Fin 4 → Bool

/-- Checks if a position is a corner -/
def isCorner (p : Position) : Bool :=
  (p.x = 0 ∧ p.y = 0) ∨ (p.x = 0 ∧ p.y = 3) ∨ (p.x = 3 ∧ p.y = 0) ∨ (p.x = 3 ∧ p.y = 3)

/-- Performs a single hop in the given direction with wrap-around -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up    => ⟨p.x, (p.y + 1) % 4⟩
  | Direction.Down  => ⟨p.x, (p.y - 1 + 4) % 4⟩
  | Direction.Left  => ⟨(p.x - 1 + 4) % 4, p.y⟩
  | Direction.Right => ⟨(p.x + 1) % 4, p.y⟩

/-- Calculates the probability of reaching a corner within n hops -/
def probReachCorner (start : Position) (n : Nat) : Rat :=
  sorry

/-- The main theorem to prove -/
theorem frog_corner_probability :
  probReachCorner ⟨1, 1⟩ 5 = 15/16 :=
sorry

end NUMINAMATH_CALUDE_frog_corner_probability_l2236_223671


namespace NUMINAMATH_CALUDE_sum_complex_exp_argument_l2236_223687

/-- The argument of the sum of five complex exponentials -/
theorem sum_complex_exp_argument :
  let z : ℂ := Complex.exp (11 * Real.pi * Complex.I / 100) +
               Complex.exp (31 * Real.pi * Complex.I / 100) +
               Complex.exp (51 * Real.pi * Complex.I / 100) +
               Complex.exp (71 * Real.pi * Complex.I / 100) +
               Complex.exp (91 * Real.pi * Complex.I / 100)
  0 ≤ Complex.arg z ∧ Complex.arg z < 2 * Real.pi →
  Complex.arg z = 51 * Real.pi / 100 :=
by sorry

end NUMINAMATH_CALUDE_sum_complex_exp_argument_l2236_223687


namespace NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l2236_223613

theorem max_value_x_cubed_over_y_fourth (x y : ℝ) 
  (h1 : 3 ≤ x * y^2) (h2 : x * y^2 ≤ 8) 
  (h3 : 4 ≤ x^2 / y) (h4 : x^2 / y ≤ 9) : 
  (∃ (a b : ℝ), 3 ≤ a * b^2 ∧ a * b^2 ≤ 8 ∧ 4 ≤ a^2 / b ∧ a^2 / b ≤ 9 ∧ a^3 / b^4 = 27) ∧ 
  (∀ (z w : ℝ), 3 ≤ z * w^2 → z * w^2 ≤ 8 → 4 ≤ z^2 / w → z^2 / w ≤ 9 → z^3 / w^4 ≤ 27) :=
sorry

end NUMINAMATH_CALUDE_max_value_x_cubed_over_y_fourth_l2236_223613


namespace NUMINAMATH_CALUDE_binary_difference_digits_l2236_223639

theorem binary_difference_digits : ∃ (b : ℕ → Bool), 
  (Nat.castRingHom ℕ).toFun ((Nat.digits 2 1500).foldl (λ acc d => 2 * acc + d) 0 - 
                              (Nat.digits 2 300).foldl (λ acc d => 2 * acc + d) 0) = 
  (Nat.digits 2 1200).foldl (λ acc d => 2 * acc + d) 0 ∧
  (Nat.digits 2 1200).length = 11 :=
by sorry

end NUMINAMATH_CALUDE_binary_difference_digits_l2236_223639


namespace NUMINAMATH_CALUDE_candies_left_l2236_223673

def candies_bought_tuesday : ℕ := 3
def candies_bought_thursday : ℕ := 5
def candies_bought_friday : ℕ := 2
def candies_eaten : ℕ := 6

theorem candies_left : 
  candies_bought_tuesday + candies_bought_thursday + candies_bought_friday - candies_eaten = 4 := by
  sorry

end NUMINAMATH_CALUDE_candies_left_l2236_223673


namespace NUMINAMATH_CALUDE_fold_crease_length_l2236_223605

/-- Represents a rectangular sheet of paper -/
structure Sheet :=
  (length : ℝ)
  (width : ℝ)

/-- Represents the crease formed by folding the sheet -/
def crease_length (s : Sheet) : ℝ :=
  sorry

/-- The theorem stating the length of the crease -/
theorem fold_crease_length (s : Sheet) 
  (h1 : s.length = 8) 
  (h2 : s.width = 6) : 
  crease_length s = 7.5 :=
sorry

end NUMINAMATH_CALUDE_fold_crease_length_l2236_223605


namespace NUMINAMATH_CALUDE_min_value_theorem_l2236_223612

theorem min_value_theorem (x : ℝ) (h : x > -1) : 
  x + 4 / (x + 1) ≥ 3 ∧ ∃ y > -1, y + 4 / (y + 1) = 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l2236_223612


namespace NUMINAMATH_CALUDE_two_vertices_same_degree_l2236_223601

-- Define a graph
def Graph (α : Type) := α → α → Prop

-- Define the degree of a vertex in a graph
def degree {α : Type} (G : Graph α) (v : α) : ℕ := sorry

theorem two_vertices_same_degree {α : Type} (G : Graph α) (n : ℕ) (h : Fintype α) :
  (Fintype.card α = n) →
  (∀ v : α, degree G v < n) →
  ∃ u v : α, u ≠ v ∧ degree G u = degree G v :=
sorry

end NUMINAMATH_CALUDE_two_vertices_same_degree_l2236_223601


namespace NUMINAMATH_CALUDE_initial_money_calculation_l2236_223697

theorem initial_money_calculation (initial_money : ℚ) : 
  (2/5 : ℚ) * initial_money = 200 → initial_money = 500 := by
  sorry

end NUMINAMATH_CALUDE_initial_money_calculation_l2236_223697


namespace NUMINAMATH_CALUDE_trapezium_side_length_l2236_223663

/-- Given a trapezium with the specified properties, prove that the length of the unknown parallel side is 20 cm. -/
theorem trapezium_side_length 
  (known_side : ℝ) 
  (height : ℝ) 
  (area : ℝ) 
  (h1 : known_side = 18) 
  (h2 : height = 12) 
  (h3 : area = 228) : 
  ∃ unknown_side : ℝ, 
    area = (1/2) * (known_side + unknown_side) * height ∧ 
    unknown_side = 20 :=
sorry

end NUMINAMATH_CALUDE_trapezium_side_length_l2236_223663


namespace NUMINAMATH_CALUDE_tea_maker_capacity_l2236_223622

/-- A cylindrical tea maker with capacity x cups -/
structure TeaMaker where
  capacity : ℝ
  cylindrical : Bool

/-- Theorem: A cylindrical tea maker that contains 54 cups when 45% full has a total capacity of 120 cups -/
theorem tea_maker_capacity (tm : TeaMaker) (h1 : tm.cylindrical = true) 
    (h2 : 0.45 * tm.capacity = 54) : tm.capacity = 120 := by
  sorry

end NUMINAMATH_CALUDE_tea_maker_capacity_l2236_223622


namespace NUMINAMATH_CALUDE_min_perimeter_triangle_l2236_223678

theorem min_perimeter_triangle (a b x : ℕ) (ha : a = 24) (hb : b = 37) : 
  (a + b + x > a + b ∧ a + x > b ∧ b + x > a) → 
  (∀ y : ℕ, (a + b + y > a + b ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 75 := by sorry

end NUMINAMATH_CALUDE_min_perimeter_triangle_l2236_223678


namespace NUMINAMATH_CALUDE_polynomial_roots_l2236_223656

theorem polynomial_roots : 
  let f : ℝ → ℝ := λ x => (x + 1998) * (x + 1999) * (x + 2000) * (x + 2001) + 1
  ∀ x : ℝ, f x = 0 ↔ x = -1999.5 - Real.sqrt 5 / 2 ∨ x = -1999.5 + Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l2236_223656


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l2236_223604

theorem quadratic_equation_properties (m : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 - m*x - 1
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0) ∧
  (f (Real.sqrt 2) = 0 → f (-Real.sqrt 2 / 2) = 0 ∧ m = Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l2236_223604


namespace NUMINAMATH_CALUDE_container_volume_ratio_l2236_223662

theorem container_volume_ratio : 
  ∀ (A B C : ℚ),
  (4/5 : ℚ) * A = (3/5 : ℚ) * B →
  (3/5 : ℚ) * B = (3/4 : ℚ) * C →
  A / C = (15/16 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_container_volume_ratio_l2236_223662


namespace NUMINAMATH_CALUDE_prime_between_30_and_40_with_remainder_1_mod_6_l2236_223694

theorem prime_between_30_and_40_with_remainder_1_mod_6 (n : ℕ) 
  (prime_n : Nat.Prime n) 
  (range_n : 30 < n ∧ n < 40) 
  (mod_n : n % 6 = 1) : 
  n = 37 := by
  sorry

end NUMINAMATH_CALUDE_prime_between_30_and_40_with_remainder_1_mod_6_l2236_223694


namespace NUMINAMATH_CALUDE_inequality_iff_in_interval_l2236_223679

/-- The roots of the quadratic equation x^2 - (16/5)x - 8 = 0 --/
def a : ℝ := sorry
def b : ℝ := sorry

axiom a_lt_b : a < b
axiom b_lt_zero : b < 0
axiom roots_property : ∀ x : ℝ, x^2 - (16/5) * x - 8 = 0 ↔ (x = a ∨ x = b)

/-- The main theorem stating the equivalence between the inequality and the solution interval --/
theorem inequality_iff_in_interval (x : ℝ) : 
  1 / (x^2 + 2) + 1 / 2 > 5 / x + 21 / 10 ↔ (x < a ∨ (b < x ∧ x < 0)) :=
sorry

end NUMINAMATH_CALUDE_inequality_iff_in_interval_l2236_223679


namespace NUMINAMATH_CALUDE_complex_reciprocal_sum_l2236_223652

theorem complex_reciprocal_sum (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
sorry

end NUMINAMATH_CALUDE_complex_reciprocal_sum_l2236_223652


namespace NUMINAMATH_CALUDE_trigonometric_expression_equals_four_l2236_223690

theorem trigonometric_expression_equals_four : 
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
sorry

end NUMINAMATH_CALUDE_trigonometric_expression_equals_four_l2236_223690


namespace NUMINAMATH_CALUDE_perpendicular_parallel_relations_l2236_223648

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)

variable (α : Plane) (a b : Line)

-- State the theorem
theorem perpendicular_parallel_relations :
  (∀ a b : Line, ∀ α : Plane,
    (parallel a b ∧ perpendicular_line_plane a α → perpendicular_line_plane b α)) ∧
  (∀ a b : Line, ∀ α : Plane,
    (perpendicular_line_plane a α ∧ perpendicular_line_plane b α → parallel a b)) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_parallel_relations_l2236_223648


namespace NUMINAMATH_CALUDE_angle_complement_supplement_l2236_223664

theorem angle_complement_supplement (x : ℝ) : 
  (90 - x) = 4 * (180 - x) → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_angle_complement_supplement_l2236_223664


namespace NUMINAMATH_CALUDE_radical_simplification_l2236_223689

theorem radical_simplification (p : ℝ) :
  Real.sqrt (42 * p^2) * Real.sqrt (7 * p^2) * Real.sqrt (14 * p^2) = 14 * p^3 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l2236_223689


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l2236_223691

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point is in the fourth quadrant -/
def is_in_fourth_quadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Predicate to check if two points are symmetric with respect to the origin -/
def is_symmetric_to_origin (p1 p2 : Point) : Prop :=
  p1.x = -p2.x ∧ p1.y = -p2.y

/-- Theorem stating that if a point is in the fourth quadrant and symmetric to the origin,
    its symmetric point has negative x and positive y coordinates -/
theorem symmetric_point_coordinates (p : Point) :
  is_in_fourth_quadrant p → ∃ q : Point, is_symmetric_to_origin p q ∧ q.x < 0 ∧ q.y > 0 := by
  sorry


end NUMINAMATH_CALUDE_symmetric_point_coordinates_l2236_223691


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2236_223642

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_60_l2236_223642


namespace NUMINAMATH_CALUDE_ratio_sum_in_triangle_l2236_223614

/-- Given a triangle ABC with the following properties:
  - B is the midpoint of AC
  - D divides BC such that BD:DC = 2:1
  - E divides AB such that AE:EB = 1:3
  This theorem proves that the sum of the ratios EF/FC + AF/FD equals 13/4 -/
theorem ratio_sum_in_triangle (A B C D E F : ℝ × ℝ) : 
  let midpoint (P Q : ℝ × ℝ) := ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)
  let divide_segment (P Q : ℝ × ℝ) (r s : ℝ) := 
    ((r * Q.1 + s * P.1) / (r + s), (r * Q.2 + s * P.2) / (r + s))
  B = midpoint A C ∧
  D = divide_segment B C 2 1 ∧
  E = divide_segment A B 1 3 →
  let EF := ‖E - F‖
  let FC := ‖F - C‖
  let AF := ‖A - F‖
  let FD := ‖F - D‖
  EF / FC + AF / FD = 13 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ratio_sum_in_triangle_l2236_223614


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2236_223693

-- Define the universe type
inductive Universe : Type
  | a | b | c | d | e

-- Define the sets
def I : Set Universe := {Universe.a, Universe.b, Universe.c, Universe.d, Universe.e}
def M : Set Universe := {Universe.a, Universe.b, Universe.c}
def N : Set Universe := {Universe.b, Universe.d, Universe.e}

-- State the theorem
theorem complement_M_intersect_N :
  (I \ M) ∩ N = {Universe.d, Universe.e} := by
  sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2236_223693


namespace NUMINAMATH_CALUDE_aunt_gave_109_l2236_223625

/-- The amount of money Paula's aunt gave her -/
def money_from_aunt (shirt_cost shirt_count pant_cost money_left : ℕ) : ℕ :=
  shirt_cost * shirt_count + pant_cost + money_left

/-- Proof that Paula's aunt gave her $109 -/
theorem aunt_gave_109 :
  money_from_aunt 11 2 13 74 = 109 := by
  sorry

end NUMINAMATH_CALUDE_aunt_gave_109_l2236_223625


namespace NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_l2236_223634

/-- The sum of the first n positive even integers -/
def sumFirstEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- The sum of five consecutive even integers starting from n -/
def sumFiveConsecutiveEven (n : ℕ) : ℕ := 5 * n + 20

theorem smallest_of_five_consecutive_even :
  ∃ (n : ℕ), 
    sumFirstEvenIntegers 30 = sumFiveConsecutiveEven n ∧
    n = 182 ∧
    ∀ (m : ℕ), sumFirstEvenIntegers 30 = sumFiveConsecutiveEven m → n ≤ m :=
by sorry

end NUMINAMATH_CALUDE_smallest_of_five_consecutive_even_l2236_223634


namespace NUMINAMATH_CALUDE_complex_vector_sum_l2236_223666

theorem complex_vector_sum (z₁ z₂ z₃ : ℂ) (x y : ℝ) :
  z₁ = -1 + 2*Complex.I →
  z₂ = 1 - Complex.I →
  z₃ = 3 - 2*Complex.I →
  z₃ = x * z₁ + y * z₂ →
  x + y = 5 := by sorry

end NUMINAMATH_CALUDE_complex_vector_sum_l2236_223666


namespace NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l2236_223640

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

/-- The function f(x) = -x^2 + 2(m-1)x + 3 -/
def f (m : ℝ) (x : ℝ) : ℝ := -x^2 + 2*(m-1)*x + 3

theorem even_function_implies_m_equals_one :
  ∀ m : ℝ, IsEven (f m) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_even_function_implies_m_equals_one_l2236_223640


namespace NUMINAMATH_CALUDE_geometric_mean_solution_l2236_223685

theorem geometric_mean_solution : ∃! k : ℝ, (2 * k) ^ 2 = k * (k + 3) := by
  sorry

end NUMINAMATH_CALUDE_geometric_mean_solution_l2236_223685


namespace NUMINAMATH_CALUDE_combine_like_terms_1_combine_like_terms_2_l2236_223641

-- Problem 1
theorem combine_like_terms_1 (a : ℝ) :
  2*a^2 - 3*a - 5 + 4*a + a^2 = 3*a^2 + a - 5 := by sorry

-- Problem 2
theorem combine_like_terms_2 (m n : ℝ) :
  2*m^2 + 5/2*n^2 - 1/3*(m^2 - 6*n^2) = 5/3*m^2 + 9/2*n^2 := by sorry

end NUMINAMATH_CALUDE_combine_like_terms_1_combine_like_terms_2_l2236_223641


namespace NUMINAMATH_CALUDE_characterization_of_solution_l2236_223667

/-- A real-valued function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (x + y) * (f x - f y) = (x - y) * f (x + y)

/-- The theorem stating that any function satisfying the equation must be of the form ax^2 + bx -/
theorem characterization_of_solution :
  ∀ f : ℝ → ℝ, SatisfiesEquation f →
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x^2 + b * x :=
sorry

end NUMINAMATH_CALUDE_characterization_of_solution_l2236_223667


namespace NUMINAMATH_CALUDE_quadratic_root_difference_l2236_223686

theorem quadratic_root_difference (p q : ℝ) (hp : p > 0) (hq : q > 0) :
  (∀ x : ℝ, x^2 + p*x + q = 0 → 
    ∃ y : ℝ, y^2 + p*y + q = 0 ∧ |x - y| = 2) →
  p = 2 * Real.sqrt (q + 1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_root_difference_l2236_223686


namespace NUMINAMATH_CALUDE_problem_solution_l2236_223660

noncomputable section

def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (0, 1)
def C (θ : ℝ) : ℝ × ℝ := (2 * Real.sin θ, Real.cos θ)
def O : ℝ × ℝ := (0, 0)

def vec (p q : ℝ × ℝ) : ℝ × ℝ := (q.1 - p.1, q.2 - p.2)

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

def vec_length (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

theorem problem_solution (θ : ℝ) :
  (vec_length (vec A (C θ)) = vec_length (vec B (C θ)) → Real.tan θ = 1/2) ∧
  (dot_product (vec O A + 2 • vec O B) (vec O (C θ)) = 1 → Real.sin θ * Real.cos θ = -3/8) := by
  sorry

end

end NUMINAMATH_CALUDE_problem_solution_l2236_223660


namespace NUMINAMATH_CALUDE_S_value_S_approx_l2236_223623

/-- Define the sum S as a function of n, where n is the number of terms -/
def S (n : ℕ) : ℚ :=
  let rec aux (k : ℕ) : ℚ :=
    if k = 0 then 5005
    else (5005 - k : ℚ) + (1/2) * aux (k-1)
  aux n

/-- The main theorem stating that S(5000) is equal to 5009 - (1/2^5000) -/
theorem S_value : S 5000 = 5009 - (1/2)^5000 := by
  sorry

/-- Corollary stating that S(5000) is approximately equal to 5009 -/
theorem S_approx : abs (S 5000 - 5009) < 1 := by
  sorry

end NUMINAMATH_CALUDE_S_value_S_approx_l2236_223623


namespace NUMINAMATH_CALUDE_journal_problem_formula_l2236_223628

def f (x y : ℕ) : ℕ := 5 * x + 60 * (y - 1970) - 4

theorem journal_problem_formula (x y : ℕ) 
  (hx : 1 ≤ x ∧ x ≤ 12) (hy : 1970 ≤ y ∧ y ≤ 1989) : 
  (f 1 1970 = 1) ∧
  (∀ x' y', 1 ≤ x' ∧ x' < 12 → f (x' + 1) y' = f x' y' + 5) ∧
  (∀ y', f 1 (y' + 1) = f 1 y' + 60) →
  f x y = 5 * x + 60 * (y - 1970) - 4 :=
by sorry

end NUMINAMATH_CALUDE_journal_problem_formula_l2236_223628


namespace NUMINAMATH_CALUDE_inequality_theorem_l2236_223620

theorem inequality_theorem (a b c : ℝ) (h : a > b) : a * c^2 ≥ b * c^2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_theorem_l2236_223620


namespace NUMINAMATH_CALUDE_james_tin_collection_l2236_223618

/-- The number of tins James collected on the first day -/
def first_day_tins : ℕ := sorry

/-- The total number of tins James collected in a week -/
def total_tins : ℕ := 500

/-- The number of tins James collected on each of the last four days -/
def last_four_days_tins : ℕ := 50

theorem james_tin_collection :
  first_day_tins = 50 ∧
  first_day_tins +
  (3 * first_day_tins) +
  (3 * first_day_tins - 50) +
  (4 * last_four_days_tins) = total_tins :=
sorry

end NUMINAMATH_CALUDE_james_tin_collection_l2236_223618


namespace NUMINAMATH_CALUDE_leanna_money_l2236_223696

/-- The amount of money Leanna has to spend -/
def total_money : ℕ := 37

/-- The price of a CD -/
def cd_price : ℕ := 14

/-- The price of a cassette -/
def cassette_price : ℕ := 9

/-- Leanna can spend all her money on two CDs and a cassette -/
axiom scenario1 : 2 * cd_price + cassette_price = total_money

/-- Leanna can buy one CD and two cassettes and have $5 left over -/
axiom scenario2 : cd_price + 2 * cassette_price + 5 = total_money

theorem leanna_money : total_money = 37 := by
  sorry

end NUMINAMATH_CALUDE_leanna_money_l2236_223696


namespace NUMINAMATH_CALUDE_chessboard_coloring_theorem_l2236_223646

/-- Represents a coloring of a chessboard -/
def Coloring (n k : ℕ) := Fin (2*n) → Fin k → Fin n

/-- Checks if a coloring has a monochromatic rectangle -/
def has_monochromatic_rectangle (n k : ℕ) (c : Coloring n k) : Prop :=
  ∃ (r₁ r₂ : Fin (2*n)) (c₁ c₂ : Fin k),
    r₁ ≠ r₂ ∧ c₁ ≠ c₂ ∧
    c r₁ c₁ = c r₁ c₂ ∧ c r₁ c₁ = c r₂ c₁ ∧ c r₁ c₁ = c r₂ c₂

/-- The main theorem -/
theorem chessboard_coloring_theorem (n : ℕ) (h : n > 0) :
  ∀ k : ℕ, (k ≥ n*(2*n-1) + 1) →
    ∀ c : Coloring n k, has_monochromatic_rectangle n k c :=
sorry

end NUMINAMATH_CALUDE_chessboard_coloring_theorem_l2236_223646


namespace NUMINAMATH_CALUDE_math_contest_score_difference_l2236_223644

theorem math_contest_score_difference (score60 score75 score85 score95 : ℝ)
  (percent60 percent75 percent85 percent95 : ℝ)
  (h1 : score60 = 60)
  (h2 : score75 = 75)
  (h3 : score85 = 85)
  (h4 : score95 = 95)
  (h5 : percent60 = 0.2)
  (h6 : percent75 = 0.4)
  (h7 : percent85 = 0.25)
  (h8 : percent95 = 0.15)
  (h9 : percent60 + percent75 + percent85 + percent95 = 1) :
  let median := score75
  let mean := percent60 * score60 + percent75 * score75 + percent85 * score85 + percent95 * score95
  median - mean = -2.5 := by
  sorry

end NUMINAMATH_CALUDE_math_contest_score_difference_l2236_223644


namespace NUMINAMATH_CALUDE_inequality_preservation_l2236_223654

theorem inequality_preservation (a b c : ℝ) (h : a > b) :
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_preservation_l2236_223654


namespace NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2236_223617

theorem stewart_farm_sheep_count :
  ∀ (sheep horses : ℕ),
  sheep / horses = 3 / 7 →
  horses * 230 = 12880 →
  sheep = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_sheep_count_l2236_223617


namespace NUMINAMATH_CALUDE_substitution_result_l2236_223680

theorem substitution_result (x y : ℝ) :
  y = x - 1 ∧ x + 2*y = 7 → x + 2*x - 2 = 7 := by
  sorry

end NUMINAMATH_CALUDE_substitution_result_l2236_223680


namespace NUMINAMATH_CALUDE_vector_dot_product_l2236_223610

theorem vector_dot_product (a b : ℝ × ℝ) :
  a + b = (2, -4) →
  3 • a - b = (-10, 16) →
  a • b = -29 := by
sorry

end NUMINAMATH_CALUDE_vector_dot_product_l2236_223610


namespace NUMINAMATH_CALUDE_other_items_sales_percentage_l2236_223602

theorem other_items_sales_percentage 
  (total_sales_percentage : ℝ)
  (notebooks_sales_percentage : ℝ)
  (markers_sales_percentage : ℝ)
  (h1 : total_sales_percentage = 100)
  (h2 : notebooks_sales_percentage = 42)
  (h3 : markers_sales_percentage = 21) :
  total_sales_percentage - (notebooks_sales_percentage + markers_sales_percentage) = 37 := by
  sorry

end NUMINAMATH_CALUDE_other_items_sales_percentage_l2236_223602


namespace NUMINAMATH_CALUDE_a_minus_b_pow_2014_l2236_223631

theorem a_minus_b_pow_2014 (a b : ℝ) 
  (ha : a^3 - 6*a^2 + 15*a = 9) 
  (hb : b^3 - 3*b^2 + 6*b = -1) : 
  (a - b)^2014 = 1 := by sorry

end NUMINAMATH_CALUDE_a_minus_b_pow_2014_l2236_223631


namespace NUMINAMATH_CALUDE_ponce_lighter_than_jalen_l2236_223609

/-- Represents the weights of three people and their relationships. -/
structure WeightProblem where
  ishmael : ℝ
  ponce : ℝ
  jalen : ℝ
  ishmael_heavier : ishmael = ponce + 20
  jalen_weight : jalen = 160
  average_weight : (ishmael + ponce + jalen) / 3 = 160

/-- Theorem stating that Ponce is 10 pounds lighter than Jalen. -/
theorem ponce_lighter_than_jalen (w : WeightProblem) : w.jalen - w.ponce = 10 := by
  sorry

#check ponce_lighter_than_jalen

end NUMINAMATH_CALUDE_ponce_lighter_than_jalen_l2236_223609


namespace NUMINAMATH_CALUDE_equation_describes_cone_l2236_223635

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ

/-- Definition of a cone in spherical coordinates -/
def IsCone (c : ℝ) (f : SphericalCoord → Prop) : Prop :=
  c > 0 ∧ ∀ p : SphericalCoord, f p ↔ p.ρ = c * Real.sin p.φ

/-- The main theorem: the equation ρ = c sin(φ) describes a cone -/
theorem equation_describes_cone (c : ℝ) :
  IsCone c (fun p => p.ρ = c * Real.sin p.φ) := by
  sorry


end NUMINAMATH_CALUDE_equation_describes_cone_l2236_223635


namespace NUMINAMATH_CALUDE_halfway_between_fractions_l2236_223630

theorem halfway_between_fractions : 
  (2 : ℚ) / 9 + (1 : ℚ) / 3 = (5 : ℚ) / 9 ∧ (5 : ℚ) / 9 / 2 = (5 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_halfway_between_fractions_l2236_223630


namespace NUMINAMATH_CALUDE_abc_inequality_l2236_223616

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : a / (1 + a) + b / (1 + b) + c / (1 + c) = 1) : a * b * c ≤ 1 / 8 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l2236_223616


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l2236_223603

/-- The eccentricity of a hyperbola with the given properties is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  let C : ℝ → ℝ → Prop := λ x y => x^2 / a^2 - y^2 / b^2 = 1
  let F₁ : ℝ × ℝ := (-c, 0)
  let F₂ : ℝ × ℝ := (c, 0)
  let asymptote₁ : ℝ → ℝ := λ x => (b / a) * x
  let asymptote₂ : ℝ → ℝ := λ x => -(b / a) * x
  ∃ (G H : ℝ × ℝ) (c : ℝ),
    (∃ x, G.1 = x ∧ G.2 = asymptote₁ x) ∧ 
    (∃ x, H.1 = x ∧ H.2 = asymptote₂ x) ∧
    (G.2 - F₁.2) * (G.1 - F₂.1) = -(G.1 - F₁.1) * (G.2 - F₂.2) ∧
    H = ((G.1 + F₁.1) / 2, (G.2 + F₁.2) / 2) →
    c = 2 * a :=
by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l2236_223603


namespace NUMINAMATH_CALUDE_hyperbola_y_coordinate_comparison_l2236_223698

/-- Given two points on a hyperbola, prove that the y-coordinate of the point with smaller x-coordinate is greater -/
theorem hyperbola_y_coordinate_comparison (k : ℝ) (y₁ y₂ : ℝ) 
  (h_positive : k > 0)
  (h_point_A : y₁ = k / 2)
  (h_point_B : y₂ = k / 3) :
  y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_y_coordinate_comparison_l2236_223698


namespace NUMINAMATH_CALUDE_one_fifth_of_number_l2236_223699

theorem one_fifth_of_number (x : ℚ) : (3/10 : ℚ) * x = 12 → (1/5 : ℚ) * x = 8 := by
  sorry

end NUMINAMATH_CALUDE_one_fifth_of_number_l2236_223699


namespace NUMINAMATH_CALUDE_product_correction_l2236_223659

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

theorem product_correction (a b : ℕ) : 
  (10 ≤ a ∧ a < 100) →  -- a is a two-digit number
  (reverse_digits a * b = 221) →  -- reversed a times b is 221
  (a * b = 527 ∨ a * b = 923) :=  -- correct product is 527 or 923
by sorry

end NUMINAMATH_CALUDE_product_correction_l2236_223659


namespace NUMINAMATH_CALUDE_roots_exist_l2236_223608

-- Define the equations
def equation1 (x : ℝ) : Prop := 5 * x^2 - 10 = 40
def equation2 (x : ℝ) : Prop := (3*x - 2)^2 = (x + 3)^2
def equation3 (x : ℝ) : Prop := 2*x^2 - 18 = 3*x - 3 ∧ x ≥ 3

-- Define a function to check if a number is a root of any equation
def isRoot (x : ℝ) : Prop :=
  equation1 x ∨ equation2 x ∨ equation3 x

-- Theorem statement
theorem roots_exist :
  (∃ x : ℝ, isRoot x ∧ x < 0) ∧
  (∃ y : ℝ, isRoot y ∧ y > 0) :=
sorry

end NUMINAMATH_CALUDE_roots_exist_l2236_223608


namespace NUMINAMATH_CALUDE_hypergeometric_distribution_proof_l2236_223645

def hypergeometric_prob (N n m k : ℕ) : ℚ :=
  (Nat.choose n k * Nat.choose (N - n) (m - k)) / Nat.choose N m

theorem hypergeometric_distribution_proof (N n m : ℕ) 
  (h1 : N = 10) (h2 : n = 8) (h3 : m = 2) : 
  (hypergeometric_prob N n m 0 = 1/45) ∧
  (hypergeometric_prob N n m 1 = 16/45) ∧
  (hypergeometric_prob N n m 2 = 28/45) := by
  sorry

end NUMINAMATH_CALUDE_hypergeometric_distribution_proof_l2236_223645


namespace NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l2236_223651

theorem distance_to_origin_of_complex_number : ∃ (z : ℂ), 
  z = (2 * Complex.I) / (1 - Complex.I) ∧ 
  Complex.abs z = Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_distance_to_origin_of_complex_number_l2236_223651


namespace NUMINAMATH_CALUDE_ngo_employees_l2236_223606

/-- The number of illiterate employees -/
def num_illiterate : ℕ := 20

/-- The decrease in daily average wages of illiterate employees in Rs -/
def wage_decrease_illiterate : ℕ := 15

/-- The decrease in average salary of all employees in Rs per day -/
def avg_salary_decrease : ℕ := 10

/-- The number of literate employees -/
def num_literate : ℕ := 10

theorem ngo_employees :
  num_literate = 10 :=
by sorry

end NUMINAMATH_CALUDE_ngo_employees_l2236_223606


namespace NUMINAMATH_CALUDE_min_dot_product_on_locus_l2236_223629

/-- The locus of point P -/
def locus (x y : ℝ) : Prop :=
  Real.sqrt ((x - 1)^2 + y^2) - abs x = 1

/-- A line through F(1,0) with slope k -/
def line_through_F (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- Two points on the locus -/
structure LocusPoint where
  x : ℝ
  y : ℝ
  on_locus : locus x y

/-- The dot product of two vectors -/
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ :=
  x1 * x2 + y1 * y2

theorem min_dot_product_on_locus :
  ∀ (k : ℝ),
  k ≠ 0 →
  ∃ (A B D E : LocusPoint),
  line_through_F k A.x A.y ∧
  line_through_F k B.x B.y ∧
  line_through_F (-1/k) D.x D.y ∧
  line_through_F (-1/k) E.x E.y →
  ∀ (AD_dot_EB : ℝ),
  AD_dot_EB = dot_product (D.x - A.x) (D.y - A.y) (B.x - E.x) (B.y - E.y) →
  AD_dot_EB ≥ 16 :=
sorry

end NUMINAMATH_CALUDE_min_dot_product_on_locus_l2236_223629


namespace NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2236_223677

theorem condition_necessary_not_sufficient (a b : ℝ) :
  (∃ a b : ℝ, (a > 0 ∨ b > 0) ∧ ¬(a + b > 0 ∧ a * b > 0)) ∧
  (∀ a b : ℝ, (a + b > 0 ∧ a * b > 0) → (a > 0 ∨ b > 0)) :=
by sorry

end NUMINAMATH_CALUDE_condition_necessary_not_sufficient_l2236_223677


namespace NUMINAMATH_CALUDE_mutually_exclusive_events_l2236_223627

-- Define the sample space
def Ω : Type := Bool × Bool

-- Define the event of hitting the target at least once
def hit_at_least_once (ω : Ω) : Prop :=
  ω.1 ∨ ω.2

-- Define the event of missing the target both times
def miss_both_times (ω : Ω) : Prop :=
  ¬ω.1 ∧ ¬ω.2

-- Theorem stating that the events are mutually exclusive
theorem mutually_exclusive_events :
  ∀ ω : Ω, ¬(hit_at_least_once ω ∧ miss_both_times ω) :=
by
  sorry

end NUMINAMATH_CALUDE_mutually_exclusive_events_l2236_223627


namespace NUMINAMATH_CALUDE_small_bottle_volume_proof_l2236_223647

/-- The volume of a big bottle in ounces -/
def big_bottle_volume : ℝ := 30

/-- The cost of a big bottle in pesetas -/
def big_bottle_cost : ℝ := 2700

/-- The cost of a small bottle in pesetas -/
def small_bottle_cost : ℝ := 600

/-- The amount saved in pesetas by buying a big bottle instead of smaller bottles for the same volume -/
def savings : ℝ := 300

/-- The volume of a small bottle in ounces -/
def small_bottle_volume : ℝ := 6

theorem small_bottle_volume_proof :
  small_bottle_volume * (big_bottle_cost / big_bottle_volume) =
  small_bottle_cost + (savings / big_bottle_volume) * small_bottle_volume :=
by sorry

end NUMINAMATH_CALUDE_small_bottle_volume_proof_l2236_223647


namespace NUMINAMATH_CALUDE_luncheon_seating_capacity_l2236_223658

theorem luncheon_seating_capacity 
  (invited : ℕ) 
  (no_shows : ℕ) 
  (tables : ℕ) 
  (h1 : invited = 47) 
  (h2 : no_shows = 7) 
  (h3 : tables = 8) :
  (invited - no_shows) / tables = 5 := by
  sorry

end NUMINAMATH_CALUDE_luncheon_seating_capacity_l2236_223658


namespace NUMINAMATH_CALUDE_precy_age_l2236_223665

/-- Represents the ages of Alex and Precy -/
structure Ages where
  alex : ℕ
  precy : ℕ

/-- The conditions given in the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.alex = 15 ∧
  (ages.alex + 3) = 3 * (ages.precy + 3) ∧
  (ages.alex - 1) = 7 * (ages.precy - 1)

/-- The theorem stating that under the given conditions, Precy's age is 3 -/
theorem precy_age (ages : Ages) : problem_conditions ages → ages.precy = 3 := by
  sorry

end NUMINAMATH_CALUDE_precy_age_l2236_223665


namespace NUMINAMATH_CALUDE_barrels_for_remaining_road_l2236_223619

/-- Represents the road paving problem -/
structure RoadPaving where
  total_length : ℝ
  truckloads_per_mile : ℝ
  day1_paved : ℝ
  day2_paved : ℝ
  pitch_per_truckload : ℝ

/-- Calculates the barrels of pitch needed for the remaining road -/
def barrels_needed (rp : RoadPaving) : ℝ :=
  (rp.total_length - (rp.day1_paved + rp.day2_paved)) * rp.truckloads_per_mile * rp.pitch_per_truckload

/-- Theorem stating the number of barrels needed for the given scenario -/
theorem barrels_for_remaining_road :
  let rp : RoadPaving := {
    total_length := 16,
    truckloads_per_mile := 3,
    day1_paved := 4,
    day2_paved := 7,
    pitch_per_truckload := 0.4
  }
  barrels_needed rp = 6 := by sorry

end NUMINAMATH_CALUDE_barrels_for_remaining_road_l2236_223619


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l2236_223643

/-- The normal distribution with mean μ and standard deviation σ -/
noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The probability density function of a normal distribution -/
noncomputable def normal_pdf (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The cumulative distribution function of a normal distribution -/
noncomputable def normal_cdf (μ σ : ℝ) : ℝ → ℝ := sorry

/-- The probability of a random variable X falling within an interval [a, b] -/
noncomputable def prob_interval (μ σ : ℝ) (a b : ℝ) : ℝ :=
  normal_cdf μ σ b - normal_cdf μ σ a

theorem normal_distribution_probability (μ σ : ℝ) :
  (normal_pdf μ σ 0 = 1 / (3 * Real.sqrt (2 * Real.pi))) →
  (prob_interval μ σ (μ - σ) (μ + σ) = 0.6826) →
  (prob_interval μ σ (μ - 2*σ) (μ + 2*σ) = 0.9544) →
  (prob_interval μ σ 3 6 = 0.1359) := by sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l2236_223643


namespace NUMINAMATH_CALUDE_solve_for_a_l2236_223674

theorem solve_for_a : ∀ a : ℚ, 
  (∃ x : ℚ, (2 * a * x + 3) / (a - x) = 3 / 4 ∧ x = 1) → 
  a = -3 := by
sorry

end NUMINAMATH_CALUDE_solve_for_a_l2236_223674


namespace NUMINAMATH_CALUDE_cost_per_side_of_square_l2236_223649

/-- The cost of fencing each side of a square, given the total cost --/
theorem cost_per_side_of_square (total_cost : ℝ) (h : total_cost = 276) : 
  ∃ (side_cost : ℝ), side_cost * 4 = total_cost ∧ side_cost = 69 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_side_of_square_l2236_223649


namespace NUMINAMATH_CALUDE_polynomial_specific_value_l2236_223637

/-- A polynomial of degree 4 with specific values at 1, 2, and 3 -/
def P (a b c d : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + d

/-- The theorem stating the result for the given polynomial -/
theorem polynomial_specific_value (a b c d : ℝ) :
  P a b c d 1 = 1993 →
  P a b c d 2 = 3986 →
  P a b c d 3 = 5979 →
  -(1/4) * (P a b c d 11 + P a b c d (-7)) = 5233 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_specific_value_l2236_223637


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l2236_223633

/-- The equation of the directrix of a parabola passing through a point on a circle -/
theorem parabola_directrix_equation (y : ℝ) (p : ℝ) : 
  (1^2 - 4*1 + y^2 = 0) →  -- Point P(1, y) lies on the circle
  (p > 0) →                -- p is positive
  (1^2 = -2*p*y) →         -- Parabola passes through P(1, y)
  (-(p : ℝ) = Real.sqrt 3 / 12) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l2236_223633


namespace NUMINAMATH_CALUDE_fifteen_factorial_representation_l2236_223650

def factorial (n : ℕ) : ℕ := (List.range n).foldl (·*·) 1

theorem fifteen_factorial_representation (X Y Z : ℕ) :
  X < 10 ∧ Y < 10 ∧ Z < 10 →
  factorial 15 = 1307674300000000 + X * 100000000 + Y * 10000 + Z * 100 →
  X + Y + Z = 0 := by
sorry

end NUMINAMATH_CALUDE_fifteen_factorial_representation_l2236_223650


namespace NUMINAMATH_CALUDE_nancy_washed_19_shirts_l2236_223682

/-- The number of shirts Nancy had to wash -/
def num_shirts (machine_capacity : ℕ) (num_loads : ℕ) (num_sweaters : ℕ) : ℕ :=
  machine_capacity * num_loads - num_sweaters

/-- Proof that Nancy washed 19 shirts -/
theorem nancy_washed_19_shirts :
  num_shirts 9 3 8 = 19 := by
  sorry

end NUMINAMATH_CALUDE_nancy_washed_19_shirts_l2236_223682


namespace NUMINAMATH_CALUDE_ellipse_area_irrational_l2236_223668

-- Define the major and minor radii as rational numbers
variable (a b : ℚ)

-- Define π as an irrational constant
noncomputable def π : ℝ := Real.pi

-- Define the area of the ellipse
noncomputable def ellipseArea (a b : ℚ) : ℝ := π * (a * b)

-- Theorem statement
theorem ellipse_area_irrational (a b : ℚ) (h1 : a > 0) (h2 : b > 0) :
  Irrational (ellipseArea a b) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_area_irrational_l2236_223668


namespace NUMINAMATH_CALUDE_olivia_change_l2236_223626

def basketball_card_price : ℕ := 3
def baseball_card_price : ℕ := 4
def num_basketball_packs : ℕ := 2
def num_baseball_decks : ℕ := 5
def bill_value : ℕ := 50

def total_cost : ℕ := num_basketball_packs * basketball_card_price + num_baseball_decks * baseball_card_price

theorem olivia_change : bill_value - total_cost = 24 := by
  sorry

end NUMINAMATH_CALUDE_olivia_change_l2236_223626


namespace NUMINAMATH_CALUDE_remainder_sum_mod_five_l2236_223636

theorem remainder_sum_mod_five : (9^5 + 8^4 + 7^6) % 5 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_five_l2236_223636


namespace NUMINAMATH_CALUDE_community_families_count_l2236_223672

theorem community_families_count :
  let families_with_two_dogs : ℕ := 15
  let families_with_one_dog : ℕ := 20
  let total_animals : ℕ := 80
  let total_dogs : ℕ := families_with_two_dogs * 2 + families_with_one_dog
  let total_cats : ℕ := total_animals - total_dogs
  let families_with_cats : ℕ := total_cats / 2
  families_with_two_dogs + families_with_one_dog + families_with_cats = 50 :=
by sorry

end NUMINAMATH_CALUDE_community_families_count_l2236_223672
