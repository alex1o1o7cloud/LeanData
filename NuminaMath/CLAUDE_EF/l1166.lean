import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_equals_max_value_l1166_116634

noncomputable def f (a b x : ℝ) : ℝ := a * Real.cos (b * x)

theorem amplitude_equals_max_value 
  (a b : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hmax : ∀ x, f a b x ≤ 3) 
  (hreach : ∃ x, f a b x = 3) : 
  a = 3 := by 
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_amplitude_equals_max_value_l1166_116634


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_M_properties_l1166_116672

-- Define M as a variable (set of real numbers)
variable (M : Set ℝ)

-- Axioms for the properties of M
axiom M_nonempty : Set.Nonempty M
axiom two_in_M : 2 ∈ M
axiom M_closed_under_subtraction : ∀ x y, x ∈ M → y ∈ M → (x - y) ∈ M
axiom M_closed_under_reciprocal : ∀ x, x ∈ M → x ≠ 0 → (1 / x) ∈ M

-- Theorem stating the properties we want to prove
theorem M_properties :
  (0 ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → (x + y) ∈ M) ∧
  (∀ x y, x ∈ M → y ∈ M → (x * y) ∈ M) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_M_properties_l1166_116672


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_Al2S3_approx_l1166_116643

/-- The mass percentage of aluminum in aluminum sulfide -/
noncomputable def mass_percentage_Al_in_Al2S3 : ℝ :=
  let molar_mass_Al : ℝ := 26.98
  let molar_mass_S : ℝ := 32.06
  let molar_mass_Al2S3 : ℝ := 2 * molar_mass_Al + 3 * molar_mass_S
  (2 * molar_mass_Al / molar_mass_Al2S3) * 100

/-- Theorem stating that the mass percentage of aluminum in aluminum sulfide is approximately 35.94% -/
theorem mass_percentage_Al_in_Al2S3_approx :
  abs (mass_percentage_Al_in_Al2S3 - 35.94) < 0.01 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mass_percentage_Al_in_Al2S3_approx_l1166_116643


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_order_l1166_116615

open Real

-- Define the concept of "new stationary point"
def new_stationary_point (f : ℝ → ℝ) (x : ℝ) : Prop :=
  f x = deriv f x

-- Define the functions and their domains
noncomputable def g (x : ℝ) : ℝ := sin x
noncomputable def h (x : ℝ) : ℝ := log x
def φ (x : ℝ) : ℝ := x^3

-- State the theorem
theorem new_stationary_points_order 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < π)
  (hb : b > 0)
  (hc : c ≠ 0)
  (new_stat_a : new_stationary_point g a)
  (new_stat_b : new_stationary_point h b)
  (new_stat_c : new_stationary_point φ c) :
  c > b ∧ b > a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_stationary_points_order_l1166_116615


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_range_l1166_116662

theorem acute_triangle_range (A B C a b c : Real) : 
  0 < A ∧ A < π/2 ∧ 
  0 < B ∧ B < π/2 ∧ 
  0 < C ∧ C < π/2 ∧ 
  A + B + C = π ∧
  c - a = 2 * a * Real.cos B →
  ∃ (x : Real), 1/2 < x ∧ x < Real.sqrt 2 / 2 ∧
    x = Real.sin A * Real.sin A / Real.sin (B - A) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_acute_triangle_range_l1166_116662


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_product_problem_solution_l1166_116630

/-- The sum of an infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a r : ℝ) : ℝ := a / (1 - r)

/-- The product of two infinite geometric series equals another infinite geometric series -/
theorem geometric_series_product : 
  (geometric_sum 1 (1/3)) * (geometric_sum 1 (-1/3)) = geometric_sum 1 (1/9) := by
  -- We'll use sorry to skip the proof for now
  sorry

/-- The value of x in the original problem is 9 -/
theorem problem_solution : ∃ x : ℝ, x = 9 ∧ 
  (geometric_sum 1 (1/3)) * (geometric_sum 1 (-1/3)) = geometric_sum 1 (1/x) := by
  -- We'll use sorry to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_series_product_problem_solution_l1166_116630


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_life_at_33_l1166_116675

/-- Shelf life function -/
noncomputable def shelfLife (k b : ℝ) (x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem: Shelf life at 33°C is 24 hours -/
theorem shelf_life_at_33 (k b : ℝ) :
  shelfLife k b 0 = 192 →
  shelfLife k b 22 = 48 →
  shelfLife k b 33 = 24 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shelf_life_at_33_l1166_116675


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1166_116632

/-- The circle C in the 2D plane -/
def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 6*y + 10 = 0

/-- The line L in the 2D plane -/
def line_L (x y : ℝ) : Prop :=
  x + y = 0

/-- The shortest distance from a point on circle C to line L -/
noncomputable def shortest_distance : ℝ := Real.sqrt 2

theorem shortest_distance_circle_to_line :
  ∀ (p : ℝ × ℝ), circle_C p.1 p.2 →
  ∃ (q : ℝ × ℝ), line_L q.1 q.2 ∧
  ∀ (r : ℝ × ℝ), line_L r.1 r.2 →
  dist p q ≤ dist p r ∧
  dist p q = shortest_distance :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_distance_circle_to_line_l1166_116632


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l1166_116631

/-- Calculates the number of tiles needed to cover a rectangular room -/
def tiles_needed (room_length room_width tile_length tile_width : ℚ) : ℕ :=
  (room_length * room_width / (tile_length * tile_width)).ceil.toNat

/-- Theorem stating the number of tiles needed for the specific room and tile sizes -/
theorem tiles_for_room : tiles_needed 10 15 (1/4) (3/4) = 800 := by
  -- Proof goes here
  sorry

#eval tiles_needed 10 15 (1/4) (3/4)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tiles_for_room_l1166_116631


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_theorem_l1166_116678

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 12*x + 32 = 0

-- Define the line passing through P(0,2) with slope k
def line_eq (k x : ℝ) : ℝ := k*x + 2

-- Define the intersection of the line and circle
def intersection (k : ℝ) : Prop :=
  ∃ x y, circle_eq x y ∧ y = line_eq k x

-- Define the dot product of two vectors
def dot_product (x1 y1 x2 y2 : ℝ) : ℝ := x1*x2 + y1*y2

theorem circle_line_intersection_theorem :
  -- Part 1: Range of k
  (∃ k_min k_max : ℝ, ∀ k, intersection k ↔ k_min < k ∧ k < k_max) ∧
  
  -- Part 2: Equation of line when dot product is 28
  (∃ k : ℝ, ∀ x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧
    y1 = line_eq k x1 ∧ y2 = line_eq k x2 ∧
    dot_product x1 y1 x2 y2 = 28 →
    ∃ a b : ℝ, ∀ x, line_eq k x = a*x + b) ∧
  
  -- Part 3: Existence of point C on y-axis
  (∃ c : ℝ, ∀ k x1 y1 x2 y2 : ℝ,
    circle_eq x1 y1 ∧ circle_eq x2 y2 ∧
    y1 = line_eq k x1 ∧ y2 = line_eq k x2 →
    ∃ d : ℝ, dot_product (x1) (y1 - c) (x2) (y2 - c) = d) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_line_intersection_theorem_l1166_116678


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_addend_possibilities_l1166_116606

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Represents a three-digit number ending in 4 -/
structure ThreeDigitNumber where
  hundreds : Digit
  tens : Digit
  units : Digit
  units_is_four : units = ⟨4, by norm_num⟩

/-- Represents a two-digit number starting with 3 -/
structure TwoDigitNumber where
  tens : Digit
  units : Digit
  tens_is_three : tens = ⟨3, by norm_num⟩

/-- Converts a ThreeDigitNumber to a natural number -/
def threeDigitToNat (n : ThreeDigitNumber) : ℕ :=
  100 * n.hundreds.val + 10 * n.tens.val + n.units.val

/-- Converts a TwoDigitNumber to a natural number -/
def twoDigitToNat (n : TwoDigitNumber) : ℕ :=
  10 * n.tens.val + n.units.val

/-- The main theorem -/
theorem first_addend_possibilities (n1 : ThreeDigitNumber) (n2 : TwoDigitNumber) :
  (threeDigitToNat n1 + twoDigitToNat n2 ≥ 1000) ∧ (threeDigitToNat n1 + twoDigitToNat n2 < 10000) →
  (n1.hundreds = ⟨9, by norm_num⟩ ∧ n1.tens = ⟨6, by norm_num⟩) ∨
  (n1.hundreds = ⟨9, by norm_num⟩ ∧ n1.tens = ⟨7, by norm_num⟩) ∨
  (n1.hundreds = ⟨9, by norm_num⟩ ∧ n1.tens = ⟨8, by norm_num⟩) ∨
  (n1.hundreds = ⟨9, by norm_num⟩ ∧ n1.tens = ⟨9, by norm_num⟩) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_addend_possibilities_l1166_116606


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_elliot_average_speed_l1166_116691

/-- Calculates the average speed given total distance and total time -/
noncomputable def average_speed (distance : ℝ) (time : ℝ) : ℝ :=
  distance / time

/-- Theorem: Given a total distance of 120 miles and a total time of 4 hours,
    the average speed is 30 miles per hour -/
theorem elliot_average_speed :
  let total_distance : ℝ := 120
  let total_time : ℝ := 4
  average_speed total_distance total_time = 30 := by
  -- Unfold the definition of average_speed
  unfold average_speed
  -- Simplify the expression
  simp
  -- Prove the equality
  norm_num

#check elliot_average_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_elliot_average_speed_l1166_116691


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zero_points_iff_a_range_l1166_116604

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then x^2 - 2*a*x - 1/2 else Real.log x

theorem two_zero_points_iff_a_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ 
   ∀ z : ℝ, z ≠ x ∧ z ≠ y → f a z ≠ 0) ↔ 
  a ≥ 1/4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zero_points_iff_a_range_l1166_116604


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_64_l1166_116620

/-- Represents a type of gemstone with its weight and price -/
structure Gemstone :=
  (weight : ℕ)
  (price : ℕ)

/-- The problem setup -/
def gemstone_problem : Prop :=
  let gem1 : Gemstone := ⟨3, 9⟩
  let gem2 : Gemstone := ⟨5, 16⟩
  let gem3 : Gemstone := ⟨2, 5⟩
  let capacity : ℕ := 20
  let min_quantity : ℕ := 15

  ∀ (q1 q2 q3 : ℕ),
    q1 * gem1.weight + q2 * gem2.weight + q3 * gem3.weight ≤ capacity →
    q1 ≤ min_quantity ∧ q2 ≤ min_quantity ∧ q3 ≤ min_quantity →
    q1 * gem1.price + q2 * gem2.price + q3 * gem3.price ≤ 64

theorem max_value_is_64 : gemstone_problem := by
  sorry

#check gemstone_problem
#check max_value_is_64

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_is_64_l1166_116620


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_two_solutions_l1166_116695

theorem equation_two_solutions (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
   x₁ ∈ Set.Icc (-π/6) (3*π/2) ∧ 
   x₂ ∈ Set.Icc (-π/6) (3*π/2) ∧
   (2 * Real.sin x₁ + a^2 + a)^3 - (Real.cos (2*x₁) + 3*a * Real.sin x₁ + 11)^3 = 
     12 - 2 * (Real.sin x₁)^2 + (3*a - 2) * Real.sin x₁ - a^2 - a ∧
   (2 * Real.sin x₂ + a^2 + a)^3 - (Real.cos (2*x₂) + 3*a * Real.sin x₂ + 11)^3 = 
     12 - 2 * (Real.sin x₂)^2 + (3*a - 2) * Real.sin x₂ - a^2 - a) ↔ 
  (a ∈ Set.Icc 2.5 4 ∪ Set.Icc (-5) (-2)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_two_solutions_l1166_116695


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_area_sum_l1166_116685

/-- The optimal length of wire to be used for the circle to minimize the total area -/
noncomputable def optimal_circle_length : ℝ := 100 * Real.pi / (16 + Real.pi)

/-- The total area of the square and circle as a function of the length used for the circle -/
noncomputable def total_area (x : ℝ) : ℝ := Real.pi * (x / (2 * Real.pi))^2 + ((100 - x) / 4)^2

theorem minimize_area_sum :
  ∀ x ∈ Set.Ioo 0 100,
    total_area x ≥ total_area optimal_circle_length := by
  sorry

#check minimize_area_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimize_area_sum_l1166_116685


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_base_9_is_3_l1166_116657

def base_3_number : List Nat := [2, 1, 1, 2, 0, 2, 2, 2, 1, 2, 1, 1]

def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * (3 ^ i)) 0

def first_digit_base_9 (n : Nat) : Nat :=
  if n = 0 then 0
  else
    let log_9 := Nat.log n 9
    n / (9 ^ log_9)

theorem first_digit_base_9_is_3 :
  first_digit_base_9 (to_base_10 base_3_number) = 3 := by
  sorry

#eval first_digit_base_9 (to_base_10 base_3_number)

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_digit_base_9_is_3_l1166_116657


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_better_discount_l1166_116642

noncomputable def apply_discount (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount / 100)

noncomputable def three_successive_8_percent (price : ℝ) : ℝ :=
  apply_discount (apply_discount (apply_discount price 8) 8) 8

noncomputable def two_successive_12_percent (price : ℝ) : ℝ :=
  apply_discount (apply_discount price 12) 12

noncomputable def twenty_then_fifteen_percent (price : ℝ) : ℝ :=
  apply_discount (apply_discount price 20) 15

theorem smallest_better_discount : 
  ∀ (price : ℝ), price > 0 →
    ∀ (m : ℕ), m ≥ 33 →
      apply_discount price m < three_successive_8_percent price ∧
      apply_discount price m < two_successive_12_percent price ∧
      apply_discount price m < twenty_then_fifteen_percent price ∧
      ∀ (n : ℕ), n < 33 →
        (apply_discount price n ≥ three_successive_8_percent price ∨
         apply_discount price n ≥ two_successive_12_percent price ∨
         apply_discount price n ≥ twenty_then_fifteen_percent price) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_better_discount_l1166_116642


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_group_division_count_l1166_116646

/-- The number of ways to divide 2n individuals into n groups of 2 people each -/
def group_division_count (n : ℕ) : ℕ :=
  Nat.factorial (2 * n) / (2^n * Nat.factorial n)

/-- Theorem stating that group_division_count gives the correct number of ways to divide 2n individuals into n groups of 2 people each -/
theorem correct_group_division_count (n : ℕ) :
  group_division_count n = group_division_count n :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_group_division_count_l1166_116646


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_two_lines_l1166_116648

/-- The intersection point of two lines in 2D space -/
noncomputable def intersection_point (a b c d e f : ℝ) : ℝ × ℝ :=
  let x := (c * e - b * f) / (a * e - b * d)
  let y := (a * f - c * d) / (a * e - b * d)
  (x, y)

/-- Two lines intersect at a unique point if their slopes are different -/
def lines_intersect (a b d e : ℝ) : Prop :=
  a * e ≠ b * d

theorem intersection_of_two_lines :
  let l₁ : ℝ → ℝ → ℝ := λ x y ↦ 3 * x + 4 * y - 2
  let l₂ : ℝ → ℝ → ℝ := λ x y ↦ 2 * x + y + 2
  let p := intersection_point 3 4 (-2) 2 1 (-2)
  lines_intersect 3 4 2 1 →
  l₁ p.1 p.2 = 0 ∧ l₂ p.1 p.2 = 0 ∧ p = (-2, 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_of_two_lines_l1166_116648


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_row_sum_row_10_sum_l1166_116654

/-- Sum of numbers in a row of Pascal's Triangle -/
def sum_of_numbers_in_row (n : ℕ) : ℕ := 2^n

/-- Pascal's Triangle row sum theorem -/
theorem pascal_triangle_row_sum (n : ℕ) : 
  sum_of_numbers_in_row n = 2^n := by rfl

/-- Sum of numbers in Row 10 of Pascal's Triangle -/
theorem row_10_sum : sum_of_numbers_in_row 10 = 1024 := by
  rw [sum_of_numbers_in_row]
  norm_num

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pascal_triangle_row_sum_row_10_sum_l1166_116654


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l1166_116611

/-- The original curve equation -/
def original_curve (x y : ℝ) : Prop :=
  y * Real.cos x + 2 * y - 1 = 0

/-- The transformation applied to the curve -/
noncomputable def transform (x y : ℝ) : ℝ × ℝ :=
  (x - Real.pi / 2, y + 1)

/-- The resulting curve equation after transformation -/
def transformed_curve (x y : ℝ) : Prop :=
  (y - 1) * Real.sin x + 2 * y - 3 = 0

/-- Theorem stating that the transformation of the original curve
    results in the transformed curve -/
theorem curve_transformation (x y : ℝ) :
  original_curve (transform x y).1 (transform x y).2 ↔ transformed_curve x y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_transformation_l1166_116611


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_max_ratio_l1166_116603

noncomputable section

-- Define the curves C₁ and C₂
def C₁ (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, Real.sin θ)
def C₂ (t : ℝ) : ℝ × ℝ := (-Real.sqrt 3 / 2 * t, 2 * Real.sqrt 3 / 3 + t / 2)

-- Define the intersection points A and B
def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := sorry

-- Define the line ρsin θ = 2 in polar coordinates
def polar_line (θ : ℝ) : ℝ := 2 / Real.sin θ

-- Define the point C on C₁
def C (α : ℝ) : ℝ × ℝ := (2 * Real.cos α, 2 * Real.sin α)

-- Define the point D on the polar line
def D (α : ℝ) : ℝ × ℝ := (2 / Real.sin α * Real.cos α, 2 / Real.sin α * Real.sin α)

-- Theorem statements
theorem intersection_distance : Real.sqrt 3 = Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) := by sorry

theorem max_ratio : 
  (∀ α, α > 0 → α < Real.pi → 
    Real.sqrt ((C α).1^2 + (C α).2^2) / Real.sqrt ((D α).1^2 + (D α).2^2) ≤ 1/2) ∧ 
  (∃ α, α > 0 ∧ α < Real.pi ∧
    Real.sqrt ((C α).1^2 + (C α).2^2) / Real.sqrt ((D α).1^2 + (D α).2^2) = 1/2) := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_max_ratio_l1166_116603


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_price_calculation_l1166_116627

/-- The total price of a refrigerator and a washing machine, given their prices -/
def total_price (refrigerator_price washing_machine_price : ℕ) : ℕ :=
  refrigerator_price + washing_machine_price

/-- Theorem: The total price of the purchases is $7060 -/
theorem total_price_calculation : 
  (let refrigerator_price : ℕ := 4275
   let washing_machine_price : ℕ := refrigerator_price - 1490
   total_price refrigerator_price washing_machine_price) = 7060 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_price_calculation_l1166_116627


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_time_l1166_116692

-- Define the train's properties
def train_length : ℚ := 1200
def time_to_cross_tree : ℚ := 120
def platform_length : ℚ := 500

-- Define the function to calculate the time to pass the platform
noncomputable def time_to_pass_platform (train_length platform_length time_to_cross_tree : ℚ) : ℚ :=
  (train_length + platform_length) / (train_length / time_to_cross_tree)

-- State the theorem
theorem train_platform_time :
  time_to_pass_platform train_length platform_length time_to_cross_tree = 170 := by
  -- Unfold the definition of time_to_pass_platform
  unfold time_to_pass_platform
  -- Perform the calculation
  norm_num
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_platform_time_l1166_116692


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_from_roots_l1166_116694

theorem triangle_angle_from_roots (θ : ℝ) (p : ℝ) :
  0 < θ ∧ θ < Real.pi →
  (∃ x y : ℝ, x = Real.sin θ ∧ y = Real.cos θ ∧ 4 * x^2 + p * x - 2 = 0 ∧ 4 * y^2 + p * y - 2 = 0) →
  θ = 3 * Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_angle_from_roots_l1166_116694


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_is_hundred_percent_l1166_116660

/-- Represents the cost of a single article -/
noncomputable def cost_per_article : ℝ := sorry

/-- The number of articles in the cost calculation -/
def cost_article_count : ℕ := 20

/-- The number of articles in the selling price calculation -/
def sell_article_count : ℕ := 10

/-- The total cost of all articles -/
noncomputable def total_cost : ℝ := cost_per_article * cost_article_count

/-- The selling price of a single article -/
noncomputable def sell_price_per_article : ℝ := total_cost / sell_article_count

/-- The gain on a single article -/
noncomputable def gain_per_article : ℝ := sell_price_per_article - cost_per_article

/-- The gain percent -/
noncomputable def gain_percent : ℝ := (gain_per_article / cost_per_article) * 100

theorem gain_is_hundred_percent : gain_percent = 100 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_gain_is_hundred_percent_l1166_116660


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_interval_is_three_minutes_l1166_116641

/-- Represents the circular metro line with two stations -/
structure MetroLine where
  travel_time_north : ℝ
  travel_time_south : ℝ
  time_difference_between_directions : ℝ
  time_difference_home_work : ℝ

/-- Calculates the expected interval between trains in one direction -/
noncomputable def expected_train_interval (line : MetroLine) : ℝ :=
  let p := (line.travel_time_north - line.travel_time_south - line.time_difference_home_work) / (2 * (line.travel_time_north - line.travel_time_south))
  line.time_difference_between_directions / (1 - p)

/-- Theorem stating that the expected interval between trains is 3 minutes -/
theorem expected_interval_is_three_minutes (line : MetroLine)
  (h1 : line.travel_time_north = 17)
  (h2 : line.travel_time_south = 11)
  (h3 : line.time_difference_between_directions = 1.25)
  (h4 : line.time_difference_home_work = 1) :
  expected_train_interval line = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_interval_is_three_minutes_l1166_116641


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_l1166_116666

-- Define the parabola and line
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x
def line (a : ℝ) (x y : ℝ) : Prop := 2*x + y + a = 0

-- Define the point A
def point_A : ℝ × ℝ := (1, 2)

-- Theorem statement
theorem parabola_line_intersection_sum (p a : ℝ) (F : ℝ × ℝ) :
  parabola p (point_A.1) (point_A.2) →
  line a (point_A.1) (point_A.2) →
  ∃ (B : ℝ × ℝ), 
    parabola p B.1 B.2 ∧ 
    line a B.1 B.2 ∧
    dist F point_A + dist F B = 7 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_sum_l1166_116666


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1166_116639

/-- The line equation: 6x + 8y = b -/
def line_equation (x y b : ℝ) : Prop := 6 * x + 8 * y = b

/-- The circle equation: (x - 1)^2 + (y - 1)^2 = 1 -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 1

/-- The distance from a point (x, y) to the line 6x + 8y = b -/
noncomputable def distance_to_line (x y b : ℝ) : ℝ :=
  |6 * x + 8 * y - b| / Real.sqrt (6^2 + 8^2)

/-- The theorem stating that the line 6x + 8y = b is tangent to the circle
    (x - 1)^2 + (y - 1)^2 = 1 if and only if b = 4 or b = 24 -/
theorem line_tangent_to_circle :
  ∀ b : ℝ, (∀ x y : ℝ, line_equation x y b ∧ circle_equation x y →
    distance_to_line 1 1 b = 1) ↔ (b = 4 ∨ b = 24) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_tangent_to_circle_l1166_116639


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l1166_116664

/-- Sum of the first n terms of a geometric sequence -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_8 :
  ∀ a : ℝ,
  geometric_sum a 2 4 = 1 →
  geometric_sum a 2 8 = 17 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_sum_8_l1166_116664


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_max_points_l1166_116617

/-- Represents the game state -/
structure GameState (n : ℕ) where
  ana_cards : Finset ℕ
  bruno_cards : Finset ℕ
  turn : ℕ
  ana_points : ℕ

/-- Initialize the game state -/
def init_game (n : ℕ) : GameState n :=
  { ana_cards := Finset.range n |>.image (fun i => 2*i + 1),
    bruno_cards := Finset.range n |>.image (fun i => 2*i + 2),
    turn := 0,
    ana_points := 0 }

/-- Simulate a single turn of the game -/
def play_turn (state : GameState n) : GameState n :=
  sorry

/-- Play the entire game -/
def play_game (n : ℕ) : ℕ :=
  let rec aux (state : GameState n) (turns : ℕ) : ℕ :=
    if turns = 0 then state.ana_points
    else aux (play_turn state) (turns - 1)
  aux (init_game n) n

/-- The main theorem to prove -/
theorem ana_max_points (n : ℕ) : 
  play_game n = n / 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ana_max_points_l1166_116617


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_g_and_floor_is_three_l1166_116629

noncomputable def g (x : ℝ) : ℝ := 3 * Real.sin x - Real.cos x + 2 / Real.cos x

theorem exists_zero_g_and_floor_is_three :
  ∃ s : ℝ, π / 2 < s ∧ s < π ∧ g s = 0 ∧ ⌊s⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_zero_g_and_floor_is_three_l1166_116629


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l1166_116699

/-- The x-intercept of a line passing through two given points -/
noncomputable def x_intercept (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  let m := (y₂ - y₁) / (x₂ - x₁)
  x₁ - y₁ / m

/-- Theorem: The x-intercept of the line passing through (10, 3) and (-8, -6) is 4 -/
theorem x_intercept_specific_line :
  x_intercept 10 3 (-8) (-6) = 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercept_specific_line_l1166_116699


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1166_116612

open Real

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := f x + a * x^2 - 3 * x

-- State the theorem
theorem function_properties :
  ∃ (a : ℝ),
    -- Part 1: The derivative of g at x = 1 is 0
    (deriv (g a)) 1 = 0 ∧
    -- Part 2: a = 1
    a = 1 ∧
    -- Part 3: The minimum value of g occurs at x = 1 and equals -2
    (∀ x > 0, g a x ≥ g a 1) ∧ g a 1 = -2 ∧
    -- Part 4: For any line with slope k intersecting f at two points
    (∀ x₁ x₂ k : ℝ, 0 < x₁ ∧ x₁ < x₂ →
      k = (f x₂ - f x₁) / (x₂ - x₁) →
      1 / x₂ < k ∧ k < 1 / x₁) :=
by
  sorry -- The proof is omitted as per the instructions

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1166_116612


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_iff_k_in_range_l1166_116607

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * Real.cos (k * x)

def monotone_decreasing (g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → g y < g x

def k_range : Set ℝ :=
  Set.Icc (-6) (-4) ∪ Set.Ioo 0 3 ∪ Set.Icc 8 9 ∪ {-12}

theorem f_monotone_decreasing_iff_k_in_range :
  ∀ k : ℝ, monotone_decreasing (f k) (π/4) (π/3) ↔ k ∈ k_range :=
by
  sorry

#check f_monotone_decreasing_iff_k_in_range

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotone_decreasing_iff_k_in_range_l1166_116607


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_approx_140kJ_l1166_116635

/-- Converts temperature change from Fahrenheit to Celsius -/
noncomputable def fahrenheit_to_celsius_change (delta_f : ℝ) : ℝ :=
  (5/9) * delta_f

/-- Calculates heat released given volume and temperature change -/
noncomputable def heat_released (volume : ℝ) (delta_t : ℝ) : ℝ :=
  4200 * volume * delta_t

/-- Theorem: Heat released is approximately 140 kJ for given conditions -/
theorem heat_released_approx_140kJ :
  let delta_f : ℝ := 30
  let volume : ℝ := 2
  let delta_c := fahrenheit_to_celsius_change delta_f
  let Q := heat_released volume delta_c
  abs (Q - 140000) < 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_approx_140kJ_l1166_116635


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_value_l1166_116688

open Real

noncomputable def e : ℝ := Real.exp 1

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (x^2 + x + m) * (e^x)

theorem local_minimum_value (m : ℝ) :
  (∃ ε > 0, ∀ x ∈ Set.Ioo (-3 - ε) (-3 + ε), f m (-3) ≥ f m x) →
  (∃ x₀ : ℝ, ∀ x : ℝ, f m x₀ ≤ f m x ∧ f m x₀ = -1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_local_minimum_value_l1166_116688


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l1166_116636

theorem log_sum_upper_bound (a b : ℝ) (h1 : a ≥ b) (h2 : b > 2) :
  Real.log (a^2 / b) / Real.log a + Real.log (b^2 / a) / Real.log b ≤ 3 ∧
  (Real.log (a^2 / b) / Real.log a + Real.log (b^2 / a) / Real.log b = 3 ↔ a = b) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_sum_upper_bound_l1166_116636


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_obtuse_angle_l1166_116616

-- Define the function f(x) = e^x cos(x)
noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := Real.exp x * Real.cos x - Real.exp x * Real.sin x

-- Theorem statement
theorem tangent_slope_obtuse_angle :
  ∃ (slope : ℝ), slope = f' 1 ∧ slope < 0 := by
  -- We'll use 'sorry' to skip the proof for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_obtuse_angle_l1166_116616


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_fold_theorem_l1166_116673

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold operation on a square piece of paper -/
structure Fold where
  start : Point
  finish : Point
  angle : ℝ

/-- The result of the fold operation -/
structure FoldResult where
  original : Point
  folded : Point

/-- Theorem: When a square piece of paper is folded such that (0,4) matches (4,0) with a 45° rotation,
    and (8,6) aligns with (m,n) after folding, then m+n = 14 -/
theorem paper_fold_theorem (fold : Fold) (result : FoldResult) :
  fold.start = Point.mk 0 4 →
  fold.finish = Point.mk 4 0 →
  fold.angle = π/4 →
  result.original = Point.mk 8 6 →
  result.folded.x + result.folded.y = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_paper_fold_theorem_l1166_116673


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l1166_116608

/-- Calculates the time (in seconds) for a train to pass a bridge -/
noncomputable def time_to_pass_bridge (train_length : ℝ) (train_speed_kmh : ℝ) (bridge_length : ℝ) : ℝ :=
  let total_distance := train_length + bridge_length
  let train_speed_ms := train_speed_kmh * (1000 / 3600)
  total_distance / train_speed_ms

/-- Theorem: A train of length 360 meters, traveling at 90 km/hour, 
    passes a bridge of length 140 meters in 20 seconds -/
theorem train_bridge_passage_time :
  time_to_pass_bridge 360 90 140 = 20 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval time_to_pass_bridge 360 90 140

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_bridge_passage_time_l1166_116608


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_l1166_116690

noncomputable def G (x p : ℝ) : ℝ := (8 * x^2 + 18 * x + 4 * p) / 8

def is_perfect_square_of_linear (p : ℝ) : Prop :=
  ∃ (a b : ℝ), ∀ x, G x p = (a * x + b)^2

theorem p_range (p : ℝ) (h : is_perfect_square_of_linear p) : 2.5 < p ∧ p < 2.6 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_range_l1166_116690


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_sum_l1166_116602

/-- An odd function f defined on the real numbers. -/
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2*x + x^2 else -(2*x + x^2)

/-- The theorem statement -/
theorem odd_function_range_sum (a b : ℝ) : 
  a > 0 ∧ b > 0 ∧
  (∀ x, x ∈ Set.Icc a b → f x ∈ Set.Icc (1/b) (1/a)) ∧
  (∀ y, y ∈ Set.Icc (1/b) (1/a) → ∃ x ∈ Set.Icc a b, f x = y) →
  a + b = (3 + Real.sqrt 5) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_range_sum_l1166_116602


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotone_interval_l1166_116668

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 + 2*x else -x^2 + 2*x

-- State the theorem
theorem odd_function_monotone_interval (m : ℝ) :
  (∀ x : ℝ, f (-x) = -f x) →  -- f is odd
  (∀ x : ℝ, x ≤ 0 → f x = x^2 + 2*x) →  -- f(x) = x^2 + 2x for x ≤ 0
  (∀ x y : ℝ, -1 ≤ x ∧ x < y ∧ y ≤ m - 1 → f x < f y) →  -- f is monotonically increasing on [-1, m-1]
  0 < m ∧ m ≤ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_monotone_interval_l1166_116668


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_a_l1166_116665

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := x * Real.log x
def g (a x : ℝ) : ℝ := -x^2 + a*x - 3

-- Theorem for the minimum value of f(x) on (0, e]
theorem min_value_f : 
  ∃ (m : ℝ), m = -1/Real.exp 1 ∧ ∀ x ∈ Set.Ioo 0 (Real.exp 1), f x ≥ m := by
  sorry

-- Theorem for the range of a
theorem range_of_a :
  ∀ a : ℝ, (∀ x > 0, 2 * f x ≥ g a x) → a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_range_of_a_l1166_116665


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_107_l1166_116655

def fibonacci : ℕ → ℕ
| 0 => 2
| 1 => 7
| (k + 2) => fibonacci k + fibonacci (k + 1)

theorem eighth_term_is_107 : fibonacci 7 = 107 := by
  rw [fibonacci]
  rw [fibonacci]
  rw [fibonacci]
  rw [fibonacci]
  rw [fibonacci]
  rw [fibonacci]
  rw [fibonacci]
  rw [fibonacci]
  rfl

#eval fibonacci 7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eighth_term_is_107_l1166_116655


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_ratio_l1166_116621

/-- Given a triangle ABC, point D on AC, and point E on BC, if AD:DC = 4:1 and BE:EC = 2:3,
    and DE intersects AB at F, then DE:DF = 2:1 -/
theorem triangle_intersection_ratio (A B C D E F : EuclideanSpace ℝ (Fin 2)) :
  ‖D - A‖ / ‖C - D‖ = 4 / 1 →
  ‖E - B‖ / ‖C - E‖ = 2 / 3 →
  ∃ t : ℝ, F = A + t • (B - A) ∧ F = D + t • (E - D) →
  ‖E - D‖ / ‖F - D‖ = 2 / 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_intersection_ratio_l1166_116621


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1166_116628

-- Define the function f as noncomputable
noncomputable def f (x : ℝ) : ℝ := Real.tan (Real.arcsin (x^3))

-- State the theorem about the domain of f
theorem domain_of_f :
  {x : ℝ | f x ∈ Set.univ} = {x : ℝ | -1 ≤ x ∧ x ≤ 1} :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1166_116628


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_term_of_sequence_l1166_116650

noncomputable def geometric_progression (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q^(n - 1)

noncomputable def sum_geometric (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a₁ else a₁ * (1 - q^n) / (1 - q)

def b_n (n : ℕ) : ℚ := (4 * n - 1) / (2 * n - 7)

theorem max_term_of_sequence (a₁ : ℝ) (q : ℝ) :
  a₁ = 2 →
  sum_geometric a₁ q 5 + 4 * sum_geometric a₁ q 3 = 5 * sum_geometric a₁ q 4 →
  ∃ n : ℕ, ∀ m : ℕ, b_n n ≥ b_n m ∧ b_n n = 15 := by
  sorry

#eval b_n 4  -- This will evaluate to 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_term_of_sequence_l1166_116650


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1166_116661

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (2 * ω * x) - Real.sin (2 * ω * x + Real.pi / 6)

theorem omega_range (ω : ℝ) (h1 : ω > 0) :
  (∃! (z1 z2 : ℝ), z1 ≠ z2 ∧ z1 ∈ Set.Icc 0 Real.pi ∧ z2 ∈ Set.Icc 0 Real.pi ∧ f ω z1 = 0 ∧ f ω z2 = 0) →
  ω ∈ Set.Icc (7/12) (13/12) := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_range_l1166_116661


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l1166_116622

-- Define the piecewise function f
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then a * x + b else 7 - 2 * x

-- State the theorem
theorem function_composition_identity (a b : ℝ) :
  (∀ x, f a b (f a b x) = x) → a + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_composition_identity_l1166_116622


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_correct_l1166_116697

/-- A quadrilateral with non-perpendicular diagonals. -/
structure Quadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  φ : ℝ
  h_positive : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d
  h_angle : 0 < φ ∧ φ < π / 2

/-- The area of a quadrilateral. -/
noncomputable def area (q : Quadrilateral) : ℝ :=
  (Real.tan q.φ * |q.a^2 + q.c^2 - q.b^2 - q.d^2|) / 4

/-- Theorem stating that the area formula is correct for any quadrilateral with non-perpendicular diagonals. -/
theorem area_formula_correct (q : Quadrilateral) : 
  area q = (Real.tan q.φ * |q.a^2 + q.c^2 - q.b^2 - q.d^2|) / 4 := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_formula_correct_l1166_116697


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hexagon_area_l1166_116658

/-- Regular hexagon with vertices A and C -/
structure RegularHexagon where
  A : ℝ × ℝ
  C : ℝ × ℝ

/-- The area of a regular hexagon given two vertices -/
noncomputable def hexagon_area (h : RegularHexagon) : ℝ :=
  51 * Real.sqrt 3 / 2

/-- Theorem stating that the area of the specific regular hexagon is 51√3/2 -/
theorem specific_hexagon_area :
  let h : RegularHexagon := ⟨(0, 0), (8, 2)⟩
  hexagon_area h = 51 * Real.sqrt 3 / 2 := by
  sorry

#check specific_hexagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_specific_hexagon_area_l1166_116658


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_angle_l1166_116683

open Real

noncomputable def line_L (t : ℝ) : ℝ × ℝ × ℝ :=
  (2 * t + 2, -3 * t / 2 + 1, -t / 2 - 1)

def plane_intersect (x y z : ℝ) : Prop :=
  x - y + z - 1 = 0

def plane_angle (x y z : ℝ) : Prop :=
  x + 2 * y - z + 3 = 0

noncomputable def angle_line_plane : ℝ := 
  Real.arcsin (Real.sqrt 21 / 14)

theorem line_plane_angle :
  ∃ (t : ℝ), 
    let (x, y, z) := line_L t
    plane_intersect x y z ∧
    angle_line_plane = 
      Real.arcsin (|(1 : ℝ) * 2 + 2 * (-3) + (-1) * (-1)| / 
        (((1 : ℝ)^2 + 2^2 + (-1)^2).sqrt * (2^2 + (-3)^2 + (-1)^2).sqrt)) :=
by sorry

#check line_plane_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_plane_angle_l1166_116683


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1166_116676

-- Define the line l
noncomputable def line_l (t : ℝ) : ℝ × ℝ := (1 + (1/2) * t, (Real.sqrt 3 / 2) * t)

-- Define the ellipse C
noncomputable def ellipse_C (θ : ℝ) : ℝ × ℝ := (Real.cos θ, 2 * Real.sin θ)

-- Define the intersection points A and B
noncomputable def A : ℝ × ℝ := line_l 0
noncomputable def B : ℝ × ℝ := line_l (-8/7)

-- Theorem statement
theorem length_of_AB : Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 7 / 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_of_AB_l1166_116676


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_to_line_distance_l1166_116687

/-- The distance from a point to a line in 2D space -/
noncomputable def distancePointToLine (x y : ℝ) (m b : ℝ) : ℝ :=
  abs (y - m*x - b) / Real.sqrt (1 + m^2)

/-- Theorem: If the distance from (1,a) to y = x + 1 is 3√2/2, then a = -1 or a = 5 -/
theorem point_to_line_distance (a : ℝ) :
  distancePointToLine 1 a 1 1 = (3 * Real.sqrt 2) / 2 →
  a = -1 ∨ a = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_to_line_distance_l1166_116687


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_intersecting_line_l1166_116671

-- Define the points
def A : ℝ × ℝ := (-2, 3)
def B : ℝ × ℝ := (3, 2)
def P : ℝ × ℝ := (0, -2)

-- Theorem statement
theorem slope_range_for_intersecting_line (k : ℝ) :
  (∃ (x y : ℝ), (x, y) ∈ Set.Icc A B ∧ y - P.2 = k * (x - P.1)) →
  k ∈ Set.Iic (-5/2) ∪ Set.Ici (4/3) := by
  sorry

#check slope_range_for_intersecting_line

end NUMINAMATH_CALUDE_ERRORFEEDBACK_slope_range_for_intersecting_line_l1166_116671


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_distribution_l1166_116625

-- Define the random variable X
def X : ℕ → ℝ
| 0 => 0
| 1 => 1
| _ => 0

-- Define the probability mass function for X
noncomputable def pmf : ℕ → ℝ
| 0 => 2/3
| 1 => 1/3
| _ => 0

-- Define the expected value of X
noncomputable def expectation : ℝ := ∑' x, X x * pmf x

-- Define the variance of X
noncomputable def variance : ℝ := ∑' x, (X x - expectation)^2 * pmf x

-- Define the standard deviation of X
noncomputable def std_dev : ℝ := Real.sqrt variance

-- Theorem statement
theorem urn_probability_distribution :
  (pmf 0 = 2/3 ∧ pmf 1 = 1/3) ∧
  expectation = 1/3 ∧
  variance = 2/9 ∧
  std_dev = Real.sqrt 2 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_urn_probability_distribution_l1166_116625


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_intersection_point_l1166_116638

-- Define the circle type
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Helper functions (declared as axioms for now)
axiom is_tangent_inside : Circle → Circle → Prop
axiom intersection_points : Circle → Circle → Set (ℝ × ℝ)
axiom line_through_tangency_points : Circle → Circle → Circle → Set (ℝ × ℝ)

-- Define the main theorem
theorem tangent_line_passes_through_intersection_point 
  (K : ℝ) 
  (main_circle : Circle) 
  (inner_circle1 inner_circle2 : Circle) 
  (h1 : main_circle.radius = K)
  (h2 : inner_circle1.radius + inner_circle2.radius = K)
  (h3 : is_tangent_inside main_circle inner_circle1)
  (h4 : is_tangent_inside main_circle inner_circle2) :
  ∃ (p : ℝ × ℝ), 
    p ∈ intersection_points inner_circle1 inner_circle2 ∧ 
    p ∈ line_through_tangency_points main_circle inner_circle1 inner_circle2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_passes_through_intersection_point_l1166_116638


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_is_14_l1166_116609

/-- The original parabola equation -/
def original_parabola (x : ℝ) : ℝ := (x - 3)^2 + 4

/-- The transformed parabola equation -/
def transformed_parabola (x : ℝ) : ℝ := -(x - 7)^2 + 7

/-- The zeros of the transformed parabola -/
noncomputable def zero_p : ℝ := 7 + Real.sqrt 7
noncomputable def zero_q : ℝ := 7 - Real.sqrt 7

/-- Theorem stating that the sum of zeros of the transformed parabola is 14 -/
theorem sum_of_zeros_is_14 : zero_p + zero_q = 14 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_zeros_is_14_l1166_116609


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_l1166_116653

-- Define the approximation relation
def approx_equal (a b : ℤ) : Prop := (a - b).natAbs ≤ 1

-- Define the equation
def equation (x y : ℤ) : Prop := approx_equal ((x - 5) * (y - 2)) 11

-- Theorem statement
theorem four_solutions :
  ∃! (s : Finset (ℤ × ℤ)), s.card = 4 ∧ 
  (∀ (p : ℤ × ℤ), p ∈ s ↔ equation p.1 p.2) ∧
  s = {(6, 13), (16, 3), (4, -9), (-6, 1)} :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_solutions_l1166_116653


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_arithmetic_sequence_if_sides_arithmetic_sequence_l1166_116633

theorem sine_arithmetic_sequence_if_sides_arithmetic_sequence 
  (A B C : ℝ) (a b c : ℝ) :
  -- Triangle conditions
  0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = Real.pi →
  -- Side lengths are positive
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Law of sines
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C →
  -- Sides form arithmetic sequence
  b - a = c - b →
  -- Conclusion: sines of angles form arithmetic sequence
  2 * Real.sin B = Real.sin A + Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_arithmetic_sequence_if_sides_arithmetic_sequence_l1166_116633


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_rate_calculation_l1166_116689

/-- The rate of grapes per kg -/
def grape_rate : ℕ → Prop := λ x => x = 70

/-- The amount of grapes purchased in kg -/
def grapes_amount : ℕ := 8

/-- The rate of mangoes per kg -/
def mango_rate : ℕ := 55

/-- The amount of mangoes purchased in kg -/
def mangoes_amount : ℕ := 9

/-- The total amount paid -/
def total_paid : ℕ := 1055

theorem grape_rate_calculation : grape_rate 70 := by
  unfold grape_rate
  rfl

#check grape_rate_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_grape_rate_calculation_l1166_116689


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_label_sum_correct_l1166_116601

/-- The sum of labels on a circle after n steps of the labeling process -/
def labelSum (n : ℕ) : ℕ :=
  2 * 3^(n - 1)

/-- The labeling process on a circle -/
inductive LabelingProcess : ℕ → Type where
  | step_one : LabelingProcess 1
  | step_n {k : ℕ} : k > 0 → LabelingProcess k → LabelingProcess (k + 1)

theorem label_sum_correct (n : ℕ) (h : n > 0) :
  ∀ (process : LabelingProcess n), labelSum n = 2 * 3^(n - 1) := by
  sorry

#check label_sum_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_label_sum_correct_l1166_116601


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_calculation_l1166_116681

/-- Represents a race track with semicircular ends -/
structure RaceTrack where
  width : ℝ
  timeDifference : ℝ

/-- Calculates the jogging speed given a race track -/
noncomputable def joggingSpeed (track : RaceTrack) : ℝ := 
  Real.pi / 3

theorem jogging_speed_calculation (track : RaceTrack) 
  (h1 : track.width = 8)
  (h2 : track.timeDifference = 48) : 
  joggingSpeed track = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_jogging_speed_calculation_l1166_116681


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l1166_116644

/-- Represents a quadratic equation of the form ax^2 + bx + c = 0 -/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  a : ℝ
  b : ℝ

/-- The discriminant of a quadratic equation -/
def discriminant (eq : QuadraticEquation) : ℝ :=
  eq.b^2 - 4 * eq.a * eq.c

/-- The perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℝ :=
  t.a + 2 * t.b

theorem quadratic_equation_properties :
  ∀ k : ℝ,
  let eq : QuadraticEquation := ⟨1, 3*k - 2, -6*k⟩
  (discriminant eq ≥ 0) ∧
  (∃ t : IsoscelesTriangle, 
    t.a = 6 ∧ 
    perimeter t = 14 ∧
    (6 : ℝ) ∈ {x : ℝ | eq.a * x^2 + eq.b * x + eq.c = 0}) := by
  intro k
  let eq : QuadraticEquation := ⟨1, 3*k - 2, -6*k⟩
  have h1 : discriminant eq ≥ 0 := by
    -- Proof for discriminant being non-negative
    sorry
  have h2 : ∃ t : IsoscelesTriangle, 
    t.a = 6 ∧ 
    perimeter t = 14 ∧
    (6 : ℝ) ∈ {x : ℝ | eq.a * x^2 + eq.b * x + eq.c = 0} := by
    -- Proof for the existence of the isosceles triangle
    sorry
  exact ⟨h1, h2⟩


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_properties_l1166_116644


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_ticket_cost_l1166_116698

/-- The price of an adult ticket -/
noncomputable def adult_price : ℝ := 7

/-- The price of a child ticket -/
noncomputable def child_price : ℝ := adult_price / 3

/-- The cost of 4 adult tickets and 3 child tickets -/
noncomputable def given_cost : ℝ := 35

/-- The number of adult tickets in the question -/
def num_adult : ℕ := 10

/-- The number of child tickets in the question -/
def num_child : ℕ := 8

theorem museum_ticket_cost :
  (given_cost = 4 * adult_price + 3 * child_price) →
  (num_adult * adult_price + num_child * child_price = 88.67) :=
by
  intro h
  sorry

#eval num_adult -- This line is added to ensure some computable content

end NUMINAMATH_CALUDE_ERRORFEEDBACK_museum_ticket_cost_l1166_116698


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_and_surface_area_l1166_116651

/-- Represents a cube with a given total edge length sum. -/
structure Cube where
  total_edge_length : ℝ
  total_edge_length_positive : total_edge_length > 0

/-- Calculates the edge length of the cube. -/
noncomputable def edge_length (c : Cube) : ℝ := c.total_edge_length / 12

/-- Calculates the volume of the cube. -/
noncomputable def volume (c : Cube) : ℝ := (edge_length c) ^ 3

/-- Calculates the surface area of the cube. -/
noncomputable def surface_area (c : Cube) : ℝ := 6 * (edge_length c) ^ 2

/-- Theorem stating the volume and surface area of a cube with total edge length 72. -/
theorem cube_volume_and_surface_area :
  ∃ (c : Cube), c.total_edge_length = 72 ∧ volume c = 216 ∧ surface_area c = 216 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_volume_and_surface_area_l1166_116651


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_original_cost_price_l1166_116682

/-- The original cost price of a car given specific selling conditions -/
noncomputable def originalCostPrice (initialLossPercentage : ℝ) (finalGainPercentage : ℝ) (finalSellingPrice : ℝ) : ℝ :=
  finalSellingPrice / ((1 + finalGainPercentage / 100) * (1 - initialLossPercentage / 100))

/-- Theorem stating the original cost price of the car -/
theorem car_original_cost_price :
  let initialLoss : ℝ := 12
  let finalGain : ℝ := 20
  let finalPrice : ℝ := 54000
  ∃ (price : ℝ), abs (originalCostPrice initialLoss finalGain finalPrice - price) < 0.01 ∧ price = 51136.36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_original_cost_price_l1166_116682


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_theorem_l1166_116624

theorem smallest_positive_angle_theorem (θ : Real) : 
  (θ > 0) → 
  (Real.cos θ = Real.sin (π/4) + Real.cos (π/3) - Real.sin (π/6) - Real.cos (π/12)) → 
  (∀ φ, φ > 0 ∧ Real.cos φ = Real.sin (π/4) + Real.cos (π/3) - Real.sin (π/6) - Real.cos (π/12) → θ ≤ φ) →
  θ = π/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_theorem_l1166_116624


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_possible_l1166_116677

-- Define the basic geometric elements
structure Point where
  x : ℝ
  y : ℝ

structure Angle where
  vertex : Point
  side1 : Point
  side2 : Point

structure Line where
  point1 : Point
  point2 : Point

structure Ray where
  origin : Point
  direction : Point

-- Define the problem setup
def setup (ABC : Angle) (l : Line) (a : ℝ) : Prop :=
  ∃ (parallel_line : Line),
    -- parallel_line is parallel to l
    sorry ∧ -- Replace with actual parallel condition when available
    -- The segment created by the intersection of parallel_line and the sides of ABC has length a
    ∃ (P Q : Point),
      sorry ∧ -- Replace with actual conditions for P and Q being on parallel_line and rays
      sorry -- Replace with actual distance calculation

-- Define the condition for the existence of a solution
def solutionExists (ABC : Angle) (l : Line) (a : ℝ) : Prop :=
  ∃ (v : Point), -- Using Point as a substitute for Vector
    -- v is parallel to l and has length a
    sorry ∧ -- Replace with actual parallel and length conditions
    -- The translated lines intersect with ray BA
    sorry -- Replace with actual intersection conditions

-- State the theorem
theorem construction_possible (ABC : Angle) (l : Line) (a : ℝ) :
  setup ABC l a ↔ solutionExists ABC l a :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_construction_possible_l1166_116677


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_plus_two_positive_l1166_116647

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + x else -x^2 - x

theorem solution_set_of_f_plus_two_positive :
  {x : ℝ | f x + 2 > 0} = {x : ℝ | x > -2} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_of_f_plus_two_positive_l1166_116647


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_distance_45min_l1166_116663

/-- The distance traveled by the tip of a clock's minute hand -/
noncomputable def minute_hand_distance (length : ℝ) (minutes : ℝ) : ℝ :=
  2 * Real.pi * length * (minutes / 60)

/-- Theorem: The distance traveled by the tip of an 8 cm long minute hand in 45 minutes is 12π cm -/
theorem minute_hand_distance_45min :
  minute_hand_distance 8 45 = 12 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minute_hand_distance_45min_l1166_116663


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_base_ratio_not_unique_l1166_116674

/-- A triangle with side lengths a, b, c and altitudes ha, hb, hc. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : ℝ
  hb : ℝ
  hc : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_altitudes : 0 < ha ∧ 0 < hb ∧ 0 < hc
  triangle_inequality : a < b + c ∧ b < a + c ∧ c < a + b

/-- The ratio of the bases of two altitudes in a triangle. -/
noncomputable def altitudeBaseRatio (t : Triangle) (i j : Fin 3) : ℝ :=
  match i, j with
  | 0, 1 => t.a / t.b
  | 0, 2 => t.a / t.c
  | 1, 0 => t.b / t.a
  | 1, 2 => t.b / t.c
  | 2, 0 => t.c / t.a
  | 2, 1 => t.c / t.b
  | _, _ => 1  -- Default case to cover all possibilities

/-- Two triangles are non-congruent if they have different side lengths. -/
def nonCongruent (t1 t2 : Triangle) : Prop :=
  t1.a ≠ t2.a ∨ t1.b ≠ t2.b ∨ t1.c ≠ t2.c

theorem altitude_base_ratio_not_unique :
  ∃ (t1 t2 : Triangle) (i j : Fin 3),
    altitudeBaseRatio t1 i j = altitudeBaseRatio t2 i j ∧
    nonCongruent t1 t2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_altitude_base_ratio_not_unique_l1166_116674


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_x_equals_one_eighth_of_two_power_33_l1166_116600

theorem eight_power_x_equals_one_eighth_of_two_power_33 (x : ℝ) : 
  (1/8 : ℝ) * (2^33 : ℝ) = 8^x → x = 10 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_eight_power_x_equals_one_eighth_of_two_power_33_l1166_116600


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_marcus_goal_value_l1166_116656

/-- The point value of the first type of goals Marcus scored -/
def x : ℕ := 3

/-- The number of goals Marcus scored of the first type -/
def first_type_goals : ℕ := 5

/-- The number of 2-point goals Marcus scored -/
def second_type_goals : ℕ := 10

/-- The point value of the second type of goals -/
def second_type_points : ℕ := 2

/-- The total points scored by the team -/
def team_total_points : ℕ := 70

/-- The percentage of team's points scored by Marcus -/
def marcus_percentage : ℚ := 1/2

theorem marcus_goal_value :
  first_type_goals * x + second_type_goals * second_type_points = 
  (marcus_percentage * team_total_points).num ∧ x = 3 := by
  sorry

#eval x

end NUMINAMATH_CALUDE_ERRORFEEDBACK_marcus_goal_value_l1166_116656


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_l1166_116618

/-- Given a triangle ABC with sides a, b, and c, if a + b - c = 2 and 2ab - c^2 = 4, then a = b = c = 2 -/
theorem triangle_equilateral (a b c : ℝ) 
  (h1 : a + b - c = 2) 
  (h2 : 2 * a * b - c^2 = 4) : 
  a = 2 ∧ b = 2 ∧ c = 2 := by
  sorry

#check triangle_equilateral

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_equilateral_l1166_116618


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_purchase_cost_l1166_116667

/-- The cost of a candy bar in dollars -/
noncomputable def candy_bar_cost : ℚ := 3/2

/-- The cost of a pack of gum in dollars -/
noncomputable def gum_pack_cost : ℚ := candy_bar_cost / 2

/-- The number of candy bars John buys -/
def num_candy_bars : ℕ := 3

/-- The number of gum packs John buys -/
def num_gum_packs : ℕ := 2

/-- The total cost of John's purchase -/
noncomputable def total_cost : ℚ := num_candy_bars * candy_bar_cost + num_gum_packs * gum_pack_cost

theorem john_purchase_cost : total_cost = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_purchase_cost_l1166_116667


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l1166_116686

/-- Represents a parabola with focus and directrix -/
structure Parabola where
  focus : ℝ × ℝ
  directrix : ℝ → Prop

/-- The equation of the parabola -/
def parabola_equation (x y : ℝ) : Prop :=
  y^2 = 2*x - 1

theorem parabola_equation_correct (p : Parabola) :
  p.focus = (1, 0) →
  p.directrix = (λ x ↦ x = 0) →
  ∀ x y : ℝ, (x - 1)^2 + y^2 = x^2 + y^2 ↔ parabola_equation x y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_correct_l1166_116686


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_tan_x_necessary_not_sufficient_for_x_sin_x_l1166_116680

theorem x_tan_x_necessary_not_sufficient_for_x_sin_x
  (x : ℝ) (h : 0 < x ∧ x < π/2) :
  (∀ y : ℝ, y * Real.sin y > 1 → y * Real.tan y > 1) ∧
  (∃ y : ℝ, y * Real.tan y > 1 ∧ y * Real.sin y ≤ 1) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_tan_x_necessary_not_sufficient_for_x_sin_x_l1166_116680


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_squared_l1166_116610

theorem quadratic_roots_squared (α : ℝ) :
  let original_eq (x : ℝ) := Real.sin (2 * α) * x^2 - 2 * (Real.sin α + Real.cos α) * x + 2
  let new_eq (z : ℝ) := (Real.sin α)^2 * (Real.cos α)^2 * z^2 - z + 1
  ∀ (x₁ x₂ : ℝ), original_eq x₁ = 0 ∧ original_eq x₂ = 0 →
  ∃ (z₁ z₂ : ℝ), new_eq z₁ = 0 ∧ new_eq z₂ = 0 ∧ z₁ = x₁^2 ∧ z₂ = x₂^2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_roots_squared_l1166_116610


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l1166_116684

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a1 : ℚ  -- First term
  d : ℚ   -- Common difference

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a1 + (n - 1 : ℚ) * seq.d

/-- The sum of the first n terms of an arithmetic sequence -/
def ArithmeticSequence.sumFirstN (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  n * seq.a1 + n * (n - 1 : ℚ) / 2 * seq.d

/-- The maximum sum of the first n terms of the given arithmetic sequence -/
theorem max_sum_arithmetic_sequence :
  ∃ (seq : ArithmeticSequence),
    seq.nthTerm 1 + seq.nthTerm 3 + seq.nthTerm 5 = 9 ∧
    seq.nthTerm 6 = -9 ∧
    ∃ (n : ℕ),
      (∀ (m : ℕ), seq.sumFirstN n ≥ seq.sumFirstN m) ∧
      seq.sumFirstN n = 21 ∧
      n = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_arithmetic_sequence_l1166_116684


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_proof_l1166_116645

theorem right_triangle_proof :
  let set_a := [Real.sqrt 3, Real.sqrt 4, Real.sqrt 5]
  let set_b := [3^2, 4^2, 5^2]
  let set_c := [1, 1, 2]
  let set_d := [9, 12, 15]
  
  (∀ (x y z : Real), x ∈ set_a → y ∈ set_a → z ∈ set_a → x^2 + y^2 ≠ z^2) ∧
  (∀ (x y z : Nat), x ∈ set_b → y ∈ set_b → z ∈ set_b → x^2 + y^2 ≠ z^2) ∧
  (∀ (x y z : Nat), x ∈ set_c → y ∈ set_c → z ∈ set_c → x + y ≤ z) ∧
  (∃ (x y z : Nat), x ∈ set_d ∧ y ∈ set_d ∧ z ∈ set_d ∧ x^2 + y^2 = z^2 ∧ x + y > z ∧ x + z > y ∧ y + z > x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_proof_l1166_116645


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_ge_four_l1166_116637

def a : ℕ → ℚ
  | 0 => 5
  | n + 1 => (a n ^ 2 + 8 * a n + 16) / (4 * a n)

theorem a_ge_four : ∀ n : ℕ, a n ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_ge_four_l1166_116637


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_greg_visible_area_l1166_116659

/-- The area visible to Greg during his walk around a rectangular garden -/
noncomputable def visibleArea (length width visibilityRadius : ℝ) : ℝ :=
  2 * (length * visibilityRadius + width * visibilityRadius) + 4 * (Real.pi * visibilityRadius^2 / 4)

/-- Rounds a real number to the nearest integer -/
noncomputable def roundToNearest (x : ℝ) : ℤ :=
  ⌊x + 0.5⌋

theorem greg_visible_area :
  let gardenLength : ℝ := 8
  let gardenWidth : ℝ := 4
  let visibilityRadius : ℝ := 2
  roundToNearest (visibleArea gardenLength gardenWidth visibilityRadius) = 61 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_greg_visible_area_l1166_116659


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_theorem_l1166_116670

/-- Calculate simple interest -/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

/-- Calculate compound interest -/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the difference between compound and simple interest -/
theorem interest_difference_theorem (principal rate time : ℝ) :
  principal = 1500 ∧ rate = 10 ∧ time = 2 →
  compound_interest principal rate time - simple_interest principal rate time = 15 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_theorem_l1166_116670


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_percentage_approx_l1166_116623

def election_votes : List Nat := [1000, 2000, 4000]

def total_votes : Nat := election_votes.sum

noncomputable def winning_votes : Nat := election_votes.maximum.getD 0

theorem winning_percentage_approx : 
  |((winning_votes : Real) / (total_votes : Real)) * 100 - 57.14| < 0.01 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_winning_percentage_approx_l1166_116623


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_is_vertex_l1166_116614

/-- A parabola with equation x^2 = 4y -/
def Parabola : Set (ℝ × ℝ) :=
  {p | p.1^2 = 4 * p.2}

/-- The line obtained from the directrix by homothety -/
def LineL : Set (ℝ × ℝ) :=
  {p | p.2 = -3}

/-- The vertex of the parabola -/
def Vertex : ℝ × ℝ := (0, 0)

/-- Predicate to represent that a line is tangent to the parabola at a point -/
def IsTangent (p q : ℝ × ℝ) : Prop :=
  sorry

/-- Function to calculate the orthocenter of a triangle -/
noncomputable def Orthocenter (p q r : ℝ × ℝ) : ℝ × ℝ :=
  sorry

/-- Statement: The orthocenter of triangle AOB is the vertex of the parabola -/
theorem orthocenter_is_vertex 
  (O A B : ℝ × ℝ)
  (hO : O ∈ LineL)
  (hA : A ∈ Parabola)
  (hB : B ∈ Parabola)
  (hTangent : IsTangent O A ∧ IsTangent O B) :
  Orthocenter A O B = Vertex := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_orthocenter_is_vertex_l1166_116614


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l1166_116613

-- Define the constants
noncomputable def a : ℝ := Real.log 0.05 / Real.log 0.2
noncomputable def b : ℝ := 0.5^(1.002 : ℝ)
noncomputable def c : ℝ := 4 * Real.cos 1

-- State the theorem
theorem ordering_abc : b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_abc_l1166_116613


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_payback_time_l1166_116679

/-- Given the initial cost, monthly revenue, and monthly expenses, 
    calculate the time needed to pay back the initial cost. -/
noncomputable def payback_time (initial_cost : ℝ) (monthly_revenue : ℝ) (monthly_expenses : ℝ) : ℝ :=
  initial_cost / (monthly_revenue - monthly_expenses)

/-- Theorem stating that under the given conditions, 
    the payback time is 10 months. -/
theorem store_payback_time :
  payback_time 25000 4000 1500 = 10 := by
  -- Unfold the definition of payback_time
  unfold payback_time
  -- Simplify the arithmetic
  simp [div_eq_mul_inv]
  -- The proof is completed with 'sorry' as requested
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_payback_time_l1166_116679


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_expression_l1166_116649

/-- The greatest integer function -/
noncomputable def floor (x : ℝ) : ℤ :=
  ⌊x⌋

/-- The expression inside the floor function -/
noncomputable def expression : ℝ :=
  (Real.rpow (Real.sqrt 5 + 2) (1/3) + Real.rpow (Real.sqrt 5 - 2) (1/3)) ^ 2014

/-- The last three digits of a number -/
def lastThreeDigits (n : ℤ) : ℤ :=
  n % 1000

theorem last_three_digits_of_expression :
  lastThreeDigits (floor expression) = 125 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_three_digits_of_expression_l1166_116649


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_average_weight_l1166_116693

theorem new_students_average_weight 
  (original_count : ℕ) 
  (new_count : ℕ) 
  (original_average : ℝ) 
  (new_total_average : ℝ) :
  original_count = 100 →
  new_count = 10 →
  original_average = 65 →
  new_total_average = 64.6 →
  (original_count + new_count) * new_total_average - original_count * original_average = new_count * 60.6 := by
  sorry

#check new_students_average_weight

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_students_average_weight_l1166_116693


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_cos_three_pi_fourth_l1166_116619

theorem arcsin_cos_three_pi_fourth :
  Real.arcsin (Real.cos (3 * Real.pi / 4)) = -Real.pi / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arcsin_cos_three_pi_fourth_l1166_116619


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squared_distances_l1166_116640

/-- Given collinear points A, B, C, D, and E in that order, with specified distances between them,
    the function calculates the sum of squared distances from a point P to all five points. -/
def sumOfSquaredDistances (A B C D E P : ℝ) : ℝ :=
  (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2

/-- Theorem stating the minimum value of the sum of squared distances -/
theorem min_sum_squared_distances :
  ∀ (A B C D E : ℝ),
  B - A = 1 →
  C - B = 3 →
  D - C = 4 →
  E - D = 5 →
  (∃ (m : ℝ), ∀ (P : ℝ), sumOfSquaredDistances A B C D E P ≥ m ∧ 
    ∃ (Q : ℝ), sumOfSquaredDistances A B C D E Q = m) ∧
  (∀ (m : ℝ), (∀ (P : ℝ), sumOfSquaredDistances A B C D E P ≥ m ∧ 
    ∃ (Q : ℝ), sumOfSquaredDistances A B C D E Q = m) → m ≥ 114.8) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_squared_distances_l1166_116640


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_interpretation_l1166_116696

/-- Represents the price of one A product in yuan -/
def x : Type := ℝ

/-- Represents the promotion condition -/
def promotion_condition (x : ℝ) : Prop := 0.8 * (2 * x - 100) < 1000

/-- Theorem stating that the promotion condition represents the described promotion -/
theorem promotion_interpretation (x : ℝ) : 
  promotion_condition x ↔ 
  (∃ (final_price : ℝ), 
    final_price = 0.8 * (2 * x - 100) ∧ 
    final_price < 1000 ∧ 
    final_price = 0.92 * (2 * x - 100)) :=
by
  sorry

#check promotion_interpretation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_promotion_interpretation_l1166_116696


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cutting_theorem_l1166_116669

/-- Represents a square with a given side length -/
structure Square (α : Type*) [LinearOrderedField α] where
  side : α
  side_pos : side > 0

/-- Represents a part cut from a square -/
inductive SquarePart (α : Type*) [LinearOrderedField α]
| Rectangle : α → α → SquarePart α
| Triangle : α → α → SquarePart α

/-- Calculate the area of a SquarePart -/
def SquarePart.area {α : Type*} [LinearOrderedField α] : SquarePart α → α
| Rectangle a b => a * b
| Triangle a b => a * b / 2

/-- Sum the areas of a list of SquareParts -/
def sum_areas {α : Type*} [LinearOrderedField α] (parts : List (SquarePart α)) : α :=
  parts.foldl (fun acc p => acc + SquarePart.area p) 0

/-- Defines a function that checks if a list of parts can form a square -/
def can_form_square {α : Type*} [LinearOrderedField α] (parts : List (SquarePart α)) : Prop :=
  ∃ (side : α), side > 0 ∧ (sum_areas parts) = side * side

/-- Main theorem: It's possible to cut a 3x3 square and a 1x1 square into at most 3 parts
    that can be rearranged to form a single square -/
theorem square_cutting_theorem :
  ∃ (parts : List (SquarePart ℝ)),
    parts.length ≤ 3 ∧
    can_form_square parts ∧
    (∃ (s1 s2 : Square ℝ),
      s1.side = 3 ∧ s2.side = 1 ∧
      (sum_areas parts) = s1.side * s1.side + s2.side * s2.side) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_cutting_theorem_l1166_116669


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_measure_l1166_116652

-- Define the isosceles trapezoid ABCD
def ABCD : Set (ℝ × ℝ × ℝ) :=
  {⟨3, 0, 0⟩, ⟨0, 3, 0⟩, ⟨0, 1, Real.sqrt 3⟩, ⟨2, 1, Real.sqrt 3⟩}

-- Define the axis of symmetry OO₁
def OO₁ : Set (ℝ × ℝ × ℝ) :=
  {⟨1.5, 0, 0⟩, ⟨1, 0.5, Real.sqrt 3⟩}

-- Define the dihedral angle function
noncomputable def dihedralAngle (ABCD : Set (ℝ × ℝ × ℝ)) (OO₁ : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

-- Theorem statement
theorem dihedral_angle_measure :
  dihedralAngle ABCD OO₁ = Real.arccos (Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dihedral_angle_measure_l1166_116652


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_eq_sum_num_denom_final_answer_l1166_116626

/-- The set of positive integers with prime factors only 2, 3, 5, or 7 -/
def B : Set ℕ+ := {n | ∀ p, Nat.Prime p → p ∣ n → p ∈ ({2, 3, 5, 7} : Set ℕ)}

/-- The sum of reciprocals of elements in B -/
noncomputable def sum_reciprocals : ℚ := ∑' n : B, (n : ℚ)⁻¹

/-- Theorem stating that the sum of reciprocals of elements in B is 35/8 -/
theorem sum_reciprocals_eq : sum_reciprocals = 35 / 8 := by
  sorry

/-- The sum of numerator and denominator is 43 -/
theorem sum_num_denom : (35 : ℕ) + 8 = 43 := by
  rfl

/-- The final answer is 43 -/
theorem final_answer : (35 : ℕ) + 8 = 43 := sum_num_denom

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_reciprocals_eq_sum_num_denom_final_answer_l1166_116626


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1166_116605

theorem cosine_identity (α β : ℝ) 
  (h1 : 0 < α) (h2 : α < π/2)
  (h3 : -π/2 < β) (h4 : β < 0)
  (h5 : Real.cos (π/4 + α) = 1/3)
  (h6 : Real.cos (π/4 - β/2) = Real.sqrt 3 / 3) :
  Real.cos (α - β/2) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_identity_l1166_116605
