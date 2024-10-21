import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1145_114551

-- Define the circle in polar coordinates
def polar_circle (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ

-- Define a line in polar coordinates
def polar_line (ρ θ : ℝ) : Prop := ρ * Real.sin θ = 1

-- Theorem statement
theorem tangent_line_to_circle :
  ∃ (ρ θ : ℝ), polar_circle ρ θ ∧ polar_line ρ θ ∧
  (∀ (ρ' θ' : ℝ), polar_circle ρ' θ' → polar_line ρ' θ' → (ρ, θ) = (ρ', θ')) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_to_circle_l1145_114551


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_6_l1145_114526

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10000 ∧ n < 100000) ∧
  (∃ a b c d e : ℕ, 
    n = a * 10000 + b * 1000 + c * 100 + d * 10 + e ∧
    ({a, b, c, d, e} : Finset ℕ) = {1, 2, 3, 6, 9})

def is_divisible_by_6 (n : ℕ) : Prop :=
  n % 6 = 0

theorem smallest_valid_divisible_by_6 : 
  ∀ n : ℕ, is_valid_number n ∧ is_divisible_by_6 n → n ≥ 12396 :=
by
  sorry

#check smallest_valid_divisible_by_6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_divisible_by_6_l1145_114526


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_row10Sum_l1145_114593

/-- Definition of Pascal's Triangle -/
def pascalTriangle (n k : ℕ) : ℕ :=
  match n, k with
  | 0, _ => if k = 0 then 1 else 0
  | n+1, 0 => 1
  | n+1, k+1 => pascalTriangle n k + pascalTriangle n (k+1)

/-- Sum of numbers in a row of Pascal's Triangle -/
def rowSum (n : ℕ) : ℕ :=
  (List.range (n + 1)).foldl (λ sum k => sum + pascalTriangle n k) 0

/-- Theorem: Sum of numbers in Row 10 of Pascal's Triangle is 2^10 -/
theorem row10Sum :
  rowSum 10 = 2^10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_row10Sum_l1145_114593


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_markup_l1145_114576

/-- Calculate the percentage markup given the cost price and selling price -/
noncomputable def percentage_markup (cost_price selling_price : ℝ) : ℝ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Theorem stating that the percentage markup for a computer table
    with cost price 1250 and selling price 2000 is 60% -/
theorem computer_table_markup :
  percentage_markup 1250 2000 = 60 := by
  -- Unfold the definition of percentage_markup
  unfold percentage_markup
  -- Simplify the arithmetic expression
  simp [div_eq_mul_inv]
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_computer_table_markup_l1145_114576


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_zero_l1145_114595

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (x.sin ^ 6 + 9 * x.cos ^ 2) - Real.sqrt (x.cos ^ 6 + 9 * x.sin ^ 2)

theorem g_is_zero : ∀ x : ℝ, g x = 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_zero_l1145_114595


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1145_114557

-- Define the ⊗ operation
noncomputable def otimes (x y : ℝ) : ℝ := if x ≤ y then x else y

-- Theorem statement
theorem range_of_m (m : ℝ) : 
  (otimes (|m - 1|) m = |m - 1|) → m ≥ (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_m_l1145_114557


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_roots_l1145_114586

theorem sin_sum_of_roots (m : ℝ) (x₁ x₂ : ℝ) : 
  x₁ ∈ Set.Icc 0 (π/2) →
  x₂ ∈ Set.Icc 0 (π/2) →
  2 * Real.sin (2 * x₁) + Real.cos (2 * x₁) = m →
  2 * Real.sin (2 * x₂) + Real.cos (2 * x₂) = m →
  Real.sin (x₁ + x₂) = 2 * Real.sqrt 5 / 5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_roots_l1145_114586


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cube_volume_ratio_l1145_114563

/-- Given a cube of side length a and a sphere that passes through the vertices of one face
    of the cube and touches the sides of the opposite face, the ratio of the volume of the
    sphere to the volume of the cube is π√2/6. -/
theorem sphere_cube_volume_ratio (a : ℝ) (h : a > 0) :
  (4 / 3) * Real.pi * (a / Real.sqrt 2)^3 / a^3 = Real.pi * Real.sqrt 2 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_cube_volume_ratio_l1145_114563


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_term50_equals_273_l1145_114547

/-- Represents a sequence where each term is a sum of distinct powers of 3 up to 3^5 --/
def powerOf3Sequence (n : ℕ) : ℕ :=
  (List.range 6).foldl (λ acc i => acc + if n.bodd then 3^i else 0) 0

/-- The 50th term of the powerOf3Sequence --/
def term50 : ℕ := powerOf3Sequence 50

theorem term50_equals_273 : term50 = 273 := by
  -- Unfold definitions
  unfold term50
  unfold powerOf3Sequence
  
  -- Evaluate the function for n = 50
  -- This step would typically involve more detailed calculations
  -- For now, we'll use sorry to skip the proof
  sorry

#eval term50  -- This will compute and display the actual value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_term50_equals_273_l1145_114547


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_acceleration_l1145_114545

/-- The acceleration due to gravity in m/s² -/
def g : ℝ := 10

/-- The radius of the larger roller in meters -/
def R : ℝ := 1

/-- The radius of the smaller roller in meters -/
def r : ℝ := 0.75

/-- The mass of the plate in kg -/
def m : ℝ := 75

/-- The angle of inclination of the plate -/
noncomputable def α : ℝ := Real.arccos 0.98

/-- The magnitude of the acceleration of the plate -/
def a : ℝ := 1

/-- The direction of the acceleration of the plate -/
noncomputable def direction : ℝ := Real.arcsin 0.1

theorem plate_acceleration (no_slip : Bool) :
  no_slip → a = g * Real.sin (α / 2) ∧ direction = Real.arcsin (Real.sin (α / 2)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_plate_acceleration_l1145_114545


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l1145_114565

/-- The length of the line segment AB, where A and B are the intersection points
    of the line y = x - 2 and the parabola y^2 = 4x, is equal to 4√6. -/
theorem intersection_length :
  ∃ A B : ℝ × ℝ, 
    let line := λ (x : ℝ) => x - 2
    let parabola := λ (x : ℝ) => 4 * x
    let intersections := {p : ℝ × ℝ | p.2 = line p.1 ∧ p.2^2 = parabola p.1}
    A ∈ intersections ∧ B ∈ intersections ∧ A ≠ B ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 * Real.sqrt 6 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_length_l1145_114565


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_proof_l1145_114542

/-- Prove that the sequence a_n converges to 7 as n approaches infinity -/
theorem sequence_limit_proof (ε : ℝ) (hε : ε > 0) : 
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → |((7 : ℝ) * n - 1) / (n + 1) - 7| < ε :=
by
  -- We'll use ceil(8/ε) as our N
  let N := Nat.ceil (8 / ε)
  
  -- Provide N as the witness for the existential quantifier
  use N
  
  -- Now we need to prove the rest of the statement
  intro n hn
  
  -- Convert n to a real number for calculations
  have n_real : ℝ := n
  
  -- The main proof steps would go here
  -- For now, we'll use sorry to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_limit_proof_l1145_114542


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_n_l1145_114572

/-- The number of radios bought by the dealer -/
def n : ℕ := sorry

/-- The total cost of the radios in dollars -/
def d : ℕ := sorry

/-- The profit per radio sold (excluding donated ones) -/
def profit_per_radio : ℕ := 10

/-- The total profit achieved -/
def total_profit : ℕ := 90

/-- The fraction of cost at which donated radios are valued -/
def donation_fraction : ℚ := 1/3

/-- The number of radios donated -/
def donated_radios : ℕ := 2

/-- The profit equation -/
def profit_equation (n d : ℕ) : Prop :=
  10 * n - 20 - (4 * d) / (3 * n) = total_profit

/-- The theorem stating the least possible value of n -/
theorem least_possible_n (n d : ℕ) (h1 : d > 0) (h2 : profit_equation n d) :
  n ≥ 11 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_possible_n_l1145_114572


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1145_114598

theorem repeating_decimal_sum (c d : ℕ) : 
  (7 : ℚ) / 19 = 0.1 * c + 0.01 * d + 0.001 * c + 0.0001 * d + (1 / 9900 : ℚ) * (100 * c + d) → c + d = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_sum_l1145_114598


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AE_is_3_sqrt_2_l1145_114523

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Represents a line in 2D space -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the intersection point of two lines -/
noncomputable def intersectionPoint (l1 l2 : Line) : Point :=
  let x := (l1.b * l2.c - l2.b * l1.c) / (l1.a * l2.b - l2.a * l1.b)
  let y := (l2.a * l1.c - l1.a * l2.c) / (l1.a * l2.b - l2.a * l1.b)
  { x := x, y := y }

theorem distance_AE_is_3_sqrt_2 (A B C D : Point) :
  A.x = 0 ∧ A.y = 4 ∧
  B.x = 4 ∧ B.y = 0 ∧
  C.x = 1 ∧ C.y = 1 ∧
  D.x = 3 ∧ D.y = 3 →
  let AB : Line := { a := A.y - B.y, b := B.x - A.x, c := A.x * B.y - B.x * A.y }
  let CD : Line := { a := C.y - D.y, b := D.x - C.x, c := C.x * D.y - D.x * C.y }
  let E : Point := intersectionPoint AB CD
  distance A E = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AE_is_3_sqrt_2_l1145_114523


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_doors_is_twenty_l1145_114552

/-- Represents the structure of rooms and probabilities -/
structure RoomStructure where
  num_rooms : Nat
  return_prob : ℚ
  advance_prob : ℚ

/-- Calculates the expected number of doors passed before reaching the final room -/
noncomputable def expected_doors (rs : RoomStructure) : ℚ :=
  sorry

/-- The specific room structure in the problem -/
def problem_structure : RoomStructure :=
  { num_rooms := 3
  , return_prob := 3/4
  , advance_prob := 1/4 }

/-- Theorem stating the expected number of doors for the given problem -/
theorem expected_doors_is_twenty :
  expected_doors problem_structure = 20 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_doors_is_twenty_l1145_114552


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_198_divisibility_l1145_114573

def is_all_twos (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2 * (10^k - 1) / 9

theorem smallest_k_for_198_divisibility :
  ∃ K : ℕ, 
    (∀ k : ℕ, k < K → ¬∃ x : ℕ, is_all_twos x ∧ (Nat.digits 10 x).length = k ∧ 198 ∣ x) ∧
    (∃ x : ℕ, is_all_twos x ∧ (Nat.digits 10 x).length = K ∧ 198 ∣ x) ∧
    K = 99 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_for_198_divisibility_l1145_114573


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1145_114520

theorem sin_2alpha_value (α : ℝ) (h : Real.sin α + Real.cos α = Real.sqrt 3 / 3) : 
  Real.sin (2 * α) = -2/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_value_l1145_114520


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_7_has_two_solutions_l1145_114504

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 2 then x^2 - 2 else x + 4

theorem f_f_eq_7_has_two_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ ∀ x : ℝ, x ∈ s ↔ f (f x) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_eq_7_has_two_solutions_l1145_114504


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_l1145_114532

theorem regular_octagon_interior_angle : ∀ (n : ℕ), n = 8 → (180 * (n - 2) : ℝ) / n = 135 := by
  intro n h
  rw [h]
  norm_num

#check regular_octagon_interior_angle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_interior_angle_l1145_114532


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_range_of_g_l1145_114512

noncomputable def f (a b x : ℝ) : ℝ := x^2 - 2*a*x + b

noncomputable def g (a b x : ℝ) : ℝ := (deriv (f a b)) x + b

theorem zeros_range_of_g (a b : ℝ) :
  (∀ x, f a b x ≥ -1) ∧ (∃ x, f a b x = -1) →
  {x | g a b x = 0} = Set.Iic 1 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zeros_range_of_g_l1145_114512


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1145_114580

/-- The distance between two parallel lines -/
noncomputable def distance_parallel_lines (a b c₁ c₂ : ℝ) : ℝ :=
  |c₂ - c₁| / Real.sqrt (a^2 + b^2)

theorem distance_between_given_lines :
  let l₁ : ℝ → ℝ → ℝ := λ x y => 2*x + y + 1
  let l₂ : ℝ → ℝ → ℝ := λ x y => 4*x + 2*y + 3
  distance_parallel_lines 4 2 2 3 = Real.sqrt 5 / 10 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval distance_parallel_lines 4 2 2 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_given_lines_l1145_114580


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_policy_support_percentage_l1145_114507

theorem policy_support_percentage 
  (men_support_rate : ℚ) 
  (women_support_rate : ℚ)
  (men_surveyed : ℕ)
  (women_surveyed : ℕ)
  (h1 : men_support_rate = 55 / 100)
  (h2 : women_support_rate = 80 / 100)
  (h3 : men_surveyed = 300)
  (h4 : women_surveyed = 700) :
  let total_surveyed := men_surveyed + women_surveyed
  let total_supporters := men_support_rate * men_surveyed + women_support_rate * women_surveyed
  (total_supporters : ℚ) / total_surveyed = 725 / 1000 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_policy_support_percentage_l1145_114507


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l1145_114535

theorem k_range (k : ℝ) : 
  (2 ≤ ∫ x in (1:ℝ)..(2:ℝ), (k + 1)) ∧ (∫ x in (1:ℝ)..(2:ℝ), (k + 1) ≤ 4) → 
  1 ≤ k ∧ k ≤ 3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_k_range_l1145_114535


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_circle_l1145_114575

-- Define the parabola
def parabola (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define our circle (renamed to avoid conflict with Mathlib's circle)
def our_circle (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 16

-- Define the directrix of the parabola
def directrix (p : ℝ) (x : ℝ) : Prop := x = -p/2

-- Theorem statement
theorem parabola_tangent_circle : 
  ∃ p : ℝ, (∀ x y : ℝ, parabola p x y → ∀ x₀ : ℝ, directrix p x₀ → 
  ∃ y₀ : ℝ, our_circle x₀ y₀) → p = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_tangent_circle_l1145_114575


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integers_greater_than_18_l1145_114522

theorem max_integers_greater_than_18 (integers : Finset ℤ) (sum_condition : (integers.toList.map Int.toNat).sum = 30) 
  (count_condition : integers.card = 10) :
  (integers.filter (λ x => x > 18)).card ≤ 9 ∧ 
  ∃ (subset : Finset ℤ), subset ⊆ integers ∧ subset.card = 9 ∧ ∀ x ∈ subset, x > 18 := by
  sorry

#check max_integers_greater_than_18

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_integers_greater_than_18_l1145_114522


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_transformation_to_circle_l1145_114599

-- Define the ellipse equation
def ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

-- Define the transformation
def transformation (lambda mu : ℝ) (x y x' y' : ℝ) : Prop :=
  x' = lambda * x ∧ y' = mu * y ∧ lambda > 0 ∧ mu > 0

-- Define the circle equation
def circle' (x' y' : ℝ) : Prop := x'^2 + y'^2 = 9

theorem ellipse_transformation_to_circle :
  ∀ (lambda mu : ℝ), 
    (∀ (x y x' y' : ℝ), 
      ellipse x y → 
      transformation lambda mu x y x' y' → 
      circle' x' y') → 
    lambda = 1 ∧ mu = 3/2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_transformation_to_circle_l1145_114599


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_formula_matches_l1145_114596

def x : List ℕ := [1, 2, 3, 4, 5]
def y : List ℕ := [4, 10, 18, 28, 40]

def formula_a (x : ℕ) : ℕ := 2 * x^2 + 2 * x
def formula_b (x : ℕ) : ℕ := 3 * x^2 - x + 1
def formula_c (x : ℕ) : ℕ := x^2 + 3 * x + 1
def formula_d (x : ℕ) : ℕ := x^2 + 4 * x

theorem no_formula_matches : 
  (∀ i, i < x.length → formula_a (x[i]!) ≠ y[i]!) ∧
  (∀ i, i < x.length → formula_b (x[i]!) ≠ y[i]!) ∧
  (∀ i, i < x.length → formula_c (x[i]!) ≠ y[i]!) ∧
  (∀ i, i < x.length → formula_d (x[i]!) ≠ y[i]!) :=
by
  sorry

#check no_formula_matches

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_formula_matches_l1145_114596


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_collection_count_l1145_114597

theorem stamp_collection_count 
  (foreign : ℕ) 
  (old : ℕ) 
  (foreign_and_old : ℕ) 
  (neither : ℕ) :
  foreign = 90 →
  old = 80 →
  foreign_and_old = 20 →
  neither = 50 →
  foreign + old - foreign_and_old + neither = 200 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stamp_collection_count_l1145_114597


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_arithmetic_l1145_114537

def S (n : ℕ) : ℤ := n^2 + 2*n - 1

def a : ℕ → ℤ
  | 0 => 0  -- Add this case to handle n = 0
  | 1 => S 1
  | n + 1 => S (n + 1) - S n

theorem sequence_not_arithmetic : ¬ ∃ (d : ℤ), ∀ (n : ℕ), n > 1 → a (n + 1) - a n = d := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_not_arithmetic_l1145_114537


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_l1145_114508

theorem car_sale_profit (original_price : ℝ) (original_price_pos : original_price > 0) :
  let discount_rate := 0.3
  let profit_rate := 0.18999999999999993
  let buying_price := original_price * (1 - discount_rate)
  let selling_price := original_price * (1 + profit_rate)
  let percentage_increase := (selling_price - buying_price) / buying_price * 100
  ∃ (ε : ℝ), abs (percentage_increase - 70) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_sale_profit_l1145_114508


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l1145_114550

theorem no_such_function : 
  ¬∃ (f : ℝ → ℝ), ∀ (x y : ℝ), x > 0 → y > 0 → f y > (y - x) * (f x)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_l1145_114550


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_result_l1145_114567

-- Define the complex function f
noncomputable def f (z : ℂ) : ℂ :=
  if z.im ≠ 0 then z^2 + 1 else -z^2 - 1

-- State the theorem
theorem f_composition_equals_result : f (f (f (f (f (1 + I))))) = -48 + 249984 * I := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_equals_result_l1145_114567


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_close_roots_l1145_114591

/-- A polynomial of degree 2000 -/
noncomputable def f : ℝ → ℝ := sorry

/-- The number of real roots of f(x^2 - 1) -/
def roots_f_x2_minus_1 : ℕ := 3400

/-- The number of real roots of f(1 - x^2) -/
def roots_f_1_minus_x2 : ℕ := 2700

/-- The degree of the polynomial f -/
def degree_f : ℕ := 2000

/-- Theorem: There exist two real roots of f(x) with a difference less than 0.002 -/
theorem existence_of_close_roots :
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ f r₁ = 0 ∧ f r₂ = 0 ∧ |r₁ - r₂| < 0.002 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_close_roots_l1145_114591


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l1145_114562

theorem sum_of_squares_of_roots (a b c : ℚ) (h1 : a = 10) (h2 : b = 15) (h3 : c = -21) 
  (h4 : ∃ x y : ℝ, a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ x ≠ y) :
  ∃ x1 x2 : ℝ, a * x1^2 + b * x1 + c = 0 ∧ a * x2^2 + b * x2 + c = 0 ∧ x1 ≠ x2 ∧ x1^2 + x2^2 = 129/20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_squares_of_roots_l1145_114562


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_l1145_114519

noncomputable def f (a b x : ℝ) : ℝ := (1/3) * x^3 - x^2 + a*x + b

theorem tangent_line_implies_a_and_b :
  ∀ a b : ℝ,
  (∀ x : ℝ, deriv (f a b) x = x^2 - 2*x + a) →
  deriv (f a b) 0 = 3 →
  f a b 0 = -2 →
  a = 3 ∧ b = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_a_and_b_l1145_114519


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_product_empty_l1145_114574

noncomputable def A : Set ℝ := {x | ∃ y, y = (2 : ℝ)^x ∧ x > 0}
noncomputable def B : Set ℝ := {y | ∃ x, y = (2 : ℝ)^x ∧ x > 0}

def custom_product (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∪ B ∧ x ∉ A ∩ B}

theorem custom_product_empty : custom_product A B = ∅ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_custom_product_empty_l1145_114574


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_and_max_distance_l1145_114541

-- Define the curves C₁ and C₂
noncomputable def C₁ (α : ℝ) : ℝ × ℝ := (1 + Real.cos α, Real.sin α)

noncomputable def C₂ (θ : ℝ) : ℝ × ℝ := 
  let ρ := 4 * Real.sin θ
  (ρ * Real.cos θ, ρ * Real.sin θ)

-- State the theorem
theorem curve_intersection_and_max_distance (α₁ α₂ θ₁ θ₂ : ℝ) :
  let A := C₁ α₁
  let B := C₁ α₂
  let C := C₂ θ₁
  let D := C₂ θ₂
  -- Common points exist
  A = C ∧ B = D →
  -- Slope of line AB is 1/2
  (B.2 - A.2) / (B.1 - A.1) = 1 / 2 ∧
  -- Maximum area of triangle AOB
  ∃ (α : ℝ) (θ : ℝ),
    let A := C₁ α
    let B := C₂ θ
    (∀ (α' θ' : ℝ), ‖(C₁ α' - C₂ θ')‖ ≤ ‖(A - B)‖) →
    |1/2 * A.1 * B.2 - 1/2 * A.2 * B.1| = 3 * Real.sqrt 5 / 5 + 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_intersection_and_max_distance_l1145_114541


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_10_equals_2_l1145_114524

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 + 1 else Real.log x / Real.log 2

-- State the theorem
theorem f_f_10_equals_2 : f (f 10) = 2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_f_10_equals_2_l1145_114524


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_quadratic_l1145_114515

/-- Definition of a quadratic equation -/
noncomputable def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equations -/
noncomputable def eq_A (x : ℝ) : ℝ := x^3 + x^2 + 1
noncomputable def eq_B (x : ℝ) : ℝ := 1 / (x^2) + x + 1
noncomputable def eq_C (x : ℝ) : ℝ := x^2 - x - 8
noncomputable def eq_D (x : ℝ) : ℝ := 2 * x - 9

/-- Theorem stating that only eq_C is a quadratic equation -/
theorem only_C_is_quadratic :
  ¬(is_quadratic_equation eq_A) ∧
  ¬(is_quadratic_equation eq_B) ∧
  (is_quadratic_equation eq_C) ∧
  ¬(is_quadratic_equation eq_D) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_C_is_quadratic_l1145_114515


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_player_is_tenth_l1145_114559

def is_valid_number (n : ℕ) : Bool :=
  10 ≤ n ∧ n ≤ 99 ∧ (n % 10 + n / 10) ≠ 6 ∧ (n % 10 + n / 10) ≠ 9

def count_valid_numbers : ℕ :=
  (Finset.range 90).filter (λ i => is_valid_number (i + 10)) |>.card

theorem last_player_is_tenth (num_players : ℕ) (h : num_players = 11) :
  count_valid_numbers % num_players = 9 := by
  -- The proof goes here
  sorry

#eval count_valid_numbers
#eval count_valid_numbers % 11

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_player_is_tenth_l1145_114559


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_4_l1145_114510

noncomputable def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![3 * Real.sqrt 2 / 2, -3 / 2;
     3 / 2, 3 * Real.sqrt 2 / 2]

theorem matrix_power_4 : A ^ 4 = !![(-81 : ℝ), 0; 0, -81] := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_power_4_l1145_114510


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_twenty_equals_two_times_square_root_five_l1145_114585

theorem square_root_twenty_equals_two_times_square_root_five : 
  Real.sqrt 20 = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_root_twenty_equals_two_times_square_root_five_l1145_114585


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_33_kgs_l1145_114588

/-- The price of apples for the first 30 kgs and additional kgs -/
structure ApplePrice where
  l : ℚ  -- price per kg for first 30 kgs
  q : ℚ  -- price per kg for additional kgs

/-- Calculate the price of n kilograms of apples -/
def price (p : ApplePrice) (n : ℚ) : ℚ :=
  if n ≤ 30 then n * p.l
  else 30 * p.l + (n - 30) * p.q

/-- The theorem stating the price of 33 kgs of apples -/
theorem price_of_33_kgs (p : ApplePrice) : 
  price p 10 = 362/100 → price p 36 = 1248/100 → price p 33 = 1167/100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_price_of_33_kgs_l1145_114588


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_equivalence_l1145_114592

-- Define a circle in a 2D plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line in a 2D plane
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define a point on a line
def PointOnLine (l : Line) (t : ℝ) : ℝ × ℝ :=
  (l.point.1 + t * l.direction.1, l.point.2 + t * l.direction.2)

-- Define the distance between two points
noncomputable def Distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define a function to check if a line intersects a circle
def LineIntersectsCircle (l : Line) (c : Circle) : Prop := sorry

-- Define a function to get the tangent point from a point to a circle
noncomputable def TangentPoint (p : ℝ × ℝ) (c : Circle) : ℝ × ℝ := sorry

-- Main theorem
theorem line_circle_intersection_equivalence (l : Line) (c : Circle) :
  (¬ LineIntersectsCircle l c) ↔
  (∀ t1 t2 : ℝ,
    let A := PointOnLine l t1
    let B := PointOnLine l t2
    let AC := Distance A (TangentPoint A c)
    let BD := Distance B (TangentPoint B c)
    let AB := Distance A B
    |AC - BD| ≤ AB ∧ AB ≤ AC + BD) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_equivalence_l1145_114592


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_equation_solutions_l1145_114561

open Real Set

theorem tan_cot_equation_solutions :
  ∃ (S : Finset ℝ), S.card = 18 ∧
  (∀ θ ∈ S, 0 < θ ∧ θ < 2*π) ∧
  (∀ θ ∈ S, tan (3*π * cos θ) = 1 / tan (4*π * sin θ)) ∧
  (∀ θ ∈ Ioo 0 (2*π), tan (3*π * cos θ) = 1 / tan (4*π * sin θ) → θ ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_cot_equation_solutions_l1145_114561


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_sum_l1145_114554

theorem function_max_min_sum (f : ℝ → ℝ) (a : ℝ) :
  (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f x = Real.log (1 - x) - Real.log (1 + x) + a) →
  (∃ M N, (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), f x ≤ M) ∧
          (∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2 : ℝ), N ≤ f x) ∧
          M + N = 1) →
  a = 1/2 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_max_min_sum_l1145_114554


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l1145_114549

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines an ellipse with equation x² + 2y² = 2 -/
def IsOnEllipse (p : Point) : Prop :=
  p.x^2 + 2*p.y^2 = 2

/-- Defines the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- States that F₁ and F₂ are the foci of the ellipse -/
axiom foci_on_ellipse (F₁ F₂ : Point) : 
  ∀ (P : Point), IsOnEllipse P → distance P F₁ + distance P F₂ = 2

/-- Theorem: The minimum value of |PF₁ + PF₂| for any point P on the ellipse is 2 -/
theorem min_sum_of_distances (F₁ F₂ : Point) :
  (∃ (P : Point), IsOnEllipse P) →
  (∀ (P : Point), IsOnEllipse P → ‖(distance P F₁, distance P F₂)‖ ≥ 2) ∧
  (∃ (P : Point), IsOnEllipse P ∧ ‖(distance P F₁, distance P F₂)‖ = 2) := by
  sorry

#check min_sum_of_distances

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_sum_of_distances_l1145_114549


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_a1a6_value_l1145_114589

-- Define the geometric sequence
noncomputable def geometric_sequence (a₁ : ℝ) : ℕ → ℝ
  | 0 => a₁
  | n + 1 => geometric_sequence a₁ n * Real.sqrt 2

-- State the theorem
theorem cos_a1a6_value (a₁ : ℝ) :
  Real.sin (geometric_sequence a₁ 2 * geometric_sequence a₁ 3) = 3/5 →
  Real.cos (geometric_sequence a₁ 1 * geometric_sequence a₁ 6) = 7/25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_a1a6_value_l1145_114589


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_journey_l1145_114570

/-- Represents the distance Sandy walked in each of the first three legs of her journey -/
def distance_per_leg : ℝ := 45

/-- Sandy's final position relative to the starting point -/
def final_position : ℝ × ℝ :=
  (25, distance_per_leg)

/-- The distance from the starting point to Sandy's final position -/
def final_distance : ℝ :=
  45

theorem sandys_journey :
  distance_per_leg = 45 :=
by
  -- The proof goes here
  sorry

#check sandys_journey

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandys_journey_l1145_114570


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1145_114569

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ
  h : y^2 = 8*x

/-- The parabola y^2 = 8x -/
def parabola (p : ℝ × ℝ) : Prop := p.2^2 = 8*p.1

/-- The length between two points on a ℝ² plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: For the parabola y^2 = 8x, if a line passing through its focus
    intersects the parabola at points A and B such that x₁ + x₂ = 5,
    then the length |AB| = 9 -/
theorem parabola_intersection_length 
  (A B : ParabolaPoint) 
  (h_sum : A.x + B.x = 5) :
  distance (A.x, A.y) (B.x, B.y) = 9 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_intersection_length_l1145_114569


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_is_60_meters_l1145_114525

/-- Calculates the distance traveled downstream given boat speed, current speed, and time -/
noncomputable def distance_downstream (boat_speed current_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + current_speed) * (1000 / 3600) * time

/-- Theorem: The distance covered downstream is 60 meters -/
theorem distance_covered_is_60_meters :
  let boat_speed := (20 : ℝ)
  let current_speed := (3 : ℝ)
  let time := 9.390553103577801
  ‖distance_downstream boat_speed current_speed time - 60‖ < 1e-6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_covered_is_60_meters_l1145_114525


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_representation_l1145_114500

-- Define the problem statement
theorem polynomial_representation (n : ℕ) (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_n : n ≥ 2)
  (h_eq : ∀ x, (a₁ * x + b₁)^n + (a₂ * x + b₂)^n = (a₃ * x + b₃)^n) :
  ∃ (A B c₁ c₂ c₃ : ℝ), 
    (a₁ * x + b₁ = c₁ * (A * x + B)) ∧
    (a₂ * x + b₂ = c₂ * (A * x + B)) ∧
    (a₃ * x + b₃ = c₃ * (A * x + B)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_representation_l1145_114500


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_triangle_base_l1145_114543

-- Definitions for the theorem
def external_tangent (r p : ℝ) (C : ℝ × ℝ) : Prop := sorry
def internal_tangent (r R : ℝ) (A : ℝ × ℝ) : Prop := sorry
def isosceles_triangle (r p R : ℝ) : Prop := sorry
def angle_between_equal_sides (r p R : ℝ) : ℝ := sorry
def base_length (r p R : ℝ) : ℝ := sorry

theorem circle_tangency_triangle_base 
  (r p R : ℝ) 
  (hr_pos : r > 0) 
  (hp_pos : p > 0) 
  (hR_pos : R > 0)
  (hrp : r < p) 
  (hext : ∃ (C : ℝ × ℝ), external_tangent r p C)
  (hint1 : ∃ (A : ℝ × ℝ), internal_tangent r R A)
  (hint2 : ∃ (B : ℝ × ℝ), internal_tangent p R B)
  (hiso : isosceles_triangle r p R)
  (hang : angle_between_equal_sides r p R > π/3) :
  base_length r p R = R - r := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_tangency_triangle_base_l1145_114543


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_exists_and_determinable_l1145_114505

/-- Represents a lock -/
def Lock : Type := Fin 4

/-- Represents a key -/
def Key : Type := Fin 6

/-- Represents a configuration of keys and locks -/
def Configuration : Type := Key → Finset Lock

/-- Predicate to check if a configuration is valid -/
def is_valid_configuration (c : Configuration) : Prop :=
  (∀ k : Key, (c k).card = 2) ∧
  (∀ k1 k2 : Key, k1 ≠ k2 → c k1 ≠ c k2)

/-- Represents a test of inserting a key into a lock -/
def Test : Type := Key × Lock

/-- Function to determine the configuration given a list of tests -/
noncomputable def determine_configuration (tests : List Test) : Option Configuration :=
  sorry  -- Implementation not required for the statement

/-- Main theorem statement -/
theorem configuration_exists_and_determinable :
  ∃ (c : Configuration) (tests : List Test),
    is_valid_configuration c ∧
    tests.length ≤ 11 ∧
    determine_configuration tests = some c :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_configuration_exists_and_determinable_l1145_114505


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l1145_114531

/-- An ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse (a b : ℝ) where
  h_positive : 0 < b ∧ b < a

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Dot product of two 2D vectors -/
def dot_product (v w : Point) : ℝ := v.x * w.x + v.y * w.y

/-- Squared magnitude of a 2D vector -/
def magnitude_squared (v : Point) : ℝ := v.x^2 + v.y^2

/-- Eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse a b) : ℝ := Real.sqrt (1 - (b/a)^2)

/-- Main theorem -/
theorem ellipse_eccentricity_sqrt_two_over_two 
  (e : Ellipse a b) 
  (O A B D : Point) 
  (E : Point)
  (h_O : O.x = 0 ∧ O.y = 0)
  (h_E : E.x = 0)
  (h_AB : ∃ t : ℝ, A = Point.mk (t * B.x) (t * B.y))
  (h_AD_perp_AB : dot_product (Point.mk (D.x - A.x) (D.y - A.y)) (Point.mk (B.x - A.x) (B.y - A.y)) = 0)
  (h_E_on_DB : ∃ s : ℝ, E = Point.mk ((1-s)*D.x + s*B.x) ((1-s)*D.y + s*B.y))
  (h_OB_OE : dot_product B E = 2 * magnitude_squared E)
  (h_on_ellipse : ∀ P : Point, (P.x^2 / a^2 + P.y^2 / b^2 = 1) ↔ (P = A ∨ P = B ∨ P = D)) :
  eccentricity e = Real.sqrt 2 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_sqrt_two_over_two_l1145_114531


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l1145_114517

-- Define the binary operation ◇
noncomputable def diamond (a b : ℝ) : ℝ := a / b

-- Theorem statement
theorem diamond_equation_solution :
  ∀ x : ℝ, x ≠ 0 →
  (diamond 2023 (diamond 7 x) = 150) ↔ (x = 1050 / 2023) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diamond_equation_solution_l1145_114517


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_equivalence_l1145_114544

open Real

noncomputable def f (h : ℝ) (x : ℝ) : ℝ := x - log x + h

def interval : Set ℝ := Set.Icc (1/exp 1) (exp 2)

theorem triangle_existence_equivalence (h : ℝ) : 
  (∀ a b c, a ∈ interval → b ∈ interval → c ∈ interval →
    f h a + f h b > f h c ∧ 
    f h b + f h c > f h a ∧ 
    f h c + f h a > f h b) ↔ 
  h > exp 2 - 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_existence_equivalence_l1145_114544


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_not_congruent_different_radii_l1145_114558

/-- Two triangles in a Euclidean plane -/
structure Triangle where
  A : EuclideanSpace ℝ (Fin 2)
  B : EuclideanSpace ℝ (Fin 2)
  C : EuclideanSpace ℝ (Fin 2)

/-- The circumradius of a triangle -/
noncomputable def circumradius (t : Triangle) : ℝ := sorry

/-- The inradius of a triangle -/
noncomputable def inradius (t : Triangle) : ℝ := sorry

/-- Two triangles are similar -/
def similar (t1 t2 : Triangle) : Prop := sorry

/-- Two triangles are congruent -/
def congruent (t1 t2 : Triangle) : Prop := sorry

theorem similar_not_congruent_different_radii (t1 t2 : Triangle) :
  similar t1 t2 ∧ ¬congruent t1 t2 →
  circumradius t1 ≠ circumradius t2 ∧ inradius t1 ≠ inradius t2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_not_congruent_different_radii_l1145_114558


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1145_114513

theorem rectangle_area (perimeter : ℝ) (h1 : perimeter = 160) : ℝ :=
  -- Define the rectangle ABCD
  let side_length : ℝ := perimeter / 6
  let length : ℝ := 2 * side_length
  let width : ℝ := side_length
  let area : ℝ := length * width

  -- State the theorem
  have : area = 25600 / 9 := by
    -- Proof steps would go here
    sorry

  25600 / 9


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1145_114513


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_dm_to_m_conversion_cubic_m_to_dm_conversion_unit_conversions_l1145_114501

-- Define conversion rates
noncomputable def square_dm_to_square_m : ℚ := 1 / 100
noncomputable def cubic_m_to_cubic_dm : ℚ := 1000

-- Theorem for square decimeters to square meters conversion
theorem square_dm_to_m_conversion (x : ℚ) : 
  x * square_dm_to_square_m = x / 100 := by sorry

-- Theorem for cubic meters to cubic decimeters conversion
theorem cubic_m_to_dm_conversion (x : ℚ) : 
  x * cubic_m_to_cubic_dm = x * 1000 := by sorry

-- Main theorem
theorem unit_conversions : 
  (30 * square_dm_to_square_m = (3 / 10 : ℚ)) ∧ 
  (305 / 100 * cubic_m_to_cubic_dm = 3050) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_dm_to_m_conversion_cubic_m_to_dm_conversion_unit_conversions_l1145_114501


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1145_114594

theorem polynomial_identity (P : ℝ → ℝ) : 
  (∀ x : ℝ, (x - 1) * P (x + 1) - (x + 2) * P x = 0) →
  ∃ a : ℝ, ∀ x : ℝ, P x = a * (x^3 - x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_identity_l1145_114594


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1145_114502

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def condition1 (t : Triangle) : Prop :=
  Real.tan t.A * Real.tan t.C + Real.tan t.B * Real.tan t.C = Real.tan t.A * Real.tan t.B

def condition2 (t : Triangle) (m : Real) : Prop :=
  Real.sin t.A ^ 2 + Real.sin t.B ^ 2 = (m ^ 2 + 1) * Real.sin t.C ^ 2

-- State the theorem
theorem triangle_problem (t : Triangle) (m : Real) 
  (h1 : condition1 t) (h2 : condition2 t m) : 
  m = Real.sqrt 2 ∨ m = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l1145_114502


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_sector_l1145_114556

/-- The length of the arc of a sector of a circle -/
noncomputable def arcLength (radius : ℝ) (centralAngle : ℝ) : ℝ :=
  (centralAngle / 360) * 2 * Real.pi * radius

theorem arc_length_sector (radius centralAngle : ℝ) 
  (h1 : radius = 15)
  (h2 : centralAngle = 42) :
  arcLength radius centralAngle = (105 * Real.pi) / 30 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arc_length_sector_l1145_114556


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1145_114590

noncomputable def f (x : Real) : Real := Real.cos x * Real.cos (x + Real.pi/3)

theorem f_properties :
  ∃ (T : Real), T > 0 ∧ (∀ x, f (x + T) = f x) ∧
  (∀ S, S > 0 ∧ (∀ x, f (x + S) = f x) → T ≤ S) ∧
  T = Real.pi ∧
  ∀ (a b c : Real),
    a = 2 →
    f c = -1/4 →
    (1/2) * a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2*a*b))) = 2 * Real.sqrt 3 →
    c = 2 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1145_114590


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1145_114553

noncomputable def f (x : ℝ) := 2 * Real.sin (2 * x + Real.pi / 6)

theorem f_properties :
  (∀ x, f x = 2 * Real.sin (2 * x + Real.pi / 6)) ∧
  (∀ x, f (x + Real.pi) = f x) ∧
  (f (2 * Real.pi / 3) = -2) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 12), f x ≥ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi / 12), f x ≤ Real.sqrt 3) ∧
  (f 0 = 1) ∧
  (f (Real.pi / 12) = Real.sqrt 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1145_114553


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_equals_real_l1145_114509

-- Define set M
def M : Set ℝ := {x | x^2 + 3*x + 2 > 0}

-- Define set N
def N : Set ℝ := {x | (1/2 : ℝ)^x ≤ 4}

-- Theorem statement
theorem union_M_N_equals_real : M ∪ N = Set.univ := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_union_M_N_equals_real_l1145_114509


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_faces_formula_l1145_114516

/-- A pyramid with an equilateral triangular base -/
structure EquilateralBasePyramid where
  /-- The base of the pyramid is an equilateral triangle -/
  base_is_equilateral : Bool
  /-- One lateral face is perpendicular to the base -/
  one_face_perpendicular : Bool
  /-- The angle between the other two lateral faces and the base -/
  alpha : ℝ

/-- The cosine of the angle between the two non-perpendicular lateral faces -/
noncomputable def angle_between_faces (p : EquilateralBasePyramid) : ℝ :=
  -(1 + 3 * Real.cos (2 * p.alpha)) / 4

/-- Theorem: The cosine of the angle between the two non-perpendicular lateral faces
    is equal to -(1 + 3 * cos(2α)) / 4 -/
theorem angle_between_faces_formula (p : EquilateralBasePyramid) 
    (h1 : p.base_is_equilateral = true) 
    (h2 : p.one_face_perpendicular = true) : 
  angle_between_faces p = -(1 + 3 * Real.cos (2 * p.alpha)) / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_faces_formula_l1145_114516


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_one_less_than_negative_two_l1145_114566

theorem one_less_than_negative_two : -3 = -2 - 1 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_one_less_than_negative_two_l1145_114566


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_triangle_l1145_114587

/-- A color type with two possibilities: Blue and Red --/
inductive Color
  | Blue
  | Red

/-- A point in a plane --/
structure Point where
  x : ℝ
  y : ℝ

/-- A function that assigns a color to each point in the plane --/
def ColoringFunction := Point → Color

/-- A triangle in the plane --/
structure Triangle where
  A : Point
  B : Point
  C : Point

/-- Function to calculate the length of a side of a triangle --/
noncomputable def side_length (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x)^2 + (A.y - B.y)^2)

/-- Function to calculate the angle at vertex A of triangle ABC --/
noncomputable def angle (A B C : Point) : ℝ := sorry

/-- Predicate to check if a triangle has the required properties --/
def has_required_properties (t : Triangle) (f : ColoringFunction) : Prop :=
  -- All vertices have the same color
  f t.A = f t.B ∧ f t.B = f t.C ∧
  -- Shortest side has length 1
  min (side_length t.A t.B) (min (side_length t.B t.C) (side_length t.C t.A)) = 1 ∧
  -- Angles are in the ratio 1:2:4
  ∃ (k : ℝ), k > 0 ∧
    angle t.A t.B t.C = k ∧
    angle t.B t.C t.A = 2*k ∧
    angle t.C t.A t.B = 4*k

/-- The main theorem --/
theorem exists_special_triangle (f : ColoringFunction) :
  ∃ (t : Triangle), has_required_properties t f := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_special_triangle_l1145_114587


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_alpha_l1145_114579

theorem sin_plus_cos_alpha (α : ℝ) : 
  (∃ (r : ℝ), r > 0 ∧ r * (Real.sin α) = -1 ∧ r * (Real.cos α) = 2) →
  Real.sin α + Real.cos α = Real.sqrt 5 / 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_alpha_l1145_114579


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1145_114568

noncomputable def f (A ω x : ℝ) : ℝ := A * Real.sin (ω * x - Real.pi / 6) + 1

theorem function_properties (A ω : ℝ) (hA : A > 0) (hω : ω > 0)
  (hmax : ∀ x, f A ω x ≤ 3)
  (hsym : ∀ x, f A ω (x + Real.pi / (2 * ω)) = f A ω x) :
  (∃ k : ℤ, ∀ x, f A ω x = f A ω (Real.pi / (12 * ω) + k * Real.pi / (2 * ω))) ∧
  (Set.Icc 0 3 = Set.image (f A ω) (Set.Icc 0 (Real.pi / 2))) := by
  sorry

#check function_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l1145_114568


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_implies_y_sixth_l1145_114564

theorem cube_root_equation_implies_y_sixth (y : ℝ) (hy : y > 0) 
  (heq : (2 - y^3)^(1/3) + (2 + y^3)^(1/3) = 2) : 
  y^6 = 44/27 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_equation_implies_y_sixth_l1145_114564


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_equals_one_triangle_is_equilateral_l1145_114578

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

-- Part 1
theorem sin_C_equals_one (t : Triangle) :
  t.A + t.B + t.C = π →  -- Sum of angles in a triangle
  2 * t.B = t.A + t.C →  -- A, B, C form an arithmetic sequence
  t.a = 1 →
  t.b = Real.sqrt 3 →
  Real.sin t.C = 1 := by sorry

-- Part 2
theorem triangle_is_equilateral (t : Triangle) :
  t.A + t.B + t.C = π →  -- Sum of angles in a triangle
  2 * t.B = t.A + t.C →  -- A, B, C form an arithmetic sequence
  2 * t.b = t.a + t.c →  -- a, b, c form an arithmetic sequence
  t.A = t.B ∧ t.B = t.C ∧ t.A = π / 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_C_equals_one_triangle_is_equilateral_l1145_114578


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rebate_rate_is_ten_percent_l1145_114521

/-- Represents the investment allocation and profit ranges for a bank's investment strategy. -/
structure InvestmentStrategy where
  total_investment : ℝ
  m_allocation : ℝ
  n_allocation : ℝ
  m_profit_min : ℝ
  m_profit_max : ℝ
  n_profit_min : ℝ
  n_profit_max : ℝ
  bank_profit_min : ℝ
  bank_profit_max : ℝ

/-- Calculates the minimum rebate rate for depositors given an investment strategy. -/
noncomputable def min_rebate_rate (strategy : InvestmentStrategy) : ℝ :=
  (strategy.m_allocation * strategy.m_profit_min + 
   strategy.n_allocation * strategy.n_profit_min - 
   strategy.bank_profit_max) / strategy.total_investment

/-- Theorem stating that the minimum rebate rate for the given investment strategy is 10%. -/
theorem min_rebate_rate_is_ten_percent 
  (strategy : InvestmentStrategy)
  (h1 : strategy.m_allocation = 0.4)
  (h2 : strategy.n_allocation = 0.6)
  (h3 : strategy.m_profit_min = 0.19)
  (h4 : strategy.m_profit_max = 0.24)
  (h5 : strategy.n_profit_min = 0.29)
  (h6 : strategy.n_profit_max = 0.34)
  (h7 : strategy.bank_profit_min = 0.1)
  (h8 : strategy.bank_profit_max = 0.15)
  (h9 : strategy.total_investment = 1) -- Added assumption for total investment
  : min_rebate_rate strategy = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_rebate_rate_is_ten_percent_l1145_114521


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_2_9_3_3_5_6_l1145_114581

theorem cube_root_of_2_9_3_3_5_6 : (2^9 * 3^3 * 5^6)^(1/3) = 600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cube_root_of_2_9_3_3_5_6_l1145_114581


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1145_114582

def geometric_sequence (a : ℕ → ℚ) (r : ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n, a (n + 1) = a n * r ∧ a n ≠ 0

theorem fifth_term_value (a : ℕ → ℚ) (h : geometric_sequence a (1/2)) : a 5 = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifth_term_value_l1145_114582


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larisa_youngest_married_to_boris_l1145_114533

-- Define the people
inductive Person : Type where
  | Andrew : Person
  | Boris : Person
  | Svetlana : Person
  | Larisa : Person

-- Define the age relation
def younger_than : Person → Person → Prop := sorry

-- Define the marriage relation
def married_to : Person → Person → Prop := sorry

-- Define the conditions
axiom two_couples : 
  ∃ (p1 p2 p3 p4 : Person), 
    p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4 ∧
    married_to p1 p2 ∧ married_to p3 p4

axiom oldest_is_larisas_husband :
  ∃ (p : Person), (∀ (q : Person), q ≠ p → younger_than q p) ∧ married_to p Person.Larisa

axiom andrew_age_relation :
  younger_than Person.Andrew Person.Svetlana ∧
  younger_than Person.Larisa Person.Andrew

-- State the theorem to be proved
theorem larisa_youngest_married_to_boris :
  (∀ (p : Person), p ≠ Person.Larisa → younger_than Person.Larisa p) ∧
  married_to Person.Boris Person.Larisa ∧
  ¬married_to Person.Boris Person.Svetlana := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_larisa_youngest_married_to_boris_l1145_114533


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_range_in_interval_l1145_114546

open Real Set

theorem tan_range_in_interval :
  ∀ y : ℝ, (∃ x : ℝ, -π/4 ≤ x ∧ x ≤ π/4 ∧ x ≠ 0 ∧ y = tan x) ↔ y ∈ Ioo (-1) 0 ∪ Ioo 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_range_in_interval_l1145_114546


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bounce_height_theorem_l1145_114530

/-- The number of bounces required for a ball to reach a maximum height less than 2 feet -/
def number_of_bounces : ℕ := 6

/-- The initial height of the ball in feet -/
noncomputable def initial_height : ℝ := 800

/-- The ratio of the bounce height to the previous fall -/
noncomputable def bounce_ratio : ℝ := 1/3

/-- The target height in feet -/
noncomputable def target_height : ℝ := 2

/-- Theorem stating the conditions for the number of bounces -/
theorem bounce_height_theorem :
  (∀ k : ℕ, k < number_of_bounces → initial_height * (bounce_ratio ^ k) ≥ target_height) ∧
  (initial_height * (bounce_ratio ^ number_of_bounces) < target_height) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bounce_height_theorem_l1145_114530


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l1145_114536

theorem quartic_equation_solutions :
  {z : ℂ | z^4 - 8*z^2 + 12 = 0} = 
  {z : ℂ | z = Complex.I * Real.sqrt 6 ∨ 
           z = -Complex.I * Real.sqrt 6 ∨ 
           z = Complex.I * Real.sqrt 2 ∨ 
           z = -Complex.I * Real.sqrt 2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quartic_equation_solutions_l1145_114536


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_third_quadrant_f_specific_angle_l1145_114548

noncomputable def f (α : Real) : Real :=
  (Real.sin (Real.pi - α) * Real.cos (2*Real.pi - α) * Real.cos (-α + 3*Real.pi/2)) /
  (Real.cos (Real.pi/2 - α) * Real.sin (-Real.pi - α))

theorem f_simplification (α : Real) :
  f α = -Real.cos α := by sorry

theorem f_third_quadrant (α : Real) 
  (h1 : Real.pi < α ∧ α < 3*Real.pi/2) 
  (h2 : Real.cos (α - 3*Real.pi/2) = 1/5) :
  f α = -2 * Real.sqrt 6 / 5 := by sorry

theorem f_specific_angle :
  f (-31*Real.pi/3) = -1/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_third_quadrant_f_specific_angle_l1145_114548


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_increasing_f_l1145_114584

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.exp x + (1/2) * a * x^2 - 1

theorem min_a_for_increasing_f :
  (∀ a : ℝ, (∀ x y : ℝ, x > 0 → y > x → f a y > f a x) →
   a ≥ -Real.exp 1) ∧
  (∃ a : ℝ, a = -Real.exp 1 ∧ ∀ x y : ℝ, x > 0 → y > x → f a y > f a x) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a_for_increasing_f_l1145_114584


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_r_squared_and_residuals_l1145_114571

/-- Represents the coefficient of determination in regression analysis -/
def R_squared : ℝ → ℝ := sorry

/-- Represents the sum of squares of residuals in regression analysis -/
def sum_squares_residuals : ℝ → ℝ := sorry

/-- As R² increases, the sum of squares of residuals decreases -/
theorem regression_analysis_r_squared_and_residuals :
  ∀ (x y : ℝ), x < y → R_squared x < R_squared y → sum_squares_residuals x > sum_squares_residuals y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regression_analysis_r_squared_and_residuals_l1145_114571


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_values_l1145_114506

theorem log_equality_implies_values (a b : ℝ) 
  (ha : a ≠ 1) (hb : b ≠ 1) (ha_pos : a > 0) (hb_pos : b > 0)
  (h_log : Real.log 2 / Real.log a = Real.log 4 / Real.log (a / 2) ∧ 
           Real.log 2 / Real.log a = Real.log 3 / Real.log b) :
  a = 1/2 ∧ b = 1/3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equality_implies_values_l1145_114506


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l1145_114511

def S : Finset ℕ := Finset.range 14

theorem count_ordered_pairs : 
  (Finset.filter (fun p : Finset ℕ × Finset ℕ => 
    p.1.Nonempty ∧ 
    p.2.Nonempty ∧ 
    p.1 ∪ p.2 = S ∧ 
    p.1 ∩ p.2 = ∅ ∧ 
    p.1.card ∉ p.1 ∧ 
    p.2.card ∉ p.2) (Finset.product (Finset.powerset S) (Finset.powerset S))).card = 3172 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_ordered_pairs_l1145_114511


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_abc_l1145_114514

def S : Set Char := {'a', 'b', 'c'}

theorem proper_subsets_of_abc :
  {A : Set Char | A ⊂ S} = {∅, {'a'}, {'b'}, {'c'}, {'a', 'b'}, {'a', 'c'}, {'b', 'c'}} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_proper_subsets_of_abc_l1145_114514


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l1145_114583

-- Define the even function f
noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x + φ) - Real.cos (2 * x + φ)

-- Define the function g as a translation of f
noncomputable def g (x : ℝ) (φ : ℝ) : ℝ := f (x - Real.pi / 6) φ

-- State the theorem
theorem g_monotone_increasing (φ : ℝ) (h : 0 < φ ∧ φ < Real.pi) :
  ∃ (a b : ℝ), a = -Real.pi/3 ∧ b = Real.pi/6 ∧
  ∀ (x y : ℝ), a < x ∧ x < y ∧ y < b → g x φ < g y φ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_monotone_increasing_l1145_114583


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a4_l1145_114529

/-- An arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  d : ℚ
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of first n terms of an arithmetic sequence -/
noncomputable def S (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem min_a4 (seq : ArithmeticSequence) 
  (h1 : S seq 4 ≤ 4) 
  (h2 : S seq 5 ≥ 15) : 
  seq.a 4 ≥ 7 ∧ ∃ (seq' : ArithmeticSequence), seq'.a 4 = 7 ∧ S seq' 4 ≤ 4 ∧ S seq' 5 ≥ 15 := by
  sorry

#check min_a4

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_a4_l1145_114529


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_is_4_sqrt_5_l1145_114560

/-- The line l: (2m+1)x+(m+1)y-7m-4=0 (m∈R) -/
def line (m x y : ℝ) : Prop :=
  (2*m + 1)*x + (m + 1)*y - 7*m - 4 = 0

/-- The circle C: (x-1)²+(y-2)²=25 -/
def circle_eq (x y : ℝ) : Prop :=
  (x - 1)^2 + (y - 2)^2 = 25

/-- The length of the shortest chord intercepted by the circle on the line -/
noncomputable def shortest_chord_length : ℝ := 4 * Real.sqrt 5

/-- Theorem stating that the length of the shortest chord intercepted by the circle on the line is 4√5 -/
theorem shortest_chord_length_is_4_sqrt_5 :
  ∀ m : ℝ, ∃ x y : ℝ,
    line m x y ∧ circle_eq x y ∧
    shortest_chord_length = 4 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_chord_length_is_4_sqrt_5_l1145_114560


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stand_distance_l1145_114527

-- Define the walking speeds
noncomputable def speed1 : ℝ := 5
noncomputable def speed2 : ℝ := 6

-- Define the time differences in hours
noncomputable def timeDiff1 : ℝ := 12 / 60  -- 12 minutes converted to hours
noncomputable def timeDiff2 : ℝ := 15 / 60  -- 15 minutes converted to hours

-- Define the distance to the bus stand
noncomputable def distance : ℝ := 13.5

-- Theorem statement
theorem bus_stand_distance :
  (distance / speed1 - timeDiff1 = distance / speed2 + timeDiff2) ∧
  (distance = 13.5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bus_stand_distance_l1145_114527


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_convex_cyclic_quads_l1145_114503

/-- A convex cyclic quadrilateral with integer sides. -/
structure ConvexCyclicQuad where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  perimeter_eq : a + b + c + d = 36
  convex : a < b + c + d ∧ b < a + c + d ∧ c < a + b + d ∧ d < a + b + c

/-- The set of all valid ConvexCyclicQuad structures. -/
def ValidQuads : Set ConvexCyclicQuad :=
  {q : ConvexCyclicQuad | True}

/-- Instance to make ValidQuads a finite type -/
instance : Fintype ValidQuads :=
  sorry

/-- The main theorem stating the number of convex cyclic quadrilaterals -/
theorem count_convex_cyclic_quads :
  Fintype.card ValidQuads = 823 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_convex_cyclic_quads_l1145_114503


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_from_diagonals_l1145_114538

/-- Given a rhombus with diagonals of lengths 6 and 8, its perimeter is 20 -/
theorem rhombus_perimeter_from_diagonals :
  ∀ (d1 d2 : ℝ), d1 = 6 → d2 = 8 →
  let side := Real.sqrt ((d1/2)^2 + (d2/2)^2)
  4 * side = 20 := by
  intros d1 d2 h1 h2
  simp [h1, h2]
  norm_num
  ring
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rhombus_perimeter_from_diagonals_l1145_114538


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1145_114555

-- Define the points and vectors
variable (O B C A : EuclideanSpace ℝ (Fin 3))
variable (a b : ℝ)

-- Define the conditions
axiom A_on_BC : ∃ t : ℝ, t ∈ Set.Ioo 0 1 ∧ A = (1 - t) • B + t • C
axiom O_not_on_BC : O ∉ affineSpan ℝ {B, C}
axiom vector_equation : A - O = a • (B - O) + (2 * b) • (C - O)

-- Define the expression to be minimized
noncomputable def f (a b : ℝ) : ℝ := 2 / (3*a + 4*b) + 1 / (a + 3*b)

-- State the theorem
theorem min_value_theorem : 
  ∀ a b : ℝ, f a b ≥ 8/5 ∧ (∃ a b : ℝ, f a b = 8/5) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1145_114555


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_2023pi_over_4_l1145_114539

theorem tan_alpha_plus_2023pi_over_4 (α : Real) :
  ((-3/5 : Real)^2 + (4/5 : Real)^2 = 1) →
  (Real.cos α = -3/5) →
  (Real.sin α = 4/5) →
  Real.tan (α + 2023 * Real.pi / 4) = 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_plus_2023pi_over_4_l1145_114539


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_proof_l1145_114528

/-- The volume of a cone given its base area and height -/
noncomputable def cone_volume (base_area : ℝ) (height : ℝ) : ℝ := (1/3) * base_area * height

theorem cone_height_proof (volume : ℝ) (base_area : ℝ) (height : ℝ) 
  (h1 : volume = 18)
  (h2 : base_area = 3)
  (h3 : volume = cone_volume base_area height) :
  height = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cone_height_proof_l1145_114528


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inscribed_circle_l1145_114534

-- Define the circle
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define the right triangle
structure RightTriangle (A B C : ℝ × ℝ) : Prop where
  right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0

-- Define the theorem
theorem right_triangle_inscribed_circle 
  (O' : ℝ × ℝ) (r : ℝ) (A B C D : ℝ × ℝ) 
  (h_circle : Circle O' r)
  (h_triangle : RightTriangle A B C)
  (h_diameter : (A.1 - B.1)^2 + (A.2 - B.2)^2 = (2*r)^2)
  (h_D_on_circle : D ∈ Circle O' r)
  (h_D_on_AC : ∃ t : ℝ, D = (A.1 + t*(C.1 - A.1), A.2 + t*(C.2 - A.2)) ∧ t > 1) :
  (A.1 - D.1)^2 + (A.2 - D.2)^2 = (A.1 - B.1)^2 + (A.2 - B.2)^2 ∧ 
  (B.1 - D.1)^2 + (B.2 - D.2)^2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_inscribed_circle_l1145_114534


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_four_l1145_114577

/-- The daily sales volume function -/
noncomputable def y (x a : ℝ) : ℝ := a / (x - 3) + 10 * (x - 6)^2

/-- The daily profit function -/
noncomputable def f (x : ℝ) : ℝ := y x 2 * (x - 3)

theorem max_profit_at_four :
  ∀ x : ℝ, 3 < x → x < 6 →
  y 5 2 = 11 →
  f 4 ≥ f x := by
  sorry

#check max_profit_at_four

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_profit_at_four_l1145_114577


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_when_a_zero_f_properties_when_a_ge_e_l1145_114518

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 2) * Real.exp (2 * x) + a * (x + 1)^2

-- Theorem for the case when a = 0
theorem f_properties_when_a_zero :
  -- Minimum value of f when a = 0
  (∃ (x : ℝ), f 0 x = -1/2 * Real.exp 3 ∧ ∀ (y : ℝ), f 0 y ≥ f 0 x) ∧
  -- No maximum value when a = 0
  (¬∃ (x : ℝ), ∀ (y : ℝ), f 0 y ≤ f 0 x) ∧
  -- For any m ≠ n where f(m) = f(n), m + n < 3
  (∀ (m n : ℝ), m ≠ n → f 0 m = f 0 n → m + n < 3) := by
sorry

-- Theorem for the case when a ≥ e
theorem f_properties_when_a_ge_e (a : ℝ) (h : a ≥ Real.exp 1) :
  -- f has a unique minimum point x₀
  ∃! (x₀ : ℝ), ∀ (y : ℝ), f a y ≥ f a x₀ ∧
  -- -3/(2e) < f(x₀) < -3/e^2
  -3 / (2 * Real.exp 1) < f a x₀ ∧ f a x₀ < -3 / (Real.exp 1)^2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_when_a_zero_f_properties_when_a_ge_e_l1145_114518


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1145_114540

noncomputable def f (x : ℝ) := (x - 5) / ((x - 3) * (x + 2))

theorem inequality_solution :
  ∀ x : ℝ, f x < 0 ↔ x ∈ Set.Iio (-2) ∪ Set.Ioo (-2) 3 ∪ Set.Ioo 3 5 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_l1145_114540
