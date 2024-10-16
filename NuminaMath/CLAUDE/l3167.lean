import Mathlib

namespace NUMINAMATH_CALUDE_triple_involution_properties_l3167_316765

/-- A function f: ℝ → ℝ satisfying f(f(f(x))) = x for all x ∈ ℝ -/
def triple_involution (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f (f x)) = x

theorem triple_involution_properties (f : ℝ → ℝ) (h : triple_involution f) :
  (∀ x y : ℝ, f x = f y → x = y) ∧ 
  (¬ (∀ x y : ℝ, x < y → f x > f y)) ∧
  ((∀ x y : ℝ, x < y → f x < f y) → ∀ x : ℝ, f x = x) :=
by sorry

end NUMINAMATH_CALUDE_triple_involution_properties_l3167_316765


namespace NUMINAMATH_CALUDE_divisibility_by_37_l3167_316763

theorem divisibility_by_37 (a b c d e f : ℕ) :
  (a < 10) → (b < 10) → (c < 10) → (d < 10) → (e < 10) → (f < 10) →
  (37 ∣ (100 * a + 10 * b + c + 100 * d + 10 * e + f)) →
  (37 ∣ (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f)) :=
by sorry

#check divisibility_by_37

end NUMINAMATH_CALUDE_divisibility_by_37_l3167_316763


namespace NUMINAMATH_CALUDE_wrong_mark_value_l3167_316716

/-- Proves that the wrongly entered mark is 85 given the conditions of the problem -/
theorem wrong_mark_value (n : ℕ) (correct_mark : ℕ) (average_increase : ℚ) 
  (h1 : n = 80)
  (h2 : correct_mark = 45)
  (h3 : average_increase = 1/2) : 
  ∃ (wrong_mark : ℕ), wrong_mark = 85 ∧ 
    (wrong_mark - correct_mark : ℚ) = n * average_increase := by
  sorry

end NUMINAMATH_CALUDE_wrong_mark_value_l3167_316716


namespace NUMINAMATH_CALUDE_amin_iff_ali_can_color_all_red_l3167_316741

-- Define a type for cell colors
inductive CellColor
| Black
| White
| Red

-- Define the table as a function from coordinates to cell colors
def Table (n : ℕ) := Fin n → Fin n → CellColor

-- Define Amin's move
def AminMove (t : Table n) (row : Fin n) : Table n :=
  sorry

-- Define Ali's move
def AliMove (t : Table n) (col : Fin n) : Table n :=
  sorry

-- Define a predicate to check if all cells are red
def AllRed (t : Table n) : Prop :=
  ∀ i j, t i j = CellColor.Red

-- Define a predicate to check if Amin can color all cells red
def AminCanColorAllRed (t : Table n) : Prop :=
  sorry

-- Define a predicate to check if Ali can color all cells red
def AliCanColorAllRed (t : Table n) : Prop :=
  sorry

-- The main theorem
theorem amin_iff_ali_can_color_all_red (n : ℕ) (t : Table n) :
  AminCanColorAllRed t ↔ AliCanColorAllRed t :=
sorry

end NUMINAMATH_CALUDE_amin_iff_ali_can_color_all_red_l3167_316741


namespace NUMINAMATH_CALUDE_container_weight_l3167_316734

/-- Given the weights of different metal bars, calculate the total weight of a container --/
theorem container_weight (copper_weight tin_weight steel_weight : ℝ) 
  (h1 : steel_weight = 2 * tin_weight)
  (h2 : steel_weight = copper_weight + 20)
  (h3 : copper_weight = 90) : 
  20 * steel_weight + 20 * copper_weight + 20 * tin_weight = 5100 := by
  sorry

#check container_weight

end NUMINAMATH_CALUDE_container_weight_l3167_316734


namespace NUMINAMATH_CALUDE_makeup_exam_probability_l3167_316737

/-- Given a class with a total number of students and a number of students who need to take a makeup exam,
    calculate the probability of a student participating in the makeup exam. -/
theorem makeup_exam_probability (total_students : ℕ) (makeup_students : ℕ) 
    (h1 : total_students = 42) (h2 : makeup_students = 3) :
    (makeup_students : ℚ) / total_students = 1 / 14 := by
  sorry

#check makeup_exam_probability

end NUMINAMATH_CALUDE_makeup_exam_probability_l3167_316737


namespace NUMINAMATH_CALUDE_triangle_EC_length_l3167_316754

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the points D and E
def D (t : Triangle) : ℝ × ℝ := sorry
def E (t : Triangle) : ℝ × ℝ := sorry

-- Define the length of a segment
def length (p q : ℝ × ℝ) : ℝ := sorry

-- Define the angle between two segments
def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define perpendicularity
def perpendicular (p q r s : ℝ × ℝ) : Prop := sorry

theorem triangle_EC_length (t : Triangle) : 
  angle t.A t.B t.C = π/4 →          -- ∠A = 45°
  length t.B t.C = 10 →              -- BC = 10
  perpendicular (D t) t.B t.A t.C → -- BD ⊥ AC
  perpendicular (E t) t.C t.A t.B → -- CE ⊥ AB
  angle (D t) t.B t.C = 2 * angle (E t) t.C t.B → -- m∠DBC = 2m∠ECB
  length (E t) t.C = 5 * Real.sqrt 6 := by
    sorry

#check triangle_EC_length

end NUMINAMATH_CALUDE_triangle_EC_length_l3167_316754


namespace NUMINAMATH_CALUDE_no_zeros_in_2_16_l3167_316788

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property of f having a unique zero point in (0, 2)
def has_unique_zero_in_0_2 (f : ℝ → ℝ) : Prop :=
  ∃! x, x ∈ (Set.Ioo 0 2) ∧ f x = 0

-- Theorem statement
theorem no_zeros_in_2_16 (h : has_unique_zero_in_0_2 f) :
  ∀ x ∈ Set.Ico 2 16, f x ≠ 0 := by
  sorry

end NUMINAMATH_CALUDE_no_zeros_in_2_16_l3167_316788


namespace NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l3167_316744

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else (n % 8) :: decimal_to_octal (n / 8)

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_101101_equals_octal_55 : 
  decimal_to_octal (binary_to_decimal binary_101101) = [5, 5] := by
  sorry

end NUMINAMATH_CALUDE_binary_101101_equals_octal_55_l3167_316744


namespace NUMINAMATH_CALUDE_find_x_l3167_316733

theorem find_x : ∃ x : ℝ, (0.5 * x = 0.25 * 1500 - 30) ∧ (x = 690) := by
  sorry

end NUMINAMATH_CALUDE_find_x_l3167_316733


namespace NUMINAMATH_CALUDE_paper_stack_height_l3167_316782

/-- Given a ream of paper with 500 sheets that is 5 cm thick,
    prove that a stack of 7.5 cm high contains 750 sheets. -/
theorem paper_stack_height (sheets_per_ream : ℕ) (ream_thickness : ℝ) (stack_height : ℝ) :
  sheets_per_ream = 500 →
  ream_thickness = 5 →
  stack_height = 7.5 →
  (stack_height / ream_thickness) * sheets_per_ream = 750 := by
  sorry

#check paper_stack_height

end NUMINAMATH_CALUDE_paper_stack_height_l3167_316782


namespace NUMINAMATH_CALUDE_smallest_n_is_three_l3167_316702

-- Define the complex number i
noncomputable def i : ℂ := Complex.I

-- Define x and y
noncomputable def x : ℂ := (-1 + i * Real.sqrt 3) / 2
noncomputable def y : ℂ := (-1 - i * Real.sqrt 3) / 2

-- Define the property we want to prove
def is_smallest_n (n : ℕ) : Prop :=
  n > 0 ∧ x^n + y^n = 2 ∧ ∀ m : ℕ, 0 < m ∧ m < n → x^m + y^m ≠ 2

-- The theorem we want to prove
theorem smallest_n_is_three : is_smallest_n 3 := by sorry

end NUMINAMATH_CALUDE_smallest_n_is_three_l3167_316702


namespace NUMINAMATH_CALUDE_symmetric_points_sum_power_l3167_316780

theorem symmetric_points_sum_power (m n : ℤ) : 
  (2*n - m = -14) → (m = 4) → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_power_l3167_316780


namespace NUMINAMATH_CALUDE_fibonacci_divisibility_l3167_316712

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

/-- Extractable in p-arithmetic -/
def extractable_in_p_arithmetic (x : ℝ) (p : ℕ) : Prop := sorry

theorem fibonacci_divisibility (p k : ℕ) (h_prime : Nat.Prime p) 
  (h_sqrt5 : extractable_in_p_arithmetic (Real.sqrt 5) p) :
  p^k ∣ fib (p^(k-1) * (p-1)) :=
sorry

end NUMINAMATH_CALUDE_fibonacci_divisibility_l3167_316712


namespace NUMINAMATH_CALUDE_circle_area_difference_l3167_316775

theorem circle_area_difference : 
  let r1 : ℝ := 30
  let d2 : ℝ := 15
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 843.75 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l3167_316775


namespace NUMINAMATH_CALUDE_five_workers_required_l3167_316749

/-- Represents the project parameters and progress -/
structure ProjectStatus :=
  (total_days : ℕ)
  (elapsed_days : ℕ)
  (initial_workers : ℕ)
  (completed_fraction : ℚ)

/-- Calculates the minimum number of workers required to complete the project on schedule -/
def min_workers_required (status : ProjectStatus) : ℕ :=
  sorry

/-- Theorem stating that for the given project status, 5 workers are required -/
theorem five_workers_required (status : ProjectStatus) 
  (h1 : status.total_days = 20)
  (h2 : status.elapsed_days = 5)
  (h3 : status.initial_workers = 10)
  (h4 : status.completed_fraction = 1/4) :
  min_workers_required status = 5 := by
  sorry

end NUMINAMATH_CALUDE_five_workers_required_l3167_316749


namespace NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3167_316767

/-- 
Given an isosceles triangle where one of the angles opposite an equal side is 40°,
prove that the largest angle measures 100°.
-/
theorem isosceles_triangle_largest_angle (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α = β →            -- The triangle is isosceles (two angles are equal)
  α = 40 →           -- One of the angles opposite an equal side is 40°
  max α (max β γ) = 100 := by  -- The largest angle measures 100°
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_largest_angle_l3167_316767


namespace NUMINAMATH_CALUDE_file_storage_problem_l3167_316756

/-- Represents the minimum number of disks required to store files -/
def min_disks (total_files : ℕ) (disk_capacity : ℚ) 
  (files_size_1 : ℕ) (size_1 : ℚ)
  (files_size_2 : ℕ) (size_2 : ℚ)
  (size_3 : ℚ) : ℕ :=
  sorry

theorem file_storage_problem :
  let total_files : ℕ := 33
  let disk_capacity : ℚ := 1.44
  let files_size_1 : ℕ := 3
  let size_1 : ℚ := 1.1
  let files_size_2 : ℕ := 15
  let size_2 : ℚ := 0.6
  let size_3 : ℚ := 0.5
  let remaining_files : ℕ := total_files - files_size_1 - files_size_2
  min_disks total_files disk_capacity files_size_1 size_1 files_size_2 size_2 size_3 = 17 :=
by sorry

end NUMINAMATH_CALUDE_file_storage_problem_l3167_316756


namespace NUMINAMATH_CALUDE_pen_notebook_cost_l3167_316709

theorem pen_notebook_cost :
  ∀ (p n : ℕ), 
    p > n ∧ 
    p > 0 ∧ 
    n > 0 ∧ 
    17 * p + 5 * n = 200 →
    p + n = 16 := by
  sorry

end NUMINAMATH_CALUDE_pen_notebook_cost_l3167_316709


namespace NUMINAMATH_CALUDE_fraction_equivalence_l3167_316766

theorem fraction_equivalence : (16 : ℝ) / (8 * 17) = 1.6 / (0.8 * 17) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l3167_316766


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3167_316728

-- Define a scalene triangle with prime side lengths
def ScaleneTriangleWithPrimeSides (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c

-- Define a function to check if the perimeter is prime
def HasPrimePerimeter (a b c : ℕ) : Prop :=
  Nat.Prime (a + b + c)

-- Theorem statement
theorem smallest_prime_perimeter_scalene_triangle :
  ∀ a b c : ℕ,
    ScaleneTriangleWithPrimeSides a b c →
    HasPrimePerimeter a b c →
    a + b + c ≥ 23 :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_scalene_triangle_l3167_316728


namespace NUMINAMATH_CALUDE_marcus_pretzels_l3167_316761

theorem marcus_pretzels (total : ℕ) (john : ℕ) (alan : ℕ) (marcus : ℕ) 
  (h1 : total = 95)
  (h2 : john = 28)
  (h3 : alan = john - 9)
  (h4 : marcus = john + 12) :
  marcus = 40 := by
  sorry

end NUMINAMATH_CALUDE_marcus_pretzels_l3167_316761


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l3167_316786

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  3 * X^5 - 2 * X^3 + 5 * X - 8 = (X^2 - 3 * X + 2) * q + (74 * X - 76) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l3167_316786


namespace NUMINAMATH_CALUDE_rectangular_field_width_l3167_316772

/-- Proves that the width of a rectangular field is 1400/29 meters given specific conditions -/
theorem rectangular_field_width (w : ℝ) : 
  w > 0 → -- width is positive
  (2*w + 2*(7/5*w) + w = 280) → -- combined perimeter equation
  w = 1400/29 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_width_l3167_316772


namespace NUMINAMATH_CALUDE_max_value_g_range_of_a_inequality_for_f_l3167_316717

noncomputable section

def f (x : ℝ) : ℝ := Real.log x

def g (x : ℝ) : ℝ := f (x + 1) - x

theorem max_value_g :
  ∀ x > -1, g x ≤ 0 ∧ ∃ x₀ > -1, g x₀ = 0 :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x ≤ a * x ∧ a * x ≤ x^2 + 1) →
  (1 / Real.exp 1 ≤ a ∧ a ≤ 2) :=
sorry

theorem inequality_for_f (x₁ x₂ : ℝ) (h : x₁ > x₂ ∧ x₂ > 0) :
  (f x₁ - f x₂) / (x₁ - x₂) > (2 * x₂) / (x₁^2 + x₂^2) :=
sorry

end NUMINAMATH_CALUDE_max_value_g_range_of_a_inequality_for_f_l3167_316717


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_18_l3167_316785

/-- The area of a triangle with vertices at (1, 2), (7, 6), and (1, 8) is 18 square units. -/
theorem triangle_area : ℝ :=
  let A : ℝ × ℝ := (1, 2)
  let B : ℝ × ℝ := (7, 6)
  let C : ℝ × ℝ := (1, 8)
  let area := abs ((A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) / 2)
  18

/-- The theorem statement. -/
theorem triangle_area_is_18 : triangle_area = 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_18_l3167_316785


namespace NUMINAMATH_CALUDE_intersection_area_l3167_316711

/-- The area of intersection between two boards of widths 5 inches and 7 inches,
    crossing at a 45-degree angle. -/
theorem intersection_area (board1_width board2_width : ℝ) (angle : ℝ) :
  board1_width = 5 →
  board2_width = 7 →
  angle = π / 4 →
  (board1_width * board2_width * Real.sin angle) = (35 * Real.sqrt 2) / 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_area_l3167_316711


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3167_316751

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := ((Complex.I - 1)^2 + 4) / (Complex.I + 1)
  Complex.im z = -3 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3167_316751


namespace NUMINAMATH_CALUDE_sum_factorials_mod_20_l3167_316732

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def sum_factorials (n : ℕ) : ℕ :=
  match n with
  | 0 => factorial 0
  | n + 1 => factorial (n + 1) + sum_factorials n

theorem sum_factorials_mod_20 : sum_factorials 50 % 20 = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_factorials_mod_20_l3167_316732


namespace NUMINAMATH_CALUDE_original_number_is_332_l3167_316727

/-- Given a three-digit number abc, returns the sum of abc, acb, bca, bac, cab, and cba -/
def sum_permutations (a b c : Nat) : Nat :=
  100 * a + 10 * b + c +
  100 * a + 10 * c + b +
  100 * b + 10 * c + a +
  100 * b + 10 * a + c +
  100 * c + 10 * a + b +
  100 * c + 10 * b + a

/-- The original number abc satisfies the given conditions -/
theorem original_number_is_332 : 
  ∃ (a b c : Nat), 
    a < 10 ∧ b < 10 ∧ c < 10 ∧ 
    sum_permutations a b c = 4332 ∧
    100 * a + 10 * b + c = 332 :=
by sorry

end NUMINAMATH_CALUDE_original_number_is_332_l3167_316727


namespace NUMINAMATH_CALUDE_function_properties_l3167_316787

-- Define the function f
def f (x : ℝ) : ℝ := |2 * x - 3|

-- State the theorem
theorem function_properties :
  (∀ x : ℝ, f x ≤ 1 ↔ 1 ≤ x ∧ x ≤ 2) ∧
  (∀ a b c x : ℝ, a > 0 → b > 0 → c > 0 → a + b + c = 1 →
    f x - 2 * |x + 3| ≤ 1/a + 1/b + 1/c) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3167_316787


namespace NUMINAMATH_CALUDE_sqrt_three_not_in_P_l3167_316720

-- Define the set P
def P : Set ℝ := {x | x^2 - Real.sqrt 2 * x ≤ 0}

-- State the theorem
theorem sqrt_three_not_in_P : Real.sqrt 3 ∉ P := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_not_in_P_l3167_316720


namespace NUMINAMATH_CALUDE_walters_age_l3167_316796

theorem walters_age (walter_age_2005 : ℕ) (grandmother_age_2005 : ℕ) : 
  walter_age_2005 = grandmother_age_2005 / 3 →
  (2005 - walter_age_2005) + (2005 - grandmother_age_2005) = 3858 →
  walter_age_2005 + 5 = 43 :=
by
  sorry

end NUMINAMATH_CALUDE_walters_age_l3167_316796


namespace NUMINAMATH_CALUDE_function_property_l3167_316798

theorem function_property (f : ℝ → ℝ) (m : ℝ)
  (h1 : ∀ x, f (2 + x) = f (-x))
  (h2 : ∀ x y, x ≥ 1 → y ≥ 1 → x < y → f y < f x)
  (h3 : f (1 - m) < f m) :
  m > 1/2 := by sorry

end NUMINAMATH_CALUDE_function_property_l3167_316798


namespace NUMINAMATH_CALUDE_same_color_plate_probability_l3167_316738

theorem same_color_plate_probability (total : ℕ) (yellow : ℕ) (green : ℕ) 
  (h1 : total = yellow + green)
  (h2 : yellow = 7)
  (h3 : green = 5) :
  (Nat.choose yellow 2 + Nat.choose green 2) / Nat.choose total 2 = 31 / 66 := by
  sorry

end NUMINAMATH_CALUDE_same_color_plate_probability_l3167_316738


namespace NUMINAMATH_CALUDE_pauls_remaining_crayons_l3167_316750

/-- Given that Paul initially had 479 crayons and lost or gave away 345 crayons,
    prove that he has 134 crayons left. -/
theorem pauls_remaining_crayons (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
    (h1 : initial = 479) 
    (h2 : lost = 345) 
    (h3 : remaining = initial - lost) : 
  remaining = 134 := by
  sorry

end NUMINAMATH_CALUDE_pauls_remaining_crayons_l3167_316750


namespace NUMINAMATH_CALUDE_winnie_keeps_remainder_l3167_316770

/-- The number of balloons Winnie keeps for herself -/
def balloons_kept (total_balloons : ℕ) (num_friends : ℕ) : ℕ :=
  total_balloons % num_friends

/-- The total number of balloons Winnie has -/
def total_balloons : ℕ := 17 + 33 + 65 + 83

/-- The number of friends Winnie has -/
def num_friends : ℕ := 10

theorem winnie_keeps_remainder :
  balloons_kept total_balloons num_friends = 8 :=
sorry

end NUMINAMATH_CALUDE_winnie_keeps_remainder_l3167_316770


namespace NUMINAMATH_CALUDE_corner_subset_exists_l3167_316724

/-- A corner is a finite set of n-tuples of positive integers with a specific property. -/
def Corner (n : ℕ) : Type :=
  {S : Set (Fin n → ℕ+) // S.Finite ∧
    ∀ a b : Fin n → ℕ+, a ∈ S → (∀ k, b k ≤ a k) → b ∈ S}

/-- The theorem states that in any infinite collection of corners,
    there exist two corners where one is a subset of the other. -/
theorem corner_subset_exists {n : ℕ} (h : n > 0) (S : Set (Corner n)) (hS : Set.Infinite S) :
  ∃ C₁ C₂ : Corner n, C₁ ∈ S ∧ C₂ ∈ S ∧ C₁.1 ⊆ C₂.1 :=
sorry

end NUMINAMATH_CALUDE_corner_subset_exists_l3167_316724


namespace NUMINAMATH_CALUDE_champion_sequences_l3167_316736

/-- The number of letters in CHAMPION -/
def num_letters : ℕ := 8

/-- The number of letters in each sequence -/
def sequence_length : ℕ := 5

/-- The number of letters available for the last position (excluding N) -/
def last_position_options : ℕ := 6

/-- The number of positions to fill after fixing the first and last -/
def middle_positions : ℕ := sequence_length - 2

/-- The number of letters available for the middle positions -/
def middle_options : ℕ := num_letters - 2

theorem champion_sequences :
  (middle_options.factorial / (middle_options - middle_positions).factorial) * last_position_options = 720 := by
  sorry

end NUMINAMATH_CALUDE_champion_sequences_l3167_316736


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l3167_316739

theorem sum_of_roots_equals_fourteen : 
  ∃ (x₁ x₂ : ℝ), (x₁ - 7)^2 = 16 ∧ (x₂ - 7)^2 = 16 ∧ x₁ + x₂ = 14 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_fourteen_l3167_316739


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3167_316714

theorem sqrt_fraction_simplification :
  (Real.sqrt 6) / (Real.sqrt 10) = (Real.sqrt 15) / 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l3167_316714


namespace NUMINAMATH_CALUDE_faulty_token_identifiable_l3167_316791

/-- Represents the possible outcomes of a weighing --/
inductive WeighingResult
  | Equal : WeighingResult
  | LeftHeavier : WeighingResult
  | RightHeavier : WeighingResult

/-- Represents a token with a nominal value and an actual weight --/
structure Token where
  nominal_value : ℕ
  actual_weight : ℕ

/-- Represents a set of four tokens --/
def TokenSet := (Token × Token × Token × Token)

/-- Represents a weighing action on the balance scale --/
def Weighing := (List Token) → (List Token) → WeighingResult

/-- Represents a strategy for determining the faulty token --/
def Strategy := TokenSet → Weighing → Weighing → Option Token

/-- States that exactly one token in the set has an incorrect weight --/
def ExactlyOneFaulty (ts : TokenSet) : Prop := sorry

/-- States that a strategy correctly identifies the faulty token --/
def StrategyCorrect (s : Strategy) : Prop := sorry

theorem faulty_token_identifiable :
  ∃ (s : Strategy), StrategyCorrect s :=
sorry

end NUMINAMATH_CALUDE_faulty_token_identifiable_l3167_316791


namespace NUMINAMATH_CALUDE_count_pairs_eq_fib_l3167_316746

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- Count of pairs (α,S) with specific properties -/
def count_pairs (n : ℕ) : ℕ :=
  sorry

theorem count_pairs_eq_fib (n : ℕ) :
  count_pairs n = n! * fib (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_count_pairs_eq_fib_l3167_316746


namespace NUMINAMATH_CALUDE_adrianna_gum_l3167_316784

/-- Calculates the remaining pieces of gum after sharing with friends -/
def remaining_gum (initial : ℕ) (additional : ℕ) (friends : ℕ) : ℕ :=
  initial + additional - friends

/-- Proves that Adrianna has 2 pieces of gum left -/
theorem adrianna_gum : remaining_gum 10 3 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_adrianna_gum_l3167_316784


namespace NUMINAMATH_CALUDE_max_value_polynomial_l3167_316762

theorem max_value_polynomial (a b : ℝ) (h : a + b = 5) :
  ∃ M : ℝ, M = 6084 / 17 ∧ 
  ∀ x y : ℝ, x + y = 5 → 
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ M ∧
  ∃ a b : ℝ, a + b = 5 ∧ 
  a^4*b + a^3*b + a^2*b + a*b + a*b^2 + a*b^3 + a*b^4 = M :=
sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l3167_316762


namespace NUMINAMATH_CALUDE_ducks_theorem_l3167_316713

def ducks_remaining (initial : ℕ) : ℕ :=
  let after_first := initial - (initial / 4)
  let after_second := after_first - (after_first / 6)
  after_second - (after_second * 3 / 10)

theorem ducks_theorem : ducks_remaining 320 = 140 := by
  sorry

end NUMINAMATH_CALUDE_ducks_theorem_l3167_316713


namespace NUMINAMATH_CALUDE_unique_salaries_l3167_316795

/-- Represents the weekly salaries of three employees -/
structure Salaries where
  n : ℝ  -- Salary of employee N
  m : ℝ  -- Salary of employee M
  p : ℝ  -- Salary of employee P

/-- Checks if the given salaries satisfy the problem conditions -/
def satisfiesConditions (s : Salaries) : Prop :=
  s.m = 1.2 * s.n ∧
  s.p = 1.5 * s.m ∧
  s.n + s.m + s.p = 1500

/-- Theorem stating that the given salaries are the unique solution -/
theorem unique_salaries : 
  ∃! s : Salaries, satisfiesConditions s ∧ 
    s.n = 375 ∧ s.m = 450 ∧ s.p = 675 := by
  sorry

end NUMINAMATH_CALUDE_unique_salaries_l3167_316795


namespace NUMINAMATH_CALUDE_not_always_valid_solution_set_l3167_316769

theorem not_always_valid_solution_set (a b : ℝ) (h : b ≠ 0) :
  ¬ (∀ x, x ∈ Set.Ioi (b / a) ↔ a * x + b > 0) :=
sorry

end NUMINAMATH_CALUDE_not_always_valid_solution_set_l3167_316769


namespace NUMINAMATH_CALUDE_symmetric_points_difference_l3167_316783

/-- Given two points A and B symmetric with respect to the origin, prove that a - b = 5 -/
theorem symmetric_points_difference (a b : ℝ) : 
  (∀ (x y : ℝ), x = -2 ∧ y = b → (x, y) = (-a, -3)) → a - b = 5 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_difference_l3167_316783


namespace NUMINAMATH_CALUDE_x_gt_2_necessary_not_sufficient_for_x_gt_5_l3167_316790

theorem x_gt_2_necessary_not_sufficient_for_x_gt_5 :
  (∀ x : ℝ, x > 5 → x > 2) ∧ (∃ x : ℝ, x > 2 ∧ x ≤ 5) := by
  sorry

end NUMINAMATH_CALUDE_x_gt_2_necessary_not_sufficient_for_x_gt_5_l3167_316790


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3167_316792

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ + a₁₀ = 16, the sum a₄ + a₆ + a₈ = 24 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 2 + a 10 = 16) : 
  a 4 + a 6 + a 8 = 24 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3167_316792


namespace NUMINAMATH_CALUDE_point_division_theorem_l3167_316719

/-- Given a line segment AB and a point P on AB such that AP:PB = 3:4,
    prove that P = (4/7)*A + (3/7)*B -/
theorem point_division_theorem (A B P : ℝ × ℝ) :
  (P.1 - A.1) / (B.1 - P.1) = 3 / 4 ∧
  (P.2 - A.2) / (B.2 - P.2) = 3 / 4 →
  P = ((4:ℝ)/7) • A + ((3:ℝ)/7) • B :=
sorry

end NUMINAMATH_CALUDE_point_division_theorem_l3167_316719


namespace NUMINAMATH_CALUDE_quadratic_vertex_l3167_316730

/-- A quadratic function passing through specific points has its vertex at x = 5 -/
theorem quadratic_vertex (a b c : ℝ) : 
  (4 = a * 2^2 + b * 2 + c) →
  (4 = a * 8^2 + b * 8 + c) →
  (13 = a * 10^2 + b * 10 + c) →
  (-b / (2 * a) = 5) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_vertex_l3167_316730


namespace NUMINAMATH_CALUDE_equation_satisfied_l3167_316799

theorem equation_satisfied (x y z : ℤ) : 
  x = z ∧ y = x + 1 → x * (x - y) + y * (y - z) + z * (z - x) = 2 := by
sorry

end NUMINAMATH_CALUDE_equation_satisfied_l3167_316799


namespace NUMINAMATH_CALUDE_square_difference_plus_fifty_l3167_316742

theorem square_difference_plus_fifty : (312^2 - 288^2) / 24 + 50 = 650 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_plus_fifty_l3167_316742


namespace NUMINAMATH_CALUDE_seat_representation_l3167_316721

/-- Represents a seat in a movie theater -/
structure Seat :=
  (row : ℕ)
  (column : ℕ)

/-- The notation for representing seats in the movie theater -/
def seat_notation (r : ℕ) (c : ℕ) : Seat := ⟨r, c⟩

/-- Theorem stating that if (5, 2) represents the seat in the 5th row and 2nd column,
    then (7, 3) represents the seat in the 7th row and 3rd column -/
theorem seat_representation :
  (seat_notation 5 2 = ⟨5, 2⟩) →
  (seat_notation 7 3 = ⟨7, 3⟩) :=
by sorry

end NUMINAMATH_CALUDE_seat_representation_l3167_316721


namespace NUMINAMATH_CALUDE_layla_earnings_l3167_316743

/-- Calculates the total earnings from babysitting given the hourly rates and hours worked for three families. -/
def total_earnings (rate1 rate2 rate3 : ℕ) (hours1 hours2 hours3 : ℕ) : ℕ :=
  rate1 * hours1 + rate2 * hours2 + rate3 * hours3

/-- Proves that Layla's total earnings from babysitting equal $273 given the specified rates and hours. -/
theorem layla_earnings : total_earnings 15 18 20 7 6 3 = 273 := by
  sorry

#eval total_earnings 15 18 20 7 6 3

end NUMINAMATH_CALUDE_layla_earnings_l3167_316743


namespace NUMINAMATH_CALUDE_baseball_football_fans_l3167_316726

theorem baseball_football_fans (total : ℕ) (baseball_only : ℕ) (football_only : ℕ) (neither : ℕ) 
  (h1 : total = 16)
  (h2 : baseball_only = 2)
  (h3 : football_only = 3)
  (h4 : neither = 6) :
  total - baseball_only - football_only - neither = 5 := by
sorry

end NUMINAMATH_CALUDE_baseball_football_fans_l3167_316726


namespace NUMINAMATH_CALUDE_sum_of_powers_l3167_316718

theorem sum_of_powers (ω : ℂ) (h1 : ω^11 = 1) (h2 : ω ≠ 1) :
  ω^10 + ω^14 + ω^18 + ω^22 + ω^26 + ω^30 + ω^34 + ω^38 + ω^42 + ω^46 + ω^50 + ω^54 + ω^58 = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_powers_l3167_316718


namespace NUMINAMATH_CALUDE_cupcakes_per_package_l3167_316723

theorem cupcakes_per_package 
  (initial_cupcakes : ℕ) 
  (eaten_cupcakes : ℕ) 
  (num_packages : ℕ) 
  (h1 : initial_cupcakes = 20)
  (h2 : eaten_cupcakes = 11)
  (h3 : num_packages = 3)
  (h4 : eaten_cupcakes < initial_cupcakes) :
  (initial_cupcakes - eaten_cupcakes) / num_packages = 3 :=
by sorry

end NUMINAMATH_CALUDE_cupcakes_per_package_l3167_316723


namespace NUMINAMATH_CALUDE_l_shaped_area_l3167_316760

/-- The area of an L-shaped region formed by subtracting two smaller squares from a larger square --/
theorem l_shaped_area (square_side : ℝ) (small_square1_side : ℝ) (small_square2_side : ℝ)
  (h1 : square_side = 6)
  (h2 : small_square1_side = 2)
  (h3 : small_square2_side = 3)
  (h4 : small_square1_side < square_side)
  (h5 : small_square2_side < square_side) :
  square_side^2 - small_square1_side^2 - small_square2_side^2 = 23 := by
  sorry

#check l_shaped_area

end NUMINAMATH_CALUDE_l_shaped_area_l3167_316760


namespace NUMINAMATH_CALUDE_austin_to_dallas_passes_three_buses_l3167_316789

/-- Represents the time in hours since midnight -/
def Time := ℝ

/-- Represents the distance between Dallas and Austin in arbitrary units -/
def Distance := ℝ

/-- Represents the schedule and movement of buses -/
structure BusSchedule where
  departure_interval : ℝ
  departure_offset : ℝ
  trip_duration : ℝ

/-- Calculates the number of buses passed during a trip -/
def buses_passed (austin_schedule dallas_schedule : BusSchedule) : ℕ :=
  sorry

theorem austin_to_dallas_passes_three_buses 
  (austin_schedule : BusSchedule) 
  (dallas_schedule : BusSchedule) : 
  austin_schedule.departure_interval = 2 ∧ 
  austin_schedule.departure_offset = 0.5 ∧
  austin_schedule.trip_duration = 6 ∧
  dallas_schedule.departure_interval = 2 ∧
  dallas_schedule.departure_offset = 0 ∧
  dallas_schedule.trip_duration = 6 →
  buses_passed austin_schedule dallas_schedule = 3 :=
sorry

end NUMINAMATH_CALUDE_austin_to_dallas_passes_three_buses_l3167_316789


namespace NUMINAMATH_CALUDE_deepak_age_l3167_316753

/-- Given the ratio of Rahul's age to Deepak's age and Rahul's future age, 
    determine Deepak's present age -/
theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →   -- Age ratio condition
  rahul_age + 6 = 38 →                     -- Rahul's future age condition
  deepak_age = 24 := by                    -- Deepak's present age to prove
sorry

end NUMINAMATH_CALUDE_deepak_age_l3167_316753


namespace NUMINAMATH_CALUDE_defective_units_count_prove_defective_units_l3167_316793

theorem defective_units_count : ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun total_units customer_a customer_b customer_c defective_units =>
    total_units = 20 ∧
    customer_a = 3 ∧
    customer_b = 5 ∧
    customer_c = 7 ∧
    defective_units = total_units - (customer_a + customer_b + customer_c) ∧
    defective_units = 5

theorem prove_defective_units : ∃ (d : ℕ), defective_units_count 20 3 5 7 d :=
  sorry

end NUMINAMATH_CALUDE_defective_units_count_prove_defective_units_l3167_316793


namespace NUMINAMATH_CALUDE_youseff_distance_to_office_l3167_316774

theorem youseff_distance_to_office (x : ℝ) 
  (walk_time : ℝ → ℝ) 
  (bike_time : ℝ → ℝ) 
  (h1 : ∀ d, walk_time d = d) 
  (h2 : ∀ d, bike_time d = d / 3) 
  (h3 : walk_time x = bike_time x + 14) : 
  x = 21 := by
sorry

end NUMINAMATH_CALUDE_youseff_distance_to_office_l3167_316774


namespace NUMINAMATH_CALUDE_equation_solution_l3167_316777

theorem equation_solution :
  ∃ x : ℝ, x - 15 = 30 ∧ x = 45 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l3167_316777


namespace NUMINAMATH_CALUDE_area_at_stage_6_l3167_316755

/-- The side length of each square -/
def square_side : ℕ := 3

/-- The number of stages -/
def num_stages : ℕ := 6

/-- The area of the rectangle at a given stage -/
def rectangle_area (stage : ℕ) : ℕ :=
  stage * square_side * square_side

/-- Theorem: The area of the rectangle at Stage 6 is 54 square inches -/
theorem area_at_stage_6 : rectangle_area num_stages = 54 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_6_l3167_316755


namespace NUMINAMATH_CALUDE_original_selling_price_l3167_316710

theorem original_selling_price (CP : ℝ) : 
  (CP * 1.25 = CP + 0.25 * CP) →  -- 25% profit condition
  (320 = CP - 0.5 * CP) →         -- 50% loss condition
  CP * 1.25 = 800 := by
sorry

end NUMINAMATH_CALUDE_original_selling_price_l3167_316710


namespace NUMINAMATH_CALUDE_grace_weekly_charge_l3167_316706

/-- Grace's weekly charge given her total earnings and work duration -/
def weekly_charge (total_earnings : ℚ) (weeks : ℕ) : ℚ :=
  total_earnings / weeks

/-- Theorem: Grace's weekly charge is $300 -/
theorem grace_weekly_charge :
  let total_earnings : ℚ := 1800
  let weeks : ℕ := 6
  weekly_charge total_earnings weeks = 300 := by
  sorry

end NUMINAMATH_CALUDE_grace_weekly_charge_l3167_316706


namespace NUMINAMATH_CALUDE_megan_markers_l3167_316748

theorem megan_markers (x : ℕ) : x + 109 = 326 → x = 217 := by
  sorry

end NUMINAMATH_CALUDE_megan_markers_l3167_316748


namespace NUMINAMATH_CALUDE_tan_alpha_value_l3167_316735

theorem tan_alpha_value (α : Real) (h : Real.tan (α - π/4) = 1/6) : Real.tan α = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l3167_316735


namespace NUMINAMATH_CALUDE_no_real_roots_l3167_316768

theorem no_real_roots : ¬∃ (x : ℝ), Real.sqrt (x + 9) - Real.sqrt (x - 6) + 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_l3167_316768


namespace NUMINAMATH_CALUDE_unique_m_satisfying_conditions_l3167_316731

theorem unique_m_satisfying_conditions : ∃! m : ℤ,
  (∃ x : ℤ, (m * x - 1) / (x - 1) = 2 + 1 / (1 - x)) ∧
  (4 - 2 * (m - 1) * (1 / 2) ≥ 0) ∧
  m ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_m_satisfying_conditions_l3167_316731


namespace NUMINAMATH_CALUDE_tan_105_degrees_l3167_316781

theorem tan_105_degrees :
  Real.tan (105 * π / 180) = -2 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_105_degrees_l3167_316781


namespace NUMINAMATH_CALUDE_existence_of_greater_indices_l3167_316722

theorem existence_of_greater_indices
  (a b c : ℕ → ℕ) :
  ∃ p q : ℕ, p > q ∧ a p ≥ a q ∧ b p ≥ b q ∧ c p ≥ c q :=
by sorry

end NUMINAMATH_CALUDE_existence_of_greater_indices_l3167_316722


namespace NUMINAMATH_CALUDE_susie_pizza_price_l3167_316794

/-- The price of a whole pizza given the conditions of Susie's pizza sales -/
theorem susie_pizza_price (price_per_slice : ℚ) (slices_sold : ℕ) (whole_pizzas_sold : ℕ) (total_revenue : ℚ) :
  price_per_slice = 3 →
  slices_sold = 24 →
  whole_pizzas_sold = 3 →
  total_revenue = 117 →
  ∃ (whole_pizza_price : ℚ), whole_pizza_price = 15 ∧
    price_per_slice * slices_sold + whole_pizza_price * whole_pizzas_sold = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_susie_pizza_price_l3167_316794


namespace NUMINAMATH_CALUDE_maryville_population_increase_l3167_316758

/-- Calculates the average annual population increase given initial and final populations and the time period. -/
def averageAnnualIncrease (initialPopulation finalPopulation : ℕ) (years : ℕ) : ℚ :=
  (finalPopulation - initialPopulation : ℚ) / years

/-- Theorem stating that the average annual population increase in Maryville between 2000 and 2005 is 3400. -/
theorem maryville_population_increase : averageAnnualIncrease 450000 467000 5 = 3400 := by
  sorry

end NUMINAMATH_CALUDE_maryville_population_increase_l3167_316758


namespace NUMINAMATH_CALUDE_complex_root_property_l3167_316747

variable (a b c d e m n : ℝ)
variable (z : ℂ)

theorem complex_root_property :
  (z = m + n * Complex.I) →
  (a * z^4 + Complex.I * b * z^2 + c * z^2 + Complex.I * d * z + e = 0) →
  (a * (-m + n * Complex.I)^4 + Complex.I * b * (-m + n * Complex.I)^2 + 
   c * (-m + n * Complex.I)^2 + Complex.I * d * (-m + n * Complex.I) + e = 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_root_property_l3167_316747


namespace NUMINAMATH_CALUDE_pentagonal_prism_faces_l3167_316779

/-- A polyhedron with pentagonal bases and lateral faces -/
structure PentagonalPrism where
  base_edges : ℕ
  base_count : ℕ
  lateral_faces : ℕ

/-- The total number of faces in a pentagonal prism -/
def total_faces (p : PentagonalPrism) : ℕ :=
  p.base_count + p.lateral_faces

/-- Theorem: A pentagonal prism has 7 faces in total -/
theorem pentagonal_prism_faces :
  ∀ (p : PentagonalPrism), 
    p.base_edges = 5 → 
    p.base_count = 2 → 
    p.lateral_faces = 5 → 
    total_faces p = 7 := by
  sorry

#check pentagonal_prism_faces

end NUMINAMATH_CALUDE_pentagonal_prism_faces_l3167_316779


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3167_316764

/-- Prove that for a hyperbola with the given properties, its eccentricity is 5/3 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (P : ℝ × ℝ),
    (P.1^2 / a^2 - P.2^2 / b^2 = 1) ∧
    (∃ (F₁ F₂ : ℝ × ℝ),
      (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
       Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 3 * b) ∧
      (Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) *
       Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 9 * a * b / 4)) →
  Real.sqrt (a^2 + b^2) / a = 5 / 3 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3167_316764


namespace NUMINAMATH_CALUDE_calculate_expression_l3167_316745

theorem calculate_expression : (Real.sqrt 3) ^ 0 + 2⁻¹ + Real.sqrt 2 * Real.cos (45 * π / 180) - |-(1/2)| = 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3167_316745


namespace NUMINAMATH_CALUDE_expression_bounds_l3167_316705

theorem expression_bounds (p q r s : ℝ) 
  (hp : 0 ≤ p ∧ p ≤ 2) (hq : 0 ≤ q ∧ q ≤ 2) (hr : 0 ≤ r ∧ r ≤ 2) (hs : 0 ≤ s ∧ s ≤ 2) :
  4 * Real.sqrt 2 ≤ Real.sqrt (p^2 + (2-q)^2) + Real.sqrt (q^2 + (2-r)^2) + 
    Real.sqrt (r^2 + (2-s)^2) + Real.sqrt (s^2 + (2-p)^2) ∧
  Real.sqrt (p^2 + (2-q)^2) + Real.sqrt (q^2 + (2-r)^2) + 
    Real.sqrt (r^2 + (2-s)^2) + Real.sqrt (s^2 + (2-p)^2) ≤ 8 ∧
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 2 ∧
    4 * Real.sqrt (t^2 + (2-t)^2) = 4 * Real.sqrt 2 ∨
    4 * Real.sqrt (t^2 + (2-t)^2) = 8 :=
by sorry


end NUMINAMATH_CALUDE_expression_bounds_l3167_316705


namespace NUMINAMATH_CALUDE_square_perimeter_contradiction_l3167_316704

theorem square_perimeter_contradiction (perimeter : ℝ) (side_length : ℝ) : 
  perimeter = 4 → side_length = 2 → perimeter ≠ 4 * side_length :=
by
  sorry

#check square_perimeter_contradiction

end NUMINAMATH_CALUDE_square_perimeter_contradiction_l3167_316704


namespace NUMINAMATH_CALUDE_quadratic_function_property_l3167_316771

/-- The quadratic function y = (x+1)(ax+2a+2) -/
def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (a * x + 2 * a + 2)

theorem quadratic_function_property (a : ℝ) (x₁ x₂ y₁ y₂ : ℝ) 
  (ha : a ≠ 0)
  (hx : x₁ + x₂ = 2)
  (horder : x₁ < x₂)
  (hy : y₁ > y₂)
  (hf₁ : f a x₁ = y₁)
  (hf₂ : f a x₂ = y₂) :
  a < -2/5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_property_l3167_316771


namespace NUMINAMATH_CALUDE_cinema_seating_l3167_316757

/-- The number of chairs occupied in a cinema row --/
def occupied_chairs (chairs_between : ℕ) : ℕ :=
  chairs_between + 2

theorem cinema_seating (chairs_between : ℕ) 
  (h : chairs_between = 30) : occupied_chairs chairs_between = 32 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seating_l3167_316757


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3167_316701

theorem negation_of_universal_proposition :
  (¬ ∀ (x : ℕ), x > 0 → (x - 1)^2 > 0) ↔ (∃ (x : ℕ), x > 0 ∧ (x - 1)^2 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3167_316701


namespace NUMINAMATH_CALUDE_function_ranges_l3167_316725

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x - 2
def g (a : ℝ) (x : ℝ) : ℝ := -x^2 + x + a

-- State the theorem
theorem function_ranges :
  ∀ a : ℝ,
  (f a (-1) = 0) →
  (∀ x₁ ∈ Set.Icc (1/4 : ℝ) 1, ∃ x₂ ∈ Set.Icc 1 2, g a x₁ > f a x₂ + 3) →
  (Set.range (f a) = Set.Ici (-9/4 : ℝ)) ∧
  (a ∈ Set.Ioi 1) :=
by sorry

end NUMINAMATH_CALUDE_function_ranges_l3167_316725


namespace NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3167_316740

/-- The number of sides in a regular nonagon -/
def n : ℕ := 9

/-- The total number of line segments (sides and diagonals) in a regular nonagon -/
def total_segments : ℕ := n.choose 2

/-- The number of diagonals in a regular nonagon -/
def num_diagonals : ℕ := total_segments - n

/-- The number of ways to choose two diagonals -/
def ways_to_choose_diagonals : ℕ := num_diagonals.choose 2

/-- The number of ways to choose four points that form intersecting diagonals -/
def intersecting_diagonals : ℕ := n.choose 4

/-- The probability that two randomly chosen diagonals intersect inside the nonagon -/
def probability_intersect : ℚ := intersecting_diagonals / ways_to_choose_diagonals

theorem nonagon_diagonal_intersection_probability :
  probability_intersect = 6 / 13 := by sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_intersection_probability_l3167_316740


namespace NUMINAMATH_CALUDE_sqrt_difference_approximation_l3167_316703

theorem sqrt_difference_approximation : 
  ∃ ε > 0, |Real.sqrt (49 + 81) - Real.sqrt (64 - 36) - 6.1| < ε :=
sorry

end NUMINAMATH_CALUDE_sqrt_difference_approximation_l3167_316703


namespace NUMINAMATH_CALUDE_tenth_term_of_sequence_l3167_316715

def inversely_proportional_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ n : ℕ, n ≥ 1 → a (n + 1) * a n = k

theorem tenth_term_of_sequence (a : ℕ → ℝ) :
  inversely_proportional_sequence a →
  a 1 = 3 →
  a 2 = 4 →
  a 10 = 4 :=
by
  sorry

end NUMINAMATH_CALUDE_tenth_term_of_sequence_l3167_316715


namespace NUMINAMATH_CALUDE_prime_sequence_l3167_316797

theorem prime_sequence (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
sorry

end NUMINAMATH_CALUDE_prime_sequence_l3167_316797


namespace NUMINAMATH_CALUDE_two_days_satisfy_l3167_316708

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- The number of days in the month -/
def monthLength : Nat := 30

/-- Function to check if a given day results in equal Tuesdays and Thursdays -/
def equalTuesdaysThursdays (startDay : DayOfWeek) : Bool :=
  sorry -- Implementation details omitted

/-- Count the number of days that satisfy the condition -/
def countSatisfyingDays : Nat :=
  sorry -- Implementation details omitted

/-- Theorem stating that exactly two days satisfy the condition -/
theorem two_days_satisfy :
  countSatisfyingDays = 2 :=
sorry

end NUMINAMATH_CALUDE_two_days_satisfy_l3167_316708


namespace NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3167_316778

theorem smallest_prime_divisor_of_sum : 
  ∃ (p : Nat), Prime p ∧ p ∣ (2^12 + 3^14 + 7^4) ∧ ∀ (q : Nat), Prime q → q ∣ (2^12 + 3^14 + 7^4) → p ≤ q :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_divisor_of_sum_l3167_316778


namespace NUMINAMATH_CALUDE_problem_solution_l3167_316752

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |a * x + 1|

-- State the theorem
theorem problem_solution :
  -- Part I
  (∃ a : ℝ, ∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) ∧
  (∀ a : ℝ, (∀ x : ℝ, f a x ≤ 3 ↔ -2 ≤ x ∧ x ≤ 1) → a = 2) ∧
  -- Part II
  (∀ x : ℝ, |f 2 x - 2 * f 2 (x/2)| ≤ 1) ∧
  (∀ k : ℝ, (∀ x : ℝ, |f 2 x - 2 * f 2 (x/2)| ≤ k) → k ≥ 1) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l3167_316752


namespace NUMINAMATH_CALUDE_largest_divisor_of_n_l3167_316776

theorem largest_divisor_of_n (n : ℕ+) (h : 450 ∣ n^2) : 
  ∀ d : ℕ, d ∣ n → d ≤ 30 ∧ 30 ∣ n := by
  sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n_l3167_316776


namespace NUMINAMATH_CALUDE_ellipse_intersection_l3167_316707

/-- Definition of an ellipse with given foci and a point on it -/
def is_ellipse (f₁ f₂ p : ℝ × ℝ) : Prop :=
  Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
  Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
  Real.sqrt ((0 - f₁.1)^2 + (0 - f₁.2)^2) +
  Real.sqrt ((0 - f₂.1)^2 + (0 - f₂.2)^2)

theorem ellipse_intersection :
  let f₁ : ℝ × ℝ := (0, 5)
  let f₂ : ℝ × ℝ := (4, 0)
  let p : ℝ × ℝ := (28/9, 0)
  is_ellipse f₁ f₂ (0, 0) → is_ellipse f₁ f₂ p :=
by sorry

end NUMINAMATH_CALUDE_ellipse_intersection_l3167_316707


namespace NUMINAMATH_CALUDE_unique_circle_circumference_equals_area_l3167_316759

theorem unique_circle_circumference_equals_area :
  ∃! r : ℝ, r > 0 ∧ 2 * Real.pi * r = Real.pi * r^2 := by sorry

end NUMINAMATH_CALUDE_unique_circle_circumference_equals_area_l3167_316759


namespace NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l3167_316700

theorem sum_of_four_consecutive_integers_divisible_by_two (n : ℤ) : 
  2 ∣ ((n - 1) + n + (n + 1) + (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_consecutive_integers_divisible_by_two_l3167_316700


namespace NUMINAMATH_CALUDE_no_common_solution_l3167_316729

theorem no_common_solution : ¬∃ x : ℝ, (263 - x = 108) ∧ (25 * x = 1950) ∧ (x / 15 = 64) := by
  sorry

end NUMINAMATH_CALUDE_no_common_solution_l3167_316729


namespace NUMINAMATH_CALUDE_prob_three_odd_dice_l3167_316773

/-- The number of dice being rolled -/
def num_dice : ℕ := 4

/-- The number of sides on each die -/
def num_sides : ℕ := 8

/-- The probability of rolling an odd number on a single die -/
def prob_odd : ℚ := 1/2

/-- The probability of rolling an even number on a single die -/
def prob_even : ℚ := 1/2

/-- The number of ways to choose 3 dice out of 4 -/
def choose_3_from_4 : ℕ := 4

theorem prob_three_odd_dice :
  (choose_3_from_4 : ℚ) * prob_odd^3 * prob_even^(num_dice - 3) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_dice_l3167_316773
