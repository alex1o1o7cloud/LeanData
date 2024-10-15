import Mathlib

namespace NUMINAMATH_CALUDE_russian_doll_purchase_l2517_251733

/-- Given a person's savings for a certain number of items at an original price,
    calculate how many items they can buy when the price drops to a new lower price. -/
theorem russian_doll_purchase (original_price new_price : ℚ) (original_quantity : ℕ) :
  original_price > 0 →
  new_price > 0 →
  new_price < original_price →
  (original_price * original_quantity) / new_price = 20 :=
by
  sorry

#check russian_doll_purchase (4 : ℚ) (3 : ℚ) 15

end NUMINAMATH_CALUDE_russian_doll_purchase_l2517_251733


namespace NUMINAMATH_CALUDE_largest_prime_to_test_primality_l2517_251776

theorem largest_prime_to_test_primality (n : ℕ) (h : 1100 ≤ n ∧ n ≤ 1150) :
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q, Nat.Prime q → q^2 ≤ n → q ≤ p) →
  (∃ p : ℕ, Nat.Prime p ∧ p^2 ≤ n ∧ ∀ q, Nat.Prime q → q^2 ≤ n → q ≤ p ∧ p = 31) :=
by sorry

end NUMINAMATH_CALUDE_largest_prime_to_test_primality_l2517_251776


namespace NUMINAMATH_CALUDE_coordinates_of_point_b_l2517_251729

/-- Given a line segment AB with length 3, parallel to the y-axis, and point A at coordinates (-1, 2),
    the coordinates of point B must be either (-1, 5) or (-1, -1). -/
theorem coordinates_of_point_b (A B : ℝ × ℝ) : 
  A = (-1, 2) → 
  (B.1 - A.1 = 0) →  -- AB is parallel to y-axis
  ((B.1 - A.1)^2 + (B.2 - A.2)^2 = 3^2) →  -- AB length is 3
  (B = (-1, 5) ∨ B = (-1, -1)) := by
sorry

end NUMINAMATH_CALUDE_coordinates_of_point_b_l2517_251729


namespace NUMINAMATH_CALUDE_not_arithmetic_sequence_x_i_l2517_251778

/-- Given real constants a and b, a geometric sequence {c_i} with common ratio ≠ 1,
    and the line ax + by + c_i = 0 intersecting the parabola y^2 = 2px (p > 0)
    forming chords with midpoints M_i(x_i, y_i), prove that {x_i} cannot be an arithmetic sequence. -/
theorem not_arithmetic_sequence_x_i 
  (a b : ℝ) 
  (c : ℕ+ → ℝ) 
  (p : ℝ) 
  (hp : p > 0)
  (hc : ∃ (r : ℝ), r ≠ 1 ∧ ∀ (i : ℕ+), c (i + 1) = r * c i)
  (x y : ℕ+ → ℝ)
  (h_intersect : ∀ (i : ℕ+), ∃ (t : ℝ), a * t + b * y i + c i = 0 ∧ (y i)^2 = 2 * p * t)
  (h_midpoint : ∀ (i : ℕ+), ∃ (t₁ t₂ : ℝ), 
    a * t₁ + b * (y i) + c i = 0 ∧ (y i)^2 = 2 * p * t₁ ∧
    a * t₂ + b * (y i) + c i = 0 ∧ (y i)^2 = 2 * p * t₂ ∧
    x i = (t₁ + t₂) / 2 ∧ y i = (y i + y i) / 2) :
  ¬ (∃ (d : ℝ), ∀ (i : ℕ+), x (i + 1) - x i = d) :=
sorry

end NUMINAMATH_CALUDE_not_arithmetic_sequence_x_i_l2517_251778


namespace NUMINAMATH_CALUDE_polynomial_remainder_theorem_remainder_equals_897_l2517_251745

def f (x : ℝ) : ℝ := 5*x^8 - 3*x^7 + 2*x^6 - 9*x^4 + 3*x^3 - 7

theorem polynomial_remainder_theorem (f : ℝ → ℝ) (a : ℝ) :
  ∃ (q : ℝ → ℝ), f = fun x ↦ (x - a) * q x + f a := by sorry

theorem remainder_equals_897 :
  ∃ (q : ℝ → ℝ), f = fun x ↦ (3*x - 6) * q x + 897 := by
  have h : ∃ (q : ℝ → ℝ), f = fun x ↦ (x - 2) * q x + f 2 := polynomial_remainder_theorem f 2
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_theorem_remainder_equals_897_l2517_251745


namespace NUMINAMATH_CALUDE_birthday_money_l2517_251704

theorem birthday_money (age : ℕ) (money : ℕ) : 
  age = 3 * 3 →
  money = 5 * age →
  money = 45 := by
sorry

end NUMINAMATH_CALUDE_birthday_money_l2517_251704


namespace NUMINAMATH_CALUDE_area_ratio_theorem_l2517_251710

-- Define the triangle XYZ
structure Triangle :=
  (X Y Z : ℝ × ℝ)
  (XY : Real)
  (YZ : Real)
  (XZ : Real)

-- Define points P and Q
structure Points (t : Triangle) :=
  (P : ℝ × ℝ)
  (Q : ℝ × ℝ)
  (XP : Real)
  (XQ : Real)

-- Define the area ratio
def areaRatio (t : Triangle) (pts : Points t) : ℚ := sorry

-- State the theorem
theorem area_ratio_theorem (t : Triangle) (pts : Points t) :
  t.XY = 30 →
  t.YZ = 45 →
  t.XZ = 54 →
  pts.XP = 18 →
  pts.XQ = 36 →
  areaRatio t pts = 27 / 50 := by sorry

end NUMINAMATH_CALUDE_area_ratio_theorem_l2517_251710


namespace NUMINAMATH_CALUDE_evaluate_expression_l2517_251724

theorem evaluate_expression :
  -(16 / 4 * 11 - 70 + 5 * 11) = -29 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l2517_251724


namespace NUMINAMATH_CALUDE_min_bing_toys_l2517_251743

/-- Represents the cost and pricing of Olympic mascot toys --/
structure OlympicToys where
  bing_cost : ℕ  -- Cost of Bing Dwen Dwen
  shuey_cost : ℕ  -- Cost of Shuey Rongrong
  bing_price : ℕ  -- Selling price of Bing Dwen Dwen
  shuey_price : ℕ  -- Selling price of Shuey Rongrong

/-- Theorem about the minimum number of Bing Dwen Dwen toys to purchase --/
theorem min_bing_toys (t : OlympicToys) 
  (h1 : 4 * t.bing_cost + 5 * t.shuey_cost = 1000)
  (h2 : 5 * t.bing_cost + 10 * t.shuey_cost = 1550)
  (h3 : t.bing_price = 180)
  (h4 : t.shuey_price = 100)
  (h5 : ∀ x : ℕ, x + (180 - x) = 180)
  (h6 : ∀ x : ℕ, x * (t.bing_price - t.bing_cost) + (180 - x) * (t.shuey_price - t.shuey_cost) ≥ 4600) :
  ∃ (min_bing : ℕ), min_bing = 100 ∧ 
    ∀ (x : ℕ), x ≥ min_bing → 
      x * (t.bing_price - t.bing_cost) + (180 - x) * (t.shuey_price - t.shuey_cost) ≥ 4600 :=
sorry

end NUMINAMATH_CALUDE_min_bing_toys_l2517_251743


namespace NUMINAMATH_CALUDE_expression_evaluation_l2517_251731

theorem expression_evaluation : 
  let x : ℝ := 2
  (2 * x + 3) * (2 * x - 3) + (x - 2)^2 - 3 * x * (x - 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2517_251731


namespace NUMINAMATH_CALUDE_disprove_propositions_l2517_251773

open Set

/-- Definition of an M point -/
def is_M_point (f : ℝ → ℝ) (c : ℝ) (a b : ℝ) : Prop :=
  ∃ I : Set ℝ, IsOpen I ∧ c ∈ I ∩ Icc a b ∧
  ∀ x ∈ I ∩ Icc a b, x ≠ c → f x < f c

/-- Main theorem stating the existence of a function that disproves both propositions -/
theorem disprove_propositions : ∃ f : ℝ → ℝ,
  (∃ a b x₀ : ℝ, x₀ ∈ Icc a b ∧ 
    (∀ x ∈ Icc a b, f x ≤ f x₀) ∧ 
    ¬is_M_point f x₀ a b) ∧
  (∀ a b : ℝ, a < b → is_M_point f b a b) ∧
  ¬StrictMono f :=
sorry

end NUMINAMATH_CALUDE_disprove_propositions_l2517_251773


namespace NUMINAMATH_CALUDE_binomial_product_l2517_251759

variable (x : ℝ)

theorem binomial_product :
  (4 * x - 3) * (2 * x + 7) = 8 * x^2 + 22 * x - 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_product_l2517_251759


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2517_251737

def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ :=
  λ n => a₁ + (n - 1) * d

theorem arithmetic_sequence_property :
  ∀ (a : ℕ → ℝ),
  (a 1 = 1) →
  (a 3 = 5) →
  (∀ n : ℕ, a n = arithmeticSequence (a 1) ((a 3 - a 1) / 2) n) →
  2 * (a 9) - (a 10) = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2517_251737


namespace NUMINAMATH_CALUDE_center_is_eight_l2517_251706

/-- Represents a 3x3 grid --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions are adjacent --/
def adjacent (p q : Fin 3 × Fin 3) : Prop :=
  (p.1 = q.1 ∧ (p.2 = q.2 + 1 ∨ p.2 + 1 = q.2)) ∨
  (p.2 = q.2 ∧ (p.1 = q.1 + 1 ∨ p.1 + 1 = q.1))

/-- Check if the grid satisfies the consecutive adjacency property --/
def consecutive_adjacent (g : Grid) : Prop :=
  ∀ n : Fin 8, ∃ p q : Fin 3 × Fin 3, 
    g p.1 p.2 = n ∧ g q.1 q.2 = n + 1 ∧ adjacent p q

/-- Sum of corner numbers in the grid --/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- Sum of numbers in the middle column --/
def middle_column_sum (g : Grid) : Nat :=
  g 0 1 + g 1 1 + g 2 1

theorem center_is_eight (g : Grid) 
  (h1 : ∀ n : Fin 9, ∃! p : Fin 3 × Fin 3, g p.1 p.2 = n)
  (h2 : consecutive_adjacent g)
  (h3 : corner_sum g = 20)
  (h4 : Even (middle_column_sum g)) :
  g 1 1 = 8 := by
  sorry

end NUMINAMATH_CALUDE_center_is_eight_l2517_251706


namespace NUMINAMATH_CALUDE_problem_solution_l2517_251785

theorem problem_solution (x y : ℝ) :
  y = (Real.sqrt (x^2 - 4) + Real.sqrt (4 - x^2) + 1) / (x - 2) →
  3 * x + 4 * y = -7 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2517_251785


namespace NUMINAMATH_CALUDE_expand_binomial_product_l2517_251703

theorem expand_binomial_product (x : ℝ) : (x + 3) * (4 * x - 8) = 4 * x^2 + 4 * x - 24 := by
  sorry

end NUMINAMATH_CALUDE_expand_binomial_product_l2517_251703


namespace NUMINAMATH_CALUDE_inequality_proof_l2517_251762

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^2 + y^4 + z^6 ≥ x*y^2 + y^2*z^3 + x*z^3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2517_251762


namespace NUMINAMATH_CALUDE_ram_original_price_l2517_251758

/-- Represents the price change of RAM due to market conditions --/
def ram_price_change (original_price : ℝ) : Prop :=
  let increased_price := original_price * 1.3
  let final_price := increased_price * 0.8
  final_price = 52

/-- Theorem stating that the original price of RAM was $50 --/
theorem ram_original_price : ∃ (price : ℝ), ram_price_change price ∧ price = 50 := by
  sorry

end NUMINAMATH_CALUDE_ram_original_price_l2517_251758


namespace NUMINAMATH_CALUDE_scale_length_difference_l2517_251777

/-- Proves that a 7 ft scale divided into 4 equal parts of 24 inches each has 12 additional inches -/
theorem scale_length_difference : 
  let scale_length_ft : ℕ := 7
  let num_parts : ℕ := 4
  let part_length_inches : ℕ := 24
  let inches_per_foot : ℕ := 12
  
  (num_parts * part_length_inches) - (scale_length_ft * inches_per_foot) = 12 := by
  sorry

end NUMINAMATH_CALUDE_scale_length_difference_l2517_251777


namespace NUMINAMATH_CALUDE_series_equals_ten_implies_k_equals_sixteen_l2517_251702

/-- The sum of the infinite geometric series with first term a and common ratio r -/
noncomputable def geometric_sum (a : ℝ) (r : ℝ) : ℝ := a / (1 - r)

/-- The series in question -/
noncomputable def series (k : ℝ) : ℝ := 
  4 + geometric_sum ((4 + k) / 5) (1 / 5)

theorem series_equals_ten_implies_k_equals_sixteen :
  ∃ k : ℝ, series k = 10 ∧ k = 16 := by sorry

end NUMINAMATH_CALUDE_series_equals_ten_implies_k_equals_sixteen_l2517_251702


namespace NUMINAMATH_CALUDE_root_equation_implies_value_l2517_251775

theorem root_equation_implies_value (m : ℝ) : 
  m^2 - 2*m - 2019 = 0 → 2*m^2 - 4*m = 4038 := by
  sorry

end NUMINAMATH_CALUDE_root_equation_implies_value_l2517_251775


namespace NUMINAMATH_CALUDE_train_speed_conversion_l2517_251769

theorem train_speed_conversion (speed_kmph : ℝ) (speed_ms : ℝ) : 
  speed_kmph = 216 → speed_ms = 60 → speed_kmph * 1000 / 3600 = speed_ms := by
  sorry

end NUMINAMATH_CALUDE_train_speed_conversion_l2517_251769


namespace NUMINAMATH_CALUDE_fg_difference_of_squares_l2517_251727

def f (x : ℝ) : ℝ := x - 2

def g (x : ℝ) : ℝ := 2 * x + 4

theorem fg_difference_of_squares : (f (g 3))^2 - (g (f 3))^2 = 28 := by
  sorry

end NUMINAMATH_CALUDE_fg_difference_of_squares_l2517_251727


namespace NUMINAMATH_CALUDE_clown_mobile_count_l2517_251791

theorem clown_mobile_count (num_mobiles : ℕ) (clowns_per_mobile : ℕ) 
  (h1 : num_mobiles = 5) 
  (h2 : clowns_per_mobile = 28) : 
  num_mobiles * clowns_per_mobile = 140 := by
sorry

end NUMINAMATH_CALUDE_clown_mobile_count_l2517_251791


namespace NUMINAMATH_CALUDE_one_four_one_not_reappear_l2517_251784

def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n
  else (n % 10) * digit_product (n / 10)

def next_numbers (n : ℕ) : Set ℕ :=
  {n + digit_product n, n - digit_product n}

def reachable_numbers (start : ℕ) : Set ℕ :=
  {n | ∃ (seq : ℕ → ℕ), seq 0 = start ∧ ∀ i, seq (i + 1) ∈ next_numbers (seq i)}

theorem one_four_one_not_reappear : 141 ∉ reachable_numbers 141 \ {141} := by
  sorry

end NUMINAMATH_CALUDE_one_four_one_not_reappear_l2517_251784


namespace NUMINAMATH_CALUDE_system_solution_l2517_251716

theorem system_solution (x y b : ℚ) : 
  5 * x - 2 * y = b →
  3 * x + 4 * y = 3 * b →
  y = 3 →
  b = 13 / 2 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l2517_251716


namespace NUMINAMATH_CALUDE_equation_solution_l2517_251789

theorem equation_solution : ∃ (x₁ x₂ : ℚ),
  x₁ = -1/3 ∧ x₂ = -2 ∧
  (∀ x : ℚ, x ≠ 3 → x ≠ 1/2 → 
    ((2*x + 4) / (x - 3) = (x + 2) / (2*x - 1) ↔ x = x₁ ∨ x = x₂)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l2517_251789


namespace NUMINAMATH_CALUDE_unique_solution_when_k_zero_l2517_251712

/-- The equation has exactly one solution when k = 0 -/
theorem unique_solution_when_k_zero :
  ∃! x : ℝ, (x + 2) / (0 * x - 1) = x :=
sorry

end NUMINAMATH_CALUDE_unique_solution_when_k_zero_l2517_251712


namespace NUMINAMATH_CALUDE_even_function_implies_a_zero_l2517_251723

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x - 4

-- State the theorem
theorem even_function_implies_a_zero :
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 :=
by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_zero_l2517_251723


namespace NUMINAMATH_CALUDE_angle_sum_bounds_l2517_251766

theorem angle_sum_bounds (x y z : Real) 
  (hx : 0 < x ∧ x < π/2) 
  (hy : 0 < y ∧ y < π/2) 
  (hz : 0 < z ∧ z < π/2) 
  (h : Real.cos x ^ 2 + Real.cos y ^ 2 + Real.cos z ^ 2 = 1) : 
  3 * π / 4 < x + y + z ∧ x + y + z < π := by
sorry

end NUMINAMATH_CALUDE_angle_sum_bounds_l2517_251766


namespace NUMINAMATH_CALUDE_Q_has_35_digits_l2517_251711

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- The product of two large numbers -/
def Q : ℕ := 6789432567123456789 * 98765432345678

/-- Theorem stating that Q has 35 digits -/
theorem Q_has_35_digits : num_digits Q = 35 := by sorry

end NUMINAMATH_CALUDE_Q_has_35_digits_l2517_251711


namespace NUMINAMATH_CALUDE_binomial_20_19_l2517_251700

theorem binomial_20_19 : Nat.choose 20 19 = 20 := by
  sorry

end NUMINAMATH_CALUDE_binomial_20_19_l2517_251700


namespace NUMINAMATH_CALUDE_less_than_implies_less_than_minus_one_l2517_251799

theorem less_than_implies_less_than_minus_one {a b : ℝ} (h : a < b) : a - 1 < b - 1 := by
  sorry

end NUMINAMATH_CALUDE_less_than_implies_less_than_minus_one_l2517_251799


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2517_251732

-- Define the sets M and N
def M : Set ℝ := {x | Real.log (x - 2) ≤ 0}
def N : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2517_251732


namespace NUMINAMATH_CALUDE_right_triangle_area_l2517_251787

theorem right_triangle_area (a b c : ℝ) (h1 : a = 24) (h2 : c = 26) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 120 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l2517_251787


namespace NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l2517_251761

theorem or_necessary_not_sufficient_for_and (p q : Prop) :
  (∀ (p q : Prop), (p ∧ q) → (p ∨ q)) ∧
  (∃ (p q : Prop), (p ∨ q) ∧ ¬(p ∧ q)) :=
by sorry

end NUMINAMATH_CALUDE_or_necessary_not_sufficient_for_and_l2517_251761


namespace NUMINAMATH_CALUDE_min_value_of_f_l2517_251764

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / (x - 1)

theorem min_value_of_f :
  ∃ (x_min : ℝ), x_min > 1 ∧
  (∀ (x : ℝ), x > 1 → f x ≥ f x_min) ∧
  f x_min = 2 * Real.sqrt 3 + 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l2517_251764


namespace NUMINAMATH_CALUDE_problem_solution_l2517_251747

theorem problem_solution (X Y : ℝ) : 
  (18 / 100 * X = 54 / 100 * 1200) → 
  (X = 4 * Y) → 
  (X = 3600 ∧ Y = 900) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2517_251747


namespace NUMINAMATH_CALUDE_fraction_sum_l2517_251755

theorem fraction_sum : (1 : ℚ) / 6 + (1 : ℚ) / 3 + (5 : ℚ) / 9 = (19 : ℚ) / 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l2517_251755


namespace NUMINAMATH_CALUDE_stating_sheets_taken_exists_l2517_251748

/-- Represents the total number of pages in Hiram's algebra notes -/
def total_pages : ℕ := 60

/-- Represents the total number of sheets in Hiram's algebra notes -/
def total_sheets : ℕ := 30

/-- Represents the average of the remaining page numbers -/
def target_average : ℕ := 21

/-- 
Theorem stating that there exists a number of consecutive sheets taken 
such that the average of the remaining page numbers is the target average
-/
theorem sheets_taken_exists : 
  ∃ c : ℕ, c > 0 ∧ c < total_sheets ∧
  ∃ b : ℕ, b ≥ 0 ∧ b + c ≤ total_sheets ∧
  (b * (2 * b + 1) + 
   ((2 * (b + c) + 1 + total_pages) * (total_pages - 2 * c - 2 * b)) / 2) / 
   (total_pages - 2 * c) = target_average :=
sorry

end NUMINAMATH_CALUDE_stating_sheets_taken_exists_l2517_251748


namespace NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l2517_251741

-- Define the lines l1 and l2
def l1 (a x y : ℝ) : Prop := a * x + 2 * y + 6 = 0
def l2 (a x y : ℝ) : Prop := x + (a - 1) * y + a^2 - 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := a * 1 + 2 * (a - 1) = 0

-- Define parallelism of lines
def parallel (a : ℝ) : Prop := a / 1 = 2 / (a - 1) ∧ a / 1 ≠ 6 / (a^2 - 1)

-- Theorem for perpendicular lines
theorem perpendicular_lines (a : ℝ) : perpendicular a → a = 2/3 :=
sorry

-- Theorem for parallel lines
theorem parallel_lines (a : ℝ) : parallel a → a = -1 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_parallel_lines_l2517_251741


namespace NUMINAMATH_CALUDE_expand_and_factor_l2517_251730

theorem expand_and_factor (a b c : ℝ) : (a + b - c) * (a - b + c) = (a + (b - c)) * (a - (b - c)) := by
  sorry

end NUMINAMATH_CALUDE_expand_and_factor_l2517_251730


namespace NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2517_251708

/-- The quadratic inequality function -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 1) * x^2 + (m - 1) * x + 2

/-- The solution set when m = 0 -/
def solution_set_m_zero : Set ℝ := {x | -2 < x ∧ x < 1}

/-- The range of m for which the solution set is ℝ -/
def m_range : Set ℝ := {m | 1 ≤ m ∧ m < 9}

theorem quadratic_inequality_theorem :
  (∀ x, x ∈ solution_set_m_zero ↔ f 0 x > 0) ∧
  (∀ m, (∀ x, f m x > 0) ↔ m ∈ m_range) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_theorem_l2517_251708


namespace NUMINAMATH_CALUDE_calculate_expression_l2517_251767

theorem calculate_expression : (28 * (9 + 2 - 5)) * 3 = 504 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l2517_251767


namespace NUMINAMATH_CALUDE_circle_equation_min_distance_l2517_251757

theorem circle_equation_min_distance (x y : ℝ) :
  (x^2 + y^2 - 64 = 0) → (∀ a b : ℝ, x^2 + y^2 ≤ a^2 + b^2) → x^2 + y^2 = 64 :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_min_distance_l2517_251757


namespace NUMINAMATH_CALUDE_fish_count_l2517_251744

theorem fish_count (bass trout bluegill : ℕ) : 
  bass = 32 →
  trout = bass / 4 →
  bluegill = 2 * bass →
  bass + trout + bluegill = 104 := by
sorry

end NUMINAMATH_CALUDE_fish_count_l2517_251744


namespace NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l2517_251779

theorem x_plus_p_equals_2p_plus_3 (x p : ℝ) (h1 : |x - 3| = p) (h2 : x > 3) :
  x + p = 2 * p + 3 := by
sorry

end NUMINAMATH_CALUDE_x_plus_p_equals_2p_plus_3_l2517_251779


namespace NUMINAMATH_CALUDE_dodecagon_area_l2517_251793

/-- Given a square with side length a, prove that the area of a regular dodecagon
    constructed outside the square, where the upper bases of trapezoids on each side
    of the square and their lateral sides form the dodecagon, is equal to (3*a^2)/2. -/
theorem dodecagon_area (a : ℝ) (a_pos : a > 0) :
  let square_side := a
  let dodecagon_area := (3 * a^2) / 2
  dodecagon_area = (3 * square_side^2) / 2 :=
by sorry

end NUMINAMATH_CALUDE_dodecagon_area_l2517_251793


namespace NUMINAMATH_CALUDE_rationalize_and_product_l2517_251780

theorem rationalize_and_product : ∃ (A B C : ℤ),
  (2 : ℝ) - Real.sqrt 5 / (3 + Real.sqrt 5) = A + B * Real.sqrt C ∧
  A * B * C = -50 := by
  sorry

end NUMINAMATH_CALUDE_rationalize_and_product_l2517_251780


namespace NUMINAMATH_CALUDE_xiao_ming_math_score_l2517_251718

theorem xiao_ming_math_score :
  let average_three := 94
  let subjects := 3
  let average_two := average_three - 1
  let total_score := average_three * subjects
  let chinese_english_score := average_two * (subjects - 1)
  total_score - chinese_english_score = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_ming_math_score_l2517_251718


namespace NUMINAMATH_CALUDE_mixed_fraction_power_product_l2517_251763

theorem mixed_fraction_power_product :
  (1 + 2/3)^4 * (-3/5)^5 = -3/5 := by sorry

end NUMINAMATH_CALUDE_mixed_fraction_power_product_l2517_251763


namespace NUMINAMATH_CALUDE_point_C_coordinates_l2517_251725

-- Define the points and vectors
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (-1, 5)

-- Define vector AB
def vecAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define vector AC in terms of AB
def vecAC : ℝ × ℝ := (2 * vecAB.1, 2 * vecAB.2)

-- Define point C
def C : ℝ × ℝ := (A.1 + vecAC.1, A.2 + vecAC.2)

-- Theorem statement
theorem point_C_coordinates : C = (-3, 9) := by
  sorry

end NUMINAMATH_CALUDE_point_C_coordinates_l2517_251725


namespace NUMINAMATH_CALUDE_friend_distribution_problem_l2517_251720

/-- The number of friends that satisfies the given conditions --/
def num_friends : ℕ := 16

/-- The total amount distributed in rupees --/
def total_amount : ℕ := 5000

/-- The decrease in amount per person if there were 8 more friends --/
def decrease_amount : ℕ := 125

theorem friend_distribution_problem :
  (total_amount / num_friends : ℚ) - (total_amount / (num_friends + 8) : ℚ) = decrease_amount ∧
  num_friends > 0 := by
  sorry

#check friend_distribution_problem

end NUMINAMATH_CALUDE_friend_distribution_problem_l2517_251720


namespace NUMINAMATH_CALUDE_range_of_m_plus_n_l2517_251701

noncomputable def f (m n x : ℝ) : ℝ := m * 2^x + x^2 + n * x

theorem range_of_m_plus_n (m n : ℝ) :
  (∃ x, f m n x = 0) ∧ 
  (∀ x, f m n x = 0 ↔ f m n (f m n x) = 0) →
  0 ≤ m + n ∧ m + n < 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_plus_n_l2517_251701


namespace NUMINAMATH_CALUDE_marias_quarters_l2517_251715

/-- Represents the number of coins of each type in Maria's piggy bank -/
structure CoinCount where
  dimes : ℕ
  quarters : ℕ
  nickels : ℕ

/-- Calculates the total value in dollars given a CoinCount -/
def totalValue (coins : CoinCount) : ℚ :=
  0.1 * coins.dimes + 0.25 * coins.quarters + 0.05 * coins.nickels

/-- The problem statement -/
theorem marias_quarters (initialCoins : CoinCount) (finalTotal : ℚ) : 
  initialCoins.dimes = 4 → 
  initialCoins.quarters = 4 → 
  initialCoins.nickels = 7 → 
  finalTotal = 3 →
  ∃ (addedQuarters : ℕ), 
    totalValue { dimes := initialCoins.dimes,
                 quarters := initialCoins.quarters + addedQuarters,
                 nickels := initialCoins.nickels } = finalTotal ∧
    addedQuarters = 5 := by
  sorry


end NUMINAMATH_CALUDE_marias_quarters_l2517_251715


namespace NUMINAMATH_CALUDE_dihedral_angle_is_120_degrees_l2517_251738

/-- A regular tetrahedron with a circumscribed sphere -/
structure RegularTetrahedronWithSphere where
  /-- The height of the tetrahedron -/
  height : ℝ
  /-- The diameter of the circumscribed sphere -/
  sphere_diameter : ℝ
  /-- The diameter of the sphere is 9 times the height of the tetrahedron -/
  sphere_diameter_relation : sphere_diameter = 9 * height

/-- The dihedral angle between two lateral faces of a regular tetrahedron -/
def dihedral_angle (t : RegularTetrahedronWithSphere) : ℝ :=
  sorry

/-- Theorem: The dihedral angle between two lateral faces of a regular tetrahedron
    with the given sphere relation is 120 degrees -/
theorem dihedral_angle_is_120_degrees (t : RegularTetrahedronWithSphere) :
  dihedral_angle t = 120 * π / 180 :=
sorry

end NUMINAMATH_CALUDE_dihedral_angle_is_120_degrees_l2517_251738


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2517_251739

theorem complex_fraction_equality : 1 + (1 / (1 + (1 / (1 + 1)))) = 5 / 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2517_251739


namespace NUMINAMATH_CALUDE_total_distance_walked_l2517_251774

-- Define the walking rate in miles per hour
def walking_rate : ℝ := 4

-- Define the total time in hours
def total_time : ℝ := 2

-- Define the break time in hours
def break_time : ℝ := 0.5

-- Define the effective walking time
def effective_walking_time : ℝ := total_time - break_time

-- Theorem to prove
theorem total_distance_walked :
  walking_rate * effective_walking_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_total_distance_walked_l2517_251774


namespace NUMINAMATH_CALUDE_unique_solution_circle_equation_l2517_251797

theorem unique_solution_circle_equation :
  ∃! (x y : ℝ), (x - 5)^2 + (y - 6)^2 + (x - y)^2 = 1/3 ∧
  x = 16/3 ∧ y = 17/3 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_circle_equation_l2517_251797


namespace NUMINAMATH_CALUDE_combined_polyhedron_faces_l2517_251798

/-- A regular tetrahedron -/
structure Tetrahedron :=
  (edge_length : ℝ)

/-- A regular octahedron -/
structure Octahedron :=
  (edge_length : ℝ)

/-- A polyhedron formed by combining a tetrahedron and an octahedron -/
structure CombinedPolyhedron :=
  (tetra : Tetrahedron)
  (octa : Octahedron)
  (combined : tetra.edge_length = octa.edge_length)

/-- The number of faces in the combined polyhedron -/
def num_faces (p : CombinedPolyhedron) : ℕ := 7

theorem combined_polyhedron_faces (p : CombinedPolyhedron) : 
  num_faces p = 7 := by sorry

end NUMINAMATH_CALUDE_combined_polyhedron_faces_l2517_251798


namespace NUMINAMATH_CALUDE_daily_earnings_l2517_251756

/-- Calculates the daily earnings of a person who works every day, given their earnings over a 4-week period. -/
theorem daily_earnings (total_earnings : ℚ) (h : total_earnings = 1960) : 
  total_earnings / (4 * 7) = 70 := by
  sorry

end NUMINAMATH_CALUDE_daily_earnings_l2517_251756


namespace NUMINAMATH_CALUDE_quadratic_vertex_and_extremum_l2517_251751

/-- Given a quadratic equation y = -x^2 + cx + d with roots -5 and 3,
    prove that its vertex is (4, 1) and it represents a maximum point. -/
theorem quadratic_vertex_and_extremum (c d : ℝ) :
  (∀ x, -x^2 + c*x + d = 0 ↔ x = -5 ∨ x = 3) →
  (∃! p : ℝ × ℝ, p.1 = 4 ∧ p.2 = 1 ∧ 
    (∀ x, -x^2 + c*x + d ≤ -p.1^2 + c*p.1 + d)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_vertex_and_extremum_l2517_251751


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_l2517_251726

theorem unique_solution_lcm_gcd : 
  ∃! n : ℕ+, n.lcm 120 = n.gcd 120 + 300 ∧ n = 180 := by sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_l2517_251726


namespace NUMINAMATH_CALUDE_perpendicular_vectors_l2517_251742

/-- Given vectors a and b in ℝ², prove that if (a + x*b) is perpendicular to (a - b), then x = -3 -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (x : ℝ) 
  (h1 : a = (3, 4))
  (h2 : b = (2, 1))
  (h3 : (a + x • b) • (a - b) = 0) :
  x = -3 := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_l2517_251742


namespace NUMINAMATH_CALUDE_simplify_expression_l2517_251796

theorem simplify_expression : -(-3) - 4 + (-5) = 3 - 4 - 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2517_251796


namespace NUMINAMATH_CALUDE_polynomial_divisibility_l2517_251790

theorem polynomial_divisibility (a b c d m : ℤ) 
  (h1 : (a * m^3 + b * m^2 + c * m + d) % 5 = 0)
  (h2 : d % 5 ≠ 0) :
  ∃ n : ℤ, (d * n^3 + c * n^2 + b * n + a) % 5 = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_divisibility_l2517_251790


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2517_251735

def A : Set ℝ := {x | 3 ≤ x ∧ x < 7}
def B : Set ℝ := {x | 2 < x ∧ x < 10}

theorem union_of_A_and_B : A ∪ B = {x | 2 < x ∧ x < 10} := by sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2517_251735


namespace NUMINAMATH_CALUDE_angle_A_value_l2517_251709

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (sum_angles : A + B + C = π)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)

-- State the theorem
theorem angle_A_value (abc : Triangle) (h : abc.b = 2 * abc.a * Real.sin abc.B) :
  abc.A = π/6 ∨ abc.A = 5*π/6 := by
  sorry

end NUMINAMATH_CALUDE_angle_A_value_l2517_251709


namespace NUMINAMATH_CALUDE_log_one_fourth_sixteen_l2517_251753

-- Define the logarithm function for an arbitrary base
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_one_fourth_sixteen : log (1/4) 16 = -2 := by sorry

end NUMINAMATH_CALUDE_log_one_fourth_sixteen_l2517_251753


namespace NUMINAMATH_CALUDE_existence_of_integer_combination_l2517_251795

theorem existence_of_integer_combination (a b c : ℝ) 
  (hab : ∃ (q : ℚ), a * b = q)
  (hbc : ∃ (q : ℚ), b * c = q)
  (hca : ∃ (q : ℚ), c * a = q) :
  ∃ (x y z : ℤ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧ a * x + b * y + c * z = 0 := by
sorry

end NUMINAMATH_CALUDE_existence_of_integer_combination_l2517_251795


namespace NUMINAMATH_CALUDE_new_years_numbers_evenness_l2517_251781

theorem new_years_numbers_evenness (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2018) :
  ((2019 - k)^12 + 2018) % 2019 = (k^12 + 2018) % 2019 :=
sorry

end NUMINAMATH_CALUDE_new_years_numbers_evenness_l2517_251781


namespace NUMINAMATH_CALUDE_kids_left_playing_result_l2517_251717

/-- The number of kids left playing soccer -/
def kids_left_playing (initial : ℝ) (left : ℝ) : ℝ :=
  initial - left

/-- Theorem stating the number of kids left playing soccer -/
theorem kids_left_playing_result :
  kids_left_playing 22.5 14.3 = 8.2 := by sorry

end NUMINAMATH_CALUDE_kids_left_playing_result_l2517_251717


namespace NUMINAMATH_CALUDE_find_divisor_l2517_251749

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (divisor : ℕ) : 
  dividend = 56 → quotient = 4 → divisor * quotient = dividend → divisor = 14 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l2517_251749


namespace NUMINAMATH_CALUDE_sixteen_to_power_divided_by_eight_l2517_251786

theorem sixteen_to_power_divided_by_eight (n : ℕ) : n = 16^1024 → n / 8 = 2^4093 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_to_power_divided_by_eight_l2517_251786


namespace NUMINAMATH_CALUDE_percentage_for_sobel_l2517_251734

/-- Represents the percentage of voters who are male -/
def male_percentage : ℝ := 60

/-- Represents the percentage of female voters who voted for Lange -/
def female_for_lange : ℝ := 35

/-- Represents the percentage of male voters who voted for Sobel -/
def male_for_sobel : ℝ := 44

/-- Theorem stating the percentage of total voters who voted for Sobel -/
theorem percentage_for_sobel :
  let female_percentage := 100 - male_percentage
  let female_for_sobel := 100 - female_for_lange
  let total_for_sobel := (male_percentage * male_for_sobel + female_percentage * female_for_sobel) / 100
  total_for_sobel = 52.4 := by sorry

end NUMINAMATH_CALUDE_percentage_for_sobel_l2517_251734


namespace NUMINAMATH_CALUDE_triangle_properties_l2517_251752

-- Define the triangle ABC
def Triangle (A B C : Real) (a b c : Real) : Prop :=
  -- Sides a, b, c are opposite to angles A, B, C respectively
  true

-- Given conditions
axiom triangle_condition {A B C a b c : Real} (h : Triangle A B C a b c) :
  2 * Real.cos C * (a * Real.cos B + b * Real.cos A) = c

axiom c_value {A B C a b c : Real} (h : Triangle A B C a b c) :
  c = Real.sqrt 7

axiom area_value {A B C a b c : Real} (h : Triangle A B C a b c) :
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2

-- Theorem to prove
theorem triangle_properties {A B C a b c : Real} (h : Triangle A B C a b c) :
  C = Real.pi / 3 ∧ a + b + c = 5 + Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2517_251752


namespace NUMINAMATH_CALUDE_three_red_and_at_least_one_white_mutually_exclusive_and_complementary_l2517_251707

/-- Represents the color of a ball -/
inductive Color
| Red
| White

/-- Represents the outcome of drawing 3 balls -/
def Outcome := (Color × Color × Color)

/-- The sample space of all possible outcomes when drawing 3 balls from a bag with 5 red and 5 white balls -/
def SampleSpace : Set Outcome := sorry

/-- Event: Draw three red balls -/
def ThreeRedBalls (outcome : Outcome) : Prop := 
  outcome = (Color.Red, Color.Red, Color.Red)

/-- Event: Draw three balls with at least one white ball -/
def AtLeastOneWhiteBall (outcome : Outcome) : Prop := 
  outcome.1 = Color.White ∨ outcome.2.1 = Color.White ∨ outcome.2.2 = Color.White

theorem three_red_and_at_least_one_white_mutually_exclusive_and_complementary :
  (∀ outcome ∈ SampleSpace, ¬(ThreeRedBalls outcome ∧ AtLeastOneWhiteBall outcome)) ∧ 
  (∀ outcome ∈ SampleSpace, ThreeRedBalls outcome ∨ AtLeastOneWhiteBall outcome) := by
  sorry

end NUMINAMATH_CALUDE_three_red_and_at_least_one_white_mutually_exclusive_and_complementary_l2517_251707


namespace NUMINAMATH_CALUDE_nancy_chips_l2517_251783

/-- Nancy's tortilla chip distribution problem -/
theorem nancy_chips (initial : ℕ) (brother sister : ℕ) (kept : ℕ) : 
  initial = 22 → brother = 7 → sister = 5 → kept = initial - (brother + sister) → kept = 10 := by
  sorry

end NUMINAMATH_CALUDE_nancy_chips_l2517_251783


namespace NUMINAMATH_CALUDE_unit_digit_product_l2517_251713

theorem unit_digit_product : (3^68 * 6^59 * 7^71) % 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_unit_digit_product_l2517_251713


namespace NUMINAMATH_CALUDE_gcf_60_90_l2517_251760

theorem gcf_60_90 : Nat.gcd 60 90 = 30 := by
  sorry

end NUMINAMATH_CALUDE_gcf_60_90_l2517_251760


namespace NUMINAMATH_CALUDE_cupcakes_per_package_l2517_251770

theorem cupcakes_per_package 
  (initial_cupcakes : ℕ) 
  (eaten_cupcakes : ℕ) 
  (total_packages : ℕ) 
  (h1 : initial_cupcakes = 39) 
  (h2 : eaten_cupcakes = 21) 
  (h3 : total_packages = 6) 
  (h4 : eaten_cupcakes < initial_cupcakes) : 
  (initial_cupcakes - eaten_cupcakes) / total_packages = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_cupcakes_per_package_l2517_251770


namespace NUMINAMATH_CALUDE_unique_modular_solution_l2517_251754

theorem unique_modular_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 8 ∧ n ≡ -2023 [ZMOD 9] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_modular_solution_l2517_251754


namespace NUMINAMATH_CALUDE_projection_vector_is_correct_l2517_251772

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D parametric form -/
structure ParametricLine where
  x : ℝ → ℝ
  y : ℝ → ℝ

/-- The line l -/
def line_l : ParametricLine :=
  { x := λ t => 2 + 3*t,
    y := λ t => 3 + 2*t }

/-- The line m -/
def line_m : ParametricLine :=
  { x := λ s => 4 + 2*s,
    y := λ s => 5 + 3*s }

/-- Direction vector of line l -/
def dir_l : Vector2D :=
  { x := 3,
    y := 2 }

/-- Direction vector of line m -/
def dir_m : Vector2D :=
  { x := 2,
    y := 3 }

/-- The vector perpendicular to the direction of line m -/
def perp_m : Vector2D :=
  { x := 3,
    y := -2 }

/-- The theorem to prove -/
theorem projection_vector_is_correct :
  ∃ (k : ℝ),
    let v : Vector2D := { x := k * perp_m.x, y := k * perp_m.y }
    v.x + v.y = 3 ∧
    v.x = 9 ∧
    v.y = -6 := by
  sorry

end NUMINAMATH_CALUDE_projection_vector_is_correct_l2517_251772


namespace NUMINAMATH_CALUDE_chess_club_female_fraction_l2517_251792

/-- Represents the chess club membership data --/
structure ChessClub where
  last_year_males : ℕ
  last_year_females : ℕ
  male_increase_rate : ℚ
  female_increase_rate : ℚ
  total_increase_rate : ℚ

/-- Calculates the fraction of female participants this year --/
def female_fraction (club : ChessClub) : ℚ :=
  let this_year_males : ℚ := club.last_year_males * (1 + club.male_increase_rate)
  let this_year_females : ℚ := club.last_year_females * (1 + club.female_increase_rate)
  this_year_females / (this_year_males + this_year_females)

/-- Theorem statement for the chess club problem --/
theorem chess_club_female_fraction :
  let club : ChessClub := {
    last_year_males := 30,
    last_year_females := 15,
    male_increase_rate := 1/10,
    female_increase_rate := 1/4,
    total_increase_rate := 3/20
  }
  female_fraction club = 19/52 := by
  sorry


end NUMINAMATH_CALUDE_chess_club_female_fraction_l2517_251792


namespace NUMINAMATH_CALUDE_airport_gate_probability_l2517_251794

/-- The number of gates in the airport --/
def num_gates : ℕ := 15

/-- The distance between adjacent gates in feet --/
def gate_distance : ℕ := 90

/-- The maximum walking distance in feet --/
def max_distance : ℕ := 450

/-- The probability of selecting two gates within the maximum distance --/
def probability : ℚ := 10 / 21

theorem airport_gate_probability :
  let total_pairs := num_gates * (num_gates - 1)
  let valid_pairs := (num_gates - max_distance / gate_distance) * (max_distance / gate_distance)
    + 2 * (max_distance / gate_distance * (max_distance / gate_distance + 1) / 2)
  (valid_pairs : ℚ) / total_pairs = probability := by sorry

end NUMINAMATH_CALUDE_airport_gate_probability_l2517_251794


namespace NUMINAMATH_CALUDE_solution_of_linear_equation_l2517_251746

theorem solution_of_linear_equation (a : ℝ) : 
  (∃ x y : ℝ, x = 3 ∧ y = 2 ∧ a * x + 2 * y = 1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_of_linear_equation_l2517_251746


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twenty_l2517_251721

theorem last_digit_of_one_over_two_to_twenty (n : ℕ) :
  n = 20 →
  ∃ k : ℕ, (1 : ℚ) / (2^n) = k * (1 / 10^n) + 5 * (1 / 10^n) :=
sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_two_to_twenty_l2517_251721


namespace NUMINAMATH_CALUDE_sequence_remainder_l2517_251771

theorem sequence_remainder (n : ℕ) : (7 * n + 4) % 7 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sequence_remainder_l2517_251771


namespace NUMINAMATH_CALUDE_solution_of_system_l2517_251765

/-- Given a system of equations with four distinct real numbers a₁, a₂, a₃, a₄,
    prove that the solution is x₁ = 1 / (a₄ - a₁), x₂ = 0, x₃ = 0, x₄ = 1 / (a₄ - a₁) -/
theorem solution_of_system (a₁ a₂ a₃ a₄ : ℝ) 
    (h_distinct : a₁ ≠ a₂ ∧ a₁ ≠ a₃ ∧ a₁ ≠ a₄ ∧ a₂ ≠ a₃ ∧ a₂ ≠ a₄ ∧ a₃ ≠ a₄) :
  ∃ (x₁ x₂ x₃ x₄ : ℝ),
    (|a₁ - a₂| * x₂ + |a₁ - a₃| * x₃ + |a₁ - a₄| * x₄ = 1) ∧
    (|a₂ - a₁| * x₁ + |a₂ - a₃| * x₃ + |a₂ - a₄| * x₄ = 1) ∧
    (|a₃ - a₁| * x₁ + |a₃ - a₂| * x₂ + |a₃ - a₄| * x₄ = 1) ∧
    (|a₄ - a₁| * x₁ + |a₄ - a₂| * x₂ + |a₄ - a₃| * x₃ = 1) ∧
    x₁ = 1 / (a₄ - a₁) ∧ x₂ = 0 ∧ x₃ = 0 ∧ x₄ = 1 / (a₄ - a₁) := by
  sorry


end NUMINAMATH_CALUDE_solution_of_system_l2517_251765


namespace NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l2517_251728

theorem capri_sun_cost_per_pouch :
  let boxes : ℕ := 10
  let pouches_per_box : ℕ := 6
  let total_cost_dollars : ℕ := 12
  let total_pouches : ℕ := boxes * pouches_per_box
  let total_cost_cents : ℕ := total_cost_dollars * 100
  total_cost_cents / total_pouches = 20 := by sorry

end NUMINAMATH_CALUDE_capri_sun_cost_per_pouch_l2517_251728


namespace NUMINAMATH_CALUDE_expansion_coefficients_l2517_251788

theorem expansion_coefficients (n : ℕ) : 
  (2^(2*n) = 2^n + 240) → 
  (∃ k, k = (Nat.choose 8 4) ∧ k = 70) ∧ 
  (∃ m, m = (2^4) ∧ m = 16) := by
  sorry

end NUMINAMATH_CALUDE_expansion_coefficients_l2517_251788


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2517_251736

/-- Given a geometric sequence {a_n}, prove that if a_1 + a_2 = 40 and a_3 + a_4 = 60, then a_7 + a_8 = 135 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
    (h_sum1 : a 1 + a 2 = 40) (h_sum2 : a 3 + a 4 = 60) : a 7 + a 8 = 135 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2517_251736


namespace NUMINAMATH_CALUDE_playground_children_count_l2517_251768

theorem playground_children_count :
  let boys : ℕ := 27
  let girls : ℕ := 35
  boys + girls = 62 := by sorry

end NUMINAMATH_CALUDE_playground_children_count_l2517_251768


namespace NUMINAMATH_CALUDE_quilt_transformation_l2517_251705

/-- Given a rectangular quilt with width 6 feet and an unknown length, and a square quilt with side length 12 feet, 
    if their areas are equal, then the length of the rectangular quilt is 24 feet. -/
theorem quilt_transformation (length : ℝ) : 
  (6 * length = 12 * 12) → length = 24 := by
  sorry

end NUMINAMATH_CALUDE_quilt_transformation_l2517_251705


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2517_251740

theorem complex_simplification_and_multiplication :
  ((4 - 3 * Complex.I) - (7 - 5 * Complex.I)) * (1 + 2 * Complex.I) = -7 - 4 * Complex.I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2517_251740


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l2517_251719

/-- Calculates the molecular weight of a compound given its composition and atomic weights -/
def molecularWeight (Ca I C H : ℕ) (wCa wI wC wH : ℝ) : ℝ :=
  Ca * wCa + I * wI + C * wC + H * wH

/-- The molecular weight of the given compound is 602.794 amu -/
theorem compound_molecular_weight :
  let Ca : ℕ := 2
  let I : ℕ := 4
  let C : ℕ := 1
  let H : ℕ := 3
  let wCa : ℝ := 40.08
  let wI : ℝ := 126.90
  let wC : ℝ := 12.01
  let wH : ℝ := 1.008
  molecularWeight Ca I C H wCa wI wC wH = 602.794 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l2517_251719


namespace NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l2517_251782

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

/-- The seventh term of an arithmetic sequence is the average of its third and eleventh terms. -/
theorem seventh_term_of_arithmetic_sequence
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a)
  (h_third_term : a 3 = 2 / 11)
  (h_eleventh_term : a 11 = 5 / 6) :
  a 7 = 67 / 132 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_arithmetic_sequence_l2517_251782


namespace NUMINAMATH_CALUDE_exactly_five_cheaper_points_l2517_251722

-- Define the cost function
def C (n : ℕ) : ℕ :=
  if n ≥ 50 then 13 * n
  else if n ≥ 20 then 14 * n
  else 15 * n

-- Define the property we want to prove
def cheaper_to_buy_more (n : ℕ) : Prop :=
  C (n + 1) < C n

-- Theorem statement
theorem exactly_five_cheaper_points :
  ∃ (S : Finset ℕ), S.card = 5 ∧ 
  (∀ n, n ∈ S ↔ cheaper_to_buy_more n) :=
sorry

end NUMINAMATH_CALUDE_exactly_five_cheaper_points_l2517_251722


namespace NUMINAMATH_CALUDE_anna_candy_purchase_l2517_251750

def candy_problem (initial_money : ℚ) (gum_price : ℚ) (gum_quantity : ℕ) 
  (chocolate_price : ℚ) (cane_price : ℚ) (cane_quantity : ℕ) (money_left : ℚ) : Prop :=
  ∃ (chocolate_quantity : ℕ),
    initial_money - 
    (gum_price * gum_quantity + 
     chocolate_price * chocolate_quantity + 
     cane_price * cane_quantity) = money_left ∧
    chocolate_quantity = 5

theorem anna_candy_purchase : 
  candy_problem 10 1 3 1 0.5 2 1 := by sorry

end NUMINAMATH_CALUDE_anna_candy_purchase_l2517_251750


namespace NUMINAMATH_CALUDE_savings_calculation_l2517_251714

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 10 / 4 →
  income = 19000 →
  savings = income - expenditure →
  savings = 11400 := by
sorry

end NUMINAMATH_CALUDE_savings_calculation_l2517_251714
