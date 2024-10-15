import Mathlib

namespace NUMINAMATH_CALUDE_stating_isosceles_triangle_with_special_bisectors_l315_31584

/-- Represents an isosceles triangle with angle bisectors -/
structure IsoscelesTriangle where
  -- Base angle of the isosceles triangle
  β : Real
  -- Ratio of the lengths of two angle bisectors
  bisector_ratio : Real

/-- 
  Theorem stating the approximate angles of an isosceles triangle 
  where one angle bisector is twice the length of another
-/
theorem isosceles_triangle_with_special_bisectors 
  (triangle : IsoscelesTriangle) 
  (h1 : triangle.bisector_ratio = 2) 
  (h2 : 76.9 ≤ triangle.β ∧ triangle.β ≤ 77.1) : 
  25.9 ≤ 180 - 2 * triangle.β ∧ 180 - 2 * triangle.β ≤ 26.1 := by
  sorry

#check isosceles_triangle_with_special_bisectors

end NUMINAMATH_CALUDE_stating_isosceles_triangle_with_special_bisectors_l315_31584


namespace NUMINAMATH_CALUDE_cost_to_feed_chickens_is_60_l315_31501

/-- Calculates the cost to feed chickens given the total number of birds and the ratio of bird types -/
def cost_to_feed_chickens (total_birds : ℕ) (duck_ratio parrot_ratio chicken_ratio : ℕ) (chicken_feed_cost : ℚ) : ℚ :=
  let total_ratio := duck_ratio + parrot_ratio + chicken_ratio
  let birds_per_ratio := total_birds / total_ratio
  let num_chickens := birds_per_ratio * chicken_ratio
  (num_chickens : ℚ) * chicken_feed_cost

/-- Theorem stating that with given conditions, the cost to feed chickens is $60 -/
theorem cost_to_feed_chickens_is_60 :
  cost_to_feed_chickens 60 2 3 5 2 = 60 := by
  sorry

end NUMINAMATH_CALUDE_cost_to_feed_chickens_is_60_l315_31501


namespace NUMINAMATH_CALUDE_dog_roaming_area_l315_31524

/-- The area a dog can roam when tied to the corner of a rectangular shed --/
theorem dog_roaming_area (shed_length shed_width leash_length : ℝ) 
  (h1 : shed_length = 4)
  (h2 : shed_width = 3)
  (h3 : leash_length = 4) : 
  let area := (3/4) * Real.pi * leash_length^2 + (1/4) * Real.pi * 1^2
  area = 12.25 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_dog_roaming_area_l315_31524


namespace NUMINAMATH_CALUDE_C₂_fixed_point_l315_31551

/-- Parabola C₁ with vertex (√2-1, 1) and focus (√2-3/4, 1) -/
def C₁ : Set (ℝ × ℝ) :=
  {p | (p.2 - 1)^2 = 2 * (p.1 - (Real.sqrt 2 - 1))}

/-- Parabola C₂ with equation y² - ay + x + 2b = 0 -/
def C₂ (a b : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2^2 - a * p.2 + p.1 + 2 * b = 0}

/-- The tangent line to C₁ at point p -/
def tangentC₁ (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | 2 * p.2 * q.2 - q.1 - 2 * (p.2 + 1) = 0}

/-- The tangent line to C₂ at point p -/
def tangentC₂ (a : ℝ) (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q | (2 * p.2 - a) * q.2 + q.1 - a * p.2 + p.1 + 4 * ((a - 2) * p.2 - p.1 - Real.sqrt 2) / 4 = 0}

/-- Perpendicularity condition for tangent lines -/
def perpendicularTangents (p : ℝ × ℝ) (a : ℝ) : Prop :=
  (p.2 - 1) * (2 * p.2 - a) = -1

theorem C₂_fixed_point (a b : ℝ) :
  (∃ p, p ∈ C₁ ∧ p ∈ C₂ a b ∧ perpendicularTangents p a) →
  (Real.sqrt 2 - 1/2, 1) ∈ C₂ a b := by
  sorry

end NUMINAMATH_CALUDE_C₂_fixed_point_l315_31551


namespace NUMINAMATH_CALUDE_binary_multiplication_theorem_l315_31522

/-- Converts a list of binary digits to a natural number -/
def binary_to_nat (bits : List Bool) : ℕ :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Converts a natural number to a list of binary digits -/
def nat_to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec to_bits (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_bits (m / 2)
  to_bits n

theorem binary_multiplication_theorem :
  let a := [true, true, false, true, true]  -- 11011₂
  let b := [true, true, true]               -- 111₂
  let result := [true, false, false, false, false, true, false, true]  -- 10000101₂
  binary_to_nat a * binary_to_nat b = binary_to_nat result := by
  sorry

#eval binary_to_nat [true, true, false, true, true]  -- Should output 27
#eval binary_to_nat [true, true, true]               -- Should output 7
#eval binary_to_nat [true, false, false, false, false, true, false, true]  -- Should output 133
#eval 27 * 7  -- Should output 189

end NUMINAMATH_CALUDE_binary_multiplication_theorem_l315_31522


namespace NUMINAMATH_CALUDE_faculty_reduction_l315_31560

theorem faculty_reduction (original : ℝ) (reduction_percentage : ℝ) : 
  original = 253.25 → 
  reduction_percentage = 0.23 →
  ⌊original - (original * reduction_percentage)⌋ = 195 := by
sorry

end NUMINAMATH_CALUDE_faculty_reduction_l315_31560


namespace NUMINAMATH_CALUDE_octahedron_projection_area_l315_31591

/-- A regular octahedron -/
structure RegularOctahedron where
  -- Add necessary fields here

/-- The area of a face of a regular octahedron -/
def face_area (o : RegularOctahedron) : ℝ :=
  sorry

/-- The area of the projection of one face onto the opposite face -/
def projection_area (o : RegularOctahedron) : ℝ :=
  sorry

/-- 
  In a regular octahedron, the perpendicular projection of one face 
  onto the plane of the opposite face covers 2/3 of the area of the opposite face
-/
theorem octahedron_projection_area (o : RegularOctahedron) :
  projection_area o = (2 / 3) * face_area o :=
sorry

end NUMINAMATH_CALUDE_octahedron_projection_area_l315_31591


namespace NUMINAMATH_CALUDE_total_seeds_equals_45_l315_31511

/-- The number of flowerbeds -/
def num_flowerbeds : ℕ := 9

/-- The number of seeds planted in each flowerbed -/
def seeds_per_flowerbed : ℕ := 5

/-- The total number of seeds planted -/
def total_seeds : ℕ := num_flowerbeds * seeds_per_flowerbed

theorem total_seeds_equals_45 : total_seeds = 45 := by
  sorry

end NUMINAMATH_CALUDE_total_seeds_equals_45_l315_31511


namespace NUMINAMATH_CALUDE_savings_increase_percentage_l315_31568

/-- Represents the financial situation of a man over two years --/
structure FinancialSituation where
  /-- Income in the first year --/
  income : ℝ
  /-- Savings rate in the first year (as a decimal) --/
  savingsRate : ℝ
  /-- Income increase rate in the second year (as a decimal) --/
  incomeIncreaseRate : ℝ

/-- Theorem stating the increase in savings percentage --/
theorem savings_increase_percentage (fs : FinancialSituation)
    (h1 : fs.savingsRate = 0.2)
    (h2 : fs.incomeIncreaseRate = 0.2)
    (h3 : fs.income > 0)
    (h4 : fs.income * (2 - fs.savingsRate) = 
          fs.income * (1 + fs.incomeIncreaseRate) * (1 - fs.savingsRate) + 
          fs.income * (1 - fs.savingsRate)) :
    (fs.income * (1 + fs.incomeIncreaseRate) * fs.savingsRate - 
     fs.income * fs.savingsRate) / 
    (fs.income * fs.savingsRate) = 1 := by
  sorry

#check savings_increase_percentage

end NUMINAMATH_CALUDE_savings_increase_percentage_l315_31568


namespace NUMINAMATH_CALUDE_solve_equation_l315_31547

theorem solve_equation (x : ℝ) : (2 * x + 7) / 6 = 13 → x = 35.5 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l315_31547


namespace NUMINAMATH_CALUDE_g_inverse_property_l315_31583

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ :=
  if x < 3 then c * x + d else 10 - 2 * x

theorem g_inverse_property (c d : ℝ) :
  (∀ x, g c d (g c d x) = x) → c + d = 9/2 := by
  sorry

end NUMINAMATH_CALUDE_g_inverse_property_l315_31583


namespace NUMINAMATH_CALUDE_sequence_sum_proof_l315_31585

-- Define the sequence a_n and its sum S_n
def S (n : ℕ) : ℕ := 2^(n+1) - 2

-- Define the sequence b_n
def b (n : ℕ) : ℝ := 2^n * n

-- Theorem statement
theorem sequence_sum_proof (n : ℕ) :
  (∀ k, S k = 2^(k+1) - 2) →
  (∀ k, b k = 2^k * k) →
  (∃ T : ℕ → ℝ, T n = (n + 1) * 2^(n + 1) - 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_proof_l315_31585


namespace NUMINAMATH_CALUDE_common_term_value_l315_31575

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a₁ : ℝ  -- First term
  a₂ : ℝ  -- Second term

/-- Represents a geometric progression -/
structure GeometricProgression where
  g₁ : ℝ  -- First term
  g₂ : ℝ  -- Second term

/-- Given arithmetic and geometric progressions, if there exists a common term, it is 37/3 -/
theorem common_term_value (x : ℝ) (ap : ArithmeticProgression) (gp : GeometricProgression) 
  (h_ap : ap.a₁ = 2*x - 3 ∧ ap.a₂ = 5*x - 11)
  (h_gp : gp.g₁ = x + 1 ∧ gp.g₂ = 2*x + 3)
  (h_common : ∃ t : ℝ, (∃ n : ℕ, t = ap.a₁ + (n - 1) * (ap.a₂ - ap.a₁)) ∧ 
                       (∃ m : ℕ, t = gp.g₁ * (gp.g₂ / gp.g₁) ^ (m - 1))) :
  ∃ t : ℝ, t = 37/3 ∧ (∃ n : ℕ, t = ap.a₁ + (n - 1) * (ap.a₂ - ap.a₁)) ∧ 
               (∃ m : ℕ, t = gp.g₁ * (gp.g₂ / gp.g₁) ^ (m - 1)) := by
  sorry

end NUMINAMATH_CALUDE_common_term_value_l315_31575


namespace NUMINAMATH_CALUDE_smallest_satisfying_number_l315_31504

def is_smallest_satisfying_number (n : ℕ) : Prop :=
  (∀ m < n, ∃ p, Nat.Prime p ∧ m % (p - 1) = 0 ∧ m % p ≠ 0) ∧
  (∀ p, Nat.Prime p → n % (p - 1) = 0 → n % p = 0)

theorem smallest_satisfying_number :
  is_smallest_satisfying_number 1806 :=
sorry

end NUMINAMATH_CALUDE_smallest_satisfying_number_l315_31504


namespace NUMINAMATH_CALUDE_consecutive_blue_red_probability_l315_31548

def num_green : ℕ := 4
def num_blue : ℕ := 3
def num_red : ℕ := 5
def total_chips : ℕ := num_green + num_blue + num_red

def probability_consecutive_blue_red : ℚ :=
  (num_blue.factorial * num_red.factorial * (Nat.choose (num_green + 2) 2)) /
  total_chips.factorial

theorem consecutive_blue_red_probability :
  probability_consecutive_blue_red = 1 / 44352 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_blue_red_probability_l315_31548


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l315_31559

theorem quadratic_roots_ratio (m : ℚ) : 
  (∃ r s : ℚ, r ≠ 0 ∧ s ≠ 0 ∧ r / s = 3 ∧ 
   r^2 + 9*r + m = 0 ∧ s^2 + 9*s + m = 0) → 
  m = 243/16 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l315_31559


namespace NUMINAMATH_CALUDE_fold_sum_l315_31516

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold on a piece of graph paper -/
structure Fold where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- Theorem: If a piece of graph paper is folded such that (0,3) matches with (5,0) 
    and (8,4) matches with (p,q), then p + q = 10 -/
theorem fold_sum (f : Fold) (h1 : f.p1 = ⟨0, 3⟩) (h2 : f.p2 = ⟨5, 0⟩) 
    (h3 : f.p3 = ⟨8, 4⟩) (h4 : f.p4 = ⟨f.p4.x, f.p4.y⟩) : 
    f.p4.x + f.p4.y = 10 := by
  sorry

end NUMINAMATH_CALUDE_fold_sum_l315_31516


namespace NUMINAMATH_CALUDE_binary_to_decimal_101101_l315_31555

theorem binary_to_decimal_101101 : 
  (1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0) = 45 := by
  sorry

end NUMINAMATH_CALUDE_binary_to_decimal_101101_l315_31555


namespace NUMINAMATH_CALUDE_expression_value_at_three_l315_31552

theorem expression_value_at_three :
  let f (x : ℝ) := (x^2 - 5*x + 4) / (x - 4)
  f 3 = 2 := by sorry

end NUMINAMATH_CALUDE_expression_value_at_three_l315_31552


namespace NUMINAMATH_CALUDE_min_value_expression_equality_condition_l315_31503

theorem min_value_expression (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq_4 : a + b + c + d = 4) : 
  (a^8 / ((a^2+b)*(a^2+c)*(a^2+d))) + 
  (b^8 / ((b^2+c)*(b^2+d)*(b^2+a))) + 
  (c^8 / ((c^2+d)*(c^2+a)*(c^2+b))) + 
  (d^8 / ((d^2+a)*(d^2+b)*(d^2+c))) ≥ (1/2) := by
  sorry

theorem equality_condition (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0) (pos_d : d > 0)
  (sum_eq_4 : a + b + c + d = 4) :
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ↔ 
  (a^8 / ((a^2+b)*(a^2+c)*(a^2+d))) + 
  (b^8 / ((b^2+c)*(b^2+d)*(b^2+a))) + 
  (c^8 / ((c^2+d)*(c^2+a)*(c^2+b))) + 
  (d^8 / ((d^2+a)*(d^2+b)*(d^2+c))) = (1/2) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_equality_condition_l315_31503


namespace NUMINAMATH_CALUDE_sin_150_degrees_l315_31545

theorem sin_150_degrees : Real.sin (150 * π / 180) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_150_degrees_l315_31545


namespace NUMINAMATH_CALUDE_remainder_of_B_l315_31549

theorem remainder_of_B (A B : ℕ) (h : B = 9 * A + 13) : B % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_B_l315_31549


namespace NUMINAMATH_CALUDE_jug_fills_ten_large_glasses_l315_31521

/-- Represents the volume of a glass -/
structure Glass :=
  (volume : ℚ)

/-- Represents a jug with a certain capacity -/
structure Jug :=
  (capacity : ℚ)

/-- Represents the problem setup -/
structure JugProblem :=
  (small_glass : Glass)
  (large_glass : Glass)
  (jug : Jug)
  (condition1 : 9 * small_glass.volume + 4 * large_glass.volume = jug.capacity)
  (condition2 : 6 * small_glass.volume + 6 * large_glass.volume = jug.capacity)

theorem jug_fills_ten_large_glasses (problem : JugProblem) :
  problem.jug.capacity = 10 * problem.large_glass.volume :=
sorry

end NUMINAMATH_CALUDE_jug_fills_ten_large_glasses_l315_31521


namespace NUMINAMATH_CALUDE_angle_measure_in_triangle_l315_31563

theorem angle_measure_in_triangle (P Q R : ℝ) (h : P + Q = 60) : P + Q + R = 180 → R = 120 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_triangle_l315_31563


namespace NUMINAMATH_CALUDE_book_pages_count_l315_31536

/-- The number of pages Cora read on Monday -/
def monday_pages : ℕ := 23

/-- The number of pages Cora read on Tuesday -/
def tuesday_pages : ℕ := 38

/-- The number of pages Cora read on Wednesday -/
def wednesday_pages : ℕ := 61

/-- The number of pages Cora will read on Thursday -/
def thursday_pages : ℕ := 12

/-- The number of pages Cora will read on Friday -/
def friday_pages : ℕ := 2 * thursday_pages

/-- The total number of pages in the book -/
def total_pages : ℕ := monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages

theorem book_pages_count : total_pages = 158 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_count_l315_31536


namespace NUMINAMATH_CALUDE_solve_equation_l315_31519

theorem solve_equation : ∃ x : ℝ, (x - 5)^4 = (1/16)⁻¹ ∧ x = 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l315_31519


namespace NUMINAMATH_CALUDE_coconut_yield_for_six_trees_l315_31596

/-- The yield of x trees in a coconut grove --/
def coconut_grove_yield (x : ℕ) (Y : ℕ) : Prop :=
  let total_trees := 3 * x
  let total_yield := (x + 3) * 60 + x * Y + (x - 3) * 180
  (total_yield : ℚ) / total_trees = 100

theorem coconut_yield_for_six_trees :
  coconut_grove_yield 6 120 :=
sorry

end NUMINAMATH_CALUDE_coconut_yield_for_six_trees_l315_31596


namespace NUMINAMATH_CALUDE_rock_collection_difference_l315_31590

theorem rock_collection_difference (joshua_rocks : ℕ) (jose_rocks : ℕ) (albert_rocks : ℕ)
  (joshua_80 : joshua_rocks = 80)
  (jose_fewer : jose_rocks < joshua_rocks)
  (albert_jose_diff : albert_rocks = jose_rocks + 20)
  (albert_joshua_diff : albert_rocks = joshua_rocks + 6) :
  joshua_rocks - jose_rocks = 14 := by
sorry

end NUMINAMATH_CALUDE_rock_collection_difference_l315_31590


namespace NUMINAMATH_CALUDE_quadratic_maximum_l315_31592

-- Define the quadratic function
def f (x : ℝ) : ℝ := -3 * x^2 + 9 * x + 24

-- State the theorem
theorem quadratic_maximum :
  (∃ (x_max : ℝ), ∀ (x : ℝ), f x ≤ f x_max) ∧
  (∃ (x_max : ℝ), f x_max = 30.75) ∧
  (∀ (x : ℝ), f x ≤ 30.75) ∧
  f (3/2) = 30.75 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_maximum_l315_31592


namespace NUMINAMATH_CALUDE_greg_age_l315_31531

/-- Given the ages and relationships of Cindy, Jan, Marcia, and Greg, prove Greg's age. -/
theorem greg_age (cindy_age : ℕ) (jan_age : ℕ) (marcia_age : ℕ) (greg_age : ℕ)
  (h1 : cindy_age = 5)
  (h2 : jan_age = cindy_age + 2)
  (h3 : marcia_age = 2 * jan_age)
  (h4 : greg_age = marcia_age + 2) :
  greg_age = 16 := by
  sorry

end NUMINAMATH_CALUDE_greg_age_l315_31531


namespace NUMINAMATH_CALUDE_group_materials_calculation_l315_31589

-- Define the given quantities
def teacher_materials : ℕ := 28
def total_products : ℕ := 93

-- Define the function to calculate group materials
def group_materials : ℕ := total_products - teacher_materials

-- Theorem statement
theorem group_materials_calculation :
  group_materials = 65 :=
sorry

end NUMINAMATH_CALUDE_group_materials_calculation_l315_31589


namespace NUMINAMATH_CALUDE_ab_nonzero_sufficient_not_necessary_for_a_nonzero_l315_31564

theorem ab_nonzero_sufficient_not_necessary_for_a_nonzero :
  (∀ a b : ℝ, a * b ≠ 0 → a ≠ 0) ∧
  (∃ a b : ℝ, a ≠ 0 ∧ a * b = 0) :=
by sorry

end NUMINAMATH_CALUDE_ab_nonzero_sufficient_not_necessary_for_a_nonzero_l315_31564


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l315_31506

/-- Represents a geometric sequence with a given first term and common ratio -/
def GeometricSequence (a : ℝ) (r : ℝ) : ℕ → ℝ := fun n => a * r ^ (n - 1)

/-- The fourth term of a geometric sequence given its first and sixth terms -/
theorem fourth_term_of_geometric_sequence (a₁ : ℝ) (a₆ : ℝ) :
  a₁ > 0 → a₆ > 0 →
  ∃ (r : ℝ), r > 0 ∧ GeometricSequence a₁ r 6 = a₆ →
  GeometricSequence a₁ r 4 = 1536 :=
by
  sorry

#check fourth_term_of_geometric_sequence 512 125

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l315_31506


namespace NUMINAMATH_CALUDE_division_remainder_l315_31543

theorem division_remainder : ∃ q : ℕ, 1234567 = 137 * q + 102 ∧ 102 < 137 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l315_31543


namespace NUMINAMATH_CALUDE_bob_candies_count_l315_31500

-- Define Bob's items
def bob_chewing_gums : ℕ := 15
def bob_chocolate_bars : ℕ := 20
def bob_assorted_candies : ℕ := 15

-- Theorem to prove
theorem bob_candies_count : bob_assorted_candies = 15 := by
  sorry

end NUMINAMATH_CALUDE_bob_candies_count_l315_31500


namespace NUMINAMATH_CALUDE_isosceles_right_triangle_area_l315_31542

theorem isosceles_right_triangle_area (side_length : ℝ) : 
  side_length = 12 →
  ∃ (r s : ℝ), 
    r > 0 ∧ s > 0 ∧
    2 * (r ^ 2 + s ^ 2) = side_length ^ 2 ∧
    4 * (r ^ 2 / 2) = 72 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_right_triangle_area_l315_31542


namespace NUMINAMATH_CALUDE_alphabet_dot_no_line_l315_31586

theorem alphabet_dot_no_line (total : ℕ) (both : ℕ) (line_no_dot : ℕ) 
  (h1 : total = 50)
  (h2 : both = 16)
  (h3 : line_no_dot = 30)
  (h4 : total = both + line_no_dot + (total - (both + line_no_dot))) :
  total - (both + line_no_dot) = 4 := by
sorry

end NUMINAMATH_CALUDE_alphabet_dot_no_line_l315_31586


namespace NUMINAMATH_CALUDE_ramanujan_identity_l315_31569

theorem ramanujan_identity : ∃ (p q r p₁ q₁ r₁ : ℕ), 
  p ≠ q ∧ p ≠ r ∧ p ≠ p₁ ∧ p ≠ q₁ ∧ p ≠ r₁ ∧
  q ≠ r ∧ q ≠ p₁ ∧ q ≠ q₁ ∧ q ≠ r₁ ∧
  r ≠ p₁ ∧ r ≠ q₁ ∧ r ≠ r₁ ∧
  p₁ ≠ q₁ ∧ p₁ ≠ r₁ ∧
  q₁ ≠ r₁ ∧
  p^2 + q^2 + r^2 = p₁^2 + q₁^2 + r₁^2 ∧
  p^4 + q^4 + r^4 = p₁^4 + q₁^4 + r₁^4 := by
  sorry

end NUMINAMATH_CALUDE_ramanujan_identity_l315_31569


namespace NUMINAMATH_CALUDE_food_price_consumption_reduction_l315_31570

theorem food_price_consumption_reduction (initial_price : ℝ) (h : initial_price > 0) :
  let price_increase_factor := 1.5
  let consumption_reduction_factor := 2/3
  initial_price * price_increase_factor * consumption_reduction_factor = initial_price :=
by sorry

end NUMINAMATH_CALUDE_food_price_consumption_reduction_l315_31570


namespace NUMINAMATH_CALUDE_mary_next_birthday_l315_31530

/-- Represents the ages of Mary, Sally, and Danielle -/
structure Ages where
  mary : ℝ
  sally : ℝ
  danielle : ℝ

/-- The conditions of the problem -/
def problem_conditions (ages : Ages) : Prop :=
  ages.mary = 1.2 * ages.sally ∧
  ages.sally = 0.6 * ages.danielle ∧
  ages.mary + ages.sally + ages.danielle = 23.2

/-- The theorem to be proved -/
theorem mary_next_birthday (ages : Ages) :
  problem_conditions ages → ⌊ages.mary⌋ + 1 = 8 :=
sorry

end NUMINAMATH_CALUDE_mary_next_birthday_l315_31530


namespace NUMINAMATH_CALUDE_candy_jar_problem_l315_31574

theorem candy_jar_problem (banana_jar grape_jar peanut_butter_jar : ℕ) : 
  banana_jar = 43 →
  grape_jar = banana_jar + 5 →
  peanut_butter_jar = 4 * grape_jar →
  peanut_butter_jar = 192 := by
sorry

end NUMINAMATH_CALUDE_candy_jar_problem_l315_31574


namespace NUMINAMATH_CALUDE_locus_of_centers_l315_31571

/-- Circle C1 with equation x^2 + y^2 = 1 -/
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- Circle C2 with equation (x - 2)^2 + y^2 = 25 -/
def C2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 25

/-- A circle is externally tangent to C1 if the distance between their centers equals the sum of their radii -/
def externally_tangent_C1 (a b r : ℝ) : Prop := a^2 + b^2 = (r + 1)^2

/-- A circle is internally tangent to C2 if the distance between their centers equals the difference of their radii -/
def internally_tangent_C2 (a b r : ℝ) : Prop := (a - 2)^2 + b^2 = (5 - r)^2

/-- The main theorem: the locus of centers (a,b) of circles externally tangent to C1 and internally tangent to C2 -/
theorem locus_of_centers (a b : ℝ) : 
  (∃ r : ℝ, externally_tangent_C1 a b r ∧ internally_tangent_C2 a b r) → 
  3 * a^2 + b^2 + 44 * a + 121 = 0 :=
sorry

end NUMINAMATH_CALUDE_locus_of_centers_l315_31571


namespace NUMINAMATH_CALUDE_students_playing_both_football_and_tennis_l315_31579

/-- Given a class of students, calculates the number of students playing both football and long tennis. -/
def students_playing_both (total : ℕ) (football : ℕ) (long_tennis : ℕ) (neither : ℕ) : ℕ :=
  football + long_tennis - (total - neither)

/-- Theorem: In a class of 36 students, where 26 play football, 20 play long tennis, and 7 play neither,
    the number of students who play both football and long tennis is 17. -/
theorem students_playing_both_football_and_tennis :
  students_playing_both 36 26 20 7 = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_football_and_tennis_l315_31579


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l315_31597

-- Define the universe set U
def U : Set Nat := {2, 3, 6, 8}

-- Define set A
def A : Set Nat := {2, 3}

-- Define set B
def B : Set Nat := {2, 6, 8}

-- Theorem statement
theorem complement_intersection_theorem :
  (Aᶜ ∩ B) = {6, 8} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l315_31597


namespace NUMINAMATH_CALUDE_unfair_coin_expected_value_l315_31572

/-- The expected value of an unfair coin flip -/
theorem unfair_coin_expected_value :
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let win_amount : ℚ := 4
  let lose_amount : ℚ := 9
  let expected_value := p_heads * win_amount - p_tails * lose_amount
  expected_value = -1/3 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_expected_value_l315_31572


namespace NUMINAMATH_CALUDE_paulines_convertibles_l315_31561

/-- Calculates the number of convertibles in Pauline's car collection --/
theorem paulines_convertibles (total : ℕ) (regular_percent trucks_percent sedans_percent sports_percent suvs_percent : ℚ) :
  total = 125 →
  regular_percent = 38/100 →
  trucks_percent = 12/100 →
  sedans_percent = 17/100 →
  sports_percent = 22/100 →
  suvs_percent = 6/100 →
  ∃ (regular trucks sedans sports suvs convertibles : ℕ),
    regular = ⌊(regular_percent * total : ℚ)⌋ ∧
    trucks = ⌊(trucks_percent * total : ℚ)⌋ ∧
    sedans = ⌊(sedans_percent * total : ℚ)⌋ ∧
    sports = ⌊(sports_percent * total : ℚ)⌋ ∧
    suvs = ⌊(suvs_percent * total : ℚ)⌋ ∧
    convertibles = total - (regular + trucks + sedans + sports + suvs) ∧
    convertibles = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_paulines_convertibles_l315_31561


namespace NUMINAMATH_CALUDE_sol_earnings_l315_31510

/-- Calculates the earnings from selling candy bars over a week -/
def candy_bar_earnings (initial_sales : ℕ) (daily_increase : ℕ) (days : ℕ) (price_cents : ℕ) : ℚ :=
  let total_bars := (List.range days).map (λ i => initial_sales + i * daily_increase) |>.sum
  (total_bars * price_cents : ℚ) / 100

/-- Theorem stating that Sol's earnings from selling candy bars over a week is $12.00 -/
theorem sol_earnings : candy_bar_earnings 10 4 6 10 = 12 := by
  sorry

end NUMINAMATH_CALUDE_sol_earnings_l315_31510


namespace NUMINAMATH_CALUDE_books_rebecca_received_l315_31576

theorem books_rebecca_received (books_initial : ℕ) (books_remaining : ℕ) 
  (h1 : books_initial = 220)
  (h2 : books_remaining = 60) : 
  ∃ (rebecca_books : ℕ), 
    rebecca_books = (books_initial - books_remaining) / 4 ∧ 
    rebecca_books = 40 := by
  sorry

end NUMINAMATH_CALUDE_books_rebecca_received_l315_31576


namespace NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l315_31513

def consecutive_integers (start : Int) (n : Nat) : List Int :=
  List.range n |>.map (λ i => start + i)

def positive_integers (list : List Int) : List Int :=
  list.filter (λ x => x > 0)

def range_of_list (list : List Int) : Int :=
  list.maximum?.getD 0 - list.minimum?.getD 0

theorem range_of_positive_integers_in_list (k : List Int) :
  k = consecutive_integers (-4) 14 →
  range_of_list (positive_integers k) = 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_positive_integers_in_list_l315_31513


namespace NUMINAMATH_CALUDE_chord_length_unit_circle_specific_chord_length_l315_31527

/-- The length of the chord cut by a line on a unit circle -/
theorem chord_length_unit_circle (a b c : ℝ) (h : a^2 + b^2 ≠ 0) :
  let line := {(x, y) : ℝ × ℝ | a * x + b * y + c = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let d := |c| / Real.sqrt (a^2 + b^2)
  2 * Real.sqrt (1 - d^2) = 8/5 :=
by sorry

/-- The specific case for the given problem -/
theorem specific_chord_length :
  let line := {(x, y) : ℝ × ℝ | 3 * x - 4 * y + 3 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 1}
  let d := 3 / 5
  2 * Real.sqrt (1 - d^2) = 8/5 :=
by sorry

end NUMINAMATH_CALUDE_chord_length_unit_circle_specific_chord_length_l315_31527


namespace NUMINAMATH_CALUDE_missy_yells_84_times_l315_31528

/-- The number of times Missy yells at her obedient dog -/
def obedient_yells : ℕ := 12

/-- The ratio of yells at the stubborn dog compared to the obedient dog -/
def stubborn_ratio : ℕ := 4

/-- The ratio of yells at the mischievous dog compared to the obedient dog -/
def mischievous_ratio : ℕ := 2

/-- The total number of times Missy yells at all three dogs -/
def total_yells : ℕ := obedient_yells + stubborn_ratio * obedient_yells + mischievous_ratio * obedient_yells

theorem missy_yells_84_times : total_yells = 84 := by
  sorry

end NUMINAMATH_CALUDE_missy_yells_84_times_l315_31528


namespace NUMINAMATH_CALUDE_optimal_investment_l315_31562

/-- Represents an investment project with maximum profit and loss rates. -/
structure Project where
  max_profit_rate : ℝ
  max_loss_rate : ℝ

/-- Represents the investment scenario with two projects and constraints. -/
structure InvestmentScenario where
  project_a : Project
  project_b : Project
  total_investment : ℝ
  max_potential_loss : ℝ

/-- Calculates the potential loss for a given investment allocation. -/
def potential_loss (scenario : InvestmentScenario) (invest_a : ℝ) (invest_b : ℝ) : ℝ :=
  invest_a * scenario.project_a.max_loss_rate + invest_b * scenario.project_b.max_loss_rate

/-- Calculates the potential profit for a given investment allocation. -/
def potential_profit (scenario : InvestmentScenario) (invest_a : ℝ) (invest_b : ℝ) : ℝ :=
  invest_a * scenario.project_a.max_profit_rate + invest_b * scenario.project_b.max_profit_rate

/-- Theorem stating that the given investment allocation maximizes potential profits
    while satisfying all constraints. -/
theorem optimal_investment (scenario : InvestmentScenario)
    (h_project_a : scenario.project_a = { max_profit_rate := 1, max_loss_rate := 0.3 })
    (h_project_b : scenario.project_b = { max_profit_rate := 0.5, max_loss_rate := 0.1 })
    (h_total_investment : scenario.total_investment = 100000)
    (h_max_potential_loss : scenario.max_potential_loss = 18000) :
    ∀ (x y : ℝ),
      x + y ≤ scenario.total_investment →
      potential_loss scenario x y ≤ scenario.max_potential_loss →
      potential_profit scenario x y ≤ potential_profit scenario 40000 60000 :=
  sorry

end NUMINAMATH_CALUDE_optimal_investment_l315_31562


namespace NUMINAMATH_CALUDE_function_difference_bound_l315_31598

theorem function_difference_bound 
  (f : Set.Icc 0 1 → ℝ)
  (h1 : f ⟨0, by norm_num⟩ = f ⟨1, by norm_num⟩)
  (h2 : ∀ (x₁ x₂ : Set.Icc 0 1), x₁ ≠ x₂ → |f x₂ - f x₁| < |x₂.val - x₁.val|) :
  ∀ (x₁ x₂ : Set.Icc 0 1), |f x₂ - f x₁| < (1 : ℝ) / 2 := by
sorry

end NUMINAMATH_CALUDE_function_difference_bound_l315_31598


namespace NUMINAMATH_CALUDE_right_rectangular_prism_diagonal_ratio_bound_right_rectangular_prism_diagonal_ratio_bound_tight_l315_31517

theorem right_rectangular_prism_diagonal_ratio_bound 
  (a b h d : ℝ) (ha : a > 0) (hb : b > 0) (hh : h > 0) 
  (hd : d^2 = a^2 + b^2 + h^2) : 
  (a + b + h) / d ≤ Real.sqrt 3 := by
sorry

theorem right_rectangular_prism_diagonal_ratio_bound_tight : 
  ∃ (a b h d : ℝ), a > 0 ∧ b > 0 ∧ h > 0 ∧ d^2 = a^2 + b^2 + h^2 ∧ 
  (a + b + h) / d = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_right_rectangular_prism_diagonal_ratio_bound_right_rectangular_prism_diagonal_ratio_bound_tight_l315_31517


namespace NUMINAMATH_CALUDE_value_in_scientific_notation_l315_31578

/-- Represents 1 billion -/
def billion : ℝ := 10^9

/-- The value we want to express in scientific notation -/
def value : ℝ := 45 * billion

/-- The scientific notation representation of the value -/
def scientific_notation : ℝ := 4.5 * 10^9

theorem value_in_scientific_notation : value = scientific_notation := by
  sorry

end NUMINAMATH_CALUDE_value_in_scientific_notation_l315_31578


namespace NUMINAMATH_CALUDE_smallest_n_candies_l315_31533

theorem smallest_n_candies : ∃ (n : ℕ), 
  (n ≥ 100 ∧ n < 1000) ∧ 
  (n + 6) % 7 = 0 ∧ 
  (n - 9) % 4 = 0 ∧
  (∀ m : ℕ, m ≥ 100 ∧ m < 1000 ∧ (m + 6) % 7 = 0 ∧ (m - 9) % 4 = 0 → n ≤ m) ∧
  n = 113 := by
sorry

end NUMINAMATH_CALUDE_smallest_n_candies_l315_31533


namespace NUMINAMATH_CALUDE_prime_square_plus_200_is_square_l315_31544

theorem prime_square_plus_200_is_square (p : ℕ) : 
  Prime p ∧ ∃ (n : ℕ), p^2 + 200 = n^2 ↔ p = 5 ∨ p = 23 := by
  sorry

end NUMINAMATH_CALUDE_prime_square_plus_200_is_square_l315_31544


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l315_31534

theorem cubic_polynomial_roots (a b c : ℤ) (r₁ r₂ r₃ : ℤ) : 
  (∀ x : ℤ, x^3 + a*x^2 + b*x + c = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃) →
  (r₁ > 2 ∧ r₂ > 2 ∧ r₃ > 2) →
  a + b + c + 1 = -2009 →
  (r₁ - 1) * (r₂ - 1) * (r₃ - 1) = 2009 →
  a = -58 := by
sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l315_31534


namespace NUMINAMATH_CALUDE_v_1010_proof_l315_31540

/-- Represents the last term of the nth group in the sequence -/
def f (n : ℕ) : ℕ := (5 * n^2 - 3 * n + 2) / 2

/-- Represents the total number of terms up to and including the nth group -/
def total_terms (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 1010th term of the sequence -/
def v_1010 : ℕ := 4991

theorem v_1010_proof : 
  ∃ (group : ℕ), 
    total_terms group ≥ 1010 ∧ 
    total_terms (group - 1) < 1010 ∧
    v_1010 = f group - (total_terms group - 1010) :=
sorry

end NUMINAMATH_CALUDE_v_1010_proof_l315_31540


namespace NUMINAMATH_CALUDE_mark_donation_shelters_l315_31538

/-- The number of shelters Mark donates soup to -/
def num_shelters (people_per_shelter : ℕ) (cans_per_person : ℕ) (total_cans : ℕ) : ℕ :=
  total_cans / (people_per_shelter * cans_per_person)

theorem mark_donation_shelters :
  num_shelters 30 10 1800 = 6 := by
  sorry

end NUMINAMATH_CALUDE_mark_donation_shelters_l315_31538


namespace NUMINAMATH_CALUDE_sin_alpha_for_point_l315_31556

theorem sin_alpha_for_point (α : Real) :
  (∃ (r : Real), r > 0 ∧ r * Real.cos α = 3 ∧ r * Real.sin α = -4) →
  Real.sin α = -4/5 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_for_point_l315_31556


namespace NUMINAMATH_CALUDE_value_of_expression_l315_31550

theorem value_of_expression (x y z : ℝ) 
  (eq1 : 2 * x + y - z = 7)
  (eq2 : x + 2 * y + z = 5)
  (eq3 : x - y + 2 * z = 3) :
  2 * x * y / 3 = 1.625 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l315_31550


namespace NUMINAMATH_CALUDE_value_added_to_numbers_l315_31507

theorem value_added_to_numbers (n : ℕ) (original_avg new_avg x : ℝ) 
  (h1 : n = 15)
  (h2 : original_avg = 40)
  (h3 : new_avg = 54)
  (h4 : n * new_avg = n * original_avg + n * x) :
  x = 14 := by
  sorry

end NUMINAMATH_CALUDE_value_added_to_numbers_l315_31507


namespace NUMINAMATH_CALUDE_estimate_total_balls_l315_31505

/-- Represents a box containing red and green balls -/
structure BallBox where
  redBalls : ℕ
  totalBalls : ℕ
  hRedBalls : redBalls > 0
  hTotalBalls : totalBalls ≥ redBalls

/-- The probability of drawing a red ball -/
def drawRedProbability (box : BallBox) : ℚ :=
  box.redBalls / box.totalBalls

theorem estimate_total_balls
  (box : BallBox)
  (hRedBalls : box.redBalls = 5)
  (hProbability : drawRedProbability box = 1/4) :
  box.totalBalls = 20 := by
sorry

end NUMINAMATH_CALUDE_estimate_total_balls_l315_31505


namespace NUMINAMATH_CALUDE_rectangle_x_value_l315_31520

/-- A rectangle in a rectangular coordinate system with given properties -/
structure Rectangle where
  x : ℝ
  area : ℝ
  h1 : area = 90

/-- The x-coordinate of the first and last vertices of the rectangle is -9 -/
theorem rectangle_x_value (rect : Rectangle) : rect.x = -9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_x_value_l315_31520


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l315_31599

theorem polynomial_division_remainder : ∃ q : Polynomial ℚ, 
  (X : Polynomial ℚ)^5 - 3*(X^3) + X^2 + 2 = 
  (X^2 - 4*X + 6) * q + (-22*X - 28) := by sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l315_31599


namespace NUMINAMATH_CALUDE_inequality_solution_set_l315_31537

theorem inequality_solution_set (x : ℝ) (h : x ≠ 0) :
  (1 / x ≤ 1 / 3) ↔ (x ≥ 3 ∨ x < 0) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l315_31537


namespace NUMINAMATH_CALUDE_soap_box_height_is_five_l315_31553

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.length * d.width * d.height

/-- Theorem: Given the carton and soap box dimensions, and the maximum number of soap boxes,
    the height of the soap box must be 5 inches -/
theorem soap_box_height_is_five
  (carton : BoxDimensions)
  (soap : BoxDimensions)
  (max_boxes : ℕ)
  (h_carton_length : carton.length = 25)
  (h_carton_width : carton.width = 48)
  (h_carton_height : carton.height = 60)
  (h_soap_length : soap.length = 8)
  (h_soap_width : soap.width = 6)
  (h_max_boxes : max_boxes = 300)
  (h_fit : max_boxes * boxVolume soap = boxVolume carton) :
  soap.height = 5 := by
  sorry

end NUMINAMATH_CALUDE_soap_box_height_is_five_l315_31553


namespace NUMINAMATH_CALUDE_marathon_practice_distance_l315_31502

/-- Calculates the total distance run given the number of days and miles per day -/
def total_distance (days : ℕ) (miles_per_day : ℕ) : ℕ :=
  days * miles_per_day

/-- Proves that running 8 miles for 9 days results in a total of 72 miles -/
theorem marathon_practice_distance :
  total_distance 9 8 = 72 := by
  sorry

end NUMINAMATH_CALUDE_marathon_practice_distance_l315_31502


namespace NUMINAMATH_CALUDE_minimum_value_implies_m_l315_31508

/-- If the function f(x) = x^2 - 2x + m has a minimum value of -2 on the interval [2, +∞),
    then m = -2. -/
theorem minimum_value_implies_m (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x^2 - 2*x + m) →
  (∀ x ≥ 2, f x ≥ -2) →
  (∃ x ≥ 2, f x = -2) →
  m = -2 := by
sorry

end NUMINAMATH_CALUDE_minimum_value_implies_m_l315_31508


namespace NUMINAMATH_CALUDE_ceiling_floor_difference_l315_31535

theorem ceiling_floor_difference : 
  ⌈(15 : ℚ) / 8 * (-34 : ℚ) / 4⌉ - ⌊(15 : ℚ) / 8 * ⌊(-34 : ℚ) / 4⌋⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ceiling_floor_difference_l315_31535


namespace NUMINAMATH_CALUDE_contractor_wage_l315_31567

/-- Contractor's wage problem -/
theorem contractor_wage
  (total_days : ℕ)
  (absent_days : ℕ)
  (daily_fine : ℚ)
  (total_amount : ℚ)
  (h1 : total_days = 30)
  (h2 : absent_days = 10)
  (h3 : daily_fine = 7.5)
  (h4 : total_amount = 425)
  : ∃ (daily_wage : ℚ),
    daily_wage * (total_days - absent_days : ℚ) - daily_fine * absent_days = total_amount ∧
    daily_wage = 25 := by
  sorry

end NUMINAMATH_CALUDE_contractor_wage_l315_31567


namespace NUMINAMATH_CALUDE_forever_alive_characterization_l315_31573

/-- Represents the state of a cell: alive or dead -/
inductive CellState
| Alive
| Dead

/-- Represents a grid of cells -/
def Grid (m n : ℕ) := Fin m → Fin n → CellState

/-- Counts the number of alive neighbors for a cell -/
def countAliveNeighbors (grid : Grid m n) (i j : Fin m) : ℕ := sorry

/-- Updates the state of a single cell based on its neighbors -/
def updateCell (grid : Grid m n) (i j : Fin m) : CellState := sorry

/-- Updates the entire grid for one time step -/
def updateGrid (grid : Grid m n) : Grid m n := sorry

/-- Checks if a grid has at least one alive cell -/
def hasAliveCell (grid : Grid m n) : Prop := sorry

/-- Represents the existence of an initial configuration that stays alive forever -/
def existsForeverAliveConfig (m n : ℕ) : Prop :=
  ∃ (initial : Grid m n), ∀ (t : ℕ), hasAliveCell (Nat.iterate updateGrid t initial)

/-- The main theorem: characterizes the pairs (m, n) for which an eternally alive configuration exists -/
theorem forever_alive_characterization (m n : ℕ) :
  existsForeverAliveConfig m n ↔ (m, n) ≠ (1, 1) ∧ (m, n) ≠ (1, 3) ∧ (m, n) ≠ (3, 1) :=
sorry

end NUMINAMATH_CALUDE_forever_alive_characterization_l315_31573


namespace NUMINAMATH_CALUDE_find_k_l315_31546

theorem find_k (k : ℝ) (h : 24 / k = 4) : k = 6 := by
  sorry

end NUMINAMATH_CALUDE_find_k_l315_31546


namespace NUMINAMATH_CALUDE_marble_probability_l315_31577

/-- Represents a box of marbles -/
structure Box where
  gold : Nat
  black : Nat

/-- The probability of selecting a gold marble from a box -/
def prob_gold (b : Box) : Rat :=
  b.gold / (b.gold + b.black)

/-- The probability of selecting a black marble from a box -/
def prob_black (b : Box) : Rat :=
  b.black / (b.gold + b.black)

/-- The initial state of the boxes -/
def initial_boxes : List Box :=
  [⟨1, 1⟩, ⟨1, 2⟩, ⟨1, 3⟩]

/-- The probability of the final outcome after the marble movements -/
def final_probability : Rat :=
  let box1 := initial_boxes[0]
  let box2 := initial_boxes[1]
  let box3 := initial_boxes[2]

  let prob_gold_to_box2 := prob_gold box1 * prob_gold (⟨box2.gold + 1, box2.black⟩) +
                           prob_black box1 * prob_gold box2
  
  let prob_black_to_box3 := 1 - prob_gold_to_box2

  prob_gold_to_box2 * prob_gold (⟨box3.gold + 1, box3.black⟩) +
  prob_black_to_box3 * prob_gold box3

theorem marble_probability :
  final_probability = 11 / 40 := by sorry

end NUMINAMATH_CALUDE_marble_probability_l315_31577


namespace NUMINAMATH_CALUDE_square_sum_implies_product_zero_l315_31593

theorem square_sum_implies_product_zero (n : ℝ) : 
  (n - 2022)^2 + (2023 - n)^2 = 1 → (n - 2022) * (2023 - n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_implies_product_zero_l315_31593


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l315_31532

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 5 }
def B : Set ℝ := { x | 3 < x ∧ x < 9 }

theorem union_of_A_and_B : A ∪ B = { x | -1 ≤ x ∧ x < 9 } := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l315_31532


namespace NUMINAMATH_CALUDE_product_sum_fraction_equality_l315_31581

theorem product_sum_fraction_equality : (3 * 4 * 5) * (1/3 + 1/4 + 1/5) = 47 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fraction_equality_l315_31581


namespace NUMINAMATH_CALUDE_smallest_operation_between_sqrt18_and_sqrt8_l315_31557

theorem smallest_operation_between_sqrt18_and_sqrt8 :
  let a := Real.sqrt 18
  let b := Real.sqrt 8
  (a - b < a + b) ∧ (a - b < a * b) ∧ (a - b < a / b) := by
  sorry

end NUMINAMATH_CALUDE_smallest_operation_between_sqrt18_and_sqrt8_l315_31557


namespace NUMINAMATH_CALUDE_car_speed_before_servicing_l315_31595

/-- The speed of a car before and after servicing -/
theorem car_speed_before_servicing (speed_serviced : ℝ) (time_serviced time_not_serviced : ℝ) 
  (h1 : speed_serviced = 90)
  (h2 : time_serviced = 3)
  (h3 : time_not_serviced = 6)
  (h4 : speed_serviced * time_serviced = speed_not_serviced * time_not_serviced) :
  speed_not_serviced = 45 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_before_servicing_l315_31595


namespace NUMINAMATH_CALUDE_winnie_yesterday_repetitions_l315_31526

/-- The number of repetitions Winnie completed yesterday -/
def yesterday_repetitions : ℕ := 86

/-- The number of repetitions Winnie completed today -/
def today_repetitions : ℕ := 73

/-- The number of repetitions Winnie fell behind by today -/
def difference : ℕ := 13

/-- Theorem: Winnie completed 86 repetitions yesterday -/
theorem winnie_yesterday_repetitions :
  yesterday_repetitions = today_repetitions + difference :=
by sorry

end NUMINAMATH_CALUDE_winnie_yesterday_repetitions_l315_31526


namespace NUMINAMATH_CALUDE_B_largest_at_200_l315_31594

/-- B_k is defined as the binomial coefficient (800 choose k) multiplied by 0.3^k -/
def B (k : ℕ) : ℝ := (Nat.choose 800 k : ℝ) * (0.3 ^ k)

/-- Theorem stating that B_k is largest when k = 200 -/
theorem B_largest_at_200 : ∀ k : ℕ, k ≤ 800 → B k ≤ B 200 :=
sorry

end NUMINAMATH_CALUDE_B_largest_at_200_l315_31594


namespace NUMINAMATH_CALUDE_prime_sum_product_l315_31515

/-- Given prime numbers a, b, and c satisfying abc + a + b + c = 99,
    prove that two of the numbers are 2 and the other is 19 -/
theorem prime_sum_product (a b c : ℕ) : 
  Prime a → Prime b → Prime c → a * b * c + a + b + c = 99 →
  ((a = 2 ∧ b = 2 ∧ c = 19) ∨ (a = 2 ∧ b = 19 ∧ c = 2) ∨ (a = 19 ∧ b = 2 ∧ c = 2)) :=
by sorry

end NUMINAMATH_CALUDE_prime_sum_product_l315_31515


namespace NUMINAMATH_CALUDE_square_construction_with_compass_l315_31587

-- Define a point in 2D space
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define a compass operation
def compassIntersection (c1 c2 : Circle) : Set Point :=
  { p : Point | (p.x - c1.center.x)^2 + (p.y - c1.center.y)^2 = c1.radius^2 ∧
                (p.x - c2.center.x)^2 + (p.y - c2.center.y)^2 = c2.radius^2 }

-- Define a square
structure Square where
  vertices : Fin 4 → Point

-- Theorem statement
theorem square_construction_with_compass :
  ∃ (s : Square), 
    (∀ i j : Fin 4, i ≠ j → 
      (s.vertices i).x^2 + (s.vertices i).y^2 = 
      (s.vertices j).x^2 + (s.vertices j).y^2) ∧
    (∀ i : Fin 4, 
      (s.vertices i).x^2 + (s.vertices i).y^2 = 
      ((s.vertices (i + 1)).x - (s.vertices i).x)^2 + 
      ((s.vertices (i + 1)).y - (s.vertices i).y)^2) :=
by
  sorry


end NUMINAMATH_CALUDE_square_construction_with_compass_l315_31587


namespace NUMINAMATH_CALUDE_crystal_lake_trail_length_l315_31541

/-- Represents the Crystal Lake Trail hike --/
structure CrystalLakeTrail where
  day1 : ℝ
  day2 : ℝ
  day3 : ℝ
  day4 : ℝ
  day5 : ℝ

/-- Conditions of the Crystal Lake Trail hike --/
def hikingConditions (hike : CrystalLakeTrail) : Prop :=
  hike.day1 + hike.day2 = 28 ∧
  (hike.day2 + hike.day3) / 2 = 15 ∧
  hike.day3 + hike.day4 + hike.day5 = 42 ∧
  hike.day1 + hike.day4 = 30

/-- Theorem stating that the total length of the Crystal Lake Trail is 70 miles --/
theorem crystal_lake_trail_length 
  (hike : CrystalLakeTrail) 
  (h : hikingConditions hike) : 
  hike.day1 + hike.day2 + hike.day3 + hike.day4 + hike.day5 = 70 := by
  sorry

end NUMINAMATH_CALUDE_crystal_lake_trail_length_l315_31541


namespace NUMINAMATH_CALUDE_polynomial_equality_l315_31558

/-- Given that 4x^4 + 8x^3 + g(x) = 2x^4 - 5x^3 + 7x + 4,
    prove that g(x) = -2x^4 - 13x^3 + 7x + 4 -/
theorem polynomial_equality (x : ℝ) (g : ℝ → ℝ) 
    (h : ∀ x, 4 * x^4 + 8 * x^3 + g x = 2 * x^4 - 5 * x^3 + 7 * x + 4) :
  g x = -2 * x^4 - 13 * x^3 + 7 * x + 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l315_31558


namespace NUMINAMATH_CALUDE_zions_dad_age_difference_l315_31518

/-- Proves that Zion's dad's age is 3 years more than 4 times Zion's age given the conditions. -/
theorem zions_dad_age_difference (zion_age : ℕ) (dad_age : ℕ) : 
  zion_age = 8 →
  dad_age > 4 * zion_age →
  dad_age + 10 = (zion_age + 10) + 27 →
  dad_age = 4 * zion_age + 3 := by
sorry

end NUMINAMATH_CALUDE_zions_dad_age_difference_l315_31518


namespace NUMINAMATH_CALUDE_dans_limes_l315_31580

theorem dans_limes (limes_picked : ℕ) (limes_given : ℕ) : limes_picked = 9 → limes_given = 4 → limes_picked + limes_given = 13 := by
  sorry

end NUMINAMATH_CALUDE_dans_limes_l315_31580


namespace NUMINAMATH_CALUDE_hundred_days_after_wednesday_is_friday_l315_31565

/-- Enumeration of days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Function to get the day of the week after a given number of days -/
def dayAfter (start : DayOfWeek) (days : Nat) : DayOfWeek :=
  match days with
  | 0 => start
  | n + 1 => nextDay (dayAfter start n)

/-- Theorem stating that 100 days after Wednesday is Friday -/
theorem hundred_days_after_wednesday_is_friday :
  dayAfter DayOfWeek.Wednesday 100 = DayOfWeek.Friday := by
  sorry


end NUMINAMATH_CALUDE_hundred_days_after_wednesday_is_friday_l315_31565


namespace NUMINAMATH_CALUDE_days_before_reinforcement_l315_31566

/-- Proves that the number of days before reinforcement arrived is 12 --/
theorem days_before_reinforcement 
  (initial_garrison : ℕ) 
  (initial_provision_days : ℕ) 
  (reinforcement : ℕ) 
  (remaining_provision_days : ℕ) 
  (h1 : initial_garrison = 1850)
  (h2 : initial_provision_days = 28)
  (h3 : reinforcement = 1110)
  (h4 : remaining_provision_days = 10) :
  (initial_garrison * initial_provision_days - 
   (initial_garrison + reinforcement) * remaining_provision_days) / initial_garrison = 12 :=
by sorry

end NUMINAMATH_CALUDE_days_before_reinforcement_l315_31566


namespace NUMINAMATH_CALUDE_total_children_l315_31539

theorem total_children (happy_children sad_children neutral_children boys girls happy_boys sad_girls neutral_boys : ℕ) : 
  happy_children = 30 →
  sad_children = 10 →
  neutral_children = 20 →
  boys = 22 →
  girls = 38 →
  happy_boys = 6 →
  sad_girls = 4 →
  neutral_boys = 10 →
  happy_children + sad_children + neutral_children = boys + girls :=
by
  sorry

end NUMINAMATH_CALUDE_total_children_l315_31539


namespace NUMINAMATH_CALUDE_number_division_problem_l315_31582

theorem number_division_problem : ∃ x : ℝ, x / 5 = 40 + x / 6 ∧ x = 7200 / 31 := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l315_31582


namespace NUMINAMATH_CALUDE_smallest_class_size_l315_31529

theorem smallest_class_size (n : ℕ) : 
  n > 9 ∧ 
  (∃ (a b c d e : ℕ), 
    a = n ∧ b = n ∧ c = n ∧ d = n + 2 ∧ e = n + 3 ∧
    a + b + c + d + e > 50) →
  (∀ m : ℕ, m > 9 ∧ 
    (∃ (a b c d e : ℕ), 
      a = m ∧ b = m ∧ c = m ∧ d = m + 2 ∧ e = m + 3 ∧
      a + b + c + d + e > 50) →
    5 * n + 5 ≤ 5 * m + 5) →
  5 * n + 5 = 55 := by
sorry

end NUMINAMATH_CALUDE_smallest_class_size_l315_31529


namespace NUMINAMATH_CALUDE_spencer_walking_distance_l315_31512

/-- The total distance walked by Spencer -/
def total_distance (initial_distance : ℝ) (first_segment : ℝ) (second_segment : ℝ) : ℝ :=
  first_segment + second_segment + initial_distance

/-- Theorem: Spencer's total walking distance is 1400 meters -/
theorem spencer_walking_distance :
  let initial_distance : ℝ := 1000
  let first_segment : ℝ := 200
  let second_segment : ℝ := 200
  total_distance initial_distance first_segment second_segment = 1400 := by
  sorry

#eval total_distance 1000 200 200

end NUMINAMATH_CALUDE_spencer_walking_distance_l315_31512


namespace NUMINAMATH_CALUDE_bill_per_person_l315_31523

def total_bill : ℚ := 139
def num_people : ℕ := 8
def tip_percentage : ℚ := 1 / 10

theorem bill_per_person : 
  ∃ (bill_share : ℚ), 
    (bill_share * num_people).ceil = 
      ((total_bill * (1 + tip_percentage)).ceil) ∧ 
    bill_share = 1911 / 100 := by
  sorry

end NUMINAMATH_CALUDE_bill_per_person_l315_31523


namespace NUMINAMATH_CALUDE_y_value_l315_31509

theorem y_value : ∃ y : ℝ, 1.5 * y - 10 = 35 ∧ y = 30 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l315_31509


namespace NUMINAMATH_CALUDE_divisors_of_216n4_l315_31588

/-- Number of positive integer divisors of a natural number -/
def num_divisors (n : ℕ) : ℕ := sorry

theorem divisors_of_216n4 (n : ℕ) (h : n > 0) (h240 : num_divisors (240 * n^3) = 240) : 
  num_divisors (216 * n^4) = 156 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_216n4_l315_31588


namespace NUMINAMATH_CALUDE_prob_heart_then_ace_is_one_ninety_eighth_l315_31554

/-- Represents a standard deck of 51 cards (missing the Ace of Spades) -/
def StandardDeck : ℕ := 51

/-- Number of hearts in the deck -/
def NumHearts : ℕ := 13

/-- Number of aces in the deck -/
def NumAces : ℕ := 3

/-- Probability of drawing a heart as the first card and an ace as the second card -/
def prob_heart_then_ace : ℚ := NumHearts / StandardDeck * NumAces / (StandardDeck - 1)

theorem prob_heart_then_ace_is_one_ninety_eighth :
  prob_heart_then_ace = 1 / 98 := by
  sorry

end NUMINAMATH_CALUDE_prob_heart_then_ace_is_one_ninety_eighth_l315_31554


namespace NUMINAMATH_CALUDE_quadratic_inequality_l315_31514

theorem quadratic_inequality (x : ℝ) : x ≥ 1 → x^2 + 3*x - 2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l315_31514


namespace NUMINAMATH_CALUDE_megan_homework_time_l315_31525

/-- The time it takes to complete all problems given the number of math problems,
    spelling problems, and problems that can be finished per hour. -/
def time_to_complete (math_problems : ℕ) (spelling_problems : ℕ) (problems_per_hour : ℕ) : ℕ :=
  (math_problems + spelling_problems) / problems_per_hour

/-- Theorem stating that with 36 math problems, 28 spelling problems,
    and the ability to finish 8 problems per hour, it takes 8 hours to complete all problems. -/
theorem megan_homework_time :
  time_to_complete 36 28 8 = 8 := by
  sorry

end NUMINAMATH_CALUDE_megan_homework_time_l315_31525
