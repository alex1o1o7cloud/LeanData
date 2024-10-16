import Mathlib

namespace NUMINAMATH_CALUDE_expression_factorization_l2895_289536

theorem expression_factorization (y : ℝ) : 
  (16 * y^6 + 36 * y^4 - 9) - (4 * y^6 - 6 * y^4 - 9) = 6 * y^4 * (2 * y^2 + 7) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2895_289536


namespace NUMINAMATH_CALUDE_greatest_sum_of_two_integers_l2895_289559

theorem greatest_sum_of_two_integers (n : ℤ) : 
  (∀ m : ℤ, m * (m + 2) < 500 → m ≤ n) →
  n * (n + 2) < 500 →
  n + (n + 2) = 44 := by
sorry

end NUMINAMATH_CALUDE_greatest_sum_of_two_integers_l2895_289559


namespace NUMINAMATH_CALUDE_max_value_sqrt_sum_l2895_289568

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  Real.sqrt (a + 1) + Real.sqrt (b + 3) ≤ 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_sum_l2895_289568


namespace NUMINAMATH_CALUDE_twenty_is_forty_percent_of_fifty_l2895_289546

theorem twenty_is_forty_percent_of_fifty :
  ∀ x : ℝ, (20 : ℝ) / x = (40 : ℝ) / 100 → x = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_twenty_is_forty_percent_of_fifty_l2895_289546


namespace NUMINAMATH_CALUDE_root_sum_zero_l2895_289597

theorem root_sum_zero (a b : ℝ) : 
  (Complex.I + 1) ^ 2 + a * (Complex.I + 1) + b = 0 → a + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_zero_l2895_289597


namespace NUMINAMATH_CALUDE_largest_inscribable_rectangle_area_l2895_289567

/-- The area of the largest inscribable rectangle between two congruent equilateral triangles
    within a rectangle of width 8 and length 12 -/
theorem largest_inscribable_rectangle_area
  (width : ℝ) (length : ℝ)
  (h_width : width = 8)
  (h_length : length = 12)
  (triangle_side : ℝ)
  (h_triangle_side : triangle_side = 8 * Real.sqrt 3 / 3)
  (inscribed_height : ℝ)
  (h_inscribed_height : inscribed_height = width - triangle_side)
  : inscribed_height * length = 96 - 32 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_largest_inscribable_rectangle_area_l2895_289567


namespace NUMINAMATH_CALUDE_systematic_sampling_identification_l2895_289511

/-- A sampling method is a function that selects elements from a population. -/
def SamplingMethod := Type → Type

/-- Systematic sampling is a method where samples are selected at regular intervals. -/
def IsSystematicSampling (m : SamplingMethod) : Prop := sorry

/-- Method 1: Sampling from numbered balls with a fixed interval. -/
def Method1 : SamplingMethod := sorry

/-- Method 2: Sampling products from a conveyor belt at fixed time intervals. -/
def Method2 : SamplingMethod := sorry

/-- Method 3: Random sampling at a shopping mall entrance. -/
def Method3 : SamplingMethod := sorry

/-- Method 4: Sampling moviegoers in specific seats. -/
def Method4 : SamplingMethod := sorry

/-- Theorem stating which methods are systematic sampling. -/
theorem systematic_sampling_identification :
  IsSystematicSampling Method1 ∧
  IsSystematicSampling Method2 ∧
  ¬IsSystematicSampling Method3 ∧
  IsSystematicSampling Method4 := by sorry

end NUMINAMATH_CALUDE_systematic_sampling_identification_l2895_289511


namespace NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2895_289554

theorem sum_of_squares_and_square_of_sum : (3 + 7)^2 + (3^2 + 7^2) = 158 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_and_square_of_sum_l2895_289554


namespace NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l2895_289549

theorem ice_cream_scoop_arrangements :
  (Finset.univ.filter (fun σ : Equiv.Perm (Fin 5) => true)).card = 120 := by
  sorry

end NUMINAMATH_CALUDE_ice_cream_scoop_arrangements_l2895_289549


namespace NUMINAMATH_CALUDE_remainder_theorem_l2895_289598

theorem remainder_theorem (P D Q R D' Q' R' D'' S T : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = Q' * D' + R')
  (h3 : R' = S + T)
  (h4 : S = D'' * T) :
  P % (D * D' * D'') = D * R' + R :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2895_289598


namespace NUMINAMATH_CALUDE_chef_leftover_potatoes_l2895_289556

/-- Given a chef's potato and fry situation, calculate the number of leftover potatoes. -/
def leftover_potatoes (fries_per_potato : ℕ) (total_potatoes : ℕ) (required_fries : ℕ) : ℕ :=
  total_potatoes - (required_fries / fries_per_potato)

/-- Prove that the chef will have 7 potatoes leftover. -/
theorem chef_leftover_potatoes :
  leftover_potatoes 25 15 200 = 7 := by
  sorry

end NUMINAMATH_CALUDE_chef_leftover_potatoes_l2895_289556


namespace NUMINAMATH_CALUDE_unique_mod_residue_l2895_289506

theorem unique_mod_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -4321 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_mod_residue_l2895_289506


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2895_289561

theorem inequality_solution_set (a : ℝ) (h : a^3 < a ∧ a < a^2) :
  {x : ℝ | x + a > 1 - a * x} = {x : ℝ | x < (1 - a) / (1 + a)} := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2895_289561


namespace NUMINAMATH_CALUDE_replaced_person_age_l2895_289551

/-- Represents a group of people with their ages -/
structure AgeGroup where
  size : ℕ
  average_age : ℝ

/-- Theorem stating the age of the replaced person -/
theorem replaced_person_age (group : AgeGroup) (h1 : group.size = 10) 
  (h2 : ∃ (new_average : ℝ), new_average = group.average_age - 3) 
  (h3 : ∃ (new_person_age : ℝ), new_person_age = 18) : 
  ∃ (replaced_age : ℝ), replaced_age = 48 := by
  sorry

end NUMINAMATH_CALUDE_replaced_person_age_l2895_289551


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l2895_289547

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (cost_per_foot : ℝ) (total_cost : ℝ) :
  area = 289 →
  cost_per_foot = 59 →
  total_cost = 4 * Real.sqrt area * cost_per_foot →
  total_cost = 4012 := by
  sorry

#check fence_cost_square_plot

end NUMINAMATH_CALUDE_fence_cost_square_plot_l2895_289547


namespace NUMINAMATH_CALUDE_cost_of_flour_l2895_289535

/-- Given the total cost of flour and cake stand, and the cost of the cake stand,
    prove that the cost of flour is $5. -/
theorem cost_of_flour (total_cost cake_stand_cost : ℕ)
  (h1 : total_cost = 33)
  (h2 : cake_stand_cost = 28) :
  total_cost - cake_stand_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_cost_of_flour_l2895_289535


namespace NUMINAMATH_CALUDE_problem_solution_l2895_289507

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + a + 1

-- Define the derivative of f
def f' (a : ℝ) (x : ℝ) : ℝ := 2*x - a

theorem problem_solution :
  -- Part 1: Find the value of a
  (∃ a : ℝ, f a 0 = -5 ∧ a = -5) ∧
  -- Part 2: Find the equation of the tangent line
  (∃ x y : ℝ,
    -- Point M(x, y) is on the curve f
    y = f (-5) x ∧
    -- Tangent line at M is parallel to 3x + 2y + 2 = 0
    f' (-5) x = -3/2 ∧
    -- Equation of the tangent line
    (24 : ℝ) * x + 16 * y - 37 = 0) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l2895_289507


namespace NUMINAMATH_CALUDE_students_playing_both_sports_l2895_289565

/-- Given a class of students, calculate the number of students who play both football and long tennis. -/
theorem students_playing_both_sports 
  (total : ℕ) 
  (football : ℕ) 
  (tennis : ℕ) 
  (neither : ℕ) 
  (h1 : total = 35) 
  (h2 : football = 26) 
  (h3 : tennis = 20) 
  (h4 : neither = 6) : 
  football + tennis - (total - neither) = 17 := by
  sorry

end NUMINAMATH_CALUDE_students_playing_both_sports_l2895_289565


namespace NUMINAMATH_CALUDE_exchange_rate_problem_l2895_289595

theorem exchange_rate_problem (d : ℕ) : 
  (3 / 2 : ℚ) * d - 72 = d → d = 144 := by sorry

end NUMINAMATH_CALUDE_exchange_rate_problem_l2895_289595


namespace NUMINAMATH_CALUDE_decrypt_ciphertext_l2895_289502

-- Define the encryption function
def encrypt (x : ℕ) : ℕ := 2^x - 2

-- State the theorem
theorem decrypt_ciphertext (y : ℕ) : 
  y = 1022 → ∃ x : ℕ, encrypt x = y ∧ x = 10 := by
  sorry

end NUMINAMATH_CALUDE_decrypt_ciphertext_l2895_289502


namespace NUMINAMATH_CALUDE_positive_intervals_l2895_289579

def f (x : ℝ) := (x + 2) * (x - 2) * (x + 1)

theorem positive_intervals (x : ℝ) : 
  f x > 0 ↔ (x > -2 ∧ x < -1) ∨ x > 2 :=
sorry

end NUMINAMATH_CALUDE_positive_intervals_l2895_289579


namespace NUMINAMATH_CALUDE_smallest_max_sum_l2895_289509

theorem smallest_max_sum (p q r s t : ℕ+) 
  (sum_constraint : p + q + r + s + t = 2015) : 
  (∃ (N : ℕ), 
    N = max (p + q) (max (q + r) (max (r + s) (s + t))) ∧ 
    N = 1005 ∧
    ∀ (M : ℕ), (M = max (p + q) (max (q + r) (max (r + s) (s + t))) → M ≥ N)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_max_sum_l2895_289509


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2895_289544

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_sum : a 1 + 3 * a 8 + a 15 = 120) : 
  2 * a 9 - a 10 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2895_289544


namespace NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l2895_289550

theorem smallest_two_digit_multiple_of_17 : ∃ n : ℕ, 
  n = 17 ∧ 
  (∀ m : ℕ, m % 17 = 0 ∧ 10 ≤ m ∧ m < 100 → n ≤ m) := by
  sorry

end NUMINAMATH_CALUDE_smallest_two_digit_multiple_of_17_l2895_289550


namespace NUMINAMATH_CALUDE_number_equality_l2895_289555

theorem number_equality (x : ℝ) (h1 : x > 0) (h2 : (2/3) * x = (25/216) * (1/x)) : x = 144/25 := by
  sorry

end NUMINAMATH_CALUDE_number_equality_l2895_289555


namespace NUMINAMATH_CALUDE_tetrahedron_bisector_ratio_l2895_289523

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents the area of a triangle -/
def triangleArea (p1 p2 p3 : Point3D) : ℝ := sorry

/-- Represents a point on an edge of the tetrahedron -/
def intersectionPoint (t : Tetrahedron) : Point3D := sorry

/-- Theorem: In a tetrahedron ABCD, where the bisector plane of the dihedral angle around edge CD
    intersects AB at point E, the ratio of AE to BE is equal to the ratio of the areas of
    triangles ACD and BCD -/
theorem tetrahedron_bisector_ratio (t : Tetrahedron) :
  let E := intersectionPoint t
  let AE := Real.sqrt ((t.A.x - E.x)^2 + (t.A.y - E.y)^2 + (t.A.z - E.z)^2)
  let BE := Real.sqrt ((t.B.x - E.x)^2 + (t.B.y - E.y)^2 + (t.B.z - E.z)^2)
  let t_ACD := triangleArea t.A t.C t.D
  let t_BCD := triangleArea t.B t.C t.D
  AE / BE = t_ACD / t_BCD := by sorry

end NUMINAMATH_CALUDE_tetrahedron_bisector_ratio_l2895_289523


namespace NUMINAMATH_CALUDE_laundry_cleaning_rate_l2895_289522

/-- Given a total number of laundry pieces and available hours, 
    calculate the number of pieces to be cleaned per hour -/
def pieces_per_hour (total_pieces : ℕ) (available_hours : ℕ) : ℕ :=
  total_pieces / available_hours

/-- Theorem stating that cleaning 80 pieces of laundry in 4 hours 
    requires cleaning 20 pieces per hour -/
theorem laundry_cleaning_rate : pieces_per_hour 80 4 = 20 := by
  sorry

end NUMINAMATH_CALUDE_laundry_cleaning_rate_l2895_289522


namespace NUMINAMATH_CALUDE_max_value_inequality_l2895_289570

theorem max_value_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (M : ℝ), M = 1/2 ∧ 
  (∀ (N : ℝ), (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
    x^3 + y^3 + z^3 - 3*x*y*z ≥ N*(|x-y|^3 + |x-z|^3 + |z-y|^3)) → N ≤ M) ∧
  (a^3 + b^3 + c^3 - 3*a*b*c ≥ M*(|a-b|^3 + |a-c|^3 + |c-b|^3)) := by
sorry

end NUMINAMATH_CALUDE_max_value_inequality_l2895_289570


namespace NUMINAMATH_CALUDE_vector_norm_difference_l2895_289537

theorem vector_norm_difference (a b : ℝ × ℝ) :
  (‖a‖ = 2) → (‖b‖ = 1) → (‖a + b‖ = Real.sqrt 3) → ‖a - b‖ = Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_vector_norm_difference_l2895_289537


namespace NUMINAMATH_CALUDE_solve_pizza_problem_l2895_289513

def pizza_problem (total_slices : ℕ) (slices_left : ℕ) (slices_per_person : ℕ) : Prop :=
  let slices_eaten := total_slices - slices_left
  slices_eaten / slices_per_person = 6

theorem solve_pizza_problem :
  pizza_problem 16 4 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_pizza_problem_l2895_289513


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_l2895_289581

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 1 + a 2 + a 3 = 1 →
  a 2 + a 3 + a 4 = 2 →
  a 6 + a 7 + a 8 = 32 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_l2895_289581


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2895_289553

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (h_arithmetic : is_arithmetic_sequence a)
  (h_sum : a 4 + a 10 + a 16 = 30) :
  a 18 - 2 * a 14 = -10 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2895_289553


namespace NUMINAMATH_CALUDE_ratio_problem_l2895_289539

theorem ratio_problem (a b c d e : ℝ) 
  (h1 : a / b = 5)
  (h2 : b / c = 1 / 4)
  (h3 : c / d = 7)
  (h4 : d / e = 1 / 2)
  (h5 : a ≠ 0) (h6 : b ≠ 0) (h7 : c ≠ 0) (h8 : d ≠ 0) (h9 : e ≠ 0) : 
  e / a = 8 / 35 := by
  sorry

end NUMINAMATH_CALUDE_ratio_problem_l2895_289539


namespace NUMINAMATH_CALUDE_x_plus_y_equals_six_l2895_289560

theorem x_plus_y_equals_six (x y : ℝ) 
  (h1 : |x| - x + y = 42)
  (h2 : x + |y| + y = 24) :
  x + y = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_six_l2895_289560


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2895_289501

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (2 * Real.sqrt 2, 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l2895_289501


namespace NUMINAMATH_CALUDE_incorrect_denominator_clearing_l2895_289527

theorem incorrect_denominator_clearing (x : ℝ) : 
  ¬((-((3*x+1)/2) - ((2*x-5)/6) > 1) ↔ (3*(3*x+1)+(2*x-5) > -6)) := by
  sorry

end NUMINAMATH_CALUDE_incorrect_denominator_clearing_l2895_289527


namespace NUMINAMATH_CALUDE_perpendicular_lines_k_value_l2895_289505

theorem perpendicular_lines_k_value (k : ℝ) :
  (((k - 3) * 2 * (k - 3) + (5 - k) * (-2) = 0) →
  (k = 1 ∨ k = 4)) ∧
  ((k = 1 ∨ k = 4) →
  ((k - 3) * 2 * (k - 3) + (5 - k) * (-2) = 0)) := by sorry

end NUMINAMATH_CALUDE_perpendicular_lines_k_value_l2895_289505


namespace NUMINAMATH_CALUDE_exactly_one_true_l2895_289585

-- Define what it means for three numbers to be in geometric progression
def in_geometric_progression (a b c : ℝ) : Prop :=
  ∃ r : ℝ, b = a * r ∧ c = b * r

-- Define the original proposition
def original_proposition : Prop :=
  ∀ a b c : ℝ, in_geometric_progression a b c → b^2 = a * c

-- Define the converse
def converse : Prop :=
  ∀ a b c : ℝ, b^2 = a * c → in_geometric_progression a b c

-- Define the inverse
def inverse : Prop :=
  ∀ a b c : ℝ, ¬(in_geometric_progression a b c) → b^2 ≠ a * c

-- Define the contrapositive
def contrapositive : Prop :=
  ∀ a b c : ℝ, b^2 ≠ a * c → ¬(in_geometric_progression a b c)

-- Theorem to prove
theorem exactly_one_true :
  (original_proposition ∧
   (converse ∨ inverse ∨ contrapositive) ∧
   ¬(converse ∧ inverse) ∧
   ¬(converse ∧ contrapositive) ∧
   ¬(inverse ∧ contrapositive)) :=
sorry

end NUMINAMATH_CALUDE_exactly_one_true_l2895_289585


namespace NUMINAMATH_CALUDE_science_book_page_count_l2895_289584

def history_book_pages : ℕ := 300

def novel_pages (history : ℕ) : ℕ := history / 2

def science_book_pages (novel : ℕ) : ℕ := 4 * novel

theorem science_book_page_count : 
  science_book_pages (novel_pages history_book_pages) = 600 := by
  sorry

end NUMINAMATH_CALUDE_science_book_page_count_l2895_289584


namespace NUMINAMATH_CALUDE_max_blocks_fit_l2895_289583

/-- The dimensions of the larger box -/
def box_dimensions : Fin 3 → ℕ
| 0 => 3  -- length
| 1 => 2  -- width
| 2 => 3  -- height
| _ => 0

/-- The dimensions of the smaller block -/
def block_dimensions : Fin 3 → ℕ
| 0 => 2  -- length
| 1 => 2  -- width
| 2 => 1  -- height
| _ => 0

/-- Calculate the volume of a rectangular object given its dimensions -/
def volume (dimensions : Fin 3 → ℕ) : ℕ :=
  dimensions 0 * dimensions 1 * dimensions 2

/-- The maximum number of blocks that can fit in the box -/
def max_blocks : ℕ := 4

/-- Theorem stating that the maximum number of blocks that can fit in the box is 4 -/
theorem max_blocks_fit :
  (volume box_dimensions ≥ max_blocks * volume block_dimensions) ∧
  (∀ n : ℕ, n > max_blocks → volume box_dimensions < n * volume block_dimensions) :=
sorry

end NUMINAMATH_CALUDE_max_blocks_fit_l2895_289583


namespace NUMINAMATH_CALUDE_bobby_paycheck_l2895_289564

/-- Calculates the final paycheck amount given the salary and deductions --/
def final_paycheck_amount (salary : ℝ) (federal_tax_rate : ℝ) (state_tax_rate : ℝ) 
  (health_insurance : ℝ) (life_insurance : ℝ) (parking_fee : ℝ) : ℝ :=
  salary - (salary * federal_tax_rate + salary * state_tax_rate + 
    health_insurance + life_insurance + parking_fee)

/-- Theorem stating that Bobby's final paycheck amount is $184 --/
theorem bobby_paycheck :
  final_paycheck_amount 450 (1/3) 0.08 50 20 10 = 184 := by
  sorry

end NUMINAMATH_CALUDE_bobby_paycheck_l2895_289564


namespace NUMINAMATH_CALUDE_roots_in_interval_l2895_289574

theorem roots_in_interval (m : ℝ) : 
  (∀ x : ℝ, 4 * x^2 - (3 * m + 1) * x - m - 2 = 0 → -1 < x ∧ x < 2) ↔ 
  -3/2 < m ∧ m < 12/7 :=
by sorry

end NUMINAMATH_CALUDE_roots_in_interval_l2895_289574


namespace NUMINAMATH_CALUDE_two_numbers_with_product_and_gcd_l2895_289572

theorem two_numbers_with_product_and_gcd 
  (a b : ℕ) 
  (h_product : a * b = 8214)
  (h_gcd : Nat.gcd a b = 37) :
  (a = 74 ∧ b = 111) ∨ (a = 111 ∧ b = 74) :=
sorry

end NUMINAMATH_CALUDE_two_numbers_with_product_and_gcd_l2895_289572


namespace NUMINAMATH_CALUDE_ln_one_eq_zero_l2895_289562

theorem ln_one_eq_zero : Real.log 1 = 0 := by sorry

end NUMINAMATH_CALUDE_ln_one_eq_zero_l2895_289562


namespace NUMINAMATH_CALUDE_taco_truck_lunch_rush_earnings_l2895_289557

/-- Calculates the total earnings of a taco truck during lunch rush -/
def taco_truck_earnings (soft_taco_price : ℕ) (hard_taco_price : ℕ) 
  (family_hard_tacos : ℕ) (family_soft_tacos : ℕ) 
  (other_customers : ℕ) (tacos_per_customer : ℕ) : ℕ :=
  (family_hard_tacos * hard_taco_price + family_soft_tacos * soft_taco_price) + 
  (other_customers * tacos_per_customer * soft_taco_price)

/-- The taco truck's earnings during lunch rush is $66 -/
theorem taco_truck_lunch_rush_earnings : 
  taco_truck_earnings 2 5 4 3 10 2 = 66 := by
  sorry

end NUMINAMATH_CALUDE_taco_truck_lunch_rush_earnings_l2895_289557


namespace NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2895_289503

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem ninth_term_of_arithmetic_sequence 
  (a : ℕ → ℚ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_third : a 3 = 5/11) 
  (h_fifteenth : a 15 = 7/8) : 
  a 9 = 117/176 := by
sorry

end NUMINAMATH_CALUDE_ninth_term_of_arithmetic_sequence_l2895_289503


namespace NUMINAMATH_CALUDE_unique_m_value_l2895_289520

theorem unique_m_value : ∃! m : ℝ, ∀ y : ℝ, 
  (y - 2 = 1) → (m * y - 2 = 4) := by
  sorry

end NUMINAMATH_CALUDE_unique_m_value_l2895_289520


namespace NUMINAMATH_CALUDE_birdhouse_flew_1200_feet_l2895_289580

/-- The distance the car was transported, in feet -/
def car_distance : ℕ := 200

/-- The distance the lawn chair was blown, in feet -/
def lawn_chair_distance : ℕ := 2 * car_distance

/-- The distance the birdhouse flew, in feet -/
def birdhouse_distance : ℕ := 3 * lawn_chair_distance

/-- Theorem stating that the birdhouse flew 1200 feet -/
theorem birdhouse_flew_1200_feet : birdhouse_distance = 1200 := by
  sorry

end NUMINAMATH_CALUDE_birdhouse_flew_1200_feet_l2895_289580


namespace NUMINAMATH_CALUDE_no_prime_sum_53_l2895_289516

theorem no_prime_sum_53 : ¬ ∃ (p q : ℕ), Prime p ∧ Prime q ∧ p + q = 53 := by sorry

end NUMINAMATH_CALUDE_no_prime_sum_53_l2895_289516


namespace NUMINAMATH_CALUDE_no_finite_k_with_zero_difference_l2895_289588

def u (n : ℕ) : ℕ := n^4 + n^2

def Δ : (ℕ → ℕ) → (ℕ → ℕ)
  | f => fun n => f (n + 1) - f n

def iteratedΔ : ℕ → (ℕ → ℕ) → (ℕ → ℕ)
  | 0 => id
  | k + 1 => Δ ∘ iteratedΔ k

theorem no_finite_k_with_zero_difference :
  ∀ k : ℕ, ∃ n : ℕ, (iteratedΔ k u) n ≠ 0 := by sorry

end NUMINAMATH_CALUDE_no_finite_k_with_zero_difference_l2895_289588


namespace NUMINAMATH_CALUDE_circle_radius_from_area_l2895_289531

theorem circle_radius_from_area (A : Real) (r : Real) :
  A = Real.pi * r^2 → A = 64 * Real.pi → r = 8 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_from_area_l2895_289531


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2895_289543

/-- A rectangular solid with prime edge lengths and volume 1001 has surface area 622 -/
theorem rectangular_solid_surface_area :
  ∀ (a b c : ℕ),
  Prime a → Prime b → Prime c →
  a * b * c = 1001 →
  2 * (a * b + b * c + c * a) = 622 := by
sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l2895_289543


namespace NUMINAMATH_CALUDE_sqrt_inequality_l2895_289599

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt (a - 2) - Real.sqrt (a - 3) > Real.sqrt a - Real.sqrt (a - 1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l2895_289599


namespace NUMINAMATH_CALUDE_repair_cost_calculation_l2895_289592

/-- Calculates the repair cost of a machine given its purchase price, transportation charges, profit percentage, and selling price. -/
theorem repair_cost_calculation (purchase_price : ℕ) (transportation_charges : ℕ) (profit_percentage : ℕ) (selling_price : ℕ) : 
  purchase_price = 11000 →
  transportation_charges = 1000 →
  profit_percentage = 50 →
  selling_price = 25500 →
  ∃ (repair_cost : ℕ), 
    repair_cost = 5000 ∧
    selling_price = (purchase_price + repair_cost + transportation_charges) * (100 + profit_percentage) / 100 :=
by sorry

end NUMINAMATH_CALUDE_repair_cost_calculation_l2895_289592


namespace NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l2895_289594

-- Problem 1
theorem ellipse_equation (x y : ℝ) :
  let equation := x^2 / 13 + y^2 / (13/9) = 1
  let center_at_origin := ∀ (t : ℝ), t^2 / 13 + 0^2 / (13/9) ≠ 1 ∧ 0^2 / 13 + t^2 / (13/9) ≠ 1
  let foci_on_x_axis := ∃ (c : ℝ), c^2 = 13 - 13/9 ∧ (c^2 / 13 + 0^2 / (13/9) = 1 ∨ (-c)^2 / 13 + 0^2 / (13/9) = 1)
  let major_axis_triple_minor := 13 = 3 * (13/9)
  let passes_through_p := 3^2 / 13 + 2^2 / (13/9) = 1
  center_at_origin ∧ foci_on_x_axis ∧ major_axis_triple_minor ∧ passes_through_p → equation :=
by sorry

-- Problem 2
theorem hyperbola_equation (x y : ℝ) :
  let equation := x^2 / 10 - y^2 / 6 = 1
  let common_asymptote := ∃ (k : ℝ), k^2 = 10/6 ∧ k^2 = 5/3
  let focal_length_8 := ∃ (c : ℝ), c^2 = 10 + 6 ∧ 2*c = 8
  common_asymptote ∧ focal_length_8 → equation :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_hyperbola_equation_l2895_289594


namespace NUMINAMATH_CALUDE_max_valid_list_length_l2895_289528

/-- A prime power is an integer of the form p^n where p is prime and n is a non-negative integer -/
def IsPrimePower (n : ℕ) : Prop := ∃ (p k : ℕ), Prime p ∧ n = p ^ k

/-- The type of lists of natural numbers where the sum of any two distinct elements is a prime power -/
def ValidList : Type := { l : List ℕ // ∀ (x y : ℕ), x ∈ l → y ∈ l → x ≠ y → IsPrimePower (x + y) }

/-- The theorem stating that the maximum length of a valid list is 4 -/
theorem max_valid_list_length :
  (∃ (l : ValidList), l.val.length = 4) ∧
  (∀ (l : ValidList), l.val.length ≤ 4) := by sorry

end NUMINAMATH_CALUDE_max_valid_list_length_l2895_289528


namespace NUMINAMATH_CALUDE_triangle_properties_l2895_289525

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) :
  (0 < a) → (0 < b) → (0 < c) →
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  (A + B + C = π) →
  (c * Real.sin B = Real.sqrt 3 * Real.cos C) →
  (a + b = 6) →
  (C = π / 3 ∧ a + b + c ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2895_289525


namespace NUMINAMATH_CALUDE_three_tangent_lines_imply_a_8_symmetry_of_circle_C_l2895_289540

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4

-- Define the line l
def line_l (m x y : ℝ) : Prop := m*x + x + 2*y - 1 + m = 0

-- Define the curve (another circle)
def curve (a x y : ℝ) : Prop := x^2 + y^2 - 2*x + 8*y + a = 0

-- Theorem 1: Three common tangent lines imply a = 8
theorem three_tangent_lines_imply_a_8 :
  (∃ (a : ℝ), (∀ x y : ℝ, curve a x y) ∧ 
  (∃! (l₁ l₂ l₃ : ℝ → ℝ → Prop), 
    (∀ x y : ℝ, (l₁ x y ∨ l₂ x y ∨ l₃ x y) → (curve a x y ∨ circle_C x y)) ∧
    (∀ x y : ℝ, (l₁ x y ∨ l₂ x y ∨ l₃ x y) → 
      (∃ ε > 0, ∀ x' y' : ℝ, ((x' - x)^2 + (y' - y)^2 < ε^2) → 
        ¬(curve a x' y' ∧ circle_C x' y'))))) →
  a = 8 :=
sorry

-- Theorem 2: Symmetry of circle C with respect to line l when m = 1
theorem symmetry_of_circle_C :
  ∀ x y : ℝ, line_l 1 x y → 
  (∃ x' y' : ℝ, circle_C x' y' ∧ 
    ((x + x')/2 = x ∧ (y + y')/2 = y) ∧ 
    (x^2 + (y-2)^2 = 4)) :=
sorry

end NUMINAMATH_CALUDE_three_tangent_lines_imply_a_8_symmetry_of_circle_C_l2895_289540


namespace NUMINAMATH_CALUDE_sandra_savings_proof_l2895_289596

-- Define the given conditions
def mother_contribution : ℝ := 4
def father_contribution : ℝ := 2 * mother_contribution
def candy_cost : ℝ := 0.5
def jelly_bean_cost : ℝ := 0.2
def candy_quantity : ℕ := 14
def jelly_bean_quantity : ℕ := 20
def money_left : ℝ := 11

-- Define Sandra's initial savings
def sandra_initial_savings : ℝ := 10

-- Theorem to prove
theorem sandra_savings_proof :
  sandra_initial_savings = 
    (candy_cost * candy_quantity + jelly_bean_cost * jelly_bean_quantity + money_left) - 
    (mother_contribution + father_contribution) := by
  sorry


end NUMINAMATH_CALUDE_sandra_savings_proof_l2895_289596


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2895_289589

/-- Given two arithmetic sequences {a_n} and {b_n} with sums A_n and B_n,
    if A_n / B_n = (7n + 45) / (n + 3) for all n, then a_5 / b_5 = 9 -/
theorem arithmetic_sequence_ratio (a b : ℕ → ℚ) (A B : ℕ → ℚ) :
  (∀ n, A n = (n / 2) * (a 1 + a n)) →
  (∀ n, B n = (n / 2) * (b 1 + b n)) →
  (∀ n, A n / B n = (7 * n + 45) / (n + 3)) →
  a 5 / b 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l2895_289589


namespace NUMINAMATH_CALUDE_average_age_of_joans_kittens_l2895_289548

/-- Represents the number of days in each month (simplified to 30 for all months) -/
def daysInMonth : ℕ := 30

/-- Calculates the age of kittens in days given their birth month -/
def kittenAge (birthMonth : ℕ) : ℕ :=
  (4 - birthMonth) * daysInMonth + 15

/-- Represents Joan's original number of kittens -/
def joansOriginalKittens : ℕ := 8

/-- Represents the number of kittens Joan gave away -/
def joansGivenAwayKittens : ℕ := 2

/-- Represents the number of neighbor's kittens Joan adopted -/
def adoptedNeighborKittens : ℕ := 3

/-- Represents the number of friend's kittens Joan adopted -/
def adoptedFriendKittens : ℕ := 1

/-- Calculates the total number of kittens Joan has after all transactions -/
def totalJoansKittens : ℕ :=
  joansOriginalKittens - joansGivenAwayKittens + adoptedNeighborKittens + adoptedFriendKittens

/-- Theorem stating that the average age of Joan's kittens on April 15th is 90 days -/
theorem average_age_of_joans_kittens :
  (joansOriginalKittens - joansGivenAwayKittens) * kittenAge 1 +
  adoptedNeighborKittens * kittenAge 2 +
  adoptedFriendKittens * kittenAge 3 =
  90 * totalJoansKittens := by sorry

end NUMINAMATH_CALUDE_average_age_of_joans_kittens_l2895_289548


namespace NUMINAMATH_CALUDE_parallel_plane_intersection_theorem_l2895_289577

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel_planes : Plane → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel_lines : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Theorem statement
theorem parallel_plane_intersection_theorem
  (α β γ : Plane) (a b : Line)
  (h1 : parallel_planes α β)
  (h2 : intersect α γ = a)
  (h3 : intersect β γ = b) :
  parallel_lines a b :=
sorry

end NUMINAMATH_CALUDE_parallel_plane_intersection_theorem_l2895_289577


namespace NUMINAMATH_CALUDE_book_pyramid_theorem_l2895_289508

/-- Represents a book pyramid with a given number of levels -/
structure BookPyramid where
  levels : ℕ
  top_level_books : ℕ
  ratio : ℚ
  total_books : ℕ

/-- Calculates the total number of books in the pyramid -/
def calculate_total (p : BookPyramid) : ℚ :=
  p.top_level_books * (1 - p.ratio ^ p.levels) / (1 - p.ratio)

/-- Theorem stating the properties of the specific book pyramid -/
theorem book_pyramid_theorem (p : BookPyramid) 
  (h1 : p.levels = 4)
  (h2 : p.ratio = 4/5)
  (h3 : p.total_books = 369) :
  p.top_level_books = 64 := by
  sorry


end NUMINAMATH_CALUDE_book_pyramid_theorem_l2895_289508


namespace NUMINAMATH_CALUDE_hotel_meal_spending_l2895_289541

theorem hotel_meal_spending (total_persons : ℕ) (regular_spenders : ℕ) (regular_amount : ℕ) 
  (extra_amount : ℕ) (total_spent : ℕ) :
  total_persons = 9 →
  regular_spenders = 8 →
  regular_amount = 12 →
  extra_amount = 8 →
  total_spent = 117 →
  ∃ x : ℕ, (regular_spenders * regular_amount) + (x + extra_amount) = total_spent ∧ x = 13 :=
by sorry

end NUMINAMATH_CALUDE_hotel_meal_spending_l2895_289541


namespace NUMINAMATH_CALUDE_ellipse_line_intersection_l2895_289514

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus_F : ℝ × ℝ := (1, 0)

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := y = k*(x - 1)

-- Define the perpendicularity condition
def perpendicular (x₁ y₁ x₂ y₂ : ℝ) : Prop := x₁*x₂ + y₁*y₂ = 0

theorem ellipse_line_intersection :
  ∃ k : ℝ, k = 2 ∨ k = -2 ∧
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    ellipse_C x₁ y₁ ∧
    ellipse_C x₂ y₂ ∧
    line_l k x₁ y₁ ∧
    line_l k x₂ y₂ ∧
    perpendicular x₁ y₁ x₂ y₂ :=
sorry

end NUMINAMATH_CALUDE_ellipse_line_intersection_l2895_289514


namespace NUMINAMATH_CALUDE_min_value_expression_l2895_289569

theorem min_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hsum : x + y = 1/2) (horder : x ≤ y ∧ y ≤ z) :
  (x + z) / (x * y * z) ≥ 48 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2895_289569


namespace NUMINAMATH_CALUDE_binomial_probability_two_l2895_289582

/-- A random variable following a binomial distribution with parameters n and p -/
structure BinomialDistribution (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ  -- The random variable

/-- The probability mass function for a binomial distribution -/
def binomialProbability (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

/-- The theorem stating that P(X=2) = 80/243 for X ~ B(6, 1/3) -/
theorem binomial_probability_two (X : BinomialDistribution 6 (1/3)) :
  binomialProbability 6 (1/3) 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_two_l2895_289582


namespace NUMINAMATH_CALUDE_greatest_b_value_l2895_289533

theorem greatest_b_value (b : ℝ) : 
  (∀ x : ℝ, -x^2 + 7*x - 10 ≥ 0 → x ≤ 5) ∧ 
  (-5^2 + 7*5 - 10 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_greatest_b_value_l2895_289533


namespace NUMINAMATH_CALUDE_percentage_increase_decrease_l2895_289534

theorem percentage_increase_decrease (α β p q : ℝ) 
  (h_pos_α : α > 0) (h_pos_β : β > 0) (h_pos_p : p > 0) (h_pos_q : q > 0) (h_q_lt_50 : q < 50) :
  (α * β * (1 + p / 100) * (1 - q / 100) > α * β) ↔ (p > 100 * q / (100 - q)) :=
by sorry

end NUMINAMATH_CALUDE_percentage_increase_decrease_l2895_289534


namespace NUMINAMATH_CALUDE_workshop_pairing_probability_l2895_289538

theorem workshop_pairing_probability (n : ℕ) (h : n = 24) :
  let total_participants := n
  let pairing_probability := (1 : ℚ) / (n - 1 : ℚ)
  pairing_probability = (1 : ℚ) / (23 : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_workshop_pairing_probability_l2895_289538


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l2895_289571

theorem geometric_sequence_property (a : ℕ → ℝ) (h_positive : ∀ n, a n > 0) 
  (h_geometric : ∃ r : ℝ, ∀ n, a (n + 1) = r * a n) 
  (h_arithmetic : (a 1 + 2 * a 2) / 2 = a 3 / 2) :
  (a 8 + a 9) / (a 6 + a 7) = 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l2895_289571


namespace NUMINAMATH_CALUDE_rotation_equivalence_l2895_289573

theorem rotation_equivalence (x : ℝ) : 
  (420 % 360 : ℝ) = (360 - x) % 360 → x < 360 → x = 300 := by
  sorry

end NUMINAMATH_CALUDE_rotation_equivalence_l2895_289573


namespace NUMINAMATH_CALUDE_sector_area_l2895_289587

/-- Given a sector with central angle 2 radians and arc length 2, its area is 1. -/
theorem sector_area (θ : Real) (l : Real) (r : Real) : 
  θ = 2 → l = 2 → l = r * θ → (1/2) * r * θ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2895_289587


namespace NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l2895_289558

/-- Represents a chessboard configuration with rooks -/
structure ChessboardWithRooks where
  size : Nat
  num_rooks : Nat
  rook_positions : List (Nat × Nat)
  different_squares : rook_positions.length = num_rooks ∧ 
                      rook_positions.Nodup

/-- Counts the number of pairs of rooks that can attack each other -/
def count_attacking_pairs (board : ChessboardWithRooks) : Nat :=
  sorry

/-- Theorem stating the minimum number of attacking pairs for a specific configuration -/
theorem min_attacking_pairs_8x8_16rooks :
  ∀ (board : ChessboardWithRooks),
    board.size = 8 ∧ 
    board.num_rooks = 16 →
    count_attacking_pairs board ≥ 16 :=
  sorry

end NUMINAMATH_CALUDE_min_attacking_pairs_8x8_16rooks_l2895_289558


namespace NUMINAMATH_CALUDE_root_implies_b_value_l2895_289590

theorem root_implies_b_value (a b : ℚ) :
  (2 + Real.sqrt 5 : ℝ) ^ 3 + a * (2 + Real.sqrt 5 : ℝ) ^ 2 + b * (2 + Real.sqrt 5 : ℝ) - 20 = 0 →
  b = -24 := by
  sorry

end NUMINAMATH_CALUDE_root_implies_b_value_l2895_289590


namespace NUMINAMATH_CALUDE_train_speed_calculation_train_speed_proof_l2895_289510

/-- The speed of two trains crossing each other -/
theorem train_speed_calculation (train_length : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := 2 * train_length
  let relative_speed := total_distance / crossing_time
  let train_speed := relative_speed / 2
  let km_per_hour := train_speed * 3.6
  km_per_hour

/-- Proof that the speed of each train is approximately 12.01 km/hr -/
theorem train_speed_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |train_speed_calculation 120 36 - 12.01| < ε :=
sorry

end NUMINAMATH_CALUDE_train_speed_calculation_train_speed_proof_l2895_289510


namespace NUMINAMATH_CALUDE_tangent_circle_intersection_distance_l2895_289504

-- Define the types for points and circles
variable (Point Circle : Type)

-- Define the intersection of two circles
variable (intersect : Circle → Circle → Point)

-- Define the tangent line at a point on a circle
variable (tangent_at : Point → Circle → Point → Prop)

-- Define a circle passing through three points
variable (circle_through : Point → Point → Point → Circle)

-- Define the distance between two points
variable (distance : Point → Point → ℝ)

-- State the theorem
theorem tangent_circle_intersection_distance
  (C₁ C₂ C₃ : Circle) (S A B P Q : Point) :
  intersect C₁ C₂ = S →
  tangent_at S C₁ A →
  tangent_at S C₂ B →
  C₃ = circle_through A B S →
  tangent_at S C₃ P →
  tangent_at S C₃ Q →
  A ≠ S →
  B ≠ S →
  P ≠ S →
  Q ≠ S →
  distance P S = distance Q S :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_intersection_distance_l2895_289504


namespace NUMINAMATH_CALUDE_unique_solution_l2895_289593

/-- The system of equations has a unique solution at (-2, -4) -/
theorem unique_solution : ∃! (x y : ℝ), 
  (x + 3*y + 14 ≤ 0) ∧ 
  (x^4 + 2*x^2*y^2 + y^4 + 64 - 20*x^2 - 20*y^2 = 8*x*y) ∧
  (x = -2) ∧ (y = -4) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l2895_289593


namespace NUMINAMATH_CALUDE_semicircle_perimeter_approx_l2895_289586

/-- The perimeter of a semicircle with radius 7 is approximately 35.99 -/
theorem semicircle_perimeter_approx :
  let r : ℝ := 7
  let perimeter : ℝ := 2 * r + π * r
  ∃ ε > 0, abs (perimeter - 35.99) < ε :=
by sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_approx_l2895_289586


namespace NUMINAMATH_CALUDE_fraction_evaluation_l2895_289563

theorem fraction_evaluation (x : ℝ) (h : x = 8) :
  (x^10 - 32*x^5 + 1024) / (x^5 - 32) = 32768 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l2895_289563


namespace NUMINAMATH_CALUDE_monomials_not_like_terms_l2895_289576

/-- Definition of a monomial -/
structure Monomial (α : Type*) [CommRing α] :=
  (coeff : α)
  (vars : List (Nat × Nat))  -- List of (variable index, exponent) pairs

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def areLikeTerms {α : Type*} [CommRing α] (m1 m2 : Monomial α) : Prop :=
  m1.vars = m2.vars

/-- Representation of the monomial -12a^2b -/
def m1 : Monomial ℚ :=
  ⟨-12, [(1, 2), (2, 1)]⟩  -- Assuming variable indices: 1 for a, 2 for b

/-- Representation of the monomial 2ab^2/3 -/
def m2 : Monomial ℚ :=
  ⟨2/3, [(1, 1), (2, 2)]⟩

theorem monomials_not_like_terms : ¬(areLikeTerms m1 m2) := by
  sorry


end NUMINAMATH_CALUDE_monomials_not_like_terms_l2895_289576


namespace NUMINAMATH_CALUDE_percentage_of_non_roses_l2895_289542

theorem percentage_of_non_roses (roses tulips daisies : ℕ) : 
  roses = 25 → tulips = 40 → daisies = 35 → 
  (tulips + daisies : ℚ) / (roses + tulips + daisies) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_non_roses_l2895_289542


namespace NUMINAMATH_CALUDE_geometryville_schools_l2895_289524

theorem geometryville_schools (n : ℕ) : 
  n > 0 → 
  let total_students := 4 * n
  let andreas_rank := (12 * n + 1) / 4
  andreas_rank > total_students / 2 →
  andreas_rank ≤ 3 * total_students / 4 →
  (∃ (teammate_rank : ℕ), 
    teammate_rank ≤ total_students / 2 ∧ 
    teammate_rank < andreas_rank) →
  (∃ (bottom_teammates : Fin 2 → ℕ), 
    ∀ i, bottom_teammates i > total_students / 2 ∧ 
         bottom_teammates i < andreas_rank) →
  n = 3 := by
sorry

end NUMINAMATH_CALUDE_geometryville_schools_l2895_289524


namespace NUMINAMATH_CALUDE_natural_number_pairs_l2895_289500

theorem natural_number_pairs (x y a n m : ℕ) :
  x + y = a^n ∧ x^2 + y^2 = a^m →
  ∃ k : ℕ, x = 2^k ∧ y = 2^k :=
by sorry

end NUMINAMATH_CALUDE_natural_number_pairs_l2895_289500


namespace NUMINAMATH_CALUDE_extended_pattern_ratio_l2895_289545

/-- Represents a square pattern of tiles -/
structure TilePattern :=
  (black : ℕ)
  (white : ℕ)

/-- Represents the extended pattern with a black border -/
def extendPattern (p : TilePattern) : TilePattern :=
  let side := Nat.sqrt (p.black + p.white)
  let newBlack := p.black + 4 * side + 4
  { black := newBlack, white := p.white }

/-- The theorem to be proved -/
theorem extended_pattern_ratio (p : TilePattern) :
  p.black = 13 ∧ p.white = 23 →
  let ep := extendPattern p
  (ep.black : ℚ) / ep.white = 41 / 23 := by
  sorry


end NUMINAMATH_CALUDE_extended_pattern_ratio_l2895_289545


namespace NUMINAMATH_CALUDE_craig_remaining_apples_l2895_289575

/-- Theorem: Craig's remaining apples after sharing -/
theorem craig_remaining_apples (initial_apples shared_apples : ℕ) 
  (h1 : initial_apples = 20)
  (h2 : shared_apples = 7) :
  initial_apples - shared_apples = 13 := by
  sorry

end NUMINAMATH_CALUDE_craig_remaining_apples_l2895_289575


namespace NUMINAMATH_CALUDE_set_union_condition_l2895_289529

-- Define the sets A and B
def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, x^2}

-- Define the theorem
theorem set_union_condition (x : ℝ) : 
  (A x ∪ B x = A x) ↔ (x = 2 ∨ x = -2 ∨ x = 0) :=
sorry

end NUMINAMATH_CALUDE_set_union_condition_l2895_289529


namespace NUMINAMATH_CALUDE_complex_product_real_l2895_289552

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 + a * Complex.I
  let z₂ : ℂ := a - 3 * Complex.I
  (z₁ * z₂).im = 0 ↔ a = 3 ∨ a = -3 := by
sorry

end NUMINAMATH_CALUDE_complex_product_real_l2895_289552


namespace NUMINAMATH_CALUDE_weight_ratio_l2895_289512

theorem weight_ratio (sam_weight tyler_weight peter_weight : ℝ) : 
  tyler_weight = sam_weight + 25 →
  sam_weight = 105 →
  peter_weight = 65 →
  peter_weight / tyler_weight = 0.5 := by
sorry

end NUMINAMATH_CALUDE_weight_ratio_l2895_289512


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2895_289521

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 864 → s^3 = 1728 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l2895_289521


namespace NUMINAMATH_CALUDE_matrix_equals_five_l2895_289518

-- Define the matrix
def matrix (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ := !![3*x, 2; 2*x, 4*x]

-- Define the determinant of a 2x2 matrix
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

-- Theorem statement
theorem matrix_equals_five (x : ℝ) : 
  det2x2 (matrix x 0 0) (matrix x 0 1) (matrix x 1 0) (matrix x 1 1) = 5 ↔ 
  x = 5/6 ∨ x = -1/2 := by
sorry

end NUMINAMATH_CALUDE_matrix_equals_five_l2895_289518


namespace NUMINAMATH_CALUDE_no_equal_digit_sum_decomposition_l2895_289517

def digit_sum (n : ℕ) : ℕ := sorry

theorem no_equal_digit_sum_decomposition :
  ¬ ∃ (B C : ℕ), B + C = 999999999 ∧ digit_sum B = digit_sum C := by sorry

end NUMINAMATH_CALUDE_no_equal_digit_sum_decomposition_l2895_289517


namespace NUMINAMATH_CALUDE_distinct_arithmetic_sequences_l2895_289526

/-- The largest prime power factor of a positive integer -/
def largest_prime_power_factor (n : ℕ+) : ℕ+ := sorry

/-- Check if two positive integers have the same largest prime power factor -/
def same_largest_prime_power_factor (m n : ℕ+) : Prop := 
  largest_prime_power_factor m = largest_prime_power_factor n

theorem distinct_arithmetic_sequences 
  (n : Fin 10000 → ℕ+) 
  (h_distinct : ∀ i j, i ≠ j → n i ≠ n j) 
  (h_same_factor : ∀ i j, same_largest_prime_power_factor (n i) (n j)) :
  ∃ a : Fin 10000 → ℤ, ∀ i j k l, i ≠ j → a i + k * (n i : ℤ) ≠ a j + l * (n j : ℤ) := by
    sorry

end NUMINAMATH_CALUDE_distinct_arithmetic_sequences_l2895_289526


namespace NUMINAMATH_CALUDE_simplify_expressions_l2895_289530

theorem simplify_expressions :
  (∀ x y : ℝ, x^2 - 5*y - 4*x^2 + y - 1 = -3*x^2 - 4*y - 1) ∧
  (∀ a b : ℝ, 7*a + 3*(a - 3*b) - 2*(b - 3*a) = 16*a - 11*b) :=
by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l2895_289530


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2895_289566

theorem closest_integer_to_cube_root_150 :
  ∀ n : ℤ, |n^3 - 150| ≥ |5^3 - 150| := by sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_150_l2895_289566


namespace NUMINAMATH_CALUDE_number_difference_l2895_289515

theorem number_difference (L S : ℕ) : L = 1495 → L = 5 * S + 4 → L - S = 1197 := by
  sorry

end NUMINAMATH_CALUDE_number_difference_l2895_289515


namespace NUMINAMATH_CALUDE_max_intersections_quad_pent_l2895_289591

/-- A polygon in a plane -/
structure Polygon :=
  (sides : ℕ)

/-- Represents the configuration of a quadrilateral and a pentagon on a plane -/
structure Configuration :=
  (quad : Polygon)
  (pent : Polygon)
  (no_vertex_on_side : Bool)

/-- Calculates the maximum number of intersection points between two polygons -/
def max_intersections (p1 p2 : Polygon) : ℕ :=
  p1.sides * p2.sides

/-- The main theorem stating the maximum number of intersections -/
theorem max_intersections_quad_pent (config : Configuration) 
  (h1 : config.quad.sides = 4)
  (h2 : config.pent.sides = 5)
  (h3 : config.no_vertex_on_side = true) :
  max_intersections config.quad config.pent = 20 := by
  sorry

#check max_intersections_quad_pent

end NUMINAMATH_CALUDE_max_intersections_quad_pent_l2895_289591


namespace NUMINAMATH_CALUDE_exists_nth_root_product_in_disc_l2895_289578

/-- A closed disc in the complex plane -/
structure ClosedDisc where
  center : ℂ
  radius : ℝ
  radius_nonneg : 0 ≤ radius

/-- A point is in a closed disc if its distance from the center is at most the radius -/
def in_closed_disc (z : ℂ) (D : ClosedDisc) : Prop :=
  Complex.abs (z - D.center) ≤ D.radius

/-- The main theorem -/
theorem exists_nth_root_product_in_disc (D : ClosedDisc) (n : ℕ) (h_n : 0 < n) 
    (z_list : List ℂ) (h_z_list : ∀ z ∈ z_list, in_closed_disc z D) :
    ∃ z : ℂ, in_closed_disc z D ∧ z^n = z_list.prod := by
  sorry

end NUMINAMATH_CALUDE_exists_nth_root_product_in_disc_l2895_289578


namespace NUMINAMATH_CALUDE_hair_cut_total_l2895_289532

theorem hair_cut_total : 
  let monday : ℚ := 38 / 100
  let tuesday : ℚ := 1 / 2
  let wednesday : ℚ := 1 / 4
  let thursday : ℚ := 87 / 100
  monday + tuesday + wednesday + thursday = 2 := by sorry

end NUMINAMATH_CALUDE_hair_cut_total_l2895_289532


namespace NUMINAMATH_CALUDE_no_positive_integer_solution_l2895_289519

theorem no_positive_integer_solution (f : ℕ+ → ℕ+) (a b : ℕ+) : 
  (∀ x, f x = x^2 + x) → 4 * (f a) ≠ f b := by
  sorry

end NUMINAMATH_CALUDE_no_positive_integer_solution_l2895_289519
