import Mathlib

namespace NUMINAMATH_CALUDE_tangential_quadrilateral_l1723_172345

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A circle in the plane -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- A quadrilateral in the plane -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Predicate to check if a quadrilateral is cyclic -/
def is_cyclic (q : Quadrilateral) : Prop := sorry

/-- Predicate to check if a quadrilateral is tangential -/
def is_tangential (q : Quadrilateral) : Prop := sorry

/-- Get the incenter of a triangle -/
def incenter (A B C : Point) : Point := sorry

/-- Check if four points are concyclic -/
def are_concyclic (A B C D : Point) : Prop := sorry

/-- Main theorem -/
theorem tangential_quadrilateral 
  (q : Quadrilateral) 
  (h1 : is_cyclic q) 
  (h2 : let I := incenter q.A q.B q.C
        let J := incenter q.A q.D q.C
        are_concyclic q.B I J q.D) : 
  is_tangential q :=
sorry

end NUMINAMATH_CALUDE_tangential_quadrilateral_l1723_172345


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l1723_172314

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Nat

/-- Converts a decimal number to its binary representation -/
def toBinary (n : Nat) : BinaryNumber :=
  sorry

/-- Converts a binary number to its decimal representation -/
def toDecimal (b : BinaryNumber) : Nat :=
  sorry

/-- Multiplies two binary numbers -/
def binaryMultiply (a b : BinaryNumber) : BinaryNumber :=
  sorry

/-- Divides a binary number by 2 -/
def binaryDivideByTwo (b : BinaryNumber) : BinaryNumber :=
  sorry

theorem binary_multiplication_division :
  let a := [1, 0, 1, 1, 0]  -- 10110₂
  let b := [1, 0, 1, 0, 0]  -- 10100₂
  let result := [1, 1, 0, 1, 1, 1, 0, 0]  -- 11011100₂
  binaryDivideByTwo (binaryMultiply a b) = result :=
sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l1723_172314


namespace NUMINAMATH_CALUDE_adi_change_l1723_172316

/-- Calculate the change Adi will receive after purchasing items and paying with a $20 bill. -/
theorem adi_change (pencil_cost notebook_cost colored_pencils_cost paid : ℚ) : 
  pencil_cost = 35/100 →
  notebook_cost = 3/2 →
  colored_pencils_cost = 11/4 →
  paid = 20 →
  paid - (pencil_cost + notebook_cost + colored_pencils_cost) = 77/5 := by
  sorry

#eval (20 : ℚ) - (35/100 + 3/2 + 11/4)

end NUMINAMATH_CALUDE_adi_change_l1723_172316


namespace NUMINAMATH_CALUDE_combined_grade4_percent_is_16_l1723_172344

/-- Represents the number of students in Pinegrove school -/
def pinegrove_students : ℕ := 120

/-- Represents the number of students in Maplewood school -/
def maplewood_students : ℕ := 180

/-- Represents the percentage of grade 4 students in Pinegrove school -/
def pinegrove_grade4_percent : ℚ := 10 / 100

/-- Represents the percentage of grade 4 students in Maplewood school -/
def maplewood_grade4_percent : ℚ := 20 / 100

/-- Represents the total number of students in both schools -/
def total_students : ℕ := pinegrove_students + maplewood_students

/-- Theorem stating that the percentage of grade 4 students in the combined schools is 16% -/
theorem combined_grade4_percent_is_16 : 
  (pinegrove_grade4_percent * pinegrove_students + maplewood_grade4_percent * maplewood_students) / total_students = 16 / 100 := by
  sorry

end NUMINAMATH_CALUDE_combined_grade4_percent_is_16_l1723_172344


namespace NUMINAMATH_CALUDE_trigonometric_identity_l1723_172340

theorem trigonometric_identity (α : Real) 
  (h : (1 + Real.sin α) * (1 - Real.cos α) = 1) : 
  (1 - Real.sin α) * (1 + Real.cos α) = 1 - Real.sin (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l1723_172340


namespace NUMINAMATH_CALUDE_divisor_problem_l1723_172368

theorem divisor_problem (D : ℕ) : 
  D > 0 ∧
  242 % D = 11 ∧
  698 % D = 18 ∧
  (242 + 698) % D = 9 →
  D = 20 := by
sorry

end NUMINAMATH_CALUDE_divisor_problem_l1723_172368


namespace NUMINAMATH_CALUDE_june_initial_stickers_l1723_172370

/-- The number of stickers June had initially -/
def june_initial : ℕ := 76

/-- The number of stickers Bonnie had initially -/
def bonnie_initial : ℕ := 63

/-- The number of stickers their grandparents gave to each of them -/
def gift : ℕ := 25

/-- The combined total of stickers after receiving the gifts -/
def total : ℕ := 189

theorem june_initial_stickers : 
  june_initial + gift + bonnie_initial + gift = total := by sorry

end NUMINAMATH_CALUDE_june_initial_stickers_l1723_172370


namespace NUMINAMATH_CALUDE_cubic_equation_roots_sum_l1723_172327

theorem cubic_equation_roots_sum (a b c : ℝ) : 
  (a^3 - 6*a^2 + 11*a - 6 = 0) → 
  (b^3 - 6*b^2 + 11*b - 6 = 0) → 
  (c^3 - 6*c^2 + 11*c - 6 = 0) → 
  (a ≠ b) → (b ≠ c) → (a ≠ c) →
  (1/a^3 + 1/b^3 + 1/c^3 = 251/216) := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_sum_l1723_172327


namespace NUMINAMATH_CALUDE_friend_gcd_l1723_172384

theorem friend_gcd (a b : ℕ) (h : ∃ k : ℕ, a * b = k^2) :
  ∃ m : ℕ, a * Nat.gcd a b = m^2 := by
sorry

end NUMINAMATH_CALUDE_friend_gcd_l1723_172384


namespace NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1723_172382

theorem smallest_x_absolute_value_equation : 
  (∃ x : ℝ, 2 * |x - 10| = 24) ∧ 
  (∀ x : ℝ, 2 * |x - 10| = 24 → x ≥ -2) ∧
  (2 * |-2 - 10| = 24) := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_absolute_value_equation_l1723_172382


namespace NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l1723_172320

theorem largest_coefficient_binomial_expansion :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ 6 →
  (Nat.choose 6 k) * (2^k) ≤ (Nat.choose 6 4) * (2^4) :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_binomial_expansion_l1723_172320


namespace NUMINAMATH_CALUDE_angle_C_is_pi_third_max_area_when_a_b_equal_c_max_area_is_sqrt_three_l1723_172313

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition √3(a - c cos B) = b sin C -/
def condition (t : Triangle) : Prop :=
  Real.sqrt 3 * (t.a - t.c * Real.cos t.B) = t.b * Real.sin t.C

theorem angle_C_is_pi_third (t : Triangle) (h : condition t) : t.C = π / 3 := by
  sorry

theorem max_area_when_a_b_equal_c (t : Triangle) (h : condition t) (hc : t.c = 2) :
  (∀ t' : Triangle, condition t' → t'.c = 2 → t.a * t.b ≥ t'.a * t'.b) →
  t.a = 2 ∧ t.b = 2 := by
  sorry

theorem max_area_is_sqrt_three (t : Triangle) (h : condition t) (hc : t.c = 2)
  (hmax : t.a = 2 ∧ t.b = 2) :
  (1 / 2) * t.a * t.b * Real.sin t.C = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_C_is_pi_third_max_area_when_a_b_equal_c_max_area_is_sqrt_three_l1723_172313


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l1723_172353

theorem fraction_equation_solution :
  ∃ x : ℚ, x - 2 ≠ 0 ∧ (2 / (x - 2) = (1 + x) / (x - 2) + 1) ∧ x = 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l1723_172353


namespace NUMINAMATH_CALUDE_monomial_count_in_expansion_l1723_172349

theorem monomial_count_in_expansion : 
  let n : ℕ := 2020
  let expression := (fun (x y z : ℝ) => (x + y + z)^n + (x - y - z)^n)
  (∃ (count : ℕ), count = 1022121 ∧ 
    count = (Finset.range (n / 2 + 1)).sum (fun i => 2 * i + 1)) := by
  sorry

end NUMINAMATH_CALUDE_monomial_count_in_expansion_l1723_172349


namespace NUMINAMATH_CALUDE_select_duty_officers_l1723_172305

def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem select_duty_officers : choose 20 3 = 1140 := by
  sorry

end NUMINAMATH_CALUDE_select_duty_officers_l1723_172305


namespace NUMINAMATH_CALUDE_carla_earnings_l1723_172341

/-- Carla's earnings over two weeks in June --/
theorem carla_earnings (hours_week1 hours_week2 : ℕ) (extra_earnings : ℚ) :
  hours_week1 = 18 →
  hours_week2 = 28 →
  extra_earnings = 63 →
  ∃ (hourly_wage : ℚ),
    hourly_wage * (hours_week2 - hours_week1 : ℚ) = extra_earnings ∧
    hourly_wage * (hours_week1 + hours_week2 : ℚ) = 289.80 := by
  sorry

#check carla_earnings

end NUMINAMATH_CALUDE_carla_earnings_l1723_172341


namespace NUMINAMATH_CALUDE_angle_properties_l1723_172371

-- Define the angle θ
def θ : Real := sorry

-- Define the point through which the terminal side of θ passes
def terminal_point : ℝ × ℝ := (4, -3)

-- State the theorem
theorem angle_properties (θ : Real) (terminal_point : ℝ × ℝ) :
  terminal_point = (4, -3) →
  Real.tan θ = -3/4 ∧
  (Real.sin (θ + Real.pi/2) + Real.cos θ) / (Real.sin θ - Real.cos (θ - Real.pi)) = 8 := by
  sorry

end NUMINAMATH_CALUDE_angle_properties_l1723_172371


namespace NUMINAMATH_CALUDE_kathleen_savings_problem_l1723_172379

/-- Kathleen's savings and spending problem -/
theorem kathleen_savings_problem (june july august clothes_cost remaining : ℕ)
  (h_june : june = 21)
  (h_july : july = 46)
  (h_august : august = 45)
  (h_clothes : clothes_cost = 54)
  (h_remaining : remaining = 46) :
  ∃ (school_supplies : ℕ),
    june + july + august = clothes_cost + school_supplies + remaining ∧ 
    school_supplies = 12 := by
  sorry

end NUMINAMATH_CALUDE_kathleen_savings_problem_l1723_172379


namespace NUMINAMATH_CALUDE_difference_of_numbers_l1723_172311

theorem difference_of_numbers (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 - y^2 = 50) : x - y = 10 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_numbers_l1723_172311


namespace NUMINAMATH_CALUDE_equation_root_range_l1723_172322

theorem equation_root_range (k : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) ∧ 
   Real.sqrt 3 * Real.sin (2 * x) + Real.cos (2 * x) = k + 1) 
  → k ∈ Set.Icc (-2) 1 := by
sorry

end NUMINAMATH_CALUDE_equation_root_range_l1723_172322


namespace NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l1723_172395

theorem dairy_farm_husk_consumption 
  (num_cows : ℕ) 
  (num_bags : ℕ) 
  (num_days : ℕ) 
  (h1 : num_cows = 30) 
  (h2 : num_bags = 30) 
  (h3 : num_days = 30) : 
  (1 : ℕ) * num_days = 30 := by
  sorry

end NUMINAMATH_CALUDE_dairy_farm_husk_consumption_l1723_172395


namespace NUMINAMATH_CALUDE_files_deleted_l1723_172319

/-- Given Dave's initial and final number of files, prove the number of files deleted. -/
theorem files_deleted (initial_files final_files : ℕ) 
  (h1 : initial_files = 24)
  (h2 : final_files = 21) :
  initial_files - final_files = 3 := by
  sorry

#check files_deleted

end NUMINAMATH_CALUDE_files_deleted_l1723_172319


namespace NUMINAMATH_CALUDE_eight_people_arrangement_l1723_172337

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where two specific people must sit together -/
def restrictedArrangements (n : ℕ) : ℕ := (Nat.factorial (n - 1)) * 2

/-- The number of ways to arrange n people in a row where two specific people cannot sit next to each other -/
def acceptableArrangements (n : ℕ) : ℕ := totalArrangements n - restrictedArrangements n

theorem eight_people_arrangement :
  acceptableArrangements 8 = 30240 := by
  sorry

end NUMINAMATH_CALUDE_eight_people_arrangement_l1723_172337


namespace NUMINAMATH_CALUDE_max_cube_sum_four_squares_l1723_172399

theorem max_cube_sum_four_squares {a b c d : ℝ} (h : a^2 + b^2 + c^2 + d^2 = 4) :
  a^3 + b^3 + c^3 + d^3 ≤ 8 ∧ ∃ (a₀ b₀ c₀ d₀ : ℝ), a₀^2 + b₀^2 + c₀^2 + d₀^2 = 4 ∧ a₀^3 + b₀^3 + c₀^3 + d₀^3 = 8 :=
by sorry

end NUMINAMATH_CALUDE_max_cube_sum_four_squares_l1723_172399


namespace NUMINAMATH_CALUDE_max_value_A_l1723_172343

/-- The function A(x, y) as defined in the problem -/
def A (x y : ℝ) : ℝ := x^4*y + x*y^4 + x^3*y + x*y^3 + x^2*y + x*y^2

/-- The theorem stating the maximum value of A(x, y) under the given constraint -/
theorem max_value_A :
  ∀ x y : ℝ, x + y = 1 → A x y ≤ 7/16 :=
by sorry

end NUMINAMATH_CALUDE_max_value_A_l1723_172343


namespace NUMINAMATH_CALUDE_semicircle_perimeter_semicircle_area_l1723_172362

-- Define constants
def rectangle_length : ℝ := 10
def rectangle_width : ℝ := 8
def π : ℝ := 3.14

-- Define the semicircle
def semicircle_diameter : ℝ := rectangle_length

-- Theorem for the perimeter of the semicircle
theorem semicircle_perimeter :
  π * semicircle_diameter / 2 + semicircle_diameter = 25.7 :=
sorry

-- Theorem for the area of the semicircle
theorem semicircle_area :
  π * (semicircle_diameter / 2)^2 / 2 = 39.25 :=
sorry

end NUMINAMATH_CALUDE_semicircle_perimeter_semicircle_area_l1723_172362


namespace NUMINAMATH_CALUDE_max_books_is_eight_l1723_172397

/-- Represents the maximum number of books borrowed by a single student in a class with the given conditions. -/
def max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) (avg_books : ℕ) : ℕ :=
  let rest_students := total_students - (zero_books + one_book + two_books)
  let total_books := total_students * avg_books
  let known_books := one_book + 2 * two_books
  let rest_books := total_books - known_books
  let min_rest_books := (rest_students - 1) * 3
  rest_books - min_rest_books

/-- Theorem stating that under the given conditions, the maximum number of books borrowed by a single student is 8. -/
theorem max_books_is_eight :
  max_books_borrowed 20 2 8 3 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_max_books_is_eight_l1723_172397


namespace NUMINAMATH_CALUDE_f_negative_nine_l1723_172363

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

theorem f_negative_nine (f : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_period : has_period f 4) 
  (h_f_one : f 1 = 1) : 
  f (-9) = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_negative_nine_l1723_172363


namespace NUMINAMATH_CALUDE_rental_fee_minimization_l1723_172338

/-- Represents the total number of buses to be rented -/
def total_buses : ℕ := 6

/-- Represents the rental fee for a Type A bus -/
def type_a_fee : ℕ := 450

/-- Represents the rental fee for a Type B bus -/
def type_b_fee : ℕ := 300

/-- Calculates the total rental fee based on the number of Type B buses -/
def rental_fee (x : ℕ) : ℕ := total_buses * type_a_fee - (type_a_fee - type_b_fee) * x

theorem rental_fee_minimization :
  ∀ x : ℕ, 0 < x → x < total_buses → x < total_buses - x →
  (∀ y : ℕ, 0 < y → y < total_buses → y < total_buses - y →
    rental_fee x ≤ rental_fee y) →
  x = 2 ∧ rental_fee x = 2400 := by sorry

end NUMINAMATH_CALUDE_rental_fee_minimization_l1723_172338


namespace NUMINAMATH_CALUDE_isosceles_triangle_on_hyperbola_x_coordinate_range_l1723_172378

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 6 - y^2 / 3 = 1

-- Define the isosceles triangle
structure IsoscelesTriangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  isIsosceles : (A.1 - C.1)^2 + (A.2 - C.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2

-- Theorem statement
theorem isosceles_triangle_on_hyperbola_x_coordinate_range 
  (triangle : IsoscelesTriangle)
  (hA : hyperbola triangle.A.1 triangle.A.2)
  (hB : hyperbola triangle.B.1 triangle.B.2)
  (hC : triangle.C.2 = 0)
  (hAB_not_perpendicular : (triangle.A.2 - triangle.B.2) ≠ 0) :
  triangle.C.1 > (3/2) * Real.sqrt 6 :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_on_hyperbola_x_coordinate_range_l1723_172378


namespace NUMINAMATH_CALUDE_second_number_value_l1723_172375

theorem second_number_value (x y : ℕ) (h1 : x + y = 33) (h2 : y = 2 * x) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_second_number_value_l1723_172375


namespace NUMINAMATH_CALUDE_orange_distribution_l1723_172361

/-- Given a number of oranges, calories per orange, and calories per person,
    calculate the number of people who can receive an equal share of the total calories. -/
def people_fed (num_oranges : ℕ) (calories_per_orange : ℕ) (calories_per_person : ℕ) : ℕ :=
  (num_oranges * calories_per_orange) / calories_per_person

/-- Prove that with 5 oranges, 80 calories per orange, and 100 calories per person,
    the number of people fed is 4. -/
theorem orange_distribution :
  people_fed 5 80 100 = 4 := by
  sorry

end NUMINAMATH_CALUDE_orange_distribution_l1723_172361


namespace NUMINAMATH_CALUDE_mary_picked_nine_lemons_l1723_172312

/-- The number of lemons picked by Sally -/
def sally_lemons : ℕ := 7

/-- The total number of lemons picked by Sally and Mary -/
def total_lemons : ℕ := 16

/-- The number of lemons picked by Mary -/
def mary_lemons : ℕ := total_lemons - sally_lemons

/-- Theorem stating that Mary picked 9 lemons -/
theorem mary_picked_nine_lemons : mary_lemons = 9 := by
  sorry

end NUMINAMATH_CALUDE_mary_picked_nine_lemons_l1723_172312


namespace NUMINAMATH_CALUDE_smallest_positive_angle_theorem_l1723_172301

theorem smallest_positive_angle_theorem (y : ℝ) : 
  (5 * Real.cos y * Real.sin y ^ 3 - 5 * Real.cos y ^ 3 * Real.sin y = 1 / 2) →
  y = (1 / 4) * Real.arcsin (2 / 5) ∧ y > 0 ∧ 
  ∀ z, z > 0 → (5 * Real.cos z * Real.sin z ^ 3 - 5 * Real.cos z ^ 3 * Real.sin z = 1 / 2) → y ≤ z :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_theorem_l1723_172301


namespace NUMINAMATH_CALUDE_construction_contract_l1723_172390

theorem construction_contract (H : ℕ) 
  (first_half : H * 3 / 5 = H - (300 + 500))
  (remaining : 500 = H - (H * 3 / 5 + 300)) : H = 2000 := by
  sorry

end NUMINAMATH_CALUDE_construction_contract_l1723_172390


namespace NUMINAMATH_CALUDE_max_placement_1002nd_round_max_placement_1001st_round_l1723_172365

/-- Represents the state of an election round -/
structure ElectionRound where
  candidateCount : Nat
  votes : List Nat

/-- Defines the election process -/
def runElection (initialRound : ElectionRound) : Nat → Option Nat :=
  sorry

/-- Theorem for the maximum initial placement allowing victory in the 1002nd round -/
theorem max_placement_1002nd_round 
  (initialCandidates : Nat) 
  (ostapInitialPlacement : Nat) :
  initialCandidates = 2002 →
  (∃ (initialVotes : List Nat), 
    initialVotes.length = initialCandidates ∧
    ostapInitialPlacement = 2001 ∧
    runElection ⟨initialCandidates, initialVotes⟩ 1002 = some ostapInitialPlacement) ∧
  (∀ k > 2001, ∀ (initialVotes : List Nat),
    initialVotes.length = initialCandidates →
    runElection ⟨initialCandidates, initialVotes⟩ 1002 ≠ some k) :=
  sorry

/-- Theorem for the maximum initial placement allowing victory in the 1001st round -/
theorem max_placement_1001st_round 
  (initialCandidates : Nat) 
  (ostapInitialPlacement : Nat) :
  initialCandidates = 2002 →
  (∃ (initialVotes : List Nat), 
    initialVotes.length = initialCandidates ∧
    ostapInitialPlacement = 1001 ∧
    runElection ⟨initialCandidates, initialVotes⟩ 1001 = some ostapInitialPlacement) ∧
  (∀ k > 1001, ∀ (initialVotes : List Nat),
    initialVotes.length = initialCandidates →
    runElection ⟨initialCandidates, initialVotes⟩ 1001 ≠ some k) :=
  sorry

end NUMINAMATH_CALUDE_max_placement_1002nd_round_max_placement_1001st_round_l1723_172365


namespace NUMINAMATH_CALUDE_friends_carrying_bananas_l1723_172398

theorem friends_carrying_bananas (total_friends : ℕ) (pears oranges apples : ℕ) : 
  total_friends = 35 →
  pears = 14 →
  oranges = 8 →
  apples = 5 →
  total_friends = pears + oranges + apples + (total_friends - (pears + oranges + apples)) →
  total_friends - (pears + oranges + apples) = 8 :=
by sorry

end NUMINAMATH_CALUDE_friends_carrying_bananas_l1723_172398


namespace NUMINAMATH_CALUDE_triangle_area_is_correct_l1723_172351

/-- The area of a triangular region bounded by the coordinate axes and the line 3x + y = 9 -/
def triangleArea : ℝ := 13.5

/-- The equation of the bounding line -/
def boundingLine (x y : ℝ) : Prop := 3 * x + y = 9

theorem triangle_area_is_correct : 
  triangleArea = 13.5 := by sorry

end NUMINAMATH_CALUDE_triangle_area_is_correct_l1723_172351


namespace NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l1723_172323

def A : Set ℝ := {x | x ≤ 1}
def B (a : ℝ) : Set ℝ := {x | x ≥ a}

theorem intersection_implies_a_equals_one (a : ℝ) :
  A ∩ B a = {1} → a = 1 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_a_equals_one_l1723_172323


namespace NUMINAMATH_CALUDE_tangent_line_at_M_l1723_172389

/-- The circle with equation x^2 + y^2 = 5 -/
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 5}

/-- The point M on the circle -/
def M : ℝ × ℝ := (2, -1)

/-- The proposed tangent line equation -/
def TangentLine (x y : ℝ) : Prop := 2*x - y - 5 = 0

/-- Theorem stating that the proposed line is tangent to the circle at M -/
theorem tangent_line_at_M :
  M ∈ Circle ∧
  TangentLine M.1 M.2 ∧
  ∀ p ∈ Circle, p ≠ M → ¬TangentLine p.1 p.2 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_M_l1723_172389


namespace NUMINAMATH_CALUDE_y_value_l1723_172364

theorem y_value (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 16) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l1723_172364


namespace NUMINAMATH_CALUDE_turtle_ratio_l1723_172328

def total_turtles : ℕ := 42
def turtles_on_sand : ℕ := 28

theorem turtle_ratio : 
  (total_turtles - turtles_on_sand) / total_turtles = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_turtle_ratio_l1723_172328


namespace NUMINAMATH_CALUDE_direct_proportion_conditions_l1723_172374

/-- A function representing a potential direct proportion -/
def f (k b x : ℝ) : ℝ := (k - 4) * x + b

/-- Definition of a direct proportion function -/
def is_direct_proportion (g : ℝ → ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, g x = m * x

/-- Theorem stating the necessary and sufficient conditions for f to be a direct proportion -/
theorem direct_proportion_conditions (k b : ℝ) :
  is_direct_proportion (f k b) ↔ k ≠ 4 ∧ b = 0 :=
sorry

end NUMINAMATH_CALUDE_direct_proportion_conditions_l1723_172374


namespace NUMINAMATH_CALUDE_largest_of_four_consecutive_even_l1723_172380

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

def consecutive_even (a b c d : ℤ) : Prop :=
  is_even a ∧ is_even b ∧ is_even c ∧ is_even d ∧
  b = a + 2 ∧ c = b + 2 ∧ d = c + 2

theorem largest_of_four_consecutive_even (a b c d : ℤ) :
  consecutive_even a b c d → a + b + c + d = 92 → d = 26 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_four_consecutive_even_l1723_172380


namespace NUMINAMATH_CALUDE_square_difference_divisible_by_13_l1723_172336

theorem square_difference_divisible_by_13 (a b : ℕ) :
  a ∈ Finset.range 1001 →
  b ∈ Finset.range 1001 →
  a + b = 1001 →
  13 ∣ (a^2 - b^2) :=
by sorry

end NUMINAMATH_CALUDE_square_difference_divisible_by_13_l1723_172336


namespace NUMINAMATH_CALUDE_valid_square_configurations_l1723_172309

/-- Represents a configuration of a 5x7 grid --/
structure GridConfiguration where
  squares : ℕ  -- number of 2x2 squares
  strips : ℕ   -- number of 1x3 strips
  corners : ℕ  -- number of three-cell corners

/-- Checks if a configuration is valid for a 5x7 grid --/
def isValidConfiguration (config : GridConfiguration) : Prop :=
  4 * config.squares + 3 * config.strips + 3 * config.corners = 35

/-- The theorem stating the valid configurations for 2x2 squares --/
theorem valid_square_configurations :
  ∀ (config : GridConfiguration),
    isValidConfiguration config →
    (config.squares = 5 ∨ config.squares = 2) :=
  sorry

end NUMINAMATH_CALUDE_valid_square_configurations_l1723_172309


namespace NUMINAMATH_CALUDE_new_person_weight_l1723_172386

theorem new_person_weight (initial_count : ℕ) (leaving_weight : ℝ) (avg_increase : ℝ) :
  initial_count = 8 →
  leaving_weight = 70 →
  avg_increase = 2.5 →
  (initial_count : ℝ) * avg_increase + leaving_weight = 90 :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l1723_172386


namespace NUMINAMATH_CALUDE_unique_pizza_combinations_l1723_172360

def number_of_toppings : ℕ := 8
def toppings_per_pizza : ℕ := 5

theorem unique_pizza_combinations : 
  (number_of_toppings.choose toppings_per_pizza) = 56 := by
  sorry

end NUMINAMATH_CALUDE_unique_pizza_combinations_l1723_172360


namespace NUMINAMATH_CALUDE_square_value_l1723_172302

theorem square_value (square : ℚ) : 8/12 = square/3 → square = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_value_l1723_172302


namespace NUMINAMATH_CALUDE_circle_angle_constraint_l1723_172330

-- Define the circle C
def C (x y : ℝ) : Prop := (x - 6)^2 + (y - 8)^2 = 1

-- Define points A and B
def A (m : ℝ) : ℝ × ℝ := (-m, 0)
def B (m : ℝ) : ℝ × ℝ := (m, 0)

-- Define the angle APB
def angle_APB (m : ℝ) (P : ℝ × ℝ) : ℝ := sorry

-- Main theorem
theorem circle_angle_constraint (m : ℝ) :
  m > 0 →
  (∀ P : ℝ × ℝ, C P.1 P.2 → angle_APB m P < 90) →
  9 < m ∧ m < 11 :=
sorry

end NUMINAMATH_CALUDE_circle_angle_constraint_l1723_172330


namespace NUMINAMATH_CALUDE_minimum_apples_in_basket_l1723_172347

theorem minimum_apples_in_basket (n : ℕ) : 
  (n % 3 = 2) ∧ (n % 4 = 2) ∧ (n % 5 = 2) → n ≥ 62 :=
by sorry

end NUMINAMATH_CALUDE_minimum_apples_in_basket_l1723_172347


namespace NUMINAMATH_CALUDE_yoojung_notebooks_l1723_172383

theorem yoojung_notebooks (initial : ℕ) : 
  (initial ≥ 5) →                        -- Ensure initial is at least 5
  (((initial - 5) / 2 : ℚ) = 4) →        -- Half of remaining after giving 5 equals 4
  (initial = 13) :=                      -- Prove initial is 13
sorry

end NUMINAMATH_CALUDE_yoojung_notebooks_l1723_172383


namespace NUMINAMATH_CALUDE_consecutive_odd_squares_same_digit_l1723_172329

theorem consecutive_odd_squares_same_digit : ∃! (n : ℕ), 
  (∃ (d : ℕ), d ∈ Finset.range 10 ∧ 
    (n - 2)^2 + n^2 + (n + 2)^2 = 1111 * d) ∧
  Odd n ∧ n = 43 := by sorry

end NUMINAMATH_CALUDE_consecutive_odd_squares_same_digit_l1723_172329


namespace NUMINAMATH_CALUDE_new_average_is_250_l1723_172335

/-- A salesperson's commission information -/
structure SalesCommission where
  totalSales : ℕ
  lastCommission : ℝ
  averageIncrease : ℝ

/-- Calculate the new average commission after a big sale -/
def newAverageCommission (sc : SalesCommission) : ℝ :=
  sorry

/-- Theorem stating the new average commission is $250 under given conditions -/
theorem new_average_is_250 (sc : SalesCommission) 
  (h1 : sc.totalSales = 6)
  (h2 : sc.lastCommission = 1000)
  (h3 : sc.averageIncrease = 150) :
  newAverageCommission sc = 250 :=
sorry

end NUMINAMATH_CALUDE_new_average_is_250_l1723_172335


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l1723_172332

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 + 4*x₁ = 5) ∧ 
  (x₂^2 + 4*x₂ = 5) ∧ 
  x₁ = 1 ∧ 
  x₂ = -5 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l1723_172332


namespace NUMINAMATH_CALUDE_subtract_from_percentage_l1723_172352

theorem subtract_from_percentage (number : ℝ) (percentage : ℝ) (subtrahend : ℝ) : 
  number = 200 → percentage = 40 → subtrahend = 30 →
  percentage / 100 * number - subtrahend = 50 := by
sorry

end NUMINAMATH_CALUDE_subtract_from_percentage_l1723_172352


namespace NUMINAMATH_CALUDE_manufacturing_costs_calculation_l1723_172310

structure Company where
  machines : ℕ
  estCharges : ℝ
  annualOutput : ℝ
  profitRate : ℝ
  closedMachines : ℝ
  profitDecrease : ℝ

def annualManufacturingCosts (c : Company) : ℝ :=
  c.annualOutput - c.estCharges

theorem manufacturing_costs_calculation (c : Company) 
  (h1 : c.machines = 14)
  (h2 : c.estCharges = 12000)
  (h3 : c.annualOutput = 70000)
  (h4 : c.profitRate = 0.125)
  (h5 : c.closedMachines = 7.14)
  (h6 : c.profitDecrease = 0.125)
  (h7 : (c.machines - c.closedMachines) / c.machines = 1 - c.profitDecrease) :
  annualManufacturingCosts c = 58000 := by
  sorry

#eval annualManufacturingCosts { 
  machines := 14, 
  estCharges := 12000, 
  annualOutput := 70000, 
  profitRate := 0.125, 
  closedMachines := 7.14, 
  profitDecrease := 0.125 
}

end NUMINAMATH_CALUDE_manufacturing_costs_calculation_l1723_172310


namespace NUMINAMATH_CALUDE_smallest_n_divisibility_l1723_172326

theorem smallest_n_divisibility : ∃ (n : ℕ), n > 0 ∧ 
  18 ∣ n^2 ∧ 1152 ∣ n^3 ∧ 
  ∀ (m : ℕ), m > 0 → 18 ∣ m^2 → 1152 ∣ m^3 → n ≤ m :=
by
  use 72
  sorry

end NUMINAMATH_CALUDE_smallest_n_divisibility_l1723_172326


namespace NUMINAMATH_CALUDE_complement_intersection_equiv_complement_union_l1723_172334

universe u

theorem complement_intersection_equiv_complement_union {U : Type u} (M N : Set U) :
  ∀ x : U, x ∈ (M ∩ N)ᶜ ↔ x ∈ Mᶜ ∪ Nᶜ := by sorry

end NUMINAMATH_CALUDE_complement_intersection_equiv_complement_union_l1723_172334


namespace NUMINAMATH_CALUDE_stone_minimum_speed_l1723_172325

/-- The minimum speed for a stone to pass through both corners of a building without touching the roof -/
theorem stone_minimum_speed (g H l : ℝ) (α : ℝ) (h_g : g > 0) (h_H : H > 0) (h_l : l > 0) (h_α : 0 < α ∧ α < π / 2) :
  ∃ v₀ : ℝ, v₀ > 0 ∧
    v₀ = Real.sqrt (g * (2 * H + l * (1 - Real.sin α) / Real.cos α)) ∧
    ∀ v : ℝ, v > v₀ →
      ∃ (x y : ℝ → ℝ), (∀ t, x t = v * Real.cos α * t ∧ y t = -g * t^2 / 2 + v * Real.sin α * t) ∧
        (∃ t₁ t₂, t₁ ≠ t₂ ∧ x t₁ = 0 ∧ y t₁ = H ∧ x t₂ = l ∧ y t₂ = H - l * Real.tan α) ∧
        (∀ t, 0 ≤ x t ∧ x t ≤ l → y t ≥ H - (x t) * Real.tan α) :=
by sorry

end NUMINAMATH_CALUDE_stone_minimum_speed_l1723_172325


namespace NUMINAMATH_CALUDE_student_council_committees_l1723_172307

theorem student_council_committees (n : ℕ) : 
  n * (n - 1) / 2 = 28 → (n.choose 4) = 70 := by
  sorry

end NUMINAMATH_CALUDE_student_council_committees_l1723_172307


namespace NUMINAMATH_CALUDE_no_two_cubes_between_squares_l1723_172355

theorem no_two_cubes_between_squares : ¬∃ (n a b : ℤ), n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n+1)^2 := by
  sorry

end NUMINAMATH_CALUDE_no_two_cubes_between_squares_l1723_172355


namespace NUMINAMATH_CALUDE_flippy_divisible_by_four_l1723_172377

/-- A four-digit number is flippy if its digits alternate between two distinct digits from the set {4, 6} -/
def is_flippy (n : ℕ) : Prop :=
  (n ≥ 1000 ∧ n < 10000) ∧
  (∃ a b : ℕ, (a = 4 ∨ a = 6) ∧ (b = 4 ∨ b = 6) ∧ a ≠ b ∧
   ((n = 1000 * a + 100 * b + 10 * a + b) ∨
    (n = 1000 * b + 100 * a + 10 * b + a)))

theorem flippy_divisible_by_four :
  ∃! n : ℕ, is_flippy n ∧ n % 4 = 0 :=
sorry

end NUMINAMATH_CALUDE_flippy_divisible_by_four_l1723_172377


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l1723_172357

/-- Represents the outcome of drawing balls from a bag -/
structure BallDrawing where
  totalBalls : Nat
  redBalls : Nat
  blackBalls : Nat
  drawCount : Nat

/-- Calculates the expectation for drawing with replacement -/
def expectationWithReplacement (bd : BallDrawing) : Rat :=
  bd.drawCount * (bd.redBalls : Rat) / bd.totalBalls

/-- Calculates the variance for drawing with replacement -/
def varianceWithReplacement (bd : BallDrawing) : Rat :=
  bd.drawCount * (bd.redBalls : Rat) / bd.totalBalls * (1 - (bd.redBalls : Rat) / bd.totalBalls)

/-- Calculates the expectation for drawing without replacement -/
noncomputable def expectationWithoutReplacement (bd : BallDrawing) : Rat :=
  sorry

/-- Calculates the variance for drawing without replacement -/
noncomputable def varianceWithoutReplacement (bd : BallDrawing) : Rat :=
  sorry

theorem ball_drawing_theorem (bd : BallDrawing) 
    (h1 : bd.totalBalls = 10)
    (h2 : bd.redBalls = 4)
    (h3 : bd.blackBalls = 6)
    (h4 : bd.drawCount = 3) :
    expectationWithReplacement bd = expectationWithoutReplacement bd ∧
    varianceWithReplacement bd > varianceWithoutReplacement bd :=
  by sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l1723_172357


namespace NUMINAMATH_CALUDE_unique_arrangement_l1723_172359

-- Define the types for people and professions
inductive Person : Type
| Andrey : Person
| Boris : Person
| Vyacheslav : Person
| Gennady : Person

inductive Profession : Type
| Architect : Profession
| Barista : Profession
| Veterinarian : Profession
| Guitarist : Profession

-- Define the seating arrangement
def Arrangement := List Person

-- Define the function to assign professions to people
def Assignment := Person → Profession

-- Define the conditions
def veterinarian_between (arr : Arrangement) (assign : Assignment) : Prop :=
  ∃ i, i > 0 ∧ i < arr.length - 1 ∧
    assign (arr.get ⟨i, sorry⟩) = Profession.Veterinarian ∧
    (assign (arr.get ⟨i-1, sorry⟩) = Profession.Architect ∨ assign (arr.get ⟨i-1, sorry⟩) = Profession.Guitarist) ∧
    (assign (arr.get ⟨i+1, sorry⟩) = Profession.Architect ∨ assign (arr.get ⟨i+1, sorry⟩) = Profession.Guitarist)

def boris_right_of_barista (arr : Arrangement) (assign : Assignment) : Prop :=
  ∃ i, i < arr.length - 1 ∧
    assign (arr.get ⟨i, sorry⟩) = Profession.Barista ∧
    arr.get ⟨i+1, sorry⟩ = Person.Boris

def vyacheslav_right_of_andrey_and_boris (arr : Arrangement) : Prop :=
  ∃ i j k, i < j ∧ j < k ∧ k < arr.length ∧
    arr.get ⟨i, sorry⟩ = Person.Andrey ∧
    arr.get ⟨j, sorry⟩ = Person.Boris ∧
    arr.get ⟨k, sorry⟩ = Person.Vyacheslav

def andrey_knows_neighbors (arr : Arrangement) : Prop :=
  ∃ i, i > 0 ∧ i < arr.length - 1 ∧ arr.get ⟨i, sorry⟩ = Person.Andrey

def guitarist_barista_not_adjacent (arr : Arrangement) (assign : Assignment) : Prop :=
  ∀ i, i < arr.length - 1 →
    ¬(assign (arr.get ⟨i, sorry⟩) = Profession.Guitarist ∧ assign (arr.get ⟨i+1, sorry⟩) = Profession.Barista) ∧
    ¬(assign (arr.get ⟨i, sorry⟩) = Profession.Barista ∧ assign (arr.get ⟨i+1, sorry⟩) = Profession.Guitarist)

-- Define the theorem
theorem unique_arrangement :
  ∀ (arr : Arrangement) (assign : Assignment),
    arr.length = 4 ∧
    veterinarian_between arr assign ∧
    boris_right_of_barista arr assign ∧
    vyacheslav_right_of_andrey_and_boris arr ∧
    andrey_knows_neighbors arr ∧
    guitarist_barista_not_adjacent arr assign →
    arr = [Person.Gennady, Person.Boris, Person.Andrey, Person.Vyacheslav] ∧
    assign Person.Gennady = Profession.Barista ∧
    assign Person.Boris = Profession.Architect ∧
    assign Person.Andrey = Profession.Veterinarian ∧
    assign Person.Vyacheslav = Profession.Guitarist :=
by sorry


end NUMINAMATH_CALUDE_unique_arrangement_l1723_172359


namespace NUMINAMATH_CALUDE_smallest_integer_with_square_half_and_cube_third_l1723_172358

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def is_perfect_cube (n : ℕ) : Prop := ∃ m : ℕ, n = m^3

theorem smallest_integer_with_square_half_and_cube_third :
  ∃! n : ℕ, n > 0 ∧ is_perfect_square (n / 2) ∧ is_perfect_cube (n / 3) ∧
  ∀ m : ℕ, m > 0 ∧ is_perfect_square (m / 2) ∧ is_perfect_cube (m / 3) → n ≤ m :=
by
  use 648
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_square_half_and_cube_third_l1723_172358


namespace NUMINAMATH_CALUDE_technicians_schedule_lcm_l1723_172366

theorem technicians_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 8 9)) = 360 := by
  sorry

end NUMINAMATH_CALUDE_technicians_schedule_lcm_l1723_172366


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1723_172339

/-- Given that x and y are always positive, x^3 and y vary inversely, 
    and y = 8 when x = 2, prove that x = 1 / (13.5^(1/3)) when y = 1728 -/
theorem inverse_variation_problem (x y : ℝ) 
  (h_positive : x > 0 ∧ y > 0)
  (h_inverse : ∃ k : ℝ, k > 0 ∧ ∀ x y, x^3 * y = k)
  (h_initial : 2^3 * 8 = (h_inverse.choose)^3)
  (h_final : y = 1728) :
  x = 1 / (13.5^(1/3)) :=
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1723_172339


namespace NUMINAMATH_CALUDE_right_triangle_area_l1723_172306

theorem right_triangle_area (r R : ℝ) (h : r > 0) (h' : R > 0) : ∃ (A : ℝ), 
  A > 0 ∧ 
  (∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (a^2 + b^2 = c^2) ∧  -- right triangle condition
    (A = (a * b) / 2) ∧  -- area of triangle
    (r = A / ((a + b + c) / 2)) ∧  -- inradius formula
    (R = c / 2) ∧  -- circumradius formula for right triangle
    (A = r * (2 * R + r))) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_area_l1723_172306


namespace NUMINAMATH_CALUDE_A_satisfies_conditions_l1723_172348

-- Define set B
def B : Set ℝ := {x : ℝ | x ≥ 0}

-- Define set A
def A : Set ℝ := {1, 2}

-- Theorem statement
theorem A_satisfies_conditions : (A ∩ B = A) := by sorry

end NUMINAMATH_CALUDE_A_satisfies_conditions_l1723_172348


namespace NUMINAMATH_CALUDE_race_probability_l1723_172387

theorem race_probability (total_cars : ℕ) (prob_Y prob_Z prob_XYZ : ℝ) : 
  total_cars = 15 →
  prob_Y = 1/8 →
  prob_Z = 1/12 →
  prob_XYZ = 0.4583333333333333 →
  ∃ (prob_X : ℝ), prob_X + prob_Y + prob_Z = prob_XYZ ∧ prob_X = 0.25 := by
  sorry

end NUMINAMATH_CALUDE_race_probability_l1723_172387


namespace NUMINAMATH_CALUDE_nested_square_root_equality_l1723_172381

theorem nested_square_root_equality : 
  Real.sqrt (49 * Real.sqrt (25 * Real.sqrt 9)) = 5 * Real.sqrt 7 * Real.sqrt (Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_nested_square_root_equality_l1723_172381


namespace NUMINAMATH_CALUDE_dryer_cost_l1723_172303

theorem dryer_cost (washer dryer : ℕ) : 
  washer + dryer = 600 →
  washer = 3 * dryer →
  dryer = 150 := by
sorry

end NUMINAMATH_CALUDE_dryer_cost_l1723_172303


namespace NUMINAMATH_CALUDE_white_pairs_coincide_l1723_172393

/-- Represents the number of triangles of each color in each half of the figure -/
structure HalfFigure where
  red : ℕ
  blue : ℕ
  white : ℕ

/-- Represents the number of coinciding pairs when the figure is folded -/
structure CoincidingPairs where
  redRed : ℕ
  blueBlue : ℕ
  redWhite : ℕ
  whiteWhite : ℕ

theorem white_pairs_coincide (half : HalfFigure) (pairs : CoincidingPairs) : 
  half.red = 4 ∧ 
  half.blue = 6 ∧ 
  half.white = 10 ∧ 
  pairs.redRed = 3 ∧ 
  pairs.blueBlue = 4 ∧ 
  pairs.redWhite = 3 → 
  pairs.whiteWhite = 5 := by
sorry

end NUMINAMATH_CALUDE_white_pairs_coincide_l1723_172393


namespace NUMINAMATH_CALUDE_three_pi_irrational_l1723_172350

/-- π is an irrational number -/
axiom pi_irrational : Irrational Real.pi

/-- The product of an irrational number and a non-zero rational number is irrational -/
axiom irrational_mul_rational {x : ℝ} (hx : Irrational x) {q : ℚ} (hq : q ≠ 0) :
  Irrational (x * ↑q)

/-- 3π is an irrational number -/
theorem three_pi_irrational : Irrational (3 * Real.pi) := by sorry

end NUMINAMATH_CALUDE_three_pi_irrational_l1723_172350


namespace NUMINAMATH_CALUDE_probability_standard_deck_l1723_172342

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (face_cards : Nat)
  (number_cards : Nat)

/-- Define a standard 52-card deck -/
def standard_deck : Deck :=
  { total_cards := 52,
    face_cards := 12,
    number_cards := 40 }

/-- Calculate the probability of drawing a face card first and a number card second -/
def probability_face_then_number (d : Deck) : Rat :=
  (d.face_cards * d.number_cards : Rat) / (d.total_cards * (d.total_cards - 1))

/-- Theorem stating the probability for a standard deck -/
theorem probability_standard_deck :
  probability_face_then_number standard_deck = 40 / 221 := by
  sorry


end NUMINAMATH_CALUDE_probability_standard_deck_l1723_172342


namespace NUMINAMATH_CALUDE_one_in_set_zero_one_l1723_172333

theorem one_in_set_zero_one : 1 ∈ ({0, 1} : Set ℕ) := by sorry

end NUMINAMATH_CALUDE_one_in_set_zero_one_l1723_172333


namespace NUMINAMATH_CALUDE_system_solution_l1723_172376

theorem system_solution :
  ∃ (x y : ℚ), (4 * x = -10 - 3 * y) ∧ (6 * x = 5 * y - 32) ∧ (x = -73/19) ∧ (y = 34/19) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1723_172376


namespace NUMINAMATH_CALUDE_line_intersects_x_axis_l1723_172321

/-- The line equation 3y - 4x = 12 -/
def line_equation (x y : ℝ) : Prop := 3 * y - 4 * x = 12

/-- The x-axis equation y = 0 -/
def x_axis (y : ℝ) : Prop := y = 0

/-- The intersection point of the line and the x-axis -/
def intersection_point : ℝ × ℝ := (-3, 0)

theorem line_intersects_x_axis :
  let (x, y) := intersection_point
  line_equation x y ∧ x_axis y := by sorry

end NUMINAMATH_CALUDE_line_intersects_x_axis_l1723_172321


namespace NUMINAMATH_CALUDE_fruit_purchase_cost_l1723_172300

theorem fruit_purchase_cost (strawberry_price : ℝ) (cherry_price : ℝ) (blueberry_price : ℝ)
  (strawberry_amount : ℝ) (cherry_amount : ℝ) (blueberry_amount : ℝ)
  (blueberry_discount : ℝ) (bag_fee : ℝ) :
  strawberry_price = 2.20 →
  cherry_price = 6 * strawberry_price →
  blueberry_price = cherry_price / 2 →
  strawberry_amount = 3 →
  cherry_amount = 4.5 →
  blueberry_amount = 6.2 →
  blueberry_discount = 0.15 →
  bag_fee = 0.75 →
  strawberry_price * strawberry_amount +
  cherry_price * cherry_amount +
  blueberry_price * blueberry_amount * (1 - blueberry_discount) +
  bag_fee = 101.53 := by
sorry

end NUMINAMATH_CALUDE_fruit_purchase_cost_l1723_172300


namespace NUMINAMATH_CALUDE_multiply_826446281_by_11_twice_l1723_172391

theorem multiply_826446281_by_11_twice :
  826446281 * 11 * 11 = 100000000001 := by
  sorry

end NUMINAMATH_CALUDE_multiply_826446281_by_11_twice_l1723_172391


namespace NUMINAMATH_CALUDE_f_sum_eq_two_l1723_172367

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * (Real.sin x) ^ 2 + b * Real.tan x + 1

theorem f_sum_eq_two (a b : ℝ) (h : f a b 2 = 5) : f a b (Real.pi - 2) + f a b Real.pi = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_eq_two_l1723_172367


namespace NUMINAMATH_CALUDE_power_five_mod_eleven_l1723_172308

theorem power_five_mod_eleven : 5^120 + 4 ≡ 5 [MOD 11] := by
  sorry

end NUMINAMATH_CALUDE_power_five_mod_eleven_l1723_172308


namespace NUMINAMATH_CALUDE_p_and_not_q_is_true_l1723_172385

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > Real.log x

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.exp x > 1

-- Theorem statement
theorem p_and_not_q_is_true : p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_p_and_not_q_is_true_l1723_172385


namespace NUMINAMATH_CALUDE_sin_2α_plus_π_6_l1723_172392

theorem sin_2α_plus_π_6 (α : ℝ) (h : Real.sin (α + π / 3) = 3 / 5) :
  Real.sin (2 * α + π / 6) = -(7 / 25) := by
  sorry

end NUMINAMATH_CALUDE_sin_2α_plus_π_6_l1723_172392


namespace NUMINAMATH_CALUDE_max_abs_difference_complex_l1723_172372

theorem max_abs_difference_complex (z₁ z₂ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : z₁ + z₂ = Complex.I * 2) : 
  ∃ (M : ℝ), M = 4 ∧ ∀ (w₁ w₂ : ℂ), Complex.abs w₁ = 1 → w₁ + w₂ = Complex.I * 2 → 
    Complex.abs (w₁ - w₂) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_abs_difference_complex_l1723_172372


namespace NUMINAMATH_CALUDE_cos_3theta_l1723_172356

theorem cos_3theta (θ : ℝ) : Complex.exp (θ * Complex.I) = (3 + Complex.I * Real.sqrt 2) / 4 → Complex.cos (3 * θ) = 9 / 64 := by
  sorry

end NUMINAMATH_CALUDE_cos_3theta_l1723_172356


namespace NUMINAMATH_CALUDE_product_of_primes_l1723_172331

theorem product_of_primes (a b c d : ℕ) : 
  Prime a ∧ Prime b ∧ Prime c ∧ Prime d ∧  -- a, b, c, d are prime
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧  -- a, b, c, d are distinct
  a + c = d ∧  -- condition (i)
  a * (a + b + c + d) = c * (d - b) ∧  -- condition (ii)
  1 + b * c + d = b * d  -- condition (iii)
  → a * b * c * d = 2002 := by
sorry

end NUMINAMATH_CALUDE_product_of_primes_l1723_172331


namespace NUMINAMATH_CALUDE_evaluate_expression_l1723_172394

theorem evaluate_expression : (-1 : ℤ) ^ (3^3) + 1 ^ (3^3) = 0 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1723_172394


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1723_172304

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℚ) (d : ℚ) (h1 : d ≠ 0) 
  (h2 : ∀ n : ℕ, a (n + 1) = a n + d)  -- arithmetic sequence definition
  (h3 : (a 3 - a 1) * (a 9 - a 3) = (a 3 - a 1)^2) :  -- geometric sequence condition
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13/16 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_ratio_l1723_172304


namespace NUMINAMATH_CALUDE_dogwood_trees_after_five_years_l1723_172354

/-- Calculates the total number of dogwood trees in the park after a given number of years -/
def total_dogwood_trees (initial_trees : ℕ) (trees_today : ℕ) (trees_tomorrow : ℕ) 
  (growth_rate_today : ℕ) (growth_rate_tomorrow : ℕ) (years : ℕ) : ℕ :=
  initial_trees + 
  (trees_today + growth_rate_today * years) + 
  (trees_tomorrow + growth_rate_tomorrow * years)

/-- Theorem stating that the total number of dogwood trees after 5 years is 130 -/
theorem dogwood_trees_after_five_years : 
  total_dogwood_trees 39 41 20 2 4 5 = 130 := by
  sorry

#eval total_dogwood_trees 39 41 20 2 4 5

end NUMINAMATH_CALUDE_dogwood_trees_after_five_years_l1723_172354


namespace NUMINAMATH_CALUDE_triangle_translation_l1723_172388

-- Define a point in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define a translation vector
structure Translation :=
  (dx : ℝ)
  (dy : ℝ)

-- Define a function to apply a translation to a point
def translate (p : Point) (t : Translation) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

-- The main theorem
theorem triangle_translation
  (A B C A' : Point)
  (h_A : A = { x := -1, y := -4 })
  (h_B : B = { x := 1, y := 1 })
  (h_C : C = { x := -1, y := 4 })
  (h_A' : A' = { x := 1, y := -1 })
  (h_translation : ∃ t : Translation, translate A t = A') :
  ∃ (B' C' : Point),
    B' = { x := 3, y := 4 } ∧
    C' = { x := 1, y := 7 } ∧
    translate B h_translation.choose = B' ∧
    translate C h_translation.choose = C' :=
sorry

end NUMINAMATH_CALUDE_triangle_translation_l1723_172388


namespace NUMINAMATH_CALUDE_f_properties_l1723_172373

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem f_properties :
  ∀ a : ℝ,
  (a = -1/2 → 
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), ∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≥ f a y) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), ∀ y ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x ≤ f a y) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = 1/2 + (Real.exp 1)^2/4) ∧
    (∃ x ∈ Set.Icc (1/Real.exp 1) (Real.exp 1), f a x = 5/4)) ∧
  ((a ≤ -1 → ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x > f a y) ∧
   (a ≥ 0 → ∀ x y : ℝ, 0 < x → 0 < y → x < y → f a x < f a y) ∧
   (-1 < a → a < 0 → 
     ∃ z : ℝ, 0 < z ∧ 
     (∀ x y : ℝ, 0 < x → x < y → y < z → f a x > f a y) ∧
     (∀ x y : ℝ, z ≤ x → x < y → f a x < f a y))) ∧
  (-1 < a → a < 0 → 
    (∀ x : ℝ, 0 < x → f a x > 1 + a / 2 * Real.log (-a)) ↔ 1/Real.exp 1 - 1 < a) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l1723_172373


namespace NUMINAMATH_CALUDE_triangle_circles_area_sum_l1723_172396

/-- The sum of the areas of three mutually externally tangent circles 
    centered at the vertices of a 6-8-10 right triangle is 56π. -/
theorem triangle_circles_area_sum : 
  ∀ (r s t : ℝ), 
    r + s = 6 →
    r + t = 8 →
    s + t = 10 →
    r > 0 → s > 0 → t > 0 →
    π * (r^2 + s^2 + t^2) = 56 * π := by
  sorry

end NUMINAMATH_CALUDE_triangle_circles_area_sum_l1723_172396


namespace NUMINAMATH_CALUDE_absent_children_l1723_172318

theorem absent_children (total_children : ℕ) (total_bananas : ℕ) : 
  total_children = 610 →
  total_bananas = 610 * 2 →
  total_bananas = (610 - (total_children - (610 - 305))) * 4 →
  610 - 305 = total_children - (610 - 305) :=
by
  sorry

end NUMINAMATH_CALUDE_absent_children_l1723_172318


namespace NUMINAMATH_CALUDE_existence_of_four_numbers_l1723_172346

theorem existence_of_four_numbers (x y : ℝ) : 
  ∃ (a₁ a₂ a₃ a₄ : ℝ), x = a₁ + a₂ + a₃ + a₄ ∧ y = 1/a₁ + 1/a₂ + 1/a₃ + 1/a₄ := by
  sorry

end NUMINAMATH_CALUDE_existence_of_four_numbers_l1723_172346


namespace NUMINAMATH_CALUDE_continued_fraction_solution_l1723_172369

theorem continued_fraction_solution :
  ∃ x : ℝ, x = 3 + 5 / (2 + 5 / x) ∧ x = (3 + Real.sqrt 69) / 2 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_solution_l1723_172369


namespace NUMINAMATH_CALUDE_tv_weekly_cost_l1723_172317

/-- Calculate the cost in cents to run a TV for a week -/
theorem tv_weekly_cost (tv_power : ℝ) (daily_usage : ℝ) (electricity_cost : ℝ) : 
  tv_power = 125 →
  daily_usage = 4 →
  electricity_cost = 14 →
  (tv_power * daily_usage * 7 / 1000 * electricity_cost) = 49 := by
  sorry

end NUMINAMATH_CALUDE_tv_weekly_cost_l1723_172317


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1723_172324

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | 0 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1723_172324


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l1723_172315

/-- An arithmetic sequence {a_n} satisfying the given conditions -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arith : ArithmeticSequence a) 
  (h_sum : a 4 + a 6 + a 8 + a 10 + a 12 = 60) : 
  a 7 - (1/3) * a 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l1723_172315
