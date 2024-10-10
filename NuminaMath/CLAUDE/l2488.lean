import Mathlib

namespace cupcake_distribution_l2488_248834

theorem cupcake_distribution (total_cupcakes : ℕ) (num_children : ℕ) 
  (h1 : total_cupcakes = 96) (h2 : num_children = 8) :
  total_cupcakes / num_children = 12 := by
  sorry

end cupcake_distribution_l2488_248834


namespace surface_area_of_modified_structure_l2488_248858

/-- Represents the dimensions of the large cube -/
def large_cube_dim : ℕ := 12

/-- Represents the dimensions of the small cubes -/
def small_cube_dim : ℕ := 2

/-- The total number of small cubes in the original structure -/
def total_small_cubes : ℕ := 64

/-- The number of small cubes removed from the structure -/
def removed_cubes : ℕ := 7

/-- The surface area of a single 2x2x2 cube before modification -/
def small_cube_surface_area : ℕ := 24

/-- The additional surface area exposed on each small cube after modification -/
def additional_exposed_area : ℕ := 6

/-- The surface area of a modified small cube -/
def modified_small_cube_area : ℕ := small_cube_surface_area + additional_exposed_area

/-- The theorem to be proved -/
theorem surface_area_of_modified_structure :
  (total_small_cubes - removed_cubes) * modified_small_cube_area = 1710 :=
sorry

end surface_area_of_modified_structure_l2488_248858


namespace hyperbola_equation_l2488_248825

/-- Given a hyperbola and a circle with specific properties, prove the equation of the hyperbola. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, y^2 / a^2 - x^2 / b^2 = 1) →
  (∃ x y : ℝ, x^2 + y^2 - 6*y + 5 = 0) →
  (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 6*y₀ + 5 = 0 ∧ 
    (∀ x y : ℝ, (y - y₀)^2 / a^2 - (x - x₀)^2 / b^2 = 1)) →
  a^2 = 5 ∧ b^2 = 4 :=
sorry

end hyperbola_equation_l2488_248825


namespace age_difference_l2488_248865

/-- Given a father and daughter whose ages sum to 54, and the daughter is 16 years old,
    prove that the difference between their ages is 22 years. -/
theorem age_difference (father_age daughter_age : ℕ) : 
  father_age + daughter_age = 54 →
  daughter_age = 16 →
  father_age - daughter_age = 22 := by
  sorry

end age_difference_l2488_248865


namespace problem_solution_l2488_248805

theorem problem_solution (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 143) : x = 15 := by
  sorry

end problem_solution_l2488_248805


namespace center_number_l2488_248840

/-- Represents a 3x3 grid with numbers from 1 to 9 --/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if two positions in the grid share an edge --/
def adjacent (p1 p2 : Fin 3 × Fin 3) : Prop :=
  (p1.1 = p2.1 ∧ (p1.2 = p2.2 + 1 ∨ p2.2 = p1.2 + 1)) ∨
  (p1.2 = p2.2 ∧ (p1.1 = p2.1 + 1 ∨ p2.1 = p1.1 + 1))

/-- Check if the grid satisfies the consecutive number constraint --/
def consecutive_constraint (g : Grid) : Prop :=
  ∀ n : Fin 9, ∃ p1 p2 : Fin 3 × Fin 3,
    g p1.1 p1.2 = n ∧ g p2.1 p2.2 = n + 1 ∧ adjacent p1 p2

/-- Sum of corner numbers in the grid --/
def corner_sum (g : Grid) : Nat :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The main theorem --/
theorem center_number (g : Grid) 
  (unique : ∀ i j k l : Fin 3, g i j = g k l → (i, j) = (k, l))
  (all_numbers : ∀ n : Fin 9, ∃ i j : Fin 3, g i j = n)
  (consec : consecutive_constraint g)
  (corners : corner_sum g = 20) :
  g 1 1 = 5 := by
  sorry

end center_number_l2488_248840


namespace second_graders_borrowed_books_l2488_248859

theorem second_graders_borrowed_books (initial_books borrowed_books : ℕ) : 
  initial_books = 75 → 
  initial_books - borrowed_books = 57 → 
  borrowed_books = 18 := by
sorry

end second_graders_borrowed_books_l2488_248859


namespace correct_distribution_l2488_248898

structure Participant where
  name : String
  initialDeposit : ℕ
  depositTime : ℕ

def totalValue : ℕ := 108000
def secondCarValue : ℕ := 48000
def secondCarSoldValue : ℕ := 42000

def calculateShare (p : Participant) (totalDays : ℕ) : ℚ :=
  (p.initialDeposit * p.depositTime : ℚ) / (totalDays * totalValue : ℚ)

def adjustedShare (share : ℚ) : ℚ :=
  share * (secondCarSoldValue : ℚ) / (secondCarValue : ℚ)

theorem correct_distribution 
  (istvan kalman laszlo miklos : Participant)
  (h1 : istvan.initialDeposit = 5000 + 4000 - 2500)
  (h2 : istvan.depositTime = 90)
  (h3 : kalman.initialDeposit = 4000)
  (h4 : kalman.depositTime = 70)
  (h5 : laszlo.initialDeposit = 2500)
  (h6 : laszlo.depositTime = 40)
  (h7 : miklos.initialDeposit = 2000)
  (h8 : miklos.depositTime = 90)
  : adjustedShare (calculateShare istvan 90) * secondCarValue = 54600 ∧
    adjustedShare (calculateShare kalman 90) * secondCarValue - 
    adjustedShare (calculateShare miklos 90) * secondCarValue = 7800 ∧
    adjustedShare (calculateShare laszlo 90) * secondCarValue = 10500 ∧
    adjustedShare (calculateShare miklos 90) * secondCarValue = 18900 := by
  sorry

#eval totalValue
#eval secondCarValue
#eval secondCarSoldValue

end correct_distribution_l2488_248898


namespace fraction_equality_l2488_248821

theorem fraction_equality (m n p r : ℚ) 
  (h1 : m / n = 21)
  (h2 : p / n = 7)
  (h3 : p / r = 1 / 7) :
  m / r = 3 / 7 := by
  sorry

end fraction_equality_l2488_248821


namespace even_heads_probability_l2488_248878

def probability_even_heads (p1 p2 : ℚ) (n1 n2 : ℕ) : ℚ :=
  let P1 := (1 + ((1 - 2*p1) / (1 - p1))^n1) / 2
  let P2 := (1 + ((1 - 2*p2) / (1 - p2))^n2) / 2
  P1 * P2 + (1 - P1) * (1 - P2)

theorem even_heads_probability :
  probability_even_heads (3/4) (1/2) 40 10 = 1/2 := by
  sorry

end even_heads_probability_l2488_248878


namespace triangular_number_difference_l2488_248833

/-- The n-th triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The difference between the 30th and 28th triangular numbers is 59 -/
theorem triangular_number_difference : triangular_number 30 - triangular_number 28 = 59 := by
  sorry

end triangular_number_difference_l2488_248833


namespace min_sum_is_twelve_l2488_248873

/-- Represents a 3x3 grid of integers -/
def Grid := Matrix (Fin 3) (Fin 3) ℕ

/-- Checks if a grid contains all numbers from 1 to 9 exactly once -/
def isValidGrid (g : Grid) : Prop :=
  ∀ n : ℕ, n ≥ 1 ∧ n ≤ 9 → (∃! (i j : Fin 3), g i j = n)

/-- Calculates the sum of a row in the grid -/
def rowSum (g : Grid) (i : Fin 3) : ℕ :=
  (g i 0) + (g i 1) + (g i 2)

/-- Calculates the sum of a column in the grid -/
def colSum (g : Grid) (j : Fin 3) : ℕ :=
  (g 0 j) + (g 1 j) + (g 2 j)

/-- Checks if all rows and columns in the grid have the same sum -/
def hasEqualSums (g : Grid) : Prop :=
  ∃ s : ℕ, (∀ i : Fin 3, rowSum g i = s) ∧ (∀ j : Fin 3, colSum g j = s)

/-- The main theorem: The minimum sum for a valid grid with equal sums is 12 -/
theorem min_sum_is_twelve :
  ∀ g : Grid, isValidGrid g → hasEqualSums g →
  ∃ s : ℕ, (∀ i : Fin 3, rowSum g i = s) ∧ (∀ j : Fin 3, colSum g j = s) ∧ s ≥ 12 :=
sorry

end min_sum_is_twelve_l2488_248873


namespace mass_of_iodine_l2488_248831

/-- The mass of 3 moles of I₂ given the atomic mass of I -/
theorem mass_of_iodine (atomic_mass_I : ℝ) (h : atomic_mass_I = 126.90) :
  let molar_mass_I2 := 2 * atomic_mass_I
  3 * molar_mass_I2 = 761.40 := by
  sorry

#check mass_of_iodine

end mass_of_iodine_l2488_248831


namespace equation_solution_l2488_248810

theorem equation_solution : 
  ∃! x : ℝ, x ≠ -4 ∧ (7 * x / (x + 4) - 5 / (x + 4) = 2 / (x + 4)) := by
  sorry

end equation_solution_l2488_248810


namespace units_digit_of_17_pow_2007_l2488_248849

theorem units_digit_of_17_pow_2007 : ∃ n : ℕ, 17^2007 ≡ 3 [ZMOD 10] :=
sorry

end units_digit_of_17_pow_2007_l2488_248849


namespace smallest_non_odd_units_digit_l2488_248896

def OddUnitsDigit : Set Nat := {1, 3, 5, 7, 9}
def AllDigits : Set Nat := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem smallest_non_odd_units_digit : 
  ∃ (d : Nat), d ∈ AllDigits ∧ d ∉ OddUnitsDigit ∧ 
  ∀ (x : Nat), x ∈ AllDigits ∧ x ∉ OddUnitsDigit → d ≤ x :=
by
  sorry

end smallest_non_odd_units_digit_l2488_248896


namespace no_solutions_in_interval_l2488_248881

theorem no_solutions_in_interval (x : ℝ) :
  x ∈ Set.Icc (π / 4) (π / 2) →
  ¬(Real.sin (x ^ Real.sin x) = Real.cos (x ^ Real.cos x)) :=
by sorry

end no_solutions_in_interval_l2488_248881


namespace min_sum_cube_relation_l2488_248811

theorem min_sum_cube_relation (m n : ℕ+) (h : 90 * m.val = n.val ^ 3) : 
  (∀ (x y : ℕ+), 90 * x.val = y.val ^ 3 → m.val + n.val ≤ x.val + y.val) → 
  m.val + n.val = 330 := by
sorry

end min_sum_cube_relation_l2488_248811


namespace no_isosceles_triangles_in_grid_l2488_248854

/-- Represents a point in a 2D grid --/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a triangle in the grid --/
structure GridTriangle where
  A : GridPoint
  B : GridPoint
  C : GridPoint

/-- Checks if a point is within the 5x5 grid --/
def isInGrid (p : GridPoint) : Prop :=
  1 ≤ p.x ∧ p.x ≤ 5 ∧ 1 ≤ p.y ∧ p.y ≤ 5

/-- Calculates the squared distance between two points --/
def squaredDistance (p1 p2 : GridPoint) : Int :=
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2

/-- Checks if a triangle is isosceles --/
def isIsosceles (t : GridTriangle) : Prop :=
  squaredDistance t.A t.B = squaredDistance t.A t.C ∨
  squaredDistance t.A t.B = squaredDistance t.B t.C ∨
  squaredDistance t.A t.C = squaredDistance t.B t.C

/-- The main theorem --/
theorem no_isosceles_triangles_in_grid :
  ∀ (A B : GridPoint),
    isInGrid A ∧ isInGrid B ∧
    A.y = B.y ∧ squaredDistance A B = 9 →
    ¬∃ (C : GridPoint), isInGrid C ∧ isIsosceles ⟨A, B, C⟩ := by
  sorry


end no_isosceles_triangles_in_grid_l2488_248854


namespace function_inequality_implies_a_bound_l2488_248856

theorem function_inequality_implies_a_bound 
  (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) 
  (hf : ∀ x, f x = x + a) 
  (hg : ∀ x, g x = x + 4/x) 
  (h : ∀ x₁ ∈ Set.Icc 1 3, ∃ x₂ ∈ Set.Icc 1 4, f x₁ ≥ g x₂) : 
  a ≥ 3 := by
sorry

end function_inequality_implies_a_bound_l2488_248856


namespace average_difference_number_of_elements_averaged_l2488_248853

/-- Given two real numbers with an average of 45, and two real numbers with an average of 90,
    prove that the difference between the third and first number is 90. -/
theorem average_difference (a b c : ℝ) (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 90) :
  c - a = 90 := by
  sorry

/-- The number of elements being averaged in both situations is 2. -/
theorem number_of_elements_averaged (n m : ℕ) 
  (h1 : ∃ (a b : ℝ), (a + b) / n = 45)
  (h2 : ∃ (b c : ℝ), (b + c) / m = 90) :
  n = 2 ∧ m = 2 := by
  sorry

end average_difference_number_of_elements_averaged_l2488_248853


namespace cos_alpha_minus_pi_third_l2488_248874

theorem cos_alpha_minus_pi_third (α : ℝ) 
  (h : Real.cos (α - π / 6) + Real.sin α = (4 / 5) * Real.sqrt 3) : 
  Real.cos (α - π / 3) = 4 / 5 := by
  sorry

end cos_alpha_minus_pi_third_l2488_248874


namespace cat_pictures_count_l2488_248827

/-- Represents the number of photos on Toby's camera roll at different stages -/
structure PhotoCount where
  initial : ℕ
  afterFirstDeletion : ℕ
  final : ℕ

/-- Represents the number of photos deleted or added at different stages -/
structure PhotoChanges where
  firstDeletion : ℕ
  catPictures : ℕ
  friendPhotos : ℕ
  secondDeletion : ℕ

/-- Theorem stating the relationship between cat pictures and friend photos -/
theorem cat_pictures_count (p : PhotoCount) (c : PhotoChanges) :
  p.initial = 63 →
  p.final = 84 →
  c.firstDeletion = 7 →
  c.secondDeletion = 3 →
  p.afterFirstDeletion = p.initial - c.firstDeletion →
  p.final = p.afterFirstDeletion + c.catPictures + c.friendPhotos - c.secondDeletion →
  c.catPictures = 31 - c.friendPhotos := by
  sorry


end cat_pictures_count_l2488_248827


namespace sum_of_yellow_and_blue_is_red_l2488_248850

theorem sum_of_yellow_and_blue_is_red (a b : ℕ) :
  ∃ m : ℕ, (4 * a + 3) + (4 * b + 2) = 4 * m + 1 := by
  sorry

end sum_of_yellow_and_blue_is_red_l2488_248850


namespace arithmetic_sequence_problem_l2488_248875

theorem arithmetic_sequence_problem (a b c : ℝ) : 
  (b - a = c - b) →  -- arithmetic sequence condition
  (a + b + c = 9) →  -- sum condition
  (a * b = 6 * c) →  -- product condition
  (a = 4 ∧ b = 3 ∧ c = 2) := by sorry

end arithmetic_sequence_problem_l2488_248875


namespace simplify_sqrt_one_minus_sin_two_alpha_l2488_248846

theorem simplify_sqrt_one_minus_sin_two_alpha (α : Real) 
  (h : π / 4 < α ∧ α < π / 2) : 
  Real.sqrt (1 - Real.sin (2 * α)) = Real.sin α - Real.cos α := by
  sorry

end simplify_sqrt_one_minus_sin_two_alpha_l2488_248846


namespace largest_equal_cost_under_500_equal_cost_242_largest_equal_cost_is_242_l2488_248800

/-- Convert a natural number to its base 3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec aux (m : ℕ) : List ℕ :=
    if m = 0 then [] else (m % 3) :: aux (m / 3)
  aux n |>.reverse

/-- Sum of digits in decimal representation -/
def sumDecimalDigits (n : ℕ) : ℕ :=
  let digits := n.repr.toList.map (λ c => c.toNat - '0'.toNat)
  digits.sum

/-- Sum of digits in base 3 representation -/
def sumBase3Digits (n : ℕ) : ℕ :=
  (toBase3 n).sum

/-- Predicate for numbers with equal cost in decimal and base 3 -/
def equalCost (n : ℕ) : Prop :=
  sumDecimalDigits n = sumBase3Digits n

theorem largest_equal_cost_under_500 :
  ∀ n : ℕ, n < 500 → n > 242 → ¬(equalCost n) :=
by sorry

theorem equal_cost_242 : equalCost 242 :=
by sorry

theorem largest_equal_cost_is_242 :
  ∃! n : ℕ, n < 500 ∧ equalCost n ∧ ∀ m : ℕ, m < 500 → equalCost m → m ≤ n :=
by sorry

end largest_equal_cost_under_500_equal_cost_242_largest_equal_cost_is_242_l2488_248800


namespace unique_prime_solution_l2488_248866

theorem unique_prime_solution : 
  ∀ p q r : ℕ, 
    Prime p → Prime q → Prime r → 
    p * q * r = 5 * (p + q + r) → 
    (p = 2 ∧ q = 5 ∧ r = 7) ∨ (p = 2 ∧ q = 7 ∧ r = 5) ∨ (p = 5 ∧ q = 2 ∧ r = 7) ∨ 
    (p = 5 ∧ q = 7 ∧ r = 2) ∨ (p = 7 ∧ q = 2 ∧ r = 5) ∨ (p = 7 ∧ q = 5 ∧ r = 2) :=
by
  sorry

#check unique_prime_solution

end unique_prime_solution_l2488_248866


namespace prob_at_least_one_of_two_is_eight_ninths_l2488_248836

/-- The number of candidates -/
def num_candidates : ℕ := 2

/-- The number of colleges -/
def num_colleges : ℕ := 3

/-- The probability of a candidate choosing any particular college -/
def prob_choose_college : ℚ := 1 / num_colleges

/-- The probability that both candidates choose the third college -/
def prob_both_choose_third : ℚ := prob_choose_college ^ num_candidates

/-- The probability that at least one of the first two colleges is selected -/
def prob_at_least_one_of_two : ℚ := 1 - prob_both_choose_third

theorem prob_at_least_one_of_two_is_eight_ninths :
  prob_at_least_one_of_two = 8 / 9 := by sorry

end prob_at_least_one_of_two_is_eight_ninths_l2488_248836


namespace unique_number_between_30_and_40_with_units_digit_2_l2488_248880

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_units_digit (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem unique_number_between_30_and_40_with_units_digit_2 :
  ∃! n : ℕ, is_two_digit n ∧ 30 < n ∧ n < 40 ∧ has_units_digit n 2 :=
by sorry

end unique_number_between_30_and_40_with_units_digit_2_l2488_248880


namespace ceiling_equality_abs_diff_l2488_248845

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- Main theorem
theorem ceiling_equality_abs_diff (x y : ℝ) :
  (∀ x y, ceiling x = ceiling y → |x - y| < 1) ∧
  (∃ x y, |x - y| < 1 ∧ ceiling x ≠ ceiling y) :=
by sorry

end ceiling_equality_abs_diff_l2488_248845


namespace sin_sum_product_identity_l2488_248804

theorem sin_sum_product_identity : 
  Real.sin (17 * π / 180) * Real.sin (223 * π / 180) + 
  Real.sin (253 * π / 180) * Real.sin (313 * π / 180) = 1 / 2 := by
  sorry

end sin_sum_product_identity_l2488_248804


namespace sum_of_roots_l2488_248806

theorem sum_of_roots (x : ℝ) : (x + 3) * (x - 4) = 12 → ∃ (x₁ x₂ : ℝ), 
  (x₁ + 3) * (x₁ - 4) = 12 ∧ 
  (x₂ + 3) * (x₂ - 4) = 12 ∧ 
  x₁ + x₂ = 1 := by
sorry

end sum_of_roots_l2488_248806


namespace sisters_sandcastle_height_is_half_foot_l2488_248885

/-- The height of Miki's sister's sandcastle given Miki's sandcastle height and the height difference -/
def sisters_sandcastle_height (mikis_height : ℝ) (height_difference : ℝ) : ℝ :=
  mikis_height - height_difference

/-- Theorem stating that Miki's sister's sandcastle height is 0.50 foot -/
theorem sisters_sandcastle_height_is_half_foot :
  sisters_sandcastle_height 0.83 0.33 = 0.50 := by
  sorry

end sisters_sandcastle_height_is_half_foot_l2488_248885


namespace linear_congruence_solvability_and_solutions_l2488_248863

/-- 
For integers a, b, and m > 0, this theorem states:
1. The congruence ax ≡ b (mod m) has solutions if and only if gcd(a,m) | b.
2. If solutions exist, they are of the form x = x₀ + k(m/d) for all integers k, 
   where d = gcd(a,m) and x₀ is a particular solution to (a/d)x ≡ (b/d) (mod m/d).
-/
theorem linear_congruence_solvability_and_solutions 
  (a b m : ℤ) (hm : m > 0) : 
  (∃ x, a * x ≡ b [ZMOD m]) ↔ (gcd a m ∣ b) ∧
  (∀ x, (a * x ≡ b [ZMOD m]) ↔ 
    ∃ (x₀ k : ℤ), x = x₀ + k * (m / gcd a m) ∧ 
    (a / gcd a m) * x₀ ≡ (b / gcd a m) [ZMOD (m / gcd a m)]) :=
by sorry

end linear_congruence_solvability_and_solutions_l2488_248863


namespace abc_inequality_l2488_248868

theorem abc_inequality (a b c : ℝ) (h : (1/4)*a^2 + (1/4)*b^2 + c^2 = 1) :
  -2 ≤ a*b + 2*b*c + 2*c*a ∧ a*b + 2*b*c + 2*c*a ≤ 4 := by
  sorry

end abc_inequality_l2488_248868


namespace norm_scalar_multiple_l2488_248839

theorem norm_scalar_multiple (v : ℝ × ℝ) (h : ‖v‖ = 5) : ‖(5 : ℝ) • v‖ = 25 := by
  sorry

end norm_scalar_multiple_l2488_248839


namespace tangent_line_problem_l2488_248818

theorem tangent_line_problem (a : ℝ) : 
  (∃ (k : ℝ), 
    (∃ (x₀ : ℝ), 
      (x₀^3 = k * (x₀ - 1)) ∧ 
      (a * x₀^2 + 15/4 * x₀ - 9 = k * (x₀ - 1)) ∧
      (3 * x₀^2 = k))) →
  (a = -25/64 ∨ a = -1) :=
sorry

end tangent_line_problem_l2488_248818


namespace inequalities_proof_l2488_248899

theorem inequalities_proof (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  (a^2 > b^2) ∧ (a^3 > b^3) ∧ (Real.sqrt (a - b) > Real.sqrt a - Real.sqrt b) := by
  sorry

end inequalities_proof_l2488_248899


namespace right_triangle_area_l2488_248815

/-- A right triangle PQR in the xy-plane with specific properties -/
structure RightTriangle where
  -- P, Q, R are points in ℝ²
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  -- PQR is a right triangle with right angle at R
  right_angle_at_R : (P.1 - R.1) * (Q.1 - R.1) + (P.2 - R.2) * (Q.2 - R.2) = 0
  -- Length of hypotenuse PQ is 50
  hypotenuse_length : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2
  -- Median through P lies along y = x - 2
  median_P : ∃ (t : ℝ), (P.1 + R.1) / 2 = t ∧ (P.2 + R.2) / 2 = t - 2
  -- Median through Q lies along y = 3x + 3
  median_Q : ∃ (t : ℝ), (Q.1 + R.1) / 2 = t ∧ (Q.2 + R.2) / 2 = 3 * t + 3

/-- The area of the right triangle PQR is 290 -/
theorem right_triangle_area (t : RightTriangle) : 
  abs ((t.P.1 - t.R.1) * (t.Q.2 - t.R.2) - (t.Q.1 - t.R.1) * (t.P.2 - t.R.2)) / 2 = 290 := by
  sorry

end right_triangle_area_l2488_248815


namespace yellow_shirt_pairs_l2488_248838

theorem yellow_shirt_pairs (blue_students : ℕ) (yellow_students : ℕ) (total_students : ℕ) (total_pairs : ℕ) (blue_blue_pairs : ℕ) : 
  blue_students = 57 →
  yellow_students = 75 →
  total_students = 132 →
  total_pairs = 66 →
  blue_blue_pairs = 23 →
  ∃ (yellow_yellow_pairs : ℕ), yellow_yellow_pairs = 32 :=
by
  sorry

end yellow_shirt_pairs_l2488_248838


namespace license_plate_increase_l2488_248861

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^4 * 10^4
  new_plates / old_plates = 6760 := by
sorry

end license_plate_increase_l2488_248861


namespace battleship_detectors_l2488_248848

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Represents a ship with width and height -/
structure Ship :=
  (width : ℕ)
  (height : ℕ)

/-- Function to calculate the minimum number of detectors required -/
def min_detectors (g : Grid) (s : Ship) : ℕ :=
  ((g.width - 1) / 3) * 2 + ((g.width - 1) % 3) + 1

/-- Theorem stating the minimum number of detectors for the Battleship problem -/
theorem battleship_detectors :
  let grid : Grid := ⟨203, 1⟩
  let ship : Ship := ⟨2, 1⟩
  min_detectors grid ship = 134 := by
  sorry

#check battleship_detectors

end battleship_detectors_l2488_248848


namespace balls_given_to_partner_l2488_248862

/-- Represents the number of tennis games played by Bertha -/
def games : ℕ := 20

/-- Represents the number of games after which one ball wears out -/
def wear_out_rate : ℕ := 10

/-- Represents the number of games after which Bertha loses a ball -/
def lose_rate : ℕ := 5

/-- Represents the number of games after which Bertha buys a canister of balls -/
def buy_rate : ℕ := 4

/-- Represents the number of balls in each canister -/
def balls_per_canister : ℕ := 3

/-- Represents the number of balls Bertha started with -/
def initial_balls : ℕ := 2

/-- Represents the number of balls Bertha has after 20 games -/
def final_balls : ℕ := 10

/-- Calculates the number of balls worn out during the games -/
def balls_worn_out : ℕ := games / wear_out_rate

/-- Calculates the number of balls lost during the games -/
def balls_lost : ℕ := games / lose_rate

/-- Calculates the number of balls bought during the games -/
def balls_bought : ℕ := (games / buy_rate) * balls_per_canister

/-- Theorem stating that Bertha gave 1 ball to her partner -/
theorem balls_given_to_partner :
  initial_balls + balls_bought - balls_worn_out - balls_lost - final_balls = 1 := by
  sorry

end balls_given_to_partner_l2488_248862


namespace max_value_sqrt_sum_l2488_248809

theorem max_value_sqrt_sum (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
  (ha2 : a ≤ 2) (hb2 : b ≤ 2) (hc2 : c ≤ 2) :
  Real.sqrt ((2 - a) * (2 - b) * (2 - c)) + Real.sqrt (a * b * c) ≤ 2 :=
sorry

end max_value_sqrt_sum_l2488_248809


namespace expression_evaluation_l2488_248887

/-- Evaluates the expression 2x^y + 5y^x - z^2 for given x, y, and z values -/
def evaluate (x y z : ℕ) : ℕ :=
  2 * (x ^ y) + 5 * (y ^ x) - (z ^ 2)

/-- Theorem stating that the expression evaluates to 42 for x=3, y=2, and z=4 -/
theorem expression_evaluation :
  evaluate 3 2 4 = 42 := by
  sorry

end expression_evaluation_l2488_248887


namespace tangent_line_equation_intersecting_line_equation_l2488_248890

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 8*y + 12 = 0

-- Define a line passing through point P(-2, 0)
def line_through_P (k : ℝ) (x y : ℝ) : Prop := y = k * (x + 2)

-- Theorem for tangent line
theorem tangent_line_equation :
  ∀ k : ℝ, (∀ x y : ℝ, circle_C x y → line_through_P k x y → (x = -2 ∨ y = (3/4)*x + 3/2)) ∧
            (∀ x y : ℝ, (x = -2 ∨ y = (3/4)*x + 3/2) → line_through_P k x y → 
             (∃! p : ℝ × ℝ, circle_C p.1 p.2 ∧ line_through_P k p.1 p.2)) :=
sorry

-- Theorem for intersecting line with chord length 2√2
theorem intersecting_line_equation :
  ∀ k : ℝ, (∀ x y : ℝ, circle_C x y → line_through_P k x y → 
            (x - y + 2 = 0 ∨ 7*x - y + 14 = 0)) ∧
           (∀ x y : ℝ, (x - y + 2 = 0 ∨ 7*x - y + 14 = 0) → line_through_P k x y → 
            (∃ A B : ℝ × ℝ, circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧ 
             line_through_P k A.1 A.2 ∧ line_through_P k B.1 B.2 ∧
             (A.1 - B.1)^2 + (A.2 - B.2)^2 = 8)) :=
sorry

end tangent_line_equation_intersecting_line_equation_l2488_248890


namespace expected_value_eight_sided_die_l2488_248855

def winnings (n : Nat) : Real := 8 - n

theorem expected_value_eight_sided_die :
  let outcomes := Finset.range 8
  let prob (k : Nat) := 1 / 8
  Finset.sum outcomes (fun k => prob k * winnings (k + 1)) = 3.5 := by sorry

end expected_value_eight_sided_die_l2488_248855


namespace garden_tilling_time_l2488_248888

/-- Calculates the time required to till a rectangular plot -/
def tillingTime (width : ℕ) (length : ℕ) (swathWidth : ℕ) (tillRate : ℚ) : ℚ :=
  let rows := width / swathWidth
  let totalDistance := rows * length
  let totalSeconds := totalDistance * tillRate
  totalSeconds / 60

theorem garden_tilling_time :
  tillingTime 110 120 2 2 = 220 := by
  sorry

end garden_tilling_time_l2488_248888


namespace smallest_x_value_l2488_248808

theorem smallest_x_value (x : ℝ) : 
  (3 * x^2 + 36 * x - 72 = x * (x + 20) + 8) → x ≥ -10 :=
by sorry

end smallest_x_value_l2488_248808


namespace isosceles_trapezoid_ratio_l2488_248877

/-- An isosceles trapezoid with specific properties -/
structure IsoscelesTrapezoid where
  /-- Length of the smaller base -/
  ab : ℝ
  /-- Length of the larger base -/
  cd : ℝ
  /-- Length of the diagonal AC -/
  ac : ℝ
  /-- Height of the trapezoid (altitude from D to AB) -/
  h : ℝ
  /-- The smaller base is less than the larger base -/
  ab_lt_cd : ab < cd
  /-- The diagonal AC is twice the length of the larger base CD -/
  ac_eq_2cd : ac = 2 * cd
  /-- The smaller base AB equals the height of the trapezoid -/
  ab_eq_h : ab = h

/-- The ratio of the smaller base to the larger base in the specific isosceles trapezoid is 3:1 -/
theorem isosceles_trapezoid_ratio (t : IsoscelesTrapezoid) : t.ab / t.cd = 3 := by
  sorry

end isosceles_trapezoid_ratio_l2488_248877


namespace log_roots_sum_l2488_248889

theorem log_roots_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h : 2 * (Real.log a)^2 + 4 * (Real.log a) + 1 = 0 ∧ 
       2 * (Real.log b)^2 + 4 * (Real.log b) + 1 = 0) : 
  (Real.log a)^2 + Real.log (a^2) + a * b = Real.exp (-2) - 1/2 := by
  sorry

end log_roots_sum_l2488_248889


namespace calculate_expression_l2488_248872

theorem calculate_expression : 
  4 * Real.sin (60 * π / 180) + (-1/3)⁻¹ - Real.sqrt 12 + abs (-5) = 2 := by
  sorry

end calculate_expression_l2488_248872


namespace equation_one_solutions_equation_two_solutions_l2488_248847

-- Equation 1: x^2 + 4x - 1 = 0
theorem equation_one_solutions (x : ℝ) :
  x^2 + 4*x - 1 = 0 ↔ x = Real.sqrt 5 - 2 ∨ x = -Real.sqrt 5 - 2 := by sorry

-- Equation 2: (x-1)^2 = 3(x-1)
theorem equation_two_solutions (x : ℝ) :
  (x - 1)^2 = 3*(x - 1) ↔ x = 1 ∨ x = 4 := by sorry

end equation_one_solutions_equation_two_solutions_l2488_248847


namespace percentage_increase_l2488_248884

theorem percentage_increase (x : ℝ) (h : x = 99.9) :
  (x - 90) / 90 * 100 = 11 := by
  sorry

end percentage_increase_l2488_248884


namespace problem_solution_l2488_248876

theorem problem_solution (x : ℝ) (hx : x^2 + 9 * (x / (x - 3))^2 = 72) :
  let y := ((x - 3)^2 * (x + 4)) / (3*x - 4)
  y = 2 ∨ y = 6 := by
sorry

end problem_solution_l2488_248876


namespace scalene_triangle_not_unique_l2488_248897

/-- Represents a scalene triangle -/
structure ScaleneTriangle where
  -- We don't need to define the specific properties of a scalene triangle here
  -- as it's not relevant for this particular proof

/-- Represents the circumscribed circle of a triangle -/
structure CircumscribedCircle where
  radius : ℝ

/-- States that a scalene triangle is not uniquely determined by two of its angles
    and the radius of its circumscribed circle -/
theorem scalene_triangle_not_unique (α β : ℝ) (r : CircumscribedCircle) :
  ∃ (t1 t2 : ScaleneTriangle), t1 ≠ t2 ∧
  (∃ (γ1 γ2 : ℝ), α + β + γ1 = π ∧ α + β + γ2 = π) :=
sorry

end scalene_triangle_not_unique_l2488_248897


namespace isosceles_side_in_equilateral_l2488_248802

/-- The length of a side of an isosceles triangle inscribed in an equilateral triangle -/
theorem isosceles_side_in_equilateral (s : ℝ) (h : s = 2) :
  let equilateral_side := s
  let isosceles_base := equilateral_side / 2
  let isosceles_side := Real.sqrt (7 / 3)
  ∃ (triangle : Set (ℝ × ℝ)),
    (∀ p ∈ triangle, p.1 ≥ 0 ∧ p.1 ≤ equilateral_side ∧ p.2 ≥ 0 ∧ p.2 ≤ equilateral_side * Real.sqrt 3 / 2) ∧
    (∃ (a b c : ℝ × ℝ), a ∈ triangle ∧ b ∈ triangle ∧ c ∈ triangle ∧
      (a.1 - b.1)^2 + (a.2 - b.2)^2 = equilateral_side^2 ∧
      (b.1 - c.1)^2 + (b.2 - c.2)^2 = equilateral_side^2 ∧
      (c.1 - a.1)^2 + (c.2 - a.2)^2 = equilateral_side^2) ∧
    (∃ (p q r : ℝ × ℝ), p ∈ triangle ∧ q ∈ triangle ∧ r ∈ triangle ∧
      (p.1 - q.1)^2 + (p.2 - q.2)^2 = isosceles_side^2 ∧
      (q.1 - r.1)^2 + (q.2 - r.2)^2 = isosceles_side^2 ∧
      (r.1 - p.1)^2 + (r.2 - p.2)^2 = isosceles_base^2) := by
  sorry


end isosceles_side_in_equilateral_l2488_248802


namespace combinable_with_sqrt_three_l2488_248886

theorem combinable_with_sqrt_three : ∃! x : ℝ, x > 0 ∧ 
  (x = Real.sqrt (3^2) ∨ x = Real.sqrt 27 ∨ x = Real.sqrt 30 ∨ x = Real.sqrt (2/3)) ∧
  ∃ (r : ℚ), x = r * Real.sqrt 3 := by
sorry

end combinable_with_sqrt_three_l2488_248886


namespace least_k_for_inequality_l2488_248879

theorem least_k_for_inequality : ∃ k : ℤ, k = 5 ∧ 
  (∀ n : ℤ, 0.0010101 * (10 : ℝ)^n > 10 → n ≥ k) ∧
  (0.0010101 * (10 : ℝ)^k > 10) := by
  sorry

end least_k_for_inequality_l2488_248879


namespace sum_of_roots_quadratic_l2488_248869

theorem sum_of_roots_quadratic (z : ℂ) : 
  (∃ z₁ z₂ : ℂ, z₁ + z₂ = 16 ∧ z₁ * z₂ = 15 ∧ z₁ ≠ z₂ ∧ z^2 - 16*z + 15 = 0 → z = z₁ ∨ z = z₂) :=
by sorry

end sum_of_roots_quadratic_l2488_248869


namespace solve_for_a_l2488_248832

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | -5/3 < x ∧ x < 1/3}

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := |a * x - 2| < 3

-- Theorem statement
theorem solve_for_a : 
  (∀ x, x ∈ solution_set a ↔ inequality a x) → a = -3 :=
sorry

end solve_for_a_l2488_248832


namespace complex_expression_equality_l2488_248822

theorem complex_expression_equality :
  (7 - 3*Complex.I) - 3*(2 - 5*Complex.I) + (1 + 2*Complex.I) = 2 + 14*Complex.I :=
by sorry

end complex_expression_equality_l2488_248822


namespace max_gcd_consecutive_terms_l2488_248807

def b (n : ℕ) : ℕ := 2^n * n.factorial + n

theorem max_gcd_consecutive_terms :
  ∀ n : ℕ, ∃ m : ℕ, m ≤ n → Nat.gcd (b m) (b (m + 1)) = 1 ∧
  ∀ k : ℕ, k ≤ n → Nat.gcd (b k) (b (k + 1)) ≤ 1 :=
sorry

end max_gcd_consecutive_terms_l2488_248807


namespace union_covers_reals_l2488_248844

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ -3 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x ≤ a - 1}

-- State the theorem
theorem union_covers_reals (a : ℝ) : A ∪ B a = Set.univ → a ≥ 3 := by
  sorry

end union_covers_reals_l2488_248844


namespace cos_2x_values_l2488_248860

theorem cos_2x_values (x : ℝ) (h : Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x - 3 * Real.cos x ^ 2 = 0) :
  Real.cos (2 * x) = -4/5 ∨ Real.cos (2 * x) = 0 := by
  sorry

end cos_2x_values_l2488_248860


namespace flower_beds_count_l2488_248852

theorem flower_beds_count (total_seeds : ℕ) (seeds_per_bed : ℕ) (h1 : total_seeds = 54) (h2 : seeds_per_bed = 6) :
  total_seeds / seeds_per_bed = 9 := by
  sorry

end flower_beds_count_l2488_248852


namespace ab_gt_ac_l2488_248828

theorem ab_gt_ac (a b c : ℝ) (h1 : c < b) (h2 : b < a) (h3 : a * c < 0) : a * b > a * c := by
  sorry

end ab_gt_ac_l2488_248828


namespace complex_magnitude_l2488_248826

theorem complex_magnitude (z : ℂ) (h : z * (1 - Complex.I) = 1 + Complex.I) : 
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end complex_magnitude_l2488_248826


namespace geometric_and_arithmetic_sequences_l2488_248894

-- Define the geometric sequence a_n
def a (n : ℕ) : ℝ := 2^n

-- Define the arithmetic sequence b_n
def b (n : ℕ) : ℝ := 12 * n - 28

-- Define the sum of the first n terms of b_n
def S (n : ℕ) : ℝ := 6 * n^2 - 22 * n

theorem geometric_and_arithmetic_sequences :
  (a 1 = 2) ∧ 
  (a 4 = 16) ∧ 
  (∀ n : ℕ, a n = 2^n) ∧
  (b 3 = a 3) ∧
  (b 5 = a 5) ∧
  (∀ n : ℕ, b n = 12 * n - 28) ∧
  (∀ n : ℕ, S n = 6 * n^2 - 22 * n) :=
by sorry

end geometric_and_arithmetic_sequences_l2488_248894


namespace unique_prime_pair_l2488_248819

def f (x : ℕ) : ℕ := x^2 + x + 1

theorem unique_prime_pair : 
  ∃! p q : ℕ, Prime p ∧ Prime q ∧ f p = f q + 242 ∧ p = 61 ∧ q = 59 := by
  sorry

end unique_prime_pair_l2488_248819


namespace yoongi_result_l2488_248891

theorem yoongi_result (x : ℝ) (h : x / 10 = 6) : x - 15 = 45 := by
  sorry

end yoongi_result_l2488_248891


namespace inequality_equivalence_l2488_248801

theorem inequality_equivalence (x : ℝ) : 
  (x / 4 ≤ 3 + 2 * x ∧ 3 + 2 * x < -3 * (1 + x)) ↔ 
  (x ≥ -12 / 7 ∧ x < -6 / 5) := by sorry

end inequality_equivalence_l2488_248801


namespace complex_number_quadrant_l2488_248823

theorem complex_number_quadrant (z : ℂ) : iz = -1 + I → z.re > 0 ∧ z.im > 0 :=
by sorry

end complex_number_quadrant_l2488_248823


namespace meaningful_fraction_l2488_248816

theorem meaningful_fraction (x : ℝ) : 
  (∃ y : ℝ, y = 1 / (x - 1)) ↔ x ≠ 1 := by sorry

end meaningful_fraction_l2488_248816


namespace area_difference_S_R_l2488_248814

/-- A square with side length 2 -/
def square : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2}

/-- An isosceles right triangle with legs of length 2 -/
def isoscelesRightTriangle : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 ∧ p.1 + p.2 ≤ 2}

/-- Region R: union of the square and 12 isosceles right triangles -/
def R : Set (ℝ × ℝ) := sorry

/-- Region S: smallest convex polygon containing R -/
def S : Set (ℝ × ℝ) := sorry

/-- The area of a set in ℝ² -/
noncomputable def area (A : Set (ℝ × ℝ)) : ℝ := sorry

theorem area_difference_S_R : area S - area R = 36 := by sorry

end area_difference_S_R_l2488_248814


namespace evaluate_expression_l2488_248820

theorem evaluate_expression : 5000 * (5000 ^ 1000) = 5000 ^ 1001 := by
  sorry

end evaluate_expression_l2488_248820


namespace bananas_to_pears_cost_equivalence_l2488_248824

/-- Given the cost relationships between bananas, apples, and pears at Lucy's Local Market,
    this theorem proves that 25 bananas cost as much as 10 pears. -/
theorem bananas_to_pears_cost_equivalence 
  (banana_apple_ratio : (5 : ℚ) * banana_cost = (3 : ℚ) * apple_cost)
  (apple_pear_ratio : (9 : ℚ) * apple_cost = (6 : ℚ) * pear_cost)
  (banana_cost apple_cost pear_cost : ℚ) :
  (25 : ℚ) * banana_cost = (10 : ℚ) * pear_cost :=
by sorry


end bananas_to_pears_cost_equivalence_l2488_248824


namespace derivative_inequality_l2488_248803

theorem derivative_inequality (a : ℝ) (ha : a > 0) (x : ℝ) (hx : x ≥ 1) :
  let f : ℝ → ℝ := λ x => a * Real.log x + x + 2
  (deriv f) x < x^2 + (a + 2) * x + 1 := by
  sorry

end derivative_inequality_l2488_248803


namespace helen_raisin_cookies_l2488_248812

/-- The number of chocolate chip cookies Helen baked yesterday -/
def yesterday_chocolate : ℕ := 519

/-- The number of raisin cookies Helen baked yesterday -/
def yesterday_raisin : ℕ := 300

/-- The number of chocolate chip cookies Helen baked today -/
def today_chocolate : ℕ := 359

/-- The difference in raisin cookies baked between yesterday and today -/
def raisin_difference : ℕ := 20

/-- The number of raisin cookies Helen baked today -/
def today_raisin : ℕ := yesterday_raisin - raisin_difference

theorem helen_raisin_cookies : today_raisin = 280 := by
  sorry

end helen_raisin_cookies_l2488_248812


namespace marble_probability_l2488_248837

/-- The probability of drawing either a green or black marble from a bag -/
theorem marble_probability (green black white : ℕ) 
  (h_green : green = 4)
  (h_black : black = 3)
  (h_white : white = 6) :
  (green + black : ℚ) / (green + black + white) = 7 / 13 := by
  sorry

end marble_probability_l2488_248837


namespace apples_bought_l2488_248830

/-- The number of apples bought by Cecile, Diane, and Emily -/
def total_apples (cecile diane emily : ℕ) : ℕ := cecile + diane + emily

/-- The theorem stating the total number of apples bought -/
theorem apples_bought (cecile diane emily : ℕ) 
  (h1 : cecile = 15)
  (h2 : diane = cecile + 20)
  (h3 : emily = ((cecile + diane) * 13) / 10) :
  total_apples cecile diane emily = 115 := by
  sorry

#check apples_bought

end apples_bought_l2488_248830


namespace set_inclusion_conditions_l2488_248829

def P : Set ℝ := {x | x^2 + 4*x = 0}
def Q (m : ℝ) : Set ℝ := {x | x^2 + 2*(m+1)*x + m^2 - 1 = 0}

theorem set_inclusion_conditions :
  (∀ m : ℝ, P ⊆ Q m ↔ m = 1) ∧
  (∀ m : ℝ, Q m ⊆ P ↔ m ≤ -1 ∨ m = 1) := by sorry

end set_inclusion_conditions_l2488_248829


namespace cubic_equation_solution_l2488_248864

theorem cubic_equation_solution :
  ∀ (x y : ℤ), y^2 = x^3 - 3*x + 2 ↔ ∃ (k : ℕ), x = k^2 - 2 ∧ (y = k*(k^2 - 3) ∨ y = -k*(k^2 - 3)) :=
by sorry

end cubic_equation_solution_l2488_248864


namespace pen_cost_l2488_248893

/-- Given the cost of pens and pencils in two different combinations, 
    prove that the cost of a single pen is 39 cents. -/
theorem pen_cost (x y : ℚ) 
  (eq1 : 3 * x + 2 * y = 183) 
  (eq2 : 5 * x + 4 * y = 327) : 
  x = 39 := by
  sorry

end pen_cost_l2488_248893


namespace green_paint_amount_l2488_248842

/-- Paint mixture ratios -/
structure PaintMixture where
  blue : ℚ
  green : ℚ
  white : ℚ
  red : ℚ

/-- Theorem: Given a paint mixture with ratio 5:3:4:2 for blue:green:white:red,
    if 10 quarts of blue paint are used, then 6 quarts of green paint should be used. -/
theorem green_paint_amount (mix : PaintMixture) 
  (ratio : mix.blue = 5 ∧ mix.green = 3 ∧ mix.white = 4 ∧ mix.red = 2) 
  (blue_amount : ℚ) (h : blue_amount = 10) : 
  (blue_amount * mix.green / mix.blue) = 6 := by
  sorry

end green_paint_amount_l2488_248842


namespace modulo_residue_problem_l2488_248841

theorem modulo_residue_problem : (513 + 3 * 68 + 9 * 289 + 2 * 34 - 10) % 17 = 7 := by
  sorry

end modulo_residue_problem_l2488_248841


namespace angle_counterexample_l2488_248851

theorem angle_counterexample : ∃ (angle1 angle2 : ℝ), 
  angle1 + angle2 = 90 ∧ angle1 = angle2 := by
  sorry

end angle_counterexample_l2488_248851


namespace ages_sum_l2488_248882

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * a * c = 162 → a + b + c = 20 := by
  sorry

end ages_sum_l2488_248882


namespace tan_sum_x_y_pi_third_l2488_248871

theorem tan_sum_x_y_pi_third (x y m : ℝ) 
  (hx : x^3 + Real.sin (2*x) = m)
  (hy : y^3 + Real.sin (2*y) = -m)
  (hx_bound : x > -π/4 ∧ x < π/4)
  (hy_bound : y > -π/4 ∧ y < π/4) :
  Real.tan (x + y + π/3) = Real.sqrt 3 := by
  sorry

end tan_sum_x_y_pi_third_l2488_248871


namespace uncovered_side_length_l2488_248817

/-- Represents a rectangular field with three sides fenced -/
structure FencedField where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing : ℝ

/-- The uncovered side of a fenced field is 40 feet given the conditions -/
theorem uncovered_side_length (field : FencedField)
  (h_area : field.area = 680)
  (h_fencing : field.fencing = 74)
  (h_area_calc : field.area = field.length * field.width)
  (h_fencing_calc : field.fencing = 2 * field.width + field.length) :
  field.length = 40 := by
  sorry

end uncovered_side_length_l2488_248817


namespace total_sums_attempted_l2488_248892

/-- Given a student's math problem attempt results, calculate the total number of sums attempted. -/
theorem total_sums_attempted (right_sums wrong_sums : ℕ) 
  (h1 : wrong_sums = 2 * right_sums) 
  (h2 : right_sums = 16) : 
  right_sums + wrong_sums = 48 :=
by sorry

end total_sums_attempted_l2488_248892


namespace angle_sum_BD_l2488_248835

-- Define the triangle and its angles
structure Triangle (A B C : Type) where
  angleA : ℝ
  angleB : ℝ
  angleC : ℝ

-- Define the configuration
structure Configuration where
  angleA : ℝ
  angleAFG : ℝ
  angleAGF : ℝ
  angleB : ℝ
  angleD : ℝ

-- Theorem statement
theorem angle_sum_BD (config : Configuration) 
  (h1 : config.angleA = 30)
  (h2 : config.angleAFG = config.angleAGF) :
  config.angleB + config.angleD = 75 := by
  sorry


end angle_sum_BD_l2488_248835


namespace inequality_proof_l2488_248857

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a * b) / (a^5 + a * b + b^5) + (b * c) / (b^5 + b * c + c^5) + (c * a) / (c^5 + c * a + a^5) ≤ 1 :=
by sorry

end inequality_proof_l2488_248857


namespace class_size_l2488_248870

theorem class_size (hindi : ℕ) (english : ℕ) (both : ℕ) (total : ℕ) : 
  hindi = 30 → 
  english = 20 → 
  both ≥ 10 → 
  total = hindi + english - both → 
  total = 40 := by
sorry

end class_size_l2488_248870


namespace y1_less_than_y2_l2488_248895

/-- The line equation y = -3x + 4 -/
def line_equation (x y : ℝ) : Prop := y = -3 * x + 4

/-- Point A with coordinates (2, y₁) -/
def point_A (y₁ : ℝ) : ℝ × ℝ := (2, y₁)

/-- Point B with coordinates (-1, y₂) -/
def point_B (y₂ : ℝ) : ℝ × ℝ := (-1, y₂)

theorem y1_less_than_y2 (y₁ y₂ : ℝ) 
  (hA : line_equation (point_A y₁).1 (point_A y₁).2)
  (hB : line_equation (point_B y₂).1 (point_B y₂).2) :
  y₁ < y₂ := by
  sorry

end y1_less_than_y2_l2488_248895


namespace cow_spots_multiple_l2488_248843

/-- 
Given a cow with spots on both sides:
* The left side has 16 spots
* The total number of spots is 71
* The right side has 16x + 7 spots, where x is some multiple

Prove that x = 3
-/
theorem cow_spots_multiple (x : ℚ) : 
  16 + (16 * x + 7) = 71 → x = 3 := by sorry

end cow_spots_multiple_l2488_248843


namespace similar_cube_volume_l2488_248867

theorem similar_cube_volume (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 27 → scale_factor = 2 → 
  (scale_factor^3 * original_volume : ℝ) = 216 := by
  sorry

end similar_cube_volume_l2488_248867


namespace locus_of_circle_centers_is_hyperbola_l2488_248883

/-- The locus of points equidistant from two fixed points forms a hyperbola --/
theorem locus_of_circle_centers_is_hyperbola 
  (M : ℝ × ℝ) -- Point M(x, y)
  (C₁ : ℝ × ℝ := (0, -1)) -- Center of circle C₁
  (C₂ : ℝ × ℝ := (0, 4)) -- Center of circle C₂
  (h : Real.sqrt ((M.1 - C₂.1)^2 + (M.2 - C₂.2)^2) - 
       Real.sqrt ((M.1 - C₁.1)^2 + (M.2 - C₁.2)^2) = 1) :
  -- The statement that M lies on a hyperbola
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ 
    (M.1^2 / a^2) - (M.2^2 / b^2) = 1 :=
sorry

end locus_of_circle_centers_is_hyperbola_l2488_248883


namespace remainder_2_pow_1984_mod_17_l2488_248813

theorem remainder_2_pow_1984_mod_17 (h1 : Prime 17) (h2 : ¬ 17 ∣ 2) :
  2^1984 ≡ 0 [ZMOD 17] := by
  sorry

end remainder_2_pow_1984_mod_17_l2488_248813
