import Mathlib

namespace ice_cream_scoop_arrangements_l417_41762

theorem ice_cream_scoop_arrangements : (Finset.range 5).card.factorial = 120 := by
  sorry

end ice_cream_scoop_arrangements_l417_41762


namespace parallelogram_area_l417_41722

/-- The area of a parallelogram with vertices at (0, 0), (3, 0), (1, 5), and (4, 5) is 15 square units. -/
theorem parallelogram_area : 
  let v1 : ℝ × ℝ := (0, 0)
  let v2 : ℝ × ℝ := (3, 0)
  let v3 : ℝ × ℝ := (1, 5)
  let v4 : ℝ × ℝ := (4, 5)
  let base : ℝ := v2.1 - v1.1
  let height : ℝ := v3.2 - v1.2
  base * height = 15 := by sorry

end parallelogram_area_l417_41722


namespace polynomial_coefficient_problem_l417_41742

theorem polynomial_coefficient_problem (x a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ a₈ a₉ : ℝ) 
  (h1 : (x + a)^9 = a₀ + a₁*(x+1) + a₂*(x+1)^2 + a₃*(x+1)^3 + a₄*(x+1)^4 + 
                    a₅*(x+1)^5 + a₆*(x+1)^6 + a₇*(x+1)^7 + a₈*(x+1)^8 + a₉*(x+1)^9)
  (h2 : a₅ = 126) :
  a = 0 ∨ a = 2 := by sorry

end polynomial_coefficient_problem_l417_41742


namespace pencil_distribution_l417_41768

theorem pencil_distribution (total_pencils : ℕ) (num_people : ℕ) 
  (h1 : total_pencils = 24) 
  (h2 : num_people = 3) : 
  total_pencils / num_people = 8 := by
  sorry

end pencil_distribution_l417_41768


namespace odd_integers_square_divisibility_l417_41756

theorem odd_integers_square_divisibility (m n : ℤ) :
  Odd m → Odd n → (m^2 - n^2 + 1) ∣ (n^2 - 1) → ∃ k : ℤ, m^2 - n^2 + 1 = k^2 := by
  sorry

end odd_integers_square_divisibility_l417_41756


namespace logistics_service_assignments_logistics_service_assignments_proof_l417_41755

theorem logistics_service_assignments : ℕ :=
  let total_students : ℕ := 5
  let total_athletes : ℕ := 3
  let athlete_A_in_own_team : Bool := true

  50

theorem logistics_service_assignments_proof :
  logistics_service_assignments = 50 := by
  sorry

end logistics_service_assignments_logistics_service_assignments_proof_l417_41755


namespace smallest_egg_solution_l417_41724

/-- Represents the egg selling scenario over 5 days -/
def egg_selling (initial_eggs : ℕ) (sold_per_day : ℕ) : Prop :=
  ∃ (remaining : ℕ → ℕ),
    remaining 0 = initial_eggs ∧
    remaining 1 = initial_eggs - sold_per_day ∧
    remaining 2 = 2 * (remaining 1) - sold_per_day ∧
    remaining 3 = 3 * (remaining 2) - sold_per_day ∧
    remaining 4 = 4 * (remaining 3) - sold_per_day ∧
    5 * (remaining 4) - sold_per_day = 0

/-- The smallest initial number of eggs and the number sold per day -/
theorem smallest_egg_solution :
  egg_selling 103 60 ∧
  ∀ n s, egg_selling n s → n ≥ 103 := by
  sorry

end smallest_egg_solution_l417_41724


namespace tan_alpha_2_implies_fraction_eq_two_thirds_l417_41784

theorem tan_alpha_2_implies_fraction_eq_two_thirds (α : Real) 
  (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2/3 := by
  sorry

end tan_alpha_2_implies_fraction_eq_two_thirds_l417_41784


namespace simplify_expression_square_root_of_expression_l417_41787

-- Part 1
theorem simplify_expression (x : ℝ) (h : 1 < x ∧ x < 4) :
  Real.sqrt ((1 - x)^2) - abs (x - 5) = 2 * x - 6 := by sorry

-- Part 2
theorem square_root_of_expression (x y : ℝ) (h : y = 1 + Real.sqrt (2*x - 1) + Real.sqrt (1 - 2*x)) :
  Real.sqrt (2*x + 3*y) = 2 ∨ Real.sqrt (2*x + 3*y) = -2 := by sorry

end simplify_expression_square_root_of_expression_l417_41787


namespace negation_of_existence_proposition_l417_41778

theorem negation_of_existence_proposition :
  (¬ ∃ x : ℝ, x^2 + 1 < 0) ↔ (∀ x : ℝ, x^2 + 1 ≥ 0) :=
by sorry

end negation_of_existence_proposition_l417_41778


namespace det_2x2_matrix_l417_41738

def matrix_det (x y : ℝ) : ℝ := x * y - 6

theorem det_2x2_matrix (x y : ℝ) :
  Matrix.det ![![x, 2], ![3, y]] = matrix_det x y := by
  sorry

end det_2x2_matrix_l417_41738


namespace triangle_abc_properties_l417_41798

theorem triangle_abc_properties (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  c = 2 →
  Real.sin C * (Real.cos B - Real.sqrt 3 * Real.sin B) = Real.sin A →
  Real.cos A = 2 * Real.sqrt 2 / 3 →
  -- Conclusions
  C = 5 * π / 6 ∧
  b = (4 * Real.sqrt 2 - 2 * Real.sqrt 3) / 3 := by
  sorry

end triangle_abc_properties_l417_41798


namespace power_function_through_point_and_value_l417_41709

/-- A power function that passes through the point (2,8) -/
def f (x : ℝ) : ℝ := x^3

theorem power_function_through_point_and_value : 
  f 2 = 8 ∧ ∃ x : ℝ, f x = 27 ∧ x = 3 := by
  sorry

end power_function_through_point_and_value_l417_41709


namespace roots_sum_inverse_squares_l417_41764

theorem roots_sum_inverse_squares (a b c : ℝ) (r s : ℂ) (h₁ : a ≠ 0) (h₂ : c ≠ 0) 
  (h₃ : a * r^2 + b * r - c = 0) (h₄ : a * s^2 + b * s - c = 0) : 
  1 / r^2 + 1 / s^2 = (b^2 + 2*a*c) / c^2 := by
sorry

end roots_sum_inverse_squares_l417_41764


namespace greatest_b_for_nonrange_l417_41702

theorem greatest_b_for_nonrange (b : ℤ) : (∀ x : ℝ, x^2 + b*x + 20 ≠ 5) ↔ b ≤ 7 :=
sorry

end greatest_b_for_nonrange_l417_41702


namespace max_value_sqrt_sum_l417_41782

theorem max_value_sqrt_sum (x y z : ℝ) 
  (sum_eq_one : x + y + z = 1)
  (x_ge : x ≥ -1/3)
  (y_ge : y ≥ -1)
  (z_ge : z ≥ -5/3) :
  ∃ (max : ℝ), max = 6 ∧ 
    ∀ (a b c : ℝ), a + b + c = 1 → a ≥ -1/3 → b ≥ -1 → c ≥ -5/3 →
      Real.sqrt (3 * a + 1) + Real.sqrt (3 * b + 3) + Real.sqrt (3 * c + 5) ≤ max ∧
      Real.sqrt (3 * x + 1) + Real.sqrt (3 * y + 3) + Real.sqrt (3 * z + 5) = max :=
by
  sorry

end max_value_sqrt_sum_l417_41782


namespace ellipse_eccentricity_special_case_l417_41700

/-- An ellipse with focal length equal to minor axis length has eccentricity √2/2 -/
theorem ellipse_eccentricity_special_case (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a > b) (h5 : c = b) (h6 : a^2 = b^2 + c^2) : 
  (c / a) = Real.sqrt 2 / 2 := by
sorry

end ellipse_eccentricity_special_case_l417_41700


namespace at_most_one_solution_l417_41759

/-- The floor function, mapping a real number to its integer part -/
noncomputable def floor (x : ℝ) : ℤ :=
  Int.floor x

/-- Theorem stating that the equation ax + b⌊x⌋ - c = 0 has at most one solution -/
theorem at_most_one_solution (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃! x, a * x + b * (floor x : ℝ) - c = 0 :=
sorry

end at_most_one_solution_l417_41759


namespace truncated_cone_radii_relation_l417_41730

/-- Represents a truncated cone with given dimensions and properties -/
structure TruncatedCone where
  top_radius : ℝ
  bottom_radius : ℝ
  section_ratio : ℝ

/-- Theorem stating the relationship between the radii of a truncated cone
    given specific conditions on its section -/
theorem truncated_cone_radii_relation (cone : TruncatedCone)
  (h1 : cone.top_radius = 5)
  (h2 : cone.section_ratio = 1/2) :
  cone.bottom_radius = 25 := by
  sorry

#check truncated_cone_radii_relation

end truncated_cone_radii_relation_l417_41730


namespace problem_statement_l417_41776

theorem problem_statement (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (sum_zero : a + b + c = 0) (sum_prod_nonzero : a * b + a * c + b * c ≠ 0) :
  (a^7 + b^7 + c^7) / (a * b * c * (a * b + a * c + b * c)) = -7 := by
  sorry

end problem_statement_l417_41776


namespace ball_distribution_theorem_l417_41748

/-- The number of ways to choose 3 boxes out of 4 -/
def choose_boxes : ℕ := 4

/-- The number of ways to distribute the extra white ball -/
def distribute_white : ℕ := 3

/-- The number of ways to distribute the extra black balls -/
def distribute_black : ℕ := 6

/-- The number of ways to distribute the extra red balls -/
def distribute_red : ℕ := 10

/-- The total number of ways to distribute the balls -/
def total_ways : ℕ := choose_boxes * distribute_white * distribute_black * distribute_red

theorem ball_distribution_theorem : total_ways = 720 := by
  sorry

end ball_distribution_theorem_l417_41748


namespace square_area_is_25_l417_41772

/-- A square in the coordinate plane with specific y-coordinates -/
structure SquareWithYCoords where
  -- The y-coordinates of the vertices
  y1 : ℝ
  y2 : ℝ
  y3 : ℝ
  y4 : ℝ
  -- Ensure the y-coordinates are distinct and in ascending order
  h1 : y1 < y2
  h2 : y2 < y3
  h3 : y3 < y4
  -- Ensure the square property (opposite sides are parallel and equal)
  h4 : y4 - y3 = y2 - y1

/-- The area of a square with specific y-coordinates is 25 -/
theorem square_area_is_25 (s : SquareWithYCoords) (h5 : s.y1 = 2) (h6 : s.y2 = 3) (h7 : s.y3 = 7) (h8 : s.y4 = 8) : 
  (s.y3 - s.y2) * (s.y3 - s.y2) = 25 := by
  sorry

end square_area_is_25_l417_41772


namespace light_bulbs_problem_l417_41743

theorem light_bulbs_problem (initial : ℕ) : 
  (initial - 16) / 2 = 12 → initial = 40 := by
  sorry

end light_bulbs_problem_l417_41743


namespace simplify_expression_l417_41760

theorem simplify_expression (a : ℝ) (h1 : a ≠ -1) (h2 : a ≠ 2) : 
  ((3 / (a + 1) - 1) / ((a - 2) / (a^2 + 2*a + 1))) = -a - 1 :=
by sorry

end simplify_expression_l417_41760


namespace equations_solvability_l417_41765

theorem equations_solvability :
  (∃ (x y z : ℕ), 
    (x % 2 = 1) ∧ (y % 2 = 1) ∧ (z % 2 = 1) ∧
    (y = x + 2) ∧ (z = y + 2) ∧
    (x + y + z = 51)) ∧
  (∃ (x y z w : ℕ),
    (x % 6 = 0) ∧ (y % 6 = 0) ∧ (z % 6 = 0) ∧ (w % 6 = 0) ∧
    (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (w > 0) ∧
    (x + y + z + w = 60)) :=
by sorry

end equations_solvability_l417_41765


namespace quadratic_inequality_properties_l417_41726

/-- Given that the solution set of ax^2 - bx + c > 0 is (-1, 2), prove the following statements -/
theorem quadratic_inequality_properties (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | a * x^2 - b * x + c > 0}) :
  (b < 0 ∧ c > 0) ∧ 
  (a - b + c > 0) ∧ 
  ({x : ℝ | a * x^2 + b * x + c > 0} = Set.Ioo (-2 : ℝ) 1) := by
  sorry

end quadratic_inequality_properties_l417_41726


namespace natural_number_decomposition_l417_41753

theorem natural_number_decomposition (x y z : ℕ) (h : x * y = z^2 + 1) :
  ∃ (a b c d : ℤ), (x : ℤ) = a^2 + b^2 ∧ (y : ℤ) = c^2 + d^2 ∧ (z : ℤ) = a * c + b * d := by
  sorry

end natural_number_decomposition_l417_41753


namespace quadratic_function_theorem_l417_41790

theorem quadratic_function_theorem (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x, f x = a * x^2 + b * x) →
  (f 0 = 0) →
  (∀ x, f (x + 1) = f x + x + 1) →
  (a = 1/2 ∧ b = 1/2) :=
by sorry

end quadratic_function_theorem_l417_41790


namespace paint_cans_used_l417_41793

def initial_capacity : ℕ := 36
def reduced_capacity : ℕ := 28
def lost_cans : ℕ := 4

theorem paint_cans_used : ℕ := by
  -- Prove that the number of cans used to paint 28 rooms is 14
  sorry

end paint_cans_used_l417_41793


namespace company_picnic_attendance_l417_41708

theorem company_picnic_attendance
  (total_employees : ℕ)
  (men_percentage : ℝ)
  (men_attendance_rate : ℝ)
  (women_attendance_rate : ℝ)
  (h1 : men_percentage = 0.55)
  (h2 : men_attendance_rate = 0.2)
  (h3 : women_attendance_rate = 0.4) :
  let women_percentage : ℝ := 1 - men_percentage
  let men_count : ℝ := men_percentage * total_employees
  let women_count : ℝ := women_percentage * total_employees
  let men_attended : ℝ := men_attendance_rate * men_count
  let women_attended : ℝ := women_attendance_rate * women_count
  let total_attended : ℝ := men_attended + women_attended
  total_attended / total_employees = 0.29 := by
sorry

end company_picnic_attendance_l417_41708


namespace qr_length_l417_41715

/-- Right triangle ABC with hypotenuse AB = 13, AC = 12, and BC = 5 -/
structure RightTriangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ
  right_angle : AB^2 = AC^2 + BC^2
  AB_eq : AB = 13
  AC_eq : AC = 12
  BC_eq : BC = 5

/-- Circle P passing through C and tangent to BC -/
structure CircleP (t : RightTriangle) where
  center : ℝ × ℝ
  radius : ℝ
  passes_through_C : True  -- Simplified condition
  tangent_to_BC : True     -- Simplified condition
  smallest : True          -- Simplified condition

/-- Points Q and R as intersections of circle P with AC and AB -/
structure Intersections (t : RightTriangle) (p : CircleP t) where
  Q : ℝ × ℝ
  R : ℝ × ℝ
  Q_on_AC : True           -- Simplified condition
  R_on_AB : True           -- Simplified condition
  Q_on_circle : True       -- Simplified condition
  R_on_circle : True       -- Simplified condition

/-- Main theorem: Length of QR is 5.42 -/
theorem qr_length (t : RightTriangle) (p : CircleP t) (i : Intersections t p) :
  Real.sqrt ((i.Q.1 - i.R.1)^2 + (i.Q.2 - i.R.2)^2) = 5.42 := by
  sorry

end qr_length_l417_41715


namespace munchausen_palindrome_exists_l417_41780

/-- A type representing a multi-digit number --/
def MultiDigitNumber := List Nat

/-- Check if a number is a palindrome --/
def isPalindrome (n : MultiDigitNumber) : Prop :=
  n = n.reverse

/-- Check if a list of numbers contains all numbers from 1 to N exactly once --/
def containsOneToN (l : List Nat) (N : Nat) : Prop :=
  l.toFinset = Finset.range N

/-- A function that represents cutting a number between digits --/
def cutBetweenDigits (n : MultiDigitNumber) : List Nat := sorry

/-- The main theorem --/
theorem munchausen_palindrome_exists :
  ∃ (n : MultiDigitNumber),
    isPalindrome n ∧
    containsOneToN (cutBetweenDigits n) 19 := by
  sorry

end munchausen_palindrome_exists_l417_41780


namespace more_girls_than_boys_l417_41731

theorem more_girls_than_boys :
  ∀ (boys girls : ℕ),
  (boys : ℚ) / (girls : ℚ) = 3 / 5 →
  boys + girls = 16 →
  girls - boys = 4 :=
by sorry

end more_girls_than_boys_l417_41731


namespace norma_laundry_ratio_l417_41781

/-- Proves the ratio of sweaters to T-shirts Norma left in the washer -/
theorem norma_laundry_ratio : 
  ∀ (S : ℕ), -- S is the number of sweaters Norma left
  -- Given conditions:
  (9 : ℕ) + S = 3 + 3 * 9 + 15 → -- Total items left = Total items found + Missing items
  (S : ℚ) / 9 = 2 / 1 := by
    sorry

end norma_laundry_ratio_l417_41781


namespace twentieth_term_of_arithmetic_sequence_l417_41740

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def isArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The 20th term of the specified arithmetic sequence is -49. -/
theorem twentieth_term_of_arithmetic_sequence :
  ∀ a : ℕ → ℤ,
  isArithmeticSequence a →
  a 1 = 8 →
  a 2 = 5 →
  a 3 = 2 →
  a 20 = -49 := by
sorry

end twentieth_term_of_arithmetic_sequence_l417_41740


namespace right_triangle_leg_square_l417_41725

theorem right_triangle_leg_square (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  c = 2*a + 1 →      -- Hypotenuse condition
  b^2 = 3*a^2 + 4*a + 1 := by
sorry

end right_triangle_leg_square_l417_41725


namespace race_distance_l417_41757

-- Define the race distance
variable (d : ℝ)

-- Define the speeds of A, B, and C
variable (a b c : ℝ)

-- Define the conditions of the race
variable (h1 : d / a = (d - 30) / b)
variable (h2 : d / b = (d - 15) / c)
variable (h3 : d / a = (d - 40) / c)

-- The theorem to prove
theorem race_distance : d = 90 := by
  sorry

end race_distance_l417_41757


namespace smallest_Y_value_l417_41747

/-- A function that checks if a natural number consists only of digits 0 and 1 -/
def only_zero_and_one (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 0 ∨ d = 1

/-- The theorem stating the smallest possible value of Y -/
theorem smallest_Y_value (S : ℕ) (hS : S > 0) (h_digits : only_zero_and_one S) (h_div : S % 15 = 0) :
  (S / 15 : ℕ) ≥ 74 :=
sorry

end smallest_Y_value_l417_41747


namespace cos_75_degrees_l417_41794

theorem cos_75_degrees :
  Real.cos (75 * π / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_degrees_l417_41794


namespace bottle_cap_count_l417_41745

theorem bottle_cap_count (caps_per_box : ℝ) (num_boxes : ℝ) 
  (h1 : caps_per_box = 35.0) 
  (h2 : num_boxes = 7.0) : 
  caps_per_box * num_boxes = 245.0 := by
  sorry

end bottle_cap_count_l417_41745


namespace binary_op_three_seven_l417_41734

def binary_op (c d : ℤ) : ℤ := 4 * c + 3 * d - c * d

theorem binary_op_three_seven : binary_op 3 7 = 12 := by
  sorry

end binary_op_three_seven_l417_41734


namespace infinitely_many_prime_divisors_l417_41750

/-- A quadratic polynomial with integer coefficients -/
structure QuadraticPolynomial where
  a : ℤ
  b : ℤ
  c : ℤ

/-- Evaluation of a quadratic polynomial at an integer -/
def QuadraticPolynomial.eval (p : QuadraticPolynomial) (x : ℤ) : ℤ :=
  p.a * x^2 + p.b * x + p.c

/-- The set of prime divisors of a polynomial's values -/
def primeDivisors (p : QuadraticPolynomial) : Set ℕ :=
  {q : ℕ | Nat.Prime q ∧ ∃ n : ℤ, (q : ℤ) ∣ p.eval n}

/-- The main theorem: there are infinitely many prime divisors for any quadratic polynomial -/
theorem infinitely_many_prime_divisors (p : QuadraticPolynomial) :
  Set.Infinite (primeDivisors p) := by
  sorry


end infinitely_many_prime_divisors_l417_41750


namespace expression_values_l417_41758

theorem expression_values (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ (x : ℝ), x ∈ ({-4, 0, 4} : Set ℝ) ∧
  x = a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| :=
by sorry

end expression_values_l417_41758


namespace final_bill_amount_l417_41767

def original_bill : ℝ := 500
def late_charge_rate : ℝ := 0.02

theorem final_bill_amount :
  let first_increase := original_bill * (1 + late_charge_rate)
  let final_bill := first_increase * (1 + late_charge_rate)
  final_bill = 520.20 := by sorry

end final_bill_amount_l417_41767


namespace fruit_brought_to_school_l417_41783

/-- 
Given:
- Mark had an initial number of fruit pieces for the week
- Mark ate a certain number of fruit pieces in the first four days
- Mark decided to keep some pieces for next week

Prove that the number of fruit pieces Mark brought to school on Friday
is equal to the initial number minus the number eaten minus the number kept for next week
-/
theorem fruit_brought_to_school (initial_fruit pieces_eaten pieces_kept : ℕ) :
  initial_fruit - pieces_eaten - pieces_kept = initial_fruit - (pieces_eaten + pieces_kept) :=
by sorry

end fruit_brought_to_school_l417_41783


namespace ellipse_standard_equation_l417_41770

/-- An ellipse with foci at (-3, 0) and (3, 0), passing through (0, 3) -/
structure Ellipse where
  /-- The equation of the ellipse in the form (x²/a² + y²/b² = 1) -/
  equation : ℝ → ℝ → Prop
  /-- The foci are at (-3, 0) and (3, 0) -/
  foci : equation (-3) 0 ∧ equation 3 0
  /-- The point (0, 3) is on the ellipse -/
  point : equation 0 3

/-- The standard form of the ellipse equation -/
def standard_equation (x y : ℝ) : Prop :=
  x^2 / 18 + y^2 / 9 = 1

/-- Theorem: The standard equation of the ellipse is x²/18 + y²/9 = 1 -/
theorem ellipse_standard_equation (e : Ellipse) : e.equation = standard_equation := by
  sorry

end ellipse_standard_equation_l417_41770


namespace theater_seats_l417_41788

theorem theater_seats (first_row : ℕ) (last_row : ℕ) (total_seats : ℕ) (num_rows : ℕ) 
  (h1 : first_row = 14)
  (h2 : last_row = 50)
  (h3 : total_seats = 416)
  (h4 : num_rows = 13) :
  ∃ (additional_seats : ℕ), 
    (additional_seats = 3) ∧ 
    (last_row = first_row + (num_rows - 1) * additional_seats) ∧
    (total_seats = (num_rows * (first_row + last_row)) / 2) :=
by sorry

end theater_seats_l417_41788


namespace infinitely_many_an_power_an_mod_8_l417_41717

theorem infinitely_many_an_power_an_mod_8 :
  ∀ k : ℕ, ∃ n : ℕ, n > k ∧ (8 * n + 3)^(8 * n + 3) ≡ 8 * n + 3 [ZMOD 8] := by
  sorry

end infinitely_many_an_power_an_mod_8_l417_41717


namespace density_of_cube_root_differences_l417_41754

theorem density_of_cube_root_differences :
  ∀ ε > 0, ∀ x : ℝ, ∃ n m : ℕ, |x - (n^(1/3) - m^(1/3))| < ε :=
sorry

end density_of_cube_root_differences_l417_41754


namespace mooncake_problem_l417_41773

-- Define the types and variables
variable (type_a_cost type_b_cost : ℝ)
variable (total_cost_per_pair : ℝ)
variable (type_a_quantity type_b_quantity : ℕ)
variable (m : ℝ)

-- Define the conditions
def conditions (type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m : ℝ) : Prop :=
  type_a_cost = 1200 ∧
  type_b_cost = 600 ∧
  total_cost_per_pair = 9 ∧
  type_a_quantity = 4 * type_b_quantity ∧
  m ≠ 0 ∧
  (type_a_cost / type_a_quantity + type_b_cost / type_b_quantity = total_cost_per_pair) ∧
  (2 * (type_a_quantity - 15 / 2 * m) + (6 - m / 5) * (type_b_quantity + 15 / 2 * m) = 1400 - 2 * m)

-- State the theorem
theorem mooncake_problem (type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m : ℝ) :
  conditions type_a_cost type_b_cost total_cost_per_pair type_a_quantity type_b_quantity m →
  type_a_quantity = 400 ∧ type_b_quantity = 100 ∧ m = 8 :=
by sorry

end mooncake_problem_l417_41773


namespace meeting_point_27_blocks_l417_41729

/-- Two people walking around a circular loop -/
def CircularWalk (total_blocks : ℕ) (speed_ratio : ℚ) : Prop :=
  ∃ (meeting_point : ℚ),
    meeting_point > 0 ∧
    meeting_point < total_blocks ∧
    meeting_point = total_blocks / (1 + speed_ratio)

/-- Theorem: In a 27-block loop with a 3:1 speed ratio, the meeting point is at 27/4 blocks -/
theorem meeting_point_27_blocks :
  CircularWalk 27 3 → (27 : ℚ) / 4 = 27 / (1 + 3) :=
by
  sorry

#check meeting_point_27_blocks

end meeting_point_27_blocks_l417_41729


namespace gel_pen_price_ratio_l417_41796

/-- Represents the price ratio of gel pens to ballpoint pens -/
def price_ratio (x y : ℕ) (b g : ℝ) : Prop :=
  let total := x * b + y * g
  (x + y) * g = 4 * total ∧ (x + y) * b = (1 / 2) * total ∧ g = 8 * b

/-- Theorem stating that under the given conditions, a gel pen costs 8 times as much as a ballpoint pen -/
theorem gel_pen_price_ratio {x y : ℕ} {b g : ℝ} (h : price_ratio x y b g) :
  g = 8 * b := by
  sorry

end gel_pen_price_ratio_l417_41796


namespace sixth_graders_and_parents_average_age_l417_41721

/-- The average age of a group of sixth-graders and their parents -/
def average_age (num_children : ℕ) (num_parents : ℕ) (avg_age_children : ℚ) (avg_age_parents : ℚ) : ℚ :=
  ((num_children : ℚ) * avg_age_children + (num_parents : ℚ) * avg_age_parents) / ((num_children + num_parents) : ℚ)

/-- Theorem stating the average age of sixth-graders and their parents -/
theorem sixth_graders_and_parents_average_age :
  average_age 45 60 12 35 = 25142857142857142 / 1000000000000000 :=
by sorry

end sixth_graders_and_parents_average_age_l417_41721


namespace initial_number_solution_l417_41711

theorem initial_number_solution : 
  ∃ x : ℤ, x - 12 * 3 * 2 = 1234490 ∧ x = 1234562 := by
  sorry

end initial_number_solution_l417_41711


namespace cream_fraction_after_pouring_l417_41723

/-- Represents the contents of a cup --/
structure CupContents where
  coffee : ℚ
  cream : ℚ

/-- Represents the state of both cups --/
structure CupState where
  cup1 : CupContents
  cup2 : CupContents

/-- Performs one round of pouring between cups --/
def pour (state : CupState) : CupState := sorry

/-- Calculates the fraction of cream in cup1 after the pouring process --/
def creamFractionInCup1 (initial : CupState) : ℚ := sorry

theorem cream_fraction_after_pouring :
  let initial := CupState.mk
    (CupContents.mk 5 0)  -- 5 oz coffee, 0 oz cream in cup1
    (CupContents.mk 0 3)  -- 0 oz coffee, 3 oz cream in cup2
  let final := pour (pour initial)
  creamFractionInCup1 final = (11 : ℚ) / 21 := by sorry

#check cream_fraction_after_pouring

end cream_fraction_after_pouring_l417_41723


namespace time_taken_by_A_l417_41735

/-- The time taken by A to reach the destination given the specified conditions -/
theorem time_taken_by_A (distance : ℝ) (speed_A speed_B : ℝ) (time_B : ℝ) : 
  speed_A / speed_B = 3 / 4 →
  time_B * 60 + 30 = speed_B * distance / speed_A →
  speed_A * (time_B * 60 + 30) / 60 = distance →
  speed_A * 2 = distance :=
by sorry

end time_taken_by_A_l417_41735


namespace max_cute_pairs_is_43_l417_41707

/-- A pair of ages (a, b) is cute if each person is at least seven years older than half the age of the other person. -/
def is_cute_pair (a b : ℕ) : Prop :=
  a ≥ b / 2 + 7 ∧ b ≥ a / 2 + 7

/-- The set of ages from 1 to 100. -/
def age_set : Finset ℕ :=
  Finset.range 100

/-- A function that returns the maximum number of pairwise disjoint cute pairs that can be formed from a set of ages. -/
def max_disjoint_cute_pairs (ages : Finset ℕ) : ℕ :=
  sorry

theorem max_cute_pairs_is_43 :
  max_disjoint_cute_pairs age_set = 43 :=
sorry

end max_cute_pairs_is_43_l417_41707


namespace x_eq_one_iff_z_purely_imaginary_l417_41763

/-- A complex number is purely imaginary if its real part is zero and its imaginary part is non-zero. -/
def IsPurelyImaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number z as a function of x. -/
def z (x : ℝ) : ℂ :=
  ⟨x^2 - 1, x + 1⟩

/-- Theorem stating that x = 1 is necessary and sufficient for z(x) to be purely imaginary. -/
theorem x_eq_one_iff_z_purely_imaginary :
  ∀ x : ℝ, x = 1 ↔ IsPurelyImaginary (z x) :=
sorry

end x_eq_one_iff_z_purely_imaginary_l417_41763


namespace inequality_group_solution_set_l417_41779

theorem inequality_group_solution_set :
  let S := {x : ℝ | 2 * x + 3 ≥ -1 ∧ 7 - 3 * x > 1}
  S = {x : ℝ | -2 ≤ x ∧ x < 2} := by sorry

end inequality_group_solution_set_l417_41779


namespace andrew_vacation_days_in_march_l417_41706

/-- Calculates the number of vacation days taken in March given the conditions of Andrew's work and vacation schedule. -/
def vacation_days_in_march (days_worked : ℕ) (days_per_vacation : ℕ) (remaining_days : ℕ) : ℕ :=
  let total_vacation_days := days_worked / days_per_vacation
  let used_vacation_days := total_vacation_days - remaining_days
  used_vacation_days / 3

theorem andrew_vacation_days_in_march :
  vacation_days_in_march 300 10 15 = 5 := by
  sorry

end andrew_vacation_days_in_march_l417_41706


namespace power_problem_l417_41769

theorem power_problem (a m n : ℕ) (h1 : a ^ m = 3) (h2 : a ^ n = 2) : 
  a ^ (2 * m + 3 * n) = 72 := by
sorry

end power_problem_l417_41769


namespace typists_productivity_l417_41771

/-- Given that 25 typists can type 60 letters in 20 minutes, prove that 75 typists 
    working at the same rate can complete 540 letters in 1 hour. -/
theorem typists_productivity (typists_base : ℕ) (letters_base : ℕ) (minutes_base : ℕ) 
  (typists_new : ℕ) (minutes_new : ℕ) :
  typists_base = 25 →
  letters_base = 60 →
  minutes_base = 20 →
  typists_new = 75 →
  minutes_new = 60 →
  (typists_new * letters_base * minutes_new) / (typists_base * minutes_base) = 540 :=
by sorry

end typists_productivity_l417_41771


namespace problem_solution_l417_41737

theorem problem_solution (a b : ℝ) : 
  ({a, b/a, 1} : Set ℝ) = {a^2, a+b, 0} → a^2003 + b^2004 = -1 := by
sorry

end problem_solution_l417_41737


namespace lee_class_b_students_l417_41704

theorem lee_class_b_students (kipling_total : ℕ) (kipling_b : ℕ) (lee_total : ℕ) 
  (h1 : kipling_total = 12)
  (h2 : kipling_b = 8)
  (h3 : lee_total = 30) :
  ∃ (lee_b : ℕ), (lee_b : ℚ) / lee_total = (kipling_b : ℚ) / kipling_total ∧ lee_b = 20 := by
  sorry


end lee_class_b_students_l417_41704


namespace circle_area_difference_l417_41746

/-- Given two circles where the smaller circle has radius 4 and the center of the larger circle
    is on the circumference of the smaller circle, the difference in areas between the larger
    and smaller circles is 48π. -/
theorem circle_area_difference (r : ℝ) (h : r = 4) : 
  π * (2 * r)^2 - π * r^2 = 48 * π := by sorry

end circle_area_difference_l417_41746


namespace exam_time_allocation_l417_41741

theorem exam_time_allocation :
  ∀ (total_time total_questions type_a_questions : ℕ) 
    (type_a_time_ratio : ℚ),
  total_time = 180 →
  total_questions = 200 →
  type_a_questions = 20 →
  type_a_time_ratio = 2 →
  ∃ (type_a_time : ℕ),
    type_a_time = 36 ∧
    type_a_time * (total_questions - type_a_questions) = 
      (total_time - type_a_time) * type_a_questions * type_a_time_ratio :=
by sorry

end exam_time_allocation_l417_41741


namespace boys_without_notebooks_l417_41713

def history_class (total_boys : ℕ) (students_with_notebooks : ℕ) (girls_with_notebooks : ℕ) : ℕ :=
  total_boys - (students_with_notebooks - girls_with_notebooks)

theorem boys_without_notebooks :
  history_class 16 20 11 = 7 :=
by sorry

end boys_without_notebooks_l417_41713


namespace triangular_gcd_bound_l417_41739

/-- The nth triangular number -/
def T (n : ℕ+) : ℕ := (n * (n + 1)) / 2

/-- Theorem: The GCD of 6T_n and n-1 is at most 3, and this bound is achievable -/
theorem triangular_gcd_bound (n : ℕ+) : 
  ∃ (m : ℕ+), Nat.gcd (6 * T m) (m - 1) = 3 ∧ 
  ∀ (k : ℕ+), Nat.gcd (6 * T k) (k - 1) ≤ 3 := by
  sorry

#check triangular_gcd_bound

end triangular_gcd_bound_l417_41739


namespace negation_of_positive_square_plus_x_positive_l417_41751

theorem negation_of_positive_square_plus_x_positive :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x > 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 + x ≤ 0) :=
by sorry

end negation_of_positive_square_plus_x_positive_l417_41751


namespace usual_time_calculation_l417_41733

/-- Represents the scenario of a person catching a bus -/
structure BusScenario where
  usual_speed : ℝ
  usual_time : ℝ
  faster_speed : ℝ
  missed_time : ℝ

/-- The theorem stating the relationship between usual time and missed time -/
theorem usual_time_calculation (scenario : BusScenario) 
  (h1 : scenario.faster_speed = (5/4) * scenario.usual_speed)
  (h2 : scenario.missed_time = scenario.usual_time + 5)
  (h3 : scenario.usual_speed * scenario.usual_time = scenario.faster_speed * scenario.missed_time) :
  scenario.usual_time = 25 := by
  sorry

#check usual_time_calculation

end usual_time_calculation_l417_41733


namespace arithmetic_sequence_sum_l417_41791

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℕ) :
  is_arithmetic_sequence a →
  a 1 = 3 →
  a 2 = 8 →
  a 3 = 13 →
  (∃ n : ℕ, a n = 33 ∧ a (n - 2) + a (n - 1) = 51) :=
by sorry

end arithmetic_sequence_sum_l417_41791


namespace compute_fraction_power_l417_41732

theorem compute_fraction_power : 8 * (2 / 3)^4 = 128 / 81 := by
  sorry

end compute_fraction_power_l417_41732


namespace sqrt_3_minus_pi_squared_l417_41792

theorem sqrt_3_minus_pi_squared (π : ℝ) (h : π > 3) : 
  Real.sqrt ((3 - π)^2) = π - 3 := by
  sorry

end sqrt_3_minus_pi_squared_l417_41792


namespace integer_roots_of_polynomial_l417_41728

def polynomial (a₂ a₁ : ℤ) (x : ℤ) : ℤ := x^3 + a₂*x^2 + a₁*x - 18

def possible_roots : Set ℤ := {-18, -9, -6, -3, -2, -1, 1, 2, 3, 6, 9, 18}

theorem integer_roots_of_polynomial (a₂ a₁ : ℤ) :
  ∀ r : ℤ, polynomial a₂ a₁ r = 0 → r ∈ possible_roots :=
sorry

end integer_roots_of_polynomial_l417_41728


namespace gumball_count_l417_41714

def gumball_machine (red : ℕ) : Prop :=
  ∃ (blue green yellow orange : ℕ),
    blue = red / 2 ∧
    green = 4 * blue ∧
    yellow = (60 * green) / 100 ∧
    orange = (red + blue) / 3 ∧
    red + blue + green + yellow + orange = 124

theorem gumball_count : gumball_machine 24 := by
  sorry

end gumball_count_l417_41714


namespace repunit_divisibility_l417_41749

theorem repunit_divisibility (p : Nat) (h_prime : Prime p) (h_not_two : p ≠ 2) (h_not_five : p ≠ 5) :
  ∃ n : Nat, ∃ k : Nat, k > 0 ∧ p ∣ (10^n - 1) / 9 :=
sorry

end repunit_divisibility_l417_41749


namespace parallelogram_area_from_complex_equations_sum_pqrs_equals_102_l417_41786

theorem parallelogram_area_from_complex_equations : ℂ → Prop :=
  fun i =>
  i * i = -1 →
  let eq1 := fun z : ℂ => z * z = 9 + 9 * Real.sqrt 7 * i
  let eq2 := fun z : ℂ => z * z = 5 + 5 * Real.sqrt 2 * i
  let solutions := {z : ℂ | eq1 z ∨ eq2 z}
  let parallelogram_area := Real.sqrt 96 * 2 - Real.sqrt 2 * 2
  (∃ (v1 v2 v3 v4 : ℂ), v1 ∈ solutions ∧ v2 ∈ solutions ∧ v3 ∈ solutions ∧ v4 ∈ solutions ∧
    (v1 - v2).im * (v3 - v4).re - (v1 - v2).re * (v3 - v4).im = parallelogram_area)

/-- The sum of p, q, r, and s is 102 -/
theorem sum_pqrs_equals_102 : 2 + 96 + 2 + 2 = 102 := by sorry

end parallelogram_area_from_complex_equations_sum_pqrs_equals_102_l417_41786


namespace set_operations_l417_41719

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}
def B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 4}

theorem set_operations :
  (A ∩ B = {x | 1 ≤ x ∧ x ≤ 4}) ∧
  ((Set.univ : Set ℝ) \ (A ∪ B) = {x | x < -1 ∨ x > 5}) ∧
  (((Set.univ : Set ℝ) \ A) ∪ ((Set.univ : Set ℝ) \ B) = {x | x < 1 ∨ x > 4}) := by
  sorry

end set_operations_l417_41719


namespace graph_sequence_periodic_l417_41718

/-- A graph on n vertices -/
def Graph (n : ℕ) := Fin n → Fin n → Prop

/-- The rule for constructing G_(n+1) from G_n -/
def nextGraph (G : Graph n) : Graph n :=
  λ i j => ∃ k, k ≠ i ∧ k ≠ j ∧ G k i ∧ G k j

/-- The sequence of graphs -/
def graphSequence (G₀ : Graph n) : ℕ → Graph n
  | 0 => G₀
  | m + 1 => nextGraph (graphSequence G₀ m)

/-- Two graphs are equal if they have the same edges -/
def graphEqual (G H : Graph n) : Prop :=
  ∀ i j, G i j ↔ H i j

theorem graph_sequence_periodic (n : ℕ) (G₀ : Graph n) :
  ∃ (m₀ T : ℕ), T ≤ 2^n ∧
    ∀ m ≥ m₀, graphEqual (graphSequence G₀ (m + T)) (graphSequence G₀ m) :=
sorry

end graph_sequence_periodic_l417_41718


namespace paintings_per_room_l417_41795

theorem paintings_per_room (total_paintings : ℕ) (num_rooms : ℕ) 
  (h1 : total_paintings = 32) 
  (h2 : num_rooms = 4) 
  (h3 : total_paintings % num_rooms = 0) : 
  total_paintings / num_rooms = 8 := by
  sorry

end paintings_per_room_l417_41795


namespace loan_period_calculation_l417_41727

/-- The time period (in years) for which A lent money to C -/
def loan_period_C : ℚ := 2/3

theorem loan_period_calculation (principal_B principal_C total_interest : ℚ) 
  (loan_period_B interest_rate : ℚ) :
  principal_B = 5000 →
  principal_C = 3000 →
  loan_period_B = 2 →
  interest_rate = 1/10 →
  total_interest = 2200 →
  principal_B * interest_rate * loan_period_B + 
  principal_C * interest_rate * loan_period_C = total_interest :=
by sorry

end loan_period_calculation_l417_41727


namespace sum_of_sides_equals_two_point_five_l417_41744

/-- Represents a polygon ABCDEFGH with given properties -/
structure Polygon where
  area : ℝ
  AB : ℝ
  BC : ℝ
  HA : ℝ

/-- The sum of lengths DE, EF, FG, and GH in the polygon -/
def sum_of_sides (p : Polygon) : ℝ := sorry

/-- Theorem stating that for a polygon with given properties, the sum of certain sides equals 2.5 -/
theorem sum_of_sides_equals_two_point_five (p : Polygon) 
  (h1 : p.area = 85)
  (h2 : p.AB = 7)
  (h3 : p.BC = 10)
  (h4 : p.HA = 6) :
  sum_of_sides p = 2.5 := by sorry

end sum_of_sides_equals_two_point_five_l417_41744


namespace sin_x_plus_y_eq_one_sixth_l417_41701

theorem sin_x_plus_y_eq_one_sixth (x y : ℝ) 
  (h1 : 3 * Real.sin x + 4 * Real.cos y = 5) 
  (h2 : 4 * Real.sin y + 3 * Real.cos x = 2) : 
  Real.sin (x + y) = 1/6 := by
  sorry

end sin_x_plus_y_eq_one_sixth_l417_41701


namespace f_strictly_decreasing_on_interval_l417_41752

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x + 1

-- Theorem statement
theorem f_strictly_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 3, StrictMonoOn f (Set.Ioo (-1 : ℝ) 3) := by
  sorry


end f_strictly_decreasing_on_interval_l417_41752


namespace meeting_arrangements_count_l417_41712

/-- Represents the number of schools in the community -/
def num_schools : ℕ := 4

/-- Represents the number of members in each school -/
def members_per_school : ℕ := 6

/-- Represents the number of representatives each school sends -/
def reps_per_school : ℕ := 2

/-- The number of ways to arrange the leadership meeting -/
def meeting_arrangements : ℕ := num_schools * (members_per_school.choose reps_per_school) * (members_per_school.choose reps_per_school)^(num_schools - 1)

/-- Theorem stating that the number of meeting arrangements is 202500 -/
theorem meeting_arrangements_count : meeting_arrangements = 202500 := by
  sorry

end meeting_arrangements_count_l417_41712


namespace flour_already_put_in_l417_41774

/-- Given a recipe that requires a certain amount of flour and the additional amount needed,
    calculate the amount of flour already put in. -/
theorem flour_already_put_in
  (recipe_requirement : ℕ)  -- Total cups of flour required by the recipe
  (additional_needed : ℕ)   -- Additional cups of flour needed
  (h1 : recipe_requirement = 7)  -- The recipe requires 7 cups of flour
  (h2 : additional_needed = 5)   -- Mary needs to add 5 more cups
  : recipe_requirement - additional_needed = 2 :=
by sorry

end flour_already_put_in_l417_41774


namespace fractional_equation_solution_range_l417_41736

theorem fractional_equation_solution_range (x a : ℝ) : 
  (1 / (x + 3) - 1 = a / (x + 3)) → -- Given equation
  (x < 0) → -- Solution for x is negative
  (a > -2 ∧ a ≠ 1) -- Range of a
  :=
by sorry

end fractional_equation_solution_range_l417_41736


namespace boyden_family_ticket_cost_l417_41716

/-- The cost of tickets for a family visit to a leisure park -/
def ticket_cost (adult_price : ℕ) (child_price : ℕ) (num_adults : ℕ) (num_children : ℕ) : ℕ :=
  adult_price * num_adults + child_price * num_children

theorem boyden_family_ticket_cost :
  let adult_price : ℕ := 19
  let child_price : ℕ := adult_price - 6
  let num_adults : ℕ := 2
  let num_children : ℕ := 3
  ticket_cost adult_price child_price num_adults num_children = 77 := by
  sorry

end boyden_family_ticket_cost_l417_41716


namespace balloon_unique_arrangements_l417_41720

def balloon_arrangements : ℕ :=
  Nat.factorial 7 / (Nat.factorial 2 * Nat.factorial 3)

theorem balloon_unique_arrangements :
  balloon_arrangements = 420 := by
  sorry

end balloon_unique_arrangements_l417_41720


namespace inequality_for_increasing_function_l417_41799

theorem inequality_for_increasing_function (f : ℝ → ℝ) (a b : ℝ) 
  (h_increasing : ∀ x y, x < y → f x < f y) 
  (h_sum : a + b ≤ 0) : 
  f a + f b ≤ f (-a) + f (-b) := by
  sorry

end inequality_for_increasing_function_l417_41799


namespace cos_equation_solution_l417_41789

theorem cos_equation_solution (x : ℝ) : 
  (Real.cos x - 3 * Real.cos (4 * x))^2 = 16 + Real.sin (3 * x)^2 ↔ 
  ∃ k : ℤ, x = π + 2 * k * π :=
sorry

end cos_equation_solution_l417_41789


namespace baseball_cards_count_l417_41705

theorem baseball_cards_count (num_friends : ℕ) (cards_per_friend : ℕ) : 
  num_friends = 5 → cards_per_friend = 91 → num_friends * cards_per_friend = 455 := by
  sorry

end baseball_cards_count_l417_41705


namespace shaded_area_of_intersecting_rectangles_l417_41703

/-- The area of the shaded region formed by two intersecting perpendicular rectangles -/
theorem shaded_area_of_intersecting_rectangles (rect1_width rect1_height rect2_width rect2_height : ℝ) 
  (h1 : rect1_width = 2 ∧ rect1_height = 10)
  (h2 : rect2_width = 3 ∧ rect2_height = 8)
  (h3 : rect1_width ≤ rect2_height ∧ rect2_width ≤ rect1_height) : 
  rect1_width * rect1_height + rect2_width * rect2_height - rect1_width * rect2_width = 38 :=
by sorry

end shaded_area_of_intersecting_rectangles_l417_41703


namespace tetrahedron_properties_l417_41785

/-- Represents a tetrahedron SABC with mutually perpendicular edges SA, SB, SC -/
structure Tetrahedron where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  perpendicular : True -- Represents that SA, SB, SC are mutually perpendicular

/-- The radius of the circumscribed sphere of the tetrahedron -/
def circumscribedSphereRadius (t : Tetrahedron) : ℝ :=
  sorry

/-- Determines if there exists a sphere with radius smaller than R that contains the tetrahedron -/
def existsSmallerSphere (t : Tetrahedron) (R : ℝ) : Prop :=
  sorry

theorem tetrahedron_properties (t : Tetrahedron) 
    (h1 : t.SA = 2) (h2 : t.SB = 3) (h3 : t.SC = 6) : 
    circumscribedSphereRadius t = 7/2 ∧ existsSmallerSphere t (7/2) :=
  sorry

end tetrahedron_properties_l417_41785


namespace factorization_x_squared_plus_5x_l417_41775

theorem factorization_x_squared_plus_5x (x : ℝ) : x^2 + 5*x = x*(x+5) := by
  sorry

end factorization_x_squared_plus_5x_l417_41775


namespace water_sulfuric_oxygen_equivalence_l417_41766

/-- Represents the number of oxygen atoms in a molecule --/
def oxygenAtoms (molecule : String) : ℕ :=
  match molecule with
  | "H2SO4" => 4
  | "H2O" => 1
  | _ => 0

/-- Theorem stating that 4n water molecules have the same number of oxygen atoms as n sulfuric acid molecules --/
theorem water_sulfuric_oxygen_equivalence (n : ℕ) :
  n * oxygenAtoms "H2SO4" = 4 * n * oxygenAtoms "H2O" :=
by sorry


end water_sulfuric_oxygen_equivalence_l417_41766


namespace root_condition_implies_a_range_l417_41710

-- Define the function f
def f (a x : ℝ) : ℝ := x^2 + (a^2 - 1)*x + (a - 2)

-- State the theorem
theorem root_condition_implies_a_range :
  ∀ a : ℝ,
  (∃ x y : ℝ, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ x > 1 ∧ y < 1) →
  -2 < a ∧ a < 1 :=
by
  sorry


end root_condition_implies_a_range_l417_41710


namespace parabola_focus_l417_41777

/-- A parabola is defined by the equation x^2 = 4y -/
structure Parabola where
  eq : ∀ x y : ℝ, x^2 = 4*y

/-- The focus of a parabola is a point (h, k) where h and k are real numbers -/
structure Focus where
  h : ℝ
  k : ℝ

/-- Theorem: The focus of the parabola x^2 = 4y has coordinates (0, 1) -/
theorem parabola_focus (p : Parabola) : ∃ f : Focus, f.h = 0 ∧ f.k = 1 :=
sorry

end parabola_focus_l417_41777


namespace largest_integer_problem_l417_41797

theorem largest_integer_problem (a b c d : ℤ) : 
  a < b ∧ b < c ∧ c < d →  -- four different integers
  (a + b + c + d) / 4 = 74 →  -- average is 74
  a ≥ 29 →  -- smallest integer is at least 29
  d ≤ 206 :=  -- largest integer is at most 206
by sorry

end largest_integer_problem_l417_41797


namespace system_one_solution_system_two_solution_l417_41761

-- System 1
theorem system_one_solution (x y : ℝ) :
  (3 * x + 2 * y = 10 ∧ x / 2 - (y + 1) / 3 = 1) →
  (x = 3 ∧ y = 1/2) :=
by sorry

-- System 2
theorem system_two_solution (x y : ℝ) :
  (4 * x - 5 * y = 3 ∧ (x - 2 * y) / 0.4 = 0.6) →
  (x = 1.6 ∧ y = 0.68) :=
by sorry

end system_one_solution_system_two_solution_l417_41761
