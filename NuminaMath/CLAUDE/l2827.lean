import Mathlib

namespace NUMINAMATH_CALUDE_previous_salary_calculation_l2827_282737

-- Define the salary increase rate
def salary_increase_rate : ℝ := 1.05

-- Define the new salary
def new_salary : ℝ := 2100

-- Theorem statement
theorem previous_salary_calculation :
  ∃ (previous_salary : ℝ),
    salary_increase_rate * previous_salary = new_salary ∧
    previous_salary = 2000 := by
  sorry

end NUMINAMATH_CALUDE_previous_salary_calculation_l2827_282737


namespace NUMINAMATH_CALUDE_crayons_left_l2827_282725

theorem crayons_left (initial : ℝ) (taken : ℝ) (left : ℝ) : 
  initial = 7.5 → taken = 2.25 → left = initial - taken → left = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_crayons_left_l2827_282725


namespace NUMINAMATH_CALUDE_unique_prime_pair_solution_l2827_282796

theorem unique_prime_pair_solution : 
  ∀ p q : ℕ, 
    Prime p → Prime q → 
    (3 * p^(q-1) + 1) ∣ (11^p + 17^p) → 
    p = 3 ∧ q = 3 :=
by sorry

end NUMINAMATH_CALUDE_unique_prime_pair_solution_l2827_282796


namespace NUMINAMATH_CALUDE_linda_original_amount_l2827_282778

/-- The amount of money Lucy originally had -/
def lucy_original : ℕ := 20

/-- The amount of money Linda originally had -/
def linda_original : ℕ := 10

/-- The amount of money Lucy would give to Linda -/
def transfer_amount : ℕ := 5

theorem linda_original_amount : 
  (lucy_original - transfer_amount = linda_original + transfer_amount) →
  linda_original = 10 := by
sorry

end NUMINAMATH_CALUDE_linda_original_amount_l2827_282778


namespace NUMINAMATH_CALUDE_square_of_negative_integer_is_positive_l2827_282757

theorem square_of_negative_integer_is_positive (P : Int) (h : P < 0) : P^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_integer_is_positive_l2827_282757


namespace NUMINAMATH_CALUDE_quadratic_root_reciprocal_sum_l2827_282707

theorem quadratic_root_reciprocal_sum (x₁ x₂ : ℝ) : 
  x₁^2 - 4*x₁ - 2 = 0 → 
  x₂^2 - 4*x₂ - 2 = 0 → 
  x₁ ≠ x₂ →
  1/x₁ + 1/x₂ = -2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_reciprocal_sum_l2827_282707


namespace NUMINAMATH_CALUDE_candy_per_package_l2827_282799

/-- Given that Robin has 45 packages of candy and 405 pieces of candies in total,
    prove that there are 9 pieces of candy in each package. -/
theorem candy_per_package (packages : ℕ) (total_pieces : ℕ) 
    (h1 : packages = 45) (h2 : total_pieces = 405) : 
    total_pieces / packages = 9 := by
  sorry

end NUMINAMATH_CALUDE_candy_per_package_l2827_282799


namespace NUMINAMATH_CALUDE_basketball_shot_expectation_l2827_282732

theorem basketball_shot_expectation 
  (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) 
  (hb : 0 < b ∧ b < 1) 
  (hc : 0 < c ∧ c < 1) 
  (h_sum : a + b + c = 1) 
  (h_expect : 3 * a + 2 * b = 2) : 
  (∀ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 2 → (2 / x + 1 / (3 * y)) ≥ 16 / 3) ∧ 
  (∃ x y, x > 0 ∧ y > 0 ∧ 3 * x + 2 * y = 2 ∧ 2 / x + 1 / (3 * y) = 16 / 3) :=
by sorry

end NUMINAMATH_CALUDE_basketball_shot_expectation_l2827_282732


namespace NUMINAMATH_CALUDE_two_thirds_to_tenth_bounds_l2827_282773

theorem two_thirds_to_tenth_bounds : 1/100 < (2/3)^10 ∧ (2/3)^10 < 2/100 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_to_tenth_bounds_l2827_282773


namespace NUMINAMATH_CALUDE_rectangle_area_l2827_282739

/-- A rectangle with length thrice its breadth and perimeter 48 meters has an area of 108 square meters. -/
theorem rectangle_area (b l : ℝ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 48) : l * b = 108 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2827_282739


namespace NUMINAMATH_CALUDE_find_m_value_l2827_282714

/-- Given two functions f and g, prove that m equals 10/7 -/
theorem find_m_value (f g : ℝ → ℝ) (m : ℝ) : 
  (∀ x, f x = x^2 - 3*x + m) →
  (∀ x, g x = x^2 - 3*x + 5*m) →
  3 * f 5 = 2 * g 5 →
  m = 10/7 := by
  sorry

end NUMINAMATH_CALUDE_find_m_value_l2827_282714


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2827_282764

theorem complex_number_modulus (a : ℝ) : a < 0 → Complex.abs (3 + a * Complex.I) = 5 → a = -4 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2827_282764


namespace NUMINAMATH_CALUDE_remainder_8_pow_1996_mod_5_l2827_282770

theorem remainder_8_pow_1996_mod_5 : 8^1996 % 5 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_8_pow_1996_mod_5_l2827_282770


namespace NUMINAMATH_CALUDE_zero_not_read_in_4006530_l2827_282723

/-- Rule for reading a number -/
def readNumber (n : ℕ) : Bool :=
  sorry

/-- Checks if zero is read out in the number -/
def isZeroReadOut (n : ℕ) : Bool :=
  sorry

theorem zero_not_read_in_4006530 :
  ¬(isZeroReadOut 4006530) ∧ 
  (isZeroReadOut 4650003) ∧ 
  (isZeroReadOut 4650300) ∧ 
  (isZeroReadOut 4006053) := by
  sorry

end NUMINAMATH_CALUDE_zero_not_read_in_4006530_l2827_282723


namespace NUMINAMATH_CALUDE_cone_height_l2827_282780

theorem cone_height (s : ℝ) (a : ℝ) (h : s = 13 ∧ a = 65 * Real.pi) :
  Real.sqrt (s^2 - (a / (s * Real.pi))^2) = 12 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_l2827_282780


namespace NUMINAMATH_CALUDE_eleven_remainders_l2827_282721

theorem eleven_remainders (A : Fin 100 → ℕ) 
  (h_perm : Function.Bijective A) 
  (h_range : ∀ i : Fin 100, A i ∈ Finset.range 101 \ {0}) : 
  let B : Fin 100 → ℕ := λ i => (Finset.range i.succ).sum (λ j => A j)
  Finset.card (Finset.image (λ i => B i % 100) Finset.univ) ≥ 11 := by
sorry

end NUMINAMATH_CALUDE_eleven_remainders_l2827_282721


namespace NUMINAMATH_CALUDE_journey_time_proof_l2827_282755

/-- Proves that the time taken to complete a 224 km journey, where the first half is traveled at 21 km/hr and the second half at 24 km/hr, is equal to 10 hours. -/
theorem journey_time_proof (total_distance : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  total_distance = 224 →
  speed1 = 21 →
  speed2 = 24 →
  (total_distance / 2 / speed1) + (total_distance / 2 / speed2) = 10 := by
  sorry

#check journey_time_proof

end NUMINAMATH_CALUDE_journey_time_proof_l2827_282755


namespace NUMINAMATH_CALUDE_josh_marbles_remaining_l2827_282754

/-- The number of marbles Josh has remaining after losing some. -/
def remaining_marbles (initial : ℝ) (lost : ℝ) : ℝ :=
  initial - lost

/-- Theorem stating that Josh has 7.75 marbles remaining. -/
theorem josh_marbles_remaining :
  remaining_marbles 19.5 11.75 = 7.75 := by
  sorry

end NUMINAMATH_CALUDE_josh_marbles_remaining_l2827_282754


namespace NUMINAMATH_CALUDE_jonah_calories_per_hour_l2827_282782

/-- The number of calories Jonah burns per hour while running -/
def calories_per_hour : ℝ := 30

/-- The number of hours Jonah actually ran -/
def actual_hours : ℝ := 2

/-- The hypothetical number of hours Jonah could have run -/
def hypothetical_hours : ℝ := 5

/-- The additional calories Jonah would have burned if he ran for the hypothetical hours -/
def additional_calories : ℝ := 90

theorem jonah_calories_per_hour :
  calories_per_hour * hypothetical_hours = 
  calories_per_hour * actual_hours + additional_calories :=
sorry

end NUMINAMATH_CALUDE_jonah_calories_per_hour_l2827_282782


namespace NUMINAMATH_CALUDE_unique_solution_to_equation_l2827_282744

theorem unique_solution_to_equation : 
  ∃! (x y : ℕ+), (x.val : ℝ)^6 * (y.val : ℝ)^6 - 19 * (x.val : ℝ)^3 * (y.val : ℝ)^3 + 18 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_to_equation_l2827_282744


namespace NUMINAMATH_CALUDE_continued_fraction_equality_l2827_282798

theorem continued_fraction_equality : 
  2 + (3 / (4 + (5 / (6 + (7/8))))) = 137/52 := by
  sorry

end NUMINAMATH_CALUDE_continued_fraction_equality_l2827_282798


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2827_282790

theorem sqrt_equation_solution :
  ∃! (y : ℝ), y > 0 ∧ 3 * Real.sqrt (4 + y) + 3 * Real.sqrt (4 - y) = 6 * Real.sqrt 3 :=
by
  use 2 * Real.sqrt 3
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2827_282790


namespace NUMINAMATH_CALUDE_sector_area_l2827_282731

/-- The area of a circular sector with central angle π/3 and radius 2 is 2π/3 -/
theorem sector_area (α : Real) (r : Real) (h1 : α = π / 3) (h2 : r = 2) :
  (1 / 2) * α * r^2 = (2 * π) / 3 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2827_282731


namespace NUMINAMATH_CALUDE_widget_purchase_theorem_l2827_282792

/-- Given a person can buy exactly 6 widgets at price p, and 8 widgets at price (p - 1.15),
    prove that the total amount of money they have is 27.60 -/
theorem widget_purchase_theorem (p : ℝ) (h1 : 6 * p = 8 * (p - 1.15)) : 6 * p = 27.60 := by
  sorry

end NUMINAMATH_CALUDE_widget_purchase_theorem_l2827_282792


namespace NUMINAMATH_CALUDE_original_equals_scientific_notation_l2827_282771

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 11580000

/-- The scientific notation representation of the original number -/
def scientific_notation : ScientificNotation := {
  coefficient := 1.158
  exponent := 7
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific_notation : 
  (original_number : ℝ) = scientific_notation.coefficient * (10 : ℝ) ^ scientific_notation.exponent := by
  sorry

end NUMINAMATH_CALUDE_original_equals_scientific_notation_l2827_282771


namespace NUMINAMATH_CALUDE_system_has_four_solutions_l2827_282701

-- Define the system of equations
def equation1 (x y : ℝ) : Prop := 3 * x^2 - x * y + 3 * y^2 = 16
def equation2 (x y : ℝ) : Prop := 7 * x^2 - 4 * x * y + 7 * y^2 = 38

-- Define a solution to be a pair of real numbers satisfying both equations
def is_solution (x y : ℝ) : Prop := equation1 x y ∧ equation2 x y

-- State the theorem
theorem system_has_four_solutions :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ),
    (is_solution x₁ y₁) ∧
    (is_solution x₂ y₂) ∧
    (is_solution x₃ y₃) ∧
    (is_solution x₄ y₄) ∧
    (∀ (x y : ℝ), is_solution x y → (x = x₁ ∧ y = y₁) ∨ (x = x₂ ∧ y = y₂) ∨ (x = x₃ ∧ y = y₃) ∨ (x = x₄ ∧ y = y₄)) :=
by sorry

end NUMINAMATH_CALUDE_system_has_four_solutions_l2827_282701


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_seven_l2827_282768

theorem consecutive_integers_around_sqrt_seven (a b : ℤ) : 
  (b = a + 1) → (a < Real.sqrt 7) → (Real.sqrt 7 < b) → (a + b = 5) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_seven_l2827_282768


namespace NUMINAMATH_CALUDE_square_difference_alice_subtraction_l2827_282752

theorem square_difference (n : ℕ) : n ^ 2 - (n - 1) ^ 2 = 2 * n - 1 := by
  sorry

theorem alice_subtraction : 50 ^ 2 - 49 ^ 2 = 99 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_alice_subtraction_l2827_282752


namespace NUMINAMATH_CALUDE_complex_power_2017_l2827_282753

theorem complex_power_2017 : ((1 + Complex.I) / (1 - Complex.I)) ^ 2017 = Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_power_2017_l2827_282753


namespace NUMINAMATH_CALUDE_simple_interest_sum_l2827_282710

/-- Given a sum of money with simple interest, prove that it equals 1700 --/
theorem simple_interest_sum (P r : ℝ) 
  (h1 : P * (1 + r) = 1717)
  (h2 : P * (1 + 2 * r) = 1734) :
  P = 1700 := by sorry

end NUMINAMATH_CALUDE_simple_interest_sum_l2827_282710


namespace NUMINAMATH_CALUDE_fourth_term_is_one_l2827_282716

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℚ
  is_geometric : ∀ n : ℕ, n > 0 → ∃ q : ℚ, a (n + 1) = a n * q
  first_fifth_diff : a 1 - a 5 = -15/2
  sum_first_four : (a 1) + (a 2) + (a 3) + (a 4) = -5

/-- The fourth term of the geometric sequence is 1 -/
theorem fourth_term_is_one (seq : GeometricSequence) : seq.a 4 = 1 := by
  sorry


end NUMINAMATH_CALUDE_fourth_term_is_one_l2827_282716


namespace NUMINAMATH_CALUDE_simple_interest_duration_l2827_282719

/-- Simple interest calculation -/
theorem simple_interest_duration (P R SI : ℝ) (h1 : P = 10000) (h2 : R = 9) (h3 : SI = 900) :
  (SI * 100) / (P * R) * 12 = 12 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_duration_l2827_282719


namespace NUMINAMATH_CALUDE_complex_sum_powers_of_i_l2827_282745

theorem complex_sum_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ i + i^2 + i^3 + i^4 = 0 := by sorry

end NUMINAMATH_CALUDE_complex_sum_powers_of_i_l2827_282745


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l2827_282713

theorem absolute_value_inequality (x : ℝ) : |2*x - 1| ≤ 1 ↔ 0 ≤ x ∧ x ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l2827_282713


namespace NUMINAMATH_CALUDE_prism_with_21_edges_has_9_faces_l2827_282706

/-- A prism is a polyhedron with two congruent parallel faces (bases) and whose other faces (lateral faces) are parallelograms. -/
structure Prism where
  edges : ℕ

/-- The number of faces in a prism -/
def num_faces (p : Prism) : ℕ :=
  let lateral_faces := p.edges / 3
  lateral_faces + 2

theorem prism_with_21_edges_has_9_faces (p : Prism) (h : p.edges = 21) : num_faces p = 9 := by
  sorry

end NUMINAMATH_CALUDE_prism_with_21_edges_has_9_faces_l2827_282706


namespace NUMINAMATH_CALUDE_merchant_savings_l2827_282762

def initial_order : ℝ := 15000

def apply_discount (amount : ℝ) (discount : ℝ) : ℝ :=
  amount * (1 - discount)

def option1_discounts : List ℝ := [0.1, 0.3, 0.2]
def option2_discounts : List ℝ := [0.25, 0.15, 0.05]

def apply_successive_discounts (amount : ℝ) (discounts : List ℝ) : ℝ :=
  discounts.foldl apply_discount amount

theorem merchant_savings :
  apply_successive_discounts initial_order option2_discounts -
  apply_successive_discounts initial_order option1_discounts = 1524.38 := by
  sorry

end NUMINAMATH_CALUDE_merchant_savings_l2827_282762


namespace NUMINAMATH_CALUDE_gcf_of_24_and_16_l2827_282708

theorem gcf_of_24_and_16 :
  let n : ℕ := 24
  let m : ℕ := 16
  let lcm_nm : ℕ := 48
  lcm n m = lcm_nm →
  Nat.gcd n m = 8 := by
sorry

end NUMINAMATH_CALUDE_gcf_of_24_and_16_l2827_282708


namespace NUMINAMATH_CALUDE_solve_for_y_l2827_282727

theorem solve_for_y (x y : ℝ) : 3 * x - 2 * y = 6 → y = (3 * x / 2) - 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2827_282727


namespace NUMINAMATH_CALUDE_reflect_across_y_axis_l2827_282722

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The reflection of a point across the y-axis. -/
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Theorem: The coordinates of a point P(x,y) with respect to the y-axis are (-x,y). -/
theorem reflect_across_y_axis (p : Point2D) :
  reflectAcrossYAxis p = { x := -p.x, y := p.y } := by
  sorry

#check reflect_across_y_axis

end NUMINAMATH_CALUDE_reflect_across_y_axis_l2827_282722


namespace NUMINAMATH_CALUDE_constant_point_on_graph_unique_constant_point_l2827_282788

/-- The quadratic function f(x) that passes through a constant point for any real m -/
def f (m : ℝ) (x : ℝ) : ℝ := 3 * x^2 - m * x + 2 * m + 1

/-- The constant point that lies on the graph of f(x) for all real m -/
def constant_point : ℝ × ℝ := (2, 13)

/-- Theorem stating that the constant_point lies on the graph of f(x) for all real m -/
theorem constant_point_on_graph :
  ∀ m : ℝ, f m (constant_point.1) = constant_point.2 :=
by sorry

/-- Theorem stating that constant_point is the unique point satisfying the condition -/
theorem unique_constant_point :
  ∀ p : ℝ × ℝ, (∀ m : ℝ, f m p.1 = p.2) → p = constant_point :=
by sorry

end NUMINAMATH_CALUDE_constant_point_on_graph_unique_constant_point_l2827_282788


namespace NUMINAMATH_CALUDE_no_solution_to_inequality_l2827_282730

theorem no_solution_to_inequality (x : ℝ) :
  x ≥ -1/4 → ¬(-1 - 1/(3*x + 4) < 2) :=
by sorry

end NUMINAMATH_CALUDE_no_solution_to_inequality_l2827_282730


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l2827_282709

theorem inverse_variation_problem (a b : ℝ) (k : ℝ) (h1 : a * b^3 = k) (h2 : 8 * 2^3 = k) :
  a * 4^3 = k → a = 1 := by
sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l2827_282709


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l2827_282769

/-- The distance between the foci of a hyperbola defined by xy = 4 is 4√2 -/
theorem hyperbola_foci_distance :
  ∃ (f₁ f₂ : ℝ × ℝ), 
    (∀ (x y : ℝ), x * y = 4 → (x - f₁.1)^2 / (f₂.1 - f₁.1)^2 - 
                               (y - f₁.2)^2 / (f₂.2 - f₁.2)^2 = 1) ∧
    Real.sqrt ((f₂.1 - f₁.1)^2 + (f₂.2 - f₁.2)^2) = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l2827_282769


namespace NUMINAMATH_CALUDE_tobias_driveways_shoveled_l2827_282746

/-- Calculates the number of driveways Tobias shoveled given his earnings and expenses. -/
theorem tobias_driveways_shoveled (
  shoe_cost : ℕ)
  (saving_months : ℕ)
  (monthly_allowance : ℕ)
  (lawn_mowing_fee : ℕ)
  (driveway_shoveling_fee : ℕ)
  (change_after_purchase : ℕ)
  (lawns_mowed : ℕ)
  (h1 : shoe_cost = 95)
  (h2 : saving_months = 3)
  (h3 : monthly_allowance = 5)
  (h4 : lawn_mowing_fee = 15)
  (h5 : driveway_shoveling_fee = 7)
  (h6 : change_after_purchase = 15)
  (h7 : lawns_mowed = 4) :
  (shoe_cost + change_after_purchase
    - saving_months * monthly_allowance
    - lawns_mowed * lawn_mowing_fee) / driveway_shoveling_fee = 5 :=
by sorry


end NUMINAMATH_CALUDE_tobias_driveways_shoveled_l2827_282746


namespace NUMINAMATH_CALUDE_matthew_crackers_l2827_282718

def crackers_problem (initial_crackers : ℕ) (friends : ℕ) (crackers_per_friend : ℕ) : Prop :=
  initial_crackers - (friends * crackers_per_friend) = 3

theorem matthew_crackers : crackers_problem 24 3 7 := by
  sorry

end NUMINAMATH_CALUDE_matthew_crackers_l2827_282718


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l2827_282740

theorem min_value_sqrt_sum_squares (a b m n : ℝ) 
  (h1 : a^2 + b^2 = 5)
  (h2 : m*a + n*b = 5) : 
  Real.sqrt (m^2 + n^2) ≥ Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_squares_l2827_282740


namespace NUMINAMATH_CALUDE_triangle_equality_l2827_282776

theorem triangle_equality (a b c : ℝ) 
  (h1 : |a - b| ≥ |c|) 
  (h2 : |b - c| ≥ |a|) 
  (h3 : |c - a| ≥ |b|) : 
  a = b + c ∨ b = c + a ∨ c = a + b := by
  sorry

end NUMINAMATH_CALUDE_triangle_equality_l2827_282776


namespace NUMINAMATH_CALUDE_three_roots_implies_m_equals_two_l2827_282742

/-- The function f(x) = x^2 - 2|x| + 2 - m -/
def f (x m : ℝ) : ℝ := x^2 - 2 * abs x + 2 - m

/-- The number of roots of f(x) for a given m -/
def num_roots (m : ℝ) : ℕ := sorry

theorem three_roots_implies_m_equals_two :
  ∀ m : ℝ, num_roots m = 3 → m = 2 := by sorry

end NUMINAMATH_CALUDE_three_roots_implies_m_equals_two_l2827_282742


namespace NUMINAMATH_CALUDE_randys_trip_length_l2827_282702

theorem randys_trip_length :
  ∀ (x : ℚ),
  (x / 4 : ℚ) + 40 + 10 + (x / 6 : ℚ) = x →
  x = 600 / 7 := by
sorry

end NUMINAMATH_CALUDE_randys_trip_length_l2827_282702


namespace NUMINAMATH_CALUDE_train_passing_time_l2827_282712

/-- Proves that a train of given length and speed takes a specific time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 180 → 
  train_speed_kmh = 54 → 
  passing_time = 12 → 
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_passing_time

end NUMINAMATH_CALUDE_train_passing_time_l2827_282712


namespace NUMINAMATH_CALUDE_platform_length_problem_solution_l2827_282761

/-- Calculates the length of a platform given train parameters --/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : ℝ :=
  let train_speed_mps := train_speed_kmph * 1000 / 3600
  let total_distance := train_speed_mps * crossing_time
  total_distance - train_length

/-- The length of the platform is 208.8 meters --/
theorem problem_solution : 
  (platform_length 180 70 20) = 208.8 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_problem_solution_l2827_282761


namespace NUMINAMATH_CALUDE_zoo_count_l2827_282720

/-- Represents the number of peacocks in the zoo -/
def num_peacocks : ℕ := 7

/-- Represents the number of tortoises in the zoo -/
def num_tortoises : ℕ := 17 - num_peacocks

/-- The total number of legs in the zoo -/
def total_legs : ℕ := 54

/-- The total number of heads in the zoo -/
def total_heads : ℕ := 17

/-- Each peacock has 2 legs -/
def peacock_legs : ℕ := 2

/-- Each peacock has 1 head -/
def peacock_head : ℕ := 1

/-- Each tortoise has 4 legs -/
def tortoise_legs : ℕ := 4

/-- Each tortoise has 1 head -/
def tortoise_head : ℕ := 1

theorem zoo_count :
  num_peacocks * peacock_legs + num_tortoises * tortoise_legs = total_legs ∧
  num_peacocks * peacock_head + num_tortoises * tortoise_head = total_heads :=
by sorry

end NUMINAMATH_CALUDE_zoo_count_l2827_282720


namespace NUMINAMATH_CALUDE_power_product_equality_l2827_282783

theorem power_product_equality : 2^4 * 3^2 * 5^2 * 7 = 6300 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l2827_282783


namespace NUMINAMATH_CALUDE_x_equation_proof_l2827_282741

theorem x_equation_proof (x : ℝ) (a b : ℕ+) 
  (h1 : x^2 + 5*x + 4/x + 1/x^2 = 34)
  (h2 : x = a + Real.sqrt b) : 
  a + b = 5 := by
  sorry

end NUMINAMATH_CALUDE_x_equation_proof_l2827_282741


namespace NUMINAMATH_CALUDE_not_tileable_rectangles_l2827_282738

/-- A domino is a 1x2 rectangle -/
structure Domino :=
  (width : Nat := 2)
  (height : Nat := 1)

/-- A rectangle with given width and height -/
structure Rectangle :=
  (width : Nat)
  (height : Nat)

/-- Predicate to check if a rectangle is (1,2)-tileable -/
def is_tileable (r : Rectangle) : Prop := sorry

/-- Theorem stating that 1xk and 2xn (where 4 ∤ n) rectangles are not (1,2)-tileable -/
theorem not_tileable_rectangles :
  ∀ (k n : Nat), 
    (¬ is_tileable ⟨1, k⟩) ∧ 
    ((¬ (4 ∣ n)) → ¬ is_tileable ⟨2, n⟩) :=
by sorry

end NUMINAMATH_CALUDE_not_tileable_rectangles_l2827_282738


namespace NUMINAMATH_CALUDE_steel_experiment_golden_ratio_l2827_282781

/-- The 0.618 method calculation for a given range -/
def golden_ratio_method (lower_bound upper_bound : ℝ) : ℝ :=
  lower_bound + (upper_bound - lower_bound) * 0.618

/-- Theorem: The 0.618 method for the given steel experiment -/
theorem steel_experiment_golden_ratio :
  let lower_bound : ℝ := 500
  let upper_bound : ℝ := 1000
  golden_ratio_method lower_bound upper_bound = 809 := by
  sorry

end NUMINAMATH_CALUDE_steel_experiment_golden_ratio_l2827_282781


namespace NUMINAMATH_CALUDE_february_first_is_friday_l2827_282791

/-- Represents days of the week -/
inductive Weekday
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a day in February -/
structure FebruaryDay where
  date : Nat
  weekday : Weekday

/-- Represents the condition of the student groups visiting Teacher Li -/
structure StudentVisit where
  day : FebruaryDay
  groupSize : Nat

/-- The main theorem -/
theorem february_first_is_friday 
  (visit : StudentVisit)
  (h1 : visit.day.weekday = Weekday.Sunday)
  (h2 : visit.day.date = 3 * visit.groupSize * visit.groupSize)
  (h3 : visit.groupSize > 1)
  : (⟨1, Weekday.Friday⟩ : FebruaryDay) = 
    {date := 1, weekday := Weekday.Friday} :=
by sorry

end NUMINAMATH_CALUDE_february_first_is_friday_l2827_282791


namespace NUMINAMATH_CALUDE_floor_times_self_eq_120_l2827_282760

theorem floor_times_self_eq_120 :
  ∃! (x : ℝ), x > 0 ∧ (⌊x⌋ : ℝ) * x = 120 ∧ x = 120 / 11 :=
by sorry

end NUMINAMATH_CALUDE_floor_times_self_eq_120_l2827_282760


namespace NUMINAMATH_CALUDE_reciprocal_of_five_eighths_l2827_282704

theorem reciprocal_of_five_eighths :
  let x : ℚ := 5 / 8
  let reciprocal (q : ℚ) : ℚ := 1 / q
  reciprocal x = 8 / 5 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_of_five_eighths_l2827_282704


namespace NUMINAMATH_CALUDE_delta_value_l2827_282736

theorem delta_value (Δ : ℤ) : 4 * (-3) = Δ + 3 → Δ = -15 := by
  sorry

end NUMINAMATH_CALUDE_delta_value_l2827_282736


namespace NUMINAMATH_CALUDE_mark_reading_time_l2827_282785

/-- Mark's daily reading time in hours -/
def daily_reading_time : ℕ := 2

/-- Number of days in a week -/
def days_per_week : ℕ := 7

/-- Mark's planned increase in weekly reading time in hours -/
def weekly_increase : ℕ := 4

/-- Mark's desired weekly reading time in hours -/
def desired_weekly_reading_time : ℕ := daily_reading_time * days_per_week + weekly_increase

theorem mark_reading_time :
  desired_weekly_reading_time = 18 := by
  sorry

end NUMINAMATH_CALUDE_mark_reading_time_l2827_282785


namespace NUMINAMATH_CALUDE_square_perimeter_ratio_l2827_282751

theorem square_perimeter_ratio (s₁ s₂ : ℝ) (h : s₁ > 0) (h' : s₂ > 0) :
  s₂ * Real.sqrt 2 = 1.5 * (s₁ * Real.sqrt 2) →
  (4 * s₂) / (4 * s₁) = 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_square_perimeter_ratio_l2827_282751


namespace NUMINAMATH_CALUDE_amphibian_count_l2827_282715

/-- The total number of amphibians observed in the pond -/
def total_amphibians (frogs salamanders tadpoles newts : ℕ) : ℕ :=
  frogs + salamanders + tadpoles + newts

/-- Theorem stating that the total number of amphibians is 42 -/
theorem amphibian_count : 
  total_amphibians 7 4 30 1 = 42 := by sorry

end NUMINAMATH_CALUDE_amphibian_count_l2827_282715


namespace NUMINAMATH_CALUDE_problem_solution_l2827_282774

theorem problem_solution (A B : ℝ) 
  (h1 : 30 - (4 * A + 5) = 3 * B) 
  (h2 : B = 2 * A) : 
  A = 2.5 ∧ B = 5 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2827_282774


namespace NUMINAMATH_CALUDE_equation_solution_l2827_282777

theorem equation_solution (x : ℝ) : x * (3 * x + 6) = 7 * (3 * x + 6) ↔ x = 7 ∨ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2827_282777


namespace NUMINAMATH_CALUDE_remaining_note_denomination_l2827_282711

theorem remaining_note_denomination 
  (total_amount : ℕ)
  (total_notes : ℕ)
  (fifty_notes : ℕ)
  (h1 : total_amount = 10350)
  (h2 : total_notes = 36)
  (h3 : fifty_notes = 17) :
  (total_amount - fifty_notes * 50) / (total_notes - fifty_notes) = 500 :=
by sorry

end NUMINAMATH_CALUDE_remaining_note_denomination_l2827_282711


namespace NUMINAMATH_CALUDE_yanni_remaining_money_l2827_282756

/-- Calculates the remaining money in cents after Yanni's transactions --/
def remaining_money_in_cents (initial_money : ℚ) (mother_gave : ℚ) (found_money : ℚ) (toy_cost : ℚ) : ℕ :=
  let total_money := initial_money + mother_gave + found_money
  let remaining_money := total_money - toy_cost
  (remaining_money * 100).floor.toNat

/-- Proves that Yanni has 15 cents left after his transactions --/
theorem yanni_remaining_money :
  remaining_money_in_cents 0.85 0.40 0.50 1.60 = 15 := by
  sorry

end NUMINAMATH_CALUDE_yanni_remaining_money_l2827_282756


namespace NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l2827_282705

/-- The vertex of a parabola in the form y = a(x-h)^2 + k is (h, k) --/
theorem parabola_vertex (a h k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * (x - h)^2 + k
  ∃! (x₀ y₀ : ℝ), (∀ x, f x ≥ f x₀) ∧ f x₀ = y₀ ∧ (x₀, y₀) = (h, k) :=
sorry

/-- The vertex of the parabola y = -2(x-3)^2 - 2 is (3, -2) --/
theorem specific_parabola_vertex :
  let f : ℝ → ℝ := λ x ↦ -2 * (x - 3)^2 - 2
  ∃! (x₀ y₀ : ℝ), (∀ x, f x ≥ f x₀) ∧ f x₀ = y₀ ∧ (x₀, y₀) = (3, -2) :=
sorry

end NUMINAMATH_CALUDE_parabola_vertex_specific_parabola_vertex_l2827_282705


namespace NUMINAMATH_CALUDE_valid_three_digit_numbers_l2827_282794

/-- The count of valid three-digit numbers -/
def valid_count : ℕ := 738

/-- The total count of three-digit numbers -/
def total_three_digit_numbers : ℕ := 900

/-- The count of three-digit numbers with exactly two non-adjacent identical digits -/
def count_two_same_not_adjacent : ℕ := 81

/-- The count of three-digit numbers with identical first and last digits -/
def count_first_last_same : ℕ := 81

/-- Theorem stating the count of valid three-digit numbers -/
theorem valid_three_digit_numbers :
  valid_count = total_three_digit_numbers - count_two_same_not_adjacent - count_first_last_same :=
by sorry

end NUMINAMATH_CALUDE_valid_three_digit_numbers_l2827_282794


namespace NUMINAMATH_CALUDE_sum_of_prime_factors_1320_l2827_282729

def sum_of_prime_factors (n : ℕ) : ℕ :=
  (Nat.factors n).toFinset.sum id

theorem sum_of_prime_factors_1320 :
  sum_of_prime_factors 1320 = 21 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_prime_factors_1320_l2827_282729


namespace NUMINAMATH_CALUDE_power_of_17_mod_26_l2827_282726

theorem power_of_17_mod_26 : 17^1999 % 26 = 17 := by
  sorry

end NUMINAMATH_CALUDE_power_of_17_mod_26_l2827_282726


namespace NUMINAMATH_CALUDE_parabola_chord_intersection_l2827_282734

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola y = x^2 -/
def parabola (p : Point) : Prop := p.y = p.x^2

/-- Represents the ratio condition AC:CB = 5:2 -/
def ratio_condition (a b c : Point) : Prop :=
  (c.x - a.x) / (b.x - c.x) = 5 / 2

theorem parabola_chord_intersection :
  ∀ (a b c : Point),
    parabola a →
    parabola b →
    c.x = 0 →
    c.y = 20 →
    ratio_condition a b c →
    ((a.x = -5 * Real.sqrt 2 ∧ b.x = 2 * Real.sqrt 2) ∨
     (a.x = 5 * Real.sqrt 2 ∧ b.x = -2 * Real.sqrt 2)) :=
by sorry

end NUMINAMATH_CALUDE_parabola_chord_intersection_l2827_282734


namespace NUMINAMATH_CALUDE_player_pay_is_23000_l2827_282795

/-- Represents the player's performance in a single game -/
structure GamePerformance :=
  (points : ℕ)
  (assists : ℕ)
  (rebounds : ℕ)
  (steals : ℕ)

/-- Calculates the base pay based on average points per game -/
def basePay (games : List GamePerformance) : ℕ :=
  if (games.map GamePerformance.points).sum / games.length ≥ 30 then 10000 else 8000

/-- Calculates the assists bonus based on total assists -/
def assistsBonus (games : List GamePerformance) : ℕ :=
  let totalAssists := (games.map GamePerformance.assists).sum
  if totalAssists ≥ 20 then 5000
  else if totalAssists ≥ 10 then 3000
  else 1000

/-- Calculates the rebounds bonus based on total rebounds -/
def reboundsBonus (games : List GamePerformance) : ℕ :=
  let totalRebounds := (games.map GamePerformance.rebounds).sum
  if totalRebounds ≥ 40 then 5000
  else if totalRebounds ≥ 20 then 3000
  else 1000

/-- Calculates the steals bonus based on total steals -/
def stealsBonus (games : List GamePerformance) : ℕ :=
  let totalSteals := (games.map GamePerformance.steals).sum
  if totalSteals ≥ 15 then 5000
  else if totalSteals ≥ 5 then 3000
  else 1000

/-- Calculates the total pay for the week -/
def totalPay (games : List GamePerformance) : ℕ :=
  basePay games + assistsBonus games + reboundsBonus games + stealsBonus games

/-- Theorem: Given the player's performance, the total pay for the week is $23,000 -/
theorem player_pay_is_23000 (games : List GamePerformance) 
  (h1 : games = [
    ⟨30, 5, 7, 3⟩, 
    ⟨28, 6, 5, 2⟩, 
    ⟨32, 4, 9, 1⟩, 
    ⟨34, 3, 11, 2⟩, 
    ⟨26, 2, 8, 3⟩
  ]) : 
  totalPay games = 23000 := by
  sorry


end NUMINAMATH_CALUDE_player_pay_is_23000_l2827_282795


namespace NUMINAMATH_CALUDE_marathon_run_solution_l2827_282772

/-- Represents the marathon run problem -/
def marathon_run (x : ℝ) : Prop :=
  let total_distance : ℝ := 95
  let total_time : ℝ := 15
  let speed1 : ℝ := 8
  let speed2 : ℝ := 6
  let speed3 : ℝ := 5
  (speed1 * x + speed2 * x + speed3 * (total_time - 2 * x) = total_distance) ∧
  (x ≥ 0) ∧ (x ≤ total_time / 2)

/-- Proves that the only solution to the marathon run problem is 5 hours at each speed -/
theorem marathon_run_solution :
  ∃! x : ℝ, marathon_run x ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_marathon_run_solution_l2827_282772


namespace NUMINAMATH_CALUDE_infinite_primes_solution_l2827_282758

theorem infinite_primes_solution (f : ℕ → ℕ) (k : ℕ) 
  (h_inj : Function.Injective f) 
  (h_bound : ∀ n, f n ≤ n^k) :
  ∃ S : Set ℕ, Set.Infinite S ∧ 
    (∀ q ∈ S, Nat.Prime q ∧ 
      ∃ p, Nat.Prime p ∧ f p ≡ 0 [MOD q]) :=
sorry

end NUMINAMATH_CALUDE_infinite_primes_solution_l2827_282758


namespace NUMINAMATH_CALUDE_slower_walking_speed_l2827_282748

theorem slower_walking_speed (usual_time : ℝ) (delay : ℝ) : 
  usual_time = 40 → delay = 10 → 
  (usual_time / (usual_time + delay)) = (4 : ℝ) / 5 := by
  sorry

end NUMINAMATH_CALUDE_slower_walking_speed_l2827_282748


namespace NUMINAMATH_CALUDE_paper_clips_remaining_l2827_282775

theorem paper_clips_remaining (initial : ℕ) (used : ℕ) (remaining : ℕ) : 
  initial = 85 → used = 59 → remaining = initial - used → remaining = 26 := by
  sorry

end NUMINAMATH_CALUDE_paper_clips_remaining_l2827_282775


namespace NUMINAMATH_CALUDE_tangent_line_and_max_value_l2827_282749

noncomputable section

-- Define the function f
def f (e a b x : ℝ) : ℝ := Real.exp (-x) * (a * x^2 + b * x + 1)

-- Define the derivative of f
def f' (e a b x : ℝ) : ℝ := 
  -Real.exp (-x) * (a * x^2 + b * x + 1) + Real.exp (-x) * (2 * a * x + b)

theorem tangent_line_and_max_value 
  (e : ℝ) (h_e : e > 0) :
  ∀ a b : ℝ, 
  (a > 0) → 
  (f' e a b (-1) = 0) →
  (
    -- Part I
    (a = 1) → 
    (∃ m c : ℝ, m = 1 ∧ c = 1 ∧ 
      ∀ x : ℝ, f e a b x = m * x + c → x = 0
    ) ∧
    -- Part II
    (a > 1/5) → 
    (∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f e a b x ≤ 4 * e) →
    (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ f e a b x = 4 * e) →
    (a = (8 * e^2 - 3) / 5 ∧ b = (12 * e^2 - 2) / 5)
  ) := by sorry

end NUMINAMATH_CALUDE_tangent_line_and_max_value_l2827_282749


namespace NUMINAMATH_CALUDE_store_revenue_comparison_l2827_282703

theorem store_revenue_comparison (december : ℝ) (november : ℝ) (january : ℝ)
  (h1 : november = (2/5) * december)
  (h2 : january = (1/5) * november) :
  december = (25/6) * ((november + january) / 2) := by
  sorry

end NUMINAMATH_CALUDE_store_revenue_comparison_l2827_282703


namespace NUMINAMATH_CALUDE_intersection_and_chord_length_l2827_282728

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 2)^2 = 4

-- Define the line l₃
def line_l₃ (x y : ℝ) : Prop :=
  4*x - 3*y - 1 = 0

-- Theorem statement
theorem intersection_and_chord_length :
  ∃ (A B : ℝ × ℝ),
    A ≠ B ∧
    circle_C A.1 A.2 ∧
    circle_C B.1 B.2 ∧
    line_l₃ A.1 A.2 ∧
    line_l₃ B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 12 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_and_chord_length_l2827_282728


namespace NUMINAMATH_CALUDE_trajectory_of_point_P_l2827_282724

/-- The trajectory of a point P on the curve ρcos θ + 2ρsin θ = 3, where 0 ≤ θ ≤ π/4 and ρ > 0,
    is a line segment with endpoints (1,1) and (3,0). -/
theorem trajectory_of_point_P (θ : ℝ) (ρ : ℝ) (h1 : 0 ≤ θ) (h2 : θ ≤ π/4) (h3 : ρ > 0)
  (h4 : ρ * Real.cos θ + 2 * ρ * Real.sin θ = 3) :
  ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧
  ρ * Real.cos θ = 3 - 2 * t ∧
  ρ * Real.sin θ = t :=
by sorry

end NUMINAMATH_CALUDE_trajectory_of_point_P_l2827_282724


namespace NUMINAMATH_CALUDE_problem_statement_l2827_282793

theorem problem_statement (d : ℕ) (h : d = 4) :
  (d^d - d*(d-2)^d)^d = 1358954496 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2827_282793


namespace NUMINAMATH_CALUDE_complex_number_location_l2827_282797

/-- The complex number z = (2+i)/(1+i) is located in Quadrant IV -/
theorem complex_number_location :
  let z : ℂ := (2 + I) / (1 + I)
  (z.re > 0) ∧ (z.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l2827_282797


namespace NUMINAMATH_CALUDE_inequality_solution_set_range_of_m_l2827_282735

-- Define the function f(x) = |x+1|
def f (x : ℝ) : ℝ := |x + 1|

-- Theorem for the solution set of the inequality
theorem inequality_solution_set :
  {x : ℝ | f x ≥ 2 * x + 1} = {x : ℝ | x ≤ 0} :=
sorry

-- Theorem for the range of m
theorem range_of_m :
  ∀ m : ℝ, (∃ x : ℝ, f (x - 2) - f (x + 6) < m) ↔ m > -8 :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_range_of_m_l2827_282735


namespace NUMINAMATH_CALUDE_haley_gives_away_48_papers_l2827_282743

/-- The number of origami papers Haley gives away -/
def total_papers (num_cousins : ℕ) (papers_per_cousin : ℕ) : ℕ :=
  num_cousins * papers_per_cousin

/-- Theorem stating that Haley gives away 48 origami papers -/
theorem haley_gives_away_48_papers : total_papers 6 8 = 48 := by
  sorry

end NUMINAMATH_CALUDE_haley_gives_away_48_papers_l2827_282743


namespace NUMINAMATH_CALUDE_race_time_calculation_l2827_282759

theorem race_time_calculation (race_length : ℝ) (distance_difference : ℝ) (time_difference : ℝ) :
  race_length = 1000 →
  distance_difference = 40 →
  time_difference = 8 →
  ∃ (time_A : ℝ),
    time_A > 0 ∧
    race_length / time_A = (race_length - distance_difference) / (time_A + time_difference) ∧
    time_A = 200 := by
  sorry

end NUMINAMATH_CALUDE_race_time_calculation_l2827_282759


namespace NUMINAMATH_CALUDE_temperature_conversion_deviation_l2827_282750

theorem temperature_conversion_deviation (C : ℝ) : 
  let F_approx := 2 * C + 30
  let F_exact := (9 / 5) * C + 32
  let deviation := (F_approx - F_exact) / F_exact
  (40 / 29 ≤ C ∧ C ≤ 360 / 11) ↔ (abs deviation ≤ 0.05) :=
by sorry

end NUMINAMATH_CALUDE_temperature_conversion_deviation_l2827_282750


namespace NUMINAMATH_CALUDE_b_over_a_squared_is_seven_l2827_282763

theorem b_over_a_squared_is_seven (a : ℕ) (k : ℕ) (b : ℕ) :
  a > 1 →
  b = a * (10^k + 1) →
  k > 0 →
  a < 10^k →
  a^2 ∣ b →
  b / a^2 = 7 := by
sorry

end NUMINAMATH_CALUDE_b_over_a_squared_is_seven_l2827_282763


namespace NUMINAMATH_CALUDE_smallest_sum_reciprocals_l2827_282787

theorem smallest_sum_reciprocals (x y : ℕ+) : 
  x ≠ y → 
  (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 20 → 
  (∀ a b : ℕ+, a ≠ b → (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 20 → (x + y : ℕ) ≤ (a + b : ℕ)) → 
  (x + y : ℕ) = 81 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_reciprocals_l2827_282787


namespace NUMINAMATH_CALUDE_cube_sum_divisibility_l2827_282733

theorem cube_sum_divisibility (x y z : ℤ) :
  7 ∣ (x^3 + y^3 + z^3) → 7 ∣ (x * y * z) := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_divisibility_l2827_282733


namespace NUMINAMATH_CALUDE_equidistant_point_equation_l2827_282700

/-- A point (x, y) is equidistant from the x-axis and y-axis if and only if y² = x² -/
theorem equidistant_point_equation (x y : ℝ) :
  (abs x = abs y) ↔ (y^2 = x^2) := by
  sorry

end NUMINAMATH_CALUDE_equidistant_point_equation_l2827_282700


namespace NUMINAMATH_CALUDE_problem_solution_l2827_282717

theorem problem_solution (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h1 : a^2 + a*b + b^2 = 9)
  (h2 : b^2 + b*c + c^2 = 52)
  (h3 : c^2 + c*a + a^2 = 49) :
  (49*b^2 - 33*b*c + 9*c^2) / a^2 = 52 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2827_282717


namespace NUMINAMATH_CALUDE_bakers_pastries_l2827_282789

/-- Baker's pastry problem -/
theorem bakers_pastries 
  (initial_cakes : ℕ)
  (sold_cakes : ℕ)
  (sold_pastries : ℕ)
  (remaining_pastries : ℕ)
  (h1 : initial_cakes = 7)
  (h2 : sold_cakes = 15)
  (h3 : sold_pastries = 103)
  (h4 : remaining_pastries = 45) :
  sold_pastries + remaining_pastries = 148 :=
sorry

end NUMINAMATH_CALUDE_bakers_pastries_l2827_282789


namespace NUMINAMATH_CALUDE_remaining_apple_pies_l2827_282767

/-- Proves the number of apple pies remaining with Cooper --/
theorem remaining_apple_pies (pies_per_day : ℕ) (days : ℕ) (pies_eaten : ℕ) : 
  pies_per_day = 7 → days = 12 → pies_eaten = 50 → 
  pies_per_day * days - pies_eaten = 34 := by
  sorry

#check remaining_apple_pies

end NUMINAMATH_CALUDE_remaining_apple_pies_l2827_282767


namespace NUMINAMATH_CALUDE_win_sector_area_l2827_282747

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 1/4) :
  p * π * r^2 = 16 * π := by
  sorry

end NUMINAMATH_CALUDE_win_sector_area_l2827_282747


namespace NUMINAMATH_CALUDE_function_inequality_l2827_282786

theorem function_inequality (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * deriv f x ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 := by
  sorry

end NUMINAMATH_CALUDE_function_inequality_l2827_282786


namespace NUMINAMATH_CALUDE_original_equals_scientific_l2827_282779

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- The number to be expressed in scientific notation -/
def original_number : ℕ := 24538

/-- The scientific notation representation of the original number -/
def scientific_form : ScientificNotation := {
  coefficient := 2.4538,
  exponent := 4,
  is_valid := by sorry
}

/-- Theorem stating that the original number is equal to its scientific notation representation -/
theorem original_equals_scientific :
  (original_number : ℝ) = scientific_form.coefficient * (10 : ℝ) ^ scientific_form.exponent :=
by sorry

end NUMINAMATH_CALUDE_original_equals_scientific_l2827_282779


namespace NUMINAMATH_CALUDE_max_value_of_f_min_value_of_f_in_interval_range_of_a_l2827_282765

noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 5) / Real.exp x

theorem max_value_of_f :
  ∃ (x : ℝ), f x = 5 ∧ ∀ (y : ℝ), f y ≤ 5 :=
sorry

theorem min_value_of_f_in_interval :
  ∃ (x : ℝ), x ≤ 0 ∧ f x = -Real.exp 3 ∧ ∀ (y : ℝ), y ≤ 0 → f y ≥ -Real.exp 3 :=
sorry

theorem range_of_a (a : ℝ) :
  (∀ (x : ℝ), x^2 + 5*x + 5 - a * Real.exp x ≥ 0) ↔ a ≤ -Real.exp 3 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_min_value_of_f_in_interval_range_of_a_l2827_282765


namespace NUMINAMATH_CALUDE_lesser_number_l2827_282784

theorem lesser_number (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 6) : y = 22 := by
  sorry

end NUMINAMATH_CALUDE_lesser_number_l2827_282784


namespace NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2827_282766

theorem smallest_x_satisfying_equation : 
  ∃ (x : ℝ), x > 0 ∧ 
    (∀ (y : ℝ), y > 0 → ⌊y^2⌋ - y * ⌊y⌋ = 8 → x ≤ y) ∧
    ⌊x^2⌋ - x * ⌊x⌋ = 8 ∧
    x = 89/9 := by
  sorry

end NUMINAMATH_CALUDE_smallest_x_satisfying_equation_l2827_282766
