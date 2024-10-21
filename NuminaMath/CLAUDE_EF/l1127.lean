import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1127_112721

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - 1)^2 + a * (Real.log x - x + 1)

-- State the theorem
theorem f_properties (a : ℝ) :
  (∀ x > 1, f a x > 0) ↔ a ≤ 2 ∧
  (a ≤ 2 →
    (∃! x, x ∈ Set.Ioo 0 2 ∧ f a x + a + 1 = 0) ↔
    (a < -2 / Real.log 2 ∨ a = -1 ∨ (0 < a ∧ a ≤ 2))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1127_112721


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l1127_112726

def game_rounds : ℕ := 8
def alex_win_prob : ℝ := 0.4

theorem game_probability (mel_win_prob chelsea_win_prob : ℝ) 
  (h1 : mel_win_prob = 3 * chelsea_win_prob)
  (h2 : alex_win_prob + mel_win_prob + chelsea_win_prob = 1) :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.0001 ∧ 
    (Nat.choose game_rounds 4 * Nat.choose 4 3 : ℝ) * 
    (alex_win_prob ^ 4 * mel_win_prob ^ 3 * chelsea_win_prob) = 0.0979 + ε := by
  sorry

#check game_probability

end NUMINAMATH_CALUDE_ERRORFEEDBACK_game_probability_l1127_112726


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l1127_112789

open Real

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := exp x / (1 + a * x^2)

-- State the theorem
theorem monotone_f_implies_a_range (a : ℝ) :
  a > 0 →
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Ioc 0 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_f_implies_a_range_l1127_112789


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_container_evaporation_l1127_112717

theorem container_evaporation (initial_content : ℚ) : 
  initial_content > 0 → 
  (initial_content * (1 - 2/3) * (1 - 1/4)) = initial_content * (1/4) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_container_evaporation_l1127_112717


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_to_equation_l1127_112777

theorem two_solutions_to_equation : 
  ∃! (s : Finset ℝ), (∀ x ∈ s, (2 : ℝ)^(x^2 - 6*x + 8) = 8) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_solutions_to_equation_l1127_112777


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_packaging_boxes_l1127_112714

/-- Represents the number of cardstock sheets used for side pieces -/
def x : ℕ := sorry

/-- Represents the number of cardstock sheets used for bottom pieces -/
def y : ℕ := sorry

/-- The total number of cardstock sheets -/
def total_sheets : ℕ := 14

/-- The number of side pieces that can be cut from one sheet -/
def side_pieces_per_sheet : ℕ := 2

/-- The number of bottom pieces that can be cut from one sheet -/
def bottom_pieces_per_sheet : ℕ := 3

/-- The number of side pieces required for one packaging box -/
def side_pieces_per_box : ℕ := 1

/-- The number of bottom pieces required for one packaging box -/
def bottom_pieces_per_box : ℕ := 2

theorem max_packaging_boxes :
  x + y = total_sheets →
  side_pieces_per_sheet * x = (bottom_pieces_per_sheet * y) / 2 →
  min (side_pieces_per_sheet * x) ((bottom_pieces_per_sheet * y) / bottom_pieces_per_box) = 12 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_packaging_boxes_l1127_112714


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_segment_l1127_112704

/-- Given three points A, B, and C in a metric space, if for any point M,
    either MA ≤ MB or MA ≤ MC, then A lies on the line segment BC. -/
theorem point_on_segment 
  {α : Type*} [MetricSpace α] [NormedAddCommGroup α] [NormedSpace ℝ α] (A B C : α) : 
  (∀ M : α, (dist M A ≤ dist M B) ∨ (dist M A ≤ dist M C)) → 
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ A = (1 - t) • B + t • C :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_segment_l1127_112704


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_from_prime_factorization_l1127_112778

-- Define a type for prime factorization
def PrimeFactorization := List (Nat × Nat)

-- Function to calculate the number of divisors
def numDivisors (factorization : PrimeFactorization) : Nat :=
  factorization.foldl (fun acc (_, α) => acc * (α + 1)) 1

-- Theorem statement
theorem num_divisors_from_prime_factorization 
  (n : Nat) 
  (factorization : PrimeFactorization) 
  (h_factorization : n = factorization.foldl (fun acc (p, α) => acc * p^α) 1) :
  (Nat.divisors n).card = numDivisors factorization := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_num_divisors_from_prime_factorization_l1127_112778


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_amicable_numbers_for_n_2_l1127_112745

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def sum_of_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).sum id

def are_amicable (a b : ℕ) : Prop := sum_of_divisors a = b ∧ sum_of_divisors b = a

def p (n : ℕ) : ℕ := 3 * 2^n - 1
def q (n : ℕ) : ℕ := 3 * 2^(n-1) - 1
def r (n : ℕ) : ℕ := 9 * 2^(2*n-1) - 1

def A (n : ℕ) : ℕ := 2^n * p n * q n
def B (n : ℕ) : ℕ := 2^n * r n

theorem amicable_numbers_for_n_2 :
  is_prime (p 2) ∧ is_prime (q 2) ∧ is_prime (r 2) →
  A 2 = 220 ∧ B 2 = 284 ∧ are_amicable (A 2) (B 2) := by
  sorry

#eval A 2
#eval B 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_amicable_numbers_for_n_2_l1127_112745


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rate_theorem_l1127_112720

noncomputable def man_rate (with_stream against_stream : ℝ) : ℝ :=
  (with_stream + against_stream) / 2

theorem man_rate_theorem (with_stream against_stream : ℝ) 
  (h1 : with_stream = 14)
  (h2 : against_stream = 4) : 
  man_rate with_stream against_stream = 9 := by
  -- Proof steps go here
  sorry

#check man_rate_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_man_rate_theorem_l1127_112720


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_imply_function_value_l1127_112727

noncomputable def f (x : ℝ) := Real.sin x + Real.sqrt 3 * Real.cos x
noncomputable def g (x : ℝ) := 6 * (Real.sin (x / 2))^2 + Real.cos x

theorem symmetry_axes_imply_function_value 
  (x₁ x₂ : ℝ) 
  (h1 : ∀ x, f (2 * x₁ - x) = f x)  -- x₁ is symmetry axis of f
  (h2 : ∀ x, g (2 * x₂ - x) = g x)  -- x₂ is symmetry axis of g
  : f (x₁ - x₂) = 2 ∨ f (x₁ - x₂) = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_axes_imply_function_value_l1127_112727


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lanche_commute_l1127_112765

/-- The speed at which Mrs. Lanche arrives exactly on time -/
def on_time_speed : ℝ := 42

/-- The ideal travel time in hours -/
noncomputable def travel_time : ℝ := 
  25 / 120 -- This is approximately 0.2083 hours

/-- The distance to Mrs. Lanche's workplace -/
noncomputable def distance : ℝ :=
  on_time_speed * travel_time

theorem lanche_commute :
  30 * (travel_time + 5 / 60) = 70 * (travel_time - 5 / 60) ∧
  distance = on_time_speed * travel_time :=
by sorry

#check lanche_commute

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lanche_commute_l1127_112765


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l1127_112763

theorem exponential_equation_solution :
  ∃ x : ℚ, (3 : ℝ) ^ (4 * (x : ℝ)^2 - 3 * (x : ℝ) + 1) = (3 : ℝ) ^ (4 * (x : ℝ)^2 + 9 * (x : ℝ) - 6) ∧ x = 7 / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exponential_equation_solution_l1127_112763


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_speed_limit_l1127_112709

/-- Represents the total cost of a trip with a designated driver -/
noncomputable def total_cost (distance : ℝ) (speed : ℝ) (gas_price : ℝ) (driver_fee : ℝ) : ℝ :=
  let time := distance / speed
  let fuel_consumption := time * (3 + speed^2 / 360)
  gas_price * fuel_consumption + driver_fee * time

/-- Proves that the minimum total cost for the given conditions is 122 yuan at 50 km/h -/
theorem min_cost_at_speed_limit (distance : ℝ) (speed_limit : ℝ) (gas_price : ℝ) (driver_fee : ℝ)
  (h_distance : distance = 45)
  (h_speed_limit : speed_limit = 50)
  (h_gas_price : gas_price = 8)
  (h_driver_fee : driver_fee = 56) :
  ∀ speed, 0 < speed ∧ speed ≤ speed_limit →
    total_cost distance speed gas_price driver_fee ≥ 122 ∧
    total_cost distance speed_limit gas_price driver_fee = 122 :=
by sorry

#check min_cost_at_speed_limit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_cost_at_speed_limit_l1127_112709


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l1127_112742

noncomputable def a : ℝ := Real.log 1.2 / Real.log 0.7
noncomputable def b : ℝ := (0.8 : ℝ) ^ (0.7 : ℝ)
noncomputable def c : ℝ := (1.2 : ℝ) ^ (0.8 : ℝ)

theorem abc_relationship : c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_abc_relationship_l1127_112742


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_years_at_simple_interest_is_two_l1127_112703

/-- The number of years for which a sum is placed at simple interest -/
noncomputable def years_at_simple_interest (principal : ℝ) (simple_rate : ℝ) (compound_principal : ℝ) (compound_rate : ℝ) (compound_years : ℕ) : ℝ :=
  let compound_interest := compound_principal * ((1 + compound_rate) ^ compound_years - 1)
  let simple_interest := compound_interest / 2
  simple_interest / (principal * simple_rate)

/-- Theorem stating that the number of years at simple interest is 2 -/
theorem years_at_simple_interest_is_two :
  years_at_simple_interest 2625.0000000000027 0.08 4000 0.10 2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_years_at_simple_interest_is_two_l1127_112703


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_slope_angle_when_distance_is_sqrt_17_l1127_112736

-- Define the circle C
def my_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 5

-- Define the line l
def my_line (m x y : ℝ) : Prop := m * x - y + 1 - m = 0

-- Define the intersection points
def intersection_points (m : ℝ) : Set (ℝ × ℝ) :=
  {(x, y) | my_circle x y ∧ my_line m x y}

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Statement 1: The line always intersects the circle at two distinct points
theorem line_intersects_circle (m : ℝ) :
  ∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points m ∧ p2 ∈ intersection_points m ∧ p1 ≠ p2 :=
by sorry

-- Statement 2: When the distance between intersection points is √17, the slope angle is 60° or 120°
theorem slope_angle_when_distance_is_sqrt_17 (m : ℝ) :
  (∃ (p1 p2 : ℝ × ℝ), p1 ∈ intersection_points m ∧ p2 ∈ intersection_points m ∧ distance p1 p2 = Real.sqrt 17) →
  (m = Real.sqrt 3 ∨ m = -Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_intersects_circle_slope_angle_when_distance_is_sqrt_17_l1127_112736


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_parity_l1127_112740

-- Define a polynomial over real numbers
variable (P : ℝ → ℝ)

-- Define what it means for a polynomial to be even
def is_even_poly (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P x = P (-x)

-- Define what it means for a polynomial to be odd
def is_odd_poly (P : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, P x = -P (-x)

-- Define what it means for a polynomial to have only even degree non-zero coefficients
def has_even_degree_coeff (P : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n % 2 ≠ 0 → (∀ x : ℝ, deriv P x = 0)

-- Define what it means for a polynomial to have only odd degree non-zero coefficients
def has_odd_degree_coeff (P : ℝ → ℝ) : Prop :=
  ∀ n : ℕ, n % 2 = 0 → (∀ x : ℝ, deriv P x = 0)

-- State the theorem
theorem polynomial_parity (P : ℝ → ℝ) :
  (is_even_poly P ↔ has_even_degree_coeff P) ∧
  (is_odd_poly P ↔ has_odd_degree_coeff P) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_parity_l1127_112740


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_34_l1127_112759

def mySequence : ℕ → ℕ
  | 0 => 4
  | n + 1 => if n % 2 = 0 then mySequence n + 1 else mySequence n + 9

theorem seventh_term_is_34 : mySequence 6 = 34 := by
  -- Expand the definition of mySequence for the first 7 terms
  have h0 : mySequence 0 = 4 := rfl
  have h1 : mySequence 1 = 5 := by simp [mySequence]
  have h2 : mySequence 2 = 14 := by simp [mySequence, h1]
  have h3 : mySequence 3 = 15 := by simp [mySequence, h2]
  have h4 : mySequence 4 = 24 := by simp [mySequence, h3]
  have h5 : mySequence 5 = 25 := by simp [mySequence, h4]
  have h6 : mySequence 6 = 34 := by simp [mySequence, h5]
  
  -- The final step
  exact h6

end NUMINAMATH_CALUDE_ERRORFEEDBACK_seventh_term_is_34_l1127_112759


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_odd_rightmost_digit_l1127_112770

def m (n : ℕ) : ℕ := n + 10

def a (n : ℕ) : ℕ := Nat.factorial (n + m n) / Nat.factorial (n - 1)

def rightmost_nonzero_digit (x : ℕ) : ℕ := x % 10

theorem smallest_k_with_odd_rightmost_digit :
  ∃ k : ℕ, k > 0 ∧
    (∀ j : ℕ, 0 < j → j < k → Even (rightmost_nonzero_digit (a j))) ∧
    Odd (rightmost_nonzero_digit (a k)) ∧
    k = 3 := by
  sorry

#check smallest_k_with_odd_rightmost_digit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_k_with_odd_rightmost_digit_l1127_112770


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l1127_112755

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + a * x
noncomputable def g (x : ℝ) : ℝ := x^3 + Real.sin x

theorem f_monotonicity_and_inequality (a : ℝ) :
  (a ≥ 0 → ∀ x y, x > -1 → y > -1 → x < y → f a x < f a y) ∧
  (a < 0 → ∀ x y, -1 < x ∧ x < y ∧ y < -1/a - 1 → f a x < f a y) ∧
  (a < 0 → ∀ x y, x > -1/a - 1 ∧ y > -1/a - 1 ∧ x < y → f a x > f a y) ∧
  (a = 0 → ∀ x, x > -1 → f 0 x ≤ g x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_monotonicity_and_inequality_l1127_112755


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_altitudes_similarity_l1127_112779

/-- Given a triangle ABC with altitudes extended to A', B', and C' such that
    AA' = BC, BB' = AC, and CC' = AB, prove that triangle A'B'C' is similar to triangle ABC. -/
theorem extended_altitudes_similarity (A B C A' B' C' : ℂ) :
  let AA' := A' - A
  let BB' := B' - B
  let CC' := C' - C
  let AB := B - A
  let BC := C - B
  let AC := C - A
  AA' = BC ∧ BB' = AC ∧ CC' = AB →
  ∃ (k : ℝ), k > 0 ∧
    Complex.abs (B' - C') / Complex.abs (B - C) = k ∧
    Complex.abs (C' - A') / Complex.abs (C - A) = k ∧
    Complex.abs (A' - B') / Complex.abs (A - B) = k :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_extended_altitudes_similarity_l1127_112779


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_tan_half_positive_l1127_112707

theorem second_quadrant_tan_half_positive (θ : Real) : 
  (π / 2 < θ) ∧ (θ < π) → Real.tan (θ / 2) > 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_quadrant_tan_half_positive_l1127_112707


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AC_not_unique_AC_distance_collinear_l1127_112730

-- Define the points A, B, and C as variables in ℝ²
variable (A B C : ℝ × ℝ)

-- Define the lengths of segments AB and BC
def AB_length : ℝ := 5
def BC_length : ℝ := 4

-- Define the distance function as noncomputable due to Real.sqrt
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

-- Theorem statement
theorem distance_AC_not_unique :
  ∃ (A B C D : ℝ × ℝ), 
    distance A B = AB_length ∧ 
    distance B C = BC_length ∧
    distance B D = BC_length ∧
    distance A C ≠ distance A D := by
  sorry

-- Additional theorem to show that AC can be either 1 or 9 when points are collinear
theorem AC_distance_collinear :
  ∃ (A B C : ℝ × ℝ),
    distance A B = AB_length ∧
    distance B C = BC_length ∧
    (distance A C = 1 ∨ distance A C = 9) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AC_not_unique_AC_distance_collinear_l1127_112730


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_7x_mod_9_l1127_112775

theorem remainder_of_7x_mod_9 (x : ℕ) (h : x % 9 = 5) : (7 * x) % 9 = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remainder_of_7x_mod_9_l1127_112775


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l1127_112772

def my_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 2010 ∧ 
  a 2 = 2011 ∧ 
  ∀ n : ℕ, n ≥ 1 → a n + a (n + 1) + a (n + 3) = 2 * n

theorem sequence_1000th_term (a : ℕ → ℤ) (h : my_sequence a) : a 1000 = 2759 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_1000th_term_l1127_112772


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_students_average_mark_l1127_112788

theorem excluded_students_average_mark 
  (total_students : ℕ) 
  (total_average : ℚ) 
  (remaining_average : ℚ) 
  (excluded_count : ℕ) 
  (h1 : total_students = 9)
  (h2 : total_average = 60)
  (h3 : remaining_average = 80)
  (h4 : excluded_count = 5)
  (h5 : excluded_count < total_students) :
  let remaining_count := total_students - excluded_count
  let total_marks := total_students * total_average
  let remaining_marks := remaining_count * remaining_average
  let excluded_marks := total_marks - remaining_marks
  excluded_marks / excluded_count = 44 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_excluded_students_average_mark_l1127_112788


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1127_112758

/-- The time taken for two trains to cross each other -/
noncomputable def time_to_cross (length1 length2 speed1 speed2 : ℝ) : ℝ :=
  (length1 + length2) / ((speed1 + speed2) * (1000 / 3600))

/-- Theorem stating the time taken for the given trains to cross each other -/
theorem trains_crossing_time :
  let length1 : ℝ := 140
  let length2 : ℝ := 190
  let speed1 : ℝ := 60
  let speed2 : ℝ := 40
  ∃ ε > 0, |time_to_cross length1 length2 speed1 speed2 - 11.88| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trains_crossing_time_l1127_112758


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1127_112737

theorem equation_solution (x : ℝ) : 
  (2 * (4 : ℝ)^(x^2 - 3*x))^2 = (2 : ℝ)^(x - 1) ↔ x = 1/4 ∨ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1127_112737


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_N_l1127_112753

theorem sum_of_digits_of_N : ∃ N : ℕ,
  (N : ℝ)^2 = (25 : ℝ)^64 * (64 : ℝ)^25 ∧
  (N.digits 10).sum = 14 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digits_of_N_l1127_112753


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l1127_112795

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the line l₁
def line_l1 : Set (ℝ × ℝ) := {p | p.1 - p.2 - 2 * Real.sqrt 2 = 0}

-- Define the line l₂
def line_l2 : Set (ℝ × ℝ) := {p | 4 * p.1 - 3 * p.2 + 5 = 0}

-- Define the point G
def point_G : ℝ × ℝ := (1, 3)

theorem circle_C_properties :
  -- Circle C has center at origin and is tangent to line l₁
  (∀ p ∈ circle_C, p.1^2 + p.2^2 = 4) ∧
  (∃ q, q ∈ circle_C ∩ line_l1 ∧ ∀ r ∈ circle_C, r ∉ line_l1 ∨ r = q) →
  -- 1. The equation of circle C is x² + y² = 4
  (∀ p, p ∈ circle_C ↔ p.1^2 + p.2^2 = 4) ∧
  -- 2. The length of the chord intercepted by circle C on line l₂ is 2√3
  (∃ a b, a ∈ circle_C ∩ line_l2 ∧ b ∈ circle_C ∩ line_l2 ∧
    Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = 2 * Real.sqrt 3) ∧
  -- 3. The equation of the line connecting the tangency points of the lines drawn from G to circle C is x + 3y - 4 = 0
  (∃ m n, m ∈ circle_C ∧ n ∈ circle_C ∧
    (m.1 - point_G.1) * (m.1 - 0) + (m.2 - point_G.2) * (m.2 - 0) = 0 ∧
    (n.1 - point_G.1) * (n.1 - 0) + (n.2 - point_G.2) * (n.2 - 0) = 0 ∧
    (∀ p, p.1 + 3 * p.2 - 4 = 0 ↔ ∃ t, p = (1 - t) • m + t • n)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_C_properties_l1127_112795


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_l1127_112705

noncomputable def floor (x : ℝ) : ℤ := Int.floor x

def problem (x y : ℝ) : Prop :=
  y = 3 * (floor x) + 4 ∧
  y = 4 * (floor (x - 3)) + 7 ∧
  ¬(∃ n : ℤ, x = n)

theorem solution (x y : ℝ) (h : problem x y) : 40 < x + y ∧ x + y < 41 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_l1127_112705


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1127_112725

/-- The cosine function with angular frequency ω and phase φ -/
noncomputable def f (ω φ x : ℝ) : ℝ := Real.cos (ω * x + φ)

/-- The smallest positive period of a cosine function -/
noncomputable def period (ω : ℝ) : ℝ := 2 * Real.pi / ω

theorem min_omega_value (ω φ : ℝ) (h_ω : ω > 0) (h_φ : 0 < φ ∧ φ < Real.pi) :
  (f ω φ (period ω) = Real.sqrt 3 / 2) →
  (f ω φ (Real.pi / 9) = 0) →
  (∀ ω' > 0, (f ω' φ (period ω') = Real.sqrt 3 / 2) → (f ω' φ (Real.pi / 9) = 0) → ω ≤ ω') →
  ω = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_omega_value_l1127_112725


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_village_city_distance_l1127_112787

/-- The distance between two pedestrians at two different points in their journey --/
noncomputable def distance_between : ℝ := 2

/-- Vikentiy's fraction of the total distance traveled when first measured --/
noncomputable def vikentiy_fraction : ℝ := 1/2

/-- Afanasy's fraction of the total distance traveled when second measured --/
noncomputable def afanasy_fraction : ℝ := 1/3

/-- The distance between the village and the city --/
noncomputable def total_distance : ℝ := 6

/-- Theorem stating the existence of speeds satisfying the problem conditions --/
theorem village_city_distance : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
  (vikentiy_fraction * total_distance - distance_between = y * (vikentiy_fraction * total_distance / x)) ∧
  ((1 - afanasy_fraction) * total_distance + distance_between = x * (afanasy_fraction * total_distance / y)) :=
by sorry

#check village_city_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_village_city_distance_l1127_112787


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_cone_l1127_112793

-- Define the cone and sphere properties
def cone_base_radius : ℝ := 2
def cone_slant_height : ℝ → ℝ := λ c ↦ c
def sphere_radius : ℝ → ℝ := λ r ↦ r

-- Define the configuration conditions
def spheres_touch_externally (r : ℝ) : Prop := sorry
def spheres_touch_lateral_surface (r : ℝ) (c : ℝ) : Prop := sorry
def two_spheres_touch_base (r : ℝ) : Prop := sorry

-- Define the theorem
theorem max_sphere_radius_in_cone (c : ℝ) (r : ℝ) :
  spheres_touch_externally r ∧
  spheres_touch_lateral_surface r c ∧
  two_spheres_touch_base r →
  r ≤ Real.sqrt 3 - 1 := by
  sorry

#check max_sphere_radius_in_cone

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sphere_radius_in_cone_l1127_112793


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l1127_112729

def M : ℕ := 31^3 + 3*31^2 + 3*31 + 1

theorem number_of_factors_of_M : (Nat.divisors M).card = 16 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_M_l1127_112729


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l1127_112746

theorem correct_statements :
  (∀ x y : ℝ, x * y > 0 → y / x + x / y ≥ 2) ∧
  (∀ x y : ℝ, x + y = 0 → (2 : ℝ)^x + (2 : ℝ)^y ≥ 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_statements_l1127_112746


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_dihedral_angle_bound_l1127_112780

/-- Represents a tetrahedron with one face area normalized to 1 and three other face areas -/
structure Tetrahedron where
  S₁ : ℝ
  S₂ : ℝ
  S₃ : ℝ
  area_positive : 0 < S₁ ∧ 0 < S₂ ∧ 0 < S₃

/-- Dihedral angles of a tetrahedron -/
noncomputable def dihedral_angles (t : Tetrahedron) : ℝ × ℝ × ℝ :=
  let α₁ := Real.arccos (1 / t.S₁)
  let α₂ := Real.arccos (1 / t.S₂)
  let α₃ := Real.arccos (1 / t.S₃)
  (α₁, α₂, α₃)

/-- The smallest dihedral angle of a tetrahedron -/
noncomputable def smallest_dihedral_angle (t : Tetrahedron) : ℝ :=
  let (α₁, α₂, α₃) := dihedral_angles t
  min α₁ (min α₂ α₃)

/-- The dihedral angle of a regular tetrahedron -/
noncomputable def regular_tetrahedron_angle : ℝ := Real.arccos (1 / 3)

theorem smallest_dihedral_angle_bound (t : Tetrahedron) :
  smallest_dihedral_angle t ≤ regular_tetrahedron_angle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_dihedral_angle_bound_l1127_112780


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_subset_with_double_exclusion_l1127_112723

def S : Set ℕ := {n | 1 ≤ n ∧ n ≤ 3000}

theorem no_subset_with_double_exclusion :
  ¬ ∃ (A : Finset ℕ), A.toSet ⊆ S ∧ A.card = 2000 ∧
    ∀ x ∈ A, 2 * x ∉ A :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_subset_with_double_exclusion_l1127_112723


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_has_winning_strategy_l1127_112744

/-- Represents a prime number -/
structure GamePrime where
  value : Nat
  is_prime : Nat.Prime value

/-- Represents the state of the game board -/
structure GameBoard where
  primes : List GamePrime
  is_valid : primes.length ≤ 500

/-- Represents a player's move -/
structure Move where
  n : Nat
  primes_to_erase : List GamePrime
  is_valid : primes_to_erase.length = n

/-- Represents the parity of the number of primes congruent to 3 mod 4 -/
inductive Parity
| Even
| Odd

/-- The game state after a move -/
def game_state_after_move (board : GameBoard) (move : Move) : GameBoard :=
  sorry

/-- The parity of the game state -/
def parity (board : GameBoard) : Parity :=
  sorry

/-- A player's strategy -/
def Strategy := GameBoard → Move

/-- Determines if a strategy is winning for player A -/
def is_winning_strategy_for_A (strategy : Strategy) : Prop :=
  sorry

/-- The main theorem: Player A has a winning strategy -/
theorem player_A_has_winning_strategy :
  ∃ (strategy : Strategy), is_winning_strategy_for_A strategy := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_player_A_has_winning_strategy_l1127_112744


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_monotonicity_intervals_l1127_112785

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 10

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem for f'(1)
theorem derivative_at_one : f' 1 = -3 := by sorry

-- Define increasing and decreasing intervals
def increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x < f y

def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ x y, x ∈ s → y ∈ s → x < y → f x > f y

-- Theorem for intervals of monotonicity
theorem monotonicity_intervals :
  (increasing_on f (Set.Iio 0)) ∧
  (decreasing_on f (Set.Ioo 0 2)) ∧
  (increasing_on f (Set.Ioi 2)) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_derivative_at_one_monotonicity_intervals_l1127_112785


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_modulus_l1127_112791

theorem pure_imaginary_modulus (a : ℝ) : 
  (∃ b : ℝ, (Complex.I + 2) / (a - Complex.I) = b * Complex.I) → Complex.abs (a + Complex.I) = Real.sqrt 5 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pure_imaginary_modulus_l1127_112791


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_50cm_l1127_112716

/-- The height of the ball after a given number of bounces -/
def ballHeight (initial_height : ℝ) (bounce_ratio : ℝ) (n : ℕ) : ℝ :=
  initial_height * (bounce_ratio ^ n)

/-- The proposition that 8 is the smallest number of bounces required -/
theorem min_bounces_to_50cm :
  let initial_height : ℝ := 400
  let bounce_ratio : ℝ := 3/4
  let target_height : ℝ := 50
  (∀ n : ℕ, n < 8 → ballHeight initial_height bounce_ratio n ≥ target_height) ∧
  (ballHeight initial_height bounce_ratio 8 < target_height) :=
by
  sorry

#check min_bounces_to_50cm

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_bounces_to_50cm_l1127_112716


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_weight_is_14_4_l1127_112781

/-- The weight of a single bicycle in pounds -/
noncomputable def bicycle_weight : ℝ := 36

/-- The number of bowling balls that weigh the same as two bicycles -/
def num_bowling_balls : ℕ := 5

/-- The number of bicycles that weigh the same as the bowling balls -/
def num_bicycles : ℕ := 2

/-- The weight of a single bowling ball in pounds -/
noncomputable def bowling_ball_weight : ℝ := (num_bicycles * bicycle_weight) / num_bowling_balls

theorem bowling_ball_weight_is_14_4 :
  bowling_ball_weight = 14.4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bowling_ball_weight_is_14_4_l1127_112781


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_f_period_one_not_center_of_symmetry_not_increasing_in_interval_l1127_112752

-- Define the nearest integer function
noncomputable def nearest_integer (x : ℝ) : ℤ :=
  ⌊x + 1/2⌋

-- Define the function f(x) = x - {x}
noncomputable def f (x : ℝ) : ℝ :=
  x - (nearest_integer x : ℝ)

-- Theorem: The domain of f is ℝ and the range is (-1/2, 1/2]
theorem f_domain_and_range :
  (∀ x : ℝ, ∃ y : ℝ, f x = y) ∧
  (∀ y : ℝ, -1/2 < y ∧ y ≤ 1/2 → ∃ x : ℝ, f x = y) ∧
  (∀ y : ℝ, (∃ x : ℝ, f x = y) → -1/2 < y ∧ y ≤ 1/2) :=
sorry

-- Theorem: The function f has a period of 1
theorem f_period_one :
  ∀ x : ℝ, f (x + 1) = f x :=
sorry

-- Theorem: Proposition ② is false (not a center of symmetry)
theorem not_center_of_symmetry :
  ∃ k : ℤ, ∃ x : ℝ, f (2 * k - x) ≠ f x :=
sorry

-- Theorem: Proposition ④ is false (not increasing in the given interval)
theorem not_increasing_in_interval :
  ∃ x₁ x₂ : ℝ, -1/2 < x₁ ∧ x₁ < x₂ ∧ x₂ ≤ 3/2 ∧ f x₁ > f x₂ :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_domain_and_range_f_period_one_not_center_of_symmetry_not_increasing_in_interval_l1127_112752


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_vectors_l1127_112761

-- Define the triangle
def triangle (A B C : Real) (a b c : Real) : Prop :=
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A + B + C = Real.pi ∧
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

-- Define the condition
noncomputable def condition (A B C : Real) (a b c : Real) : Prop :=
  (2 * a - c) * Real.cos B = b * Real.cos C

-- Define the vectors
noncomputable def m (A : Real) : Fin 2 → Real
| 0 => Real.sin A
| 1 => Real.cos (2 * A)
| _ => 0

def n (k : Real) : Fin 2 → Real
| 0 => 4 * k
| 1 => 1
| _ => 0

-- Define the dot product
def dot_product (v w : Fin 2 → Real) : Real :=
  (v 0) * (w 0) + (v 1) * (w 1)

-- State the theorem
theorem triangle_and_vectors 
  (A B C : Real) (a b c : Real) (k : Real) :
  triangle A B C a b c →
  condition A B C a b c →
  k > 1 →
  (∀ A', dot_product (m A') (n k) ≤ 5) →
  (∃ A', dot_product (m A') (n k) = 5) →
  B = Real.pi / 3 ∧ k = 3 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_and_vectors_l1127_112761


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_solutions_l1127_112769

-- Define the cubic roots
noncomputable def a : ℝ := Real.rpow 27 (1/3)
noncomputable def b : ℝ := Real.rpow 45 (1/3)
noncomputable def c : ℝ := Real.rpow 81 (1/3)

-- Define the equation
def equation (x : ℝ) : Prop :=
  (x - a) * (x - b) * (x - c) = 2/5

-- Define the theorem
theorem sum_of_cubes_of_solutions :
  ∃ (u v w : ℝ),
    u ≠ v ∧ v ≠ w ∧ u ≠ w ∧
    equation u ∧ equation v ∧ equation w ∧
    u^3 + v^3 + w^3 = 153.6 + 118.6025 * 10.8836 - 32.8708 * 10.8836 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_cubes_of_solutions_l1127_112769


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_f_sum_specific_l1127_112738

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 / (1 + x^2)

-- Theorem 1
theorem f_sum_reciprocal (x : ℝ) (hx : x ≠ 0) : f x + f (1/x) = 1 := by
  sorry

-- Theorem 2
theorem f_sum_specific : f 1 + f 2 + f (1/2) + f 3 + f (1/3) + f 4 + f (1/4) = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_reciprocal_f_sum_specific_l1127_112738


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_prime_powers_l1127_112756

theorem infinitely_many_non_prime_powers (a : ℤ) :
  Set.Infinite {n : ℕ+ | ¬ Nat.Prime (Int.natAbs (a^(2^n.val) + 2^n.val))} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_non_prime_powers_l1127_112756


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ears_per_row_is_70_l1127_112718

/-- Represents the farming scenario with given conditions -/
structure FarmingScenario where
  kids : ℕ
  seeds_per_bag : ℕ
  seeds_per_ear : ℕ
  pay_per_row : ℚ
  dinner_cost : ℚ
  bags_used : ℕ

/-- Calculates the number of ears of corn in each row -/
def ears_per_row (scenario : FarmingScenario) : ℚ :=
  let total_seeds := scenario.bags_used * scenario.seeds_per_bag
  let total_ears := total_seeds / scenario.seeds_per_ear
  let total_pay := 2 * scenario.dinner_cost
  let rows_planted := total_pay / scenario.pay_per_row
  (total_ears : ℚ) / rows_planted

/-- Theorem stating that for the given scenario, there are 70 ears of corn in each row -/
theorem ears_per_row_is_70 : 
  let scenario : FarmingScenario := {
    kids := 4,
    seeds_per_bag := 48,
    seeds_per_ear := 2,
    pay_per_row := 3/2,
    dinner_cost := 36,
    bags_used := 140
  }
  ears_per_row scenario = 70 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ears_per_row_is_70_l1127_112718


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_primes_under_factorial_l1127_112739

theorem least_multiple_of_primes_under_factorial : ∃ n : ℕ,
  (n > 0) ∧
  (n % 7 = 0) ∧
  (n % 11 = 0) ∧
  (n % 13 = 0) ∧
  (n ≤ Nat.factorial 13) ∧
  (∀ m : ℕ, m > 0 ∧ m % 7 = 0 ∧ m % 11 = 0 ∧ m % 13 = 0 ∧ m ≤ Nat.factorial 13 → n ≤ m) ∧
  n = 1001 := by
  -- Proof goes here
  sorry

#check least_multiple_of_primes_under_factorial

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_multiple_of_primes_under_factorial_l1127_112739


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l1127_112749

theorem sin_plus_cos_for_point (α : ℝ) : 
  (∃ r : ℝ, r > 0 ∧ r * Real.cos α = -3 ∧ r * Real.sin α = 4) → 
  Real.sin α + Real.cos α = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_plus_cos_for_point_l1127_112749


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segments_no_triangle_solution_l1127_112733

/-- Represents a list of segment lengths --/
def Segments := List Nat

/-- Checks if three segments can form a triangle --/
def can_form_triangle (a b c : Nat) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Checks if any three segments in the list can form a triangle --/
def any_three_form_triangle (segments : Segments) : Prop :=
  ∃ (a b c : Nat), a ∈ segments.toFinset ∧ b ∈ segments.toFinset ∧ c ∈ segments.toFinset ∧ 
    a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ can_form_triangle a b c

/-- The main theorem --/
theorem max_segments_no_triangle (total_length : Nat) (n : Nat) : Prop :=
  total_length = 144 →
  n > 2 →
  ∃ (segments : Segments),
    segments.length = n ∧
    segments.sum = total_length ∧
    (∀ s, s ∈ segments.toFinset → s ≥ 1) ∧
    ¬(any_three_form_triangle segments) ∧
    ∀ (m : Nat), m > n →
      ¬∃ (larger_segments : Segments),
        larger_segments.length = m ∧
        larger_segments.sum = total_length ∧
        (∀ s, s ∈ larger_segments.toFinset → s ≥ 1) ∧
        ¬(any_three_form_triangle larger_segments)

/-- The solution --/
theorem solution : max_segments_no_triangle 144 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_segments_no_triangle_solution_l1127_112733


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l1127_112747

/-- Given an interest rate and time period, calculates the simple interest factor -/
def simple_interest_factor (rate : ℝ) (time : ℝ) : ℝ :=
  1 + (rate * time)

/-- Given an interest rate and time period, calculates the compound interest factor -/
noncomputable def compound_interest_factor (rate : ℝ) (time : ℝ) : ℝ :=
  (1 + rate) ^ time

/-- Theorem stating that if the difference between compound and simple interest
    on a principal P at 10% for 2 years is 20, then P must be 2000 -/
theorem interest_difference_implies_principal :
  let rate : ℝ := 0.1
  let time : ℝ := 2
  let diff : ℝ := 20
  ∀ P : ℝ,
    P * (compound_interest_factor rate time - simple_interest_factor rate time) = diff →
    P = 2000 := by
  sorry

#check interest_difference_implies_principal

end NUMINAMATH_CALUDE_ERRORFEEDBACK_interest_difference_implies_principal_l1127_112747


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_in_circle_l1127_112771

/-- The area of a regular octagon inscribed in a circle with radius 3 units -/
noncomputable def regularOctagonArea : ℝ := 54 * (2 - Real.sqrt 2) * Real.sqrt 3

/-- Theorem: The area of a regular octagon inscribed in a circle with radius 3 units
    is equal to 54(2-√2)√3 square units -/
theorem regular_octagon_area_in_circle (r : ℝ) (h : r = 3) :
  regularOctagonArea = 54 * (2 - Real.sqrt 2) * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_octagon_area_in_circle_l1127_112771


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_words_already_written_l1127_112722

def report_length : ℕ := 1000
def typing_speed : ℕ := 300
def typing_time : ℚ := 1/2
def remaining_time : ℕ := 80

theorem words_already_written :
  let words_per_hour : ℚ := (typing_speed : ℚ) / typing_time
  let remaining_hours : ℚ := (remaining_time : ℚ) / 60
  let words_to_type : ℚ := words_per_hour * remaining_hours
  (report_length : ℚ) - words_to_type = 200 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_words_already_written_l1127_112722


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sequence_minus_sqrt_root_l1127_112750

noncomputable def sequence_a : ℕ → ℝ
  | 0 => 1  -- Add a case for 0
  | 1 => 1
  | n + 2 => sequence_a (n + 1) + 1 / (2 * sequence_a (n + 1))

theorem limit_sequence_minus_sqrt_root :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |sequence_a n - Real.sqrt n| < ε := by
  sorry

#check limit_sequence_minus_sqrt_root

end NUMINAMATH_CALUDE_ERRORFEEDBACK_limit_sequence_minus_sqrt_root_l1127_112750


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_good_numbers_in_set_l1127_112734

def isGoodNumber (n : ℕ) : Prop :=
  ∃ (a : Fin n → Fin n), Function.Bijective a ∧
    ∀ k : Fin n, ∃ m : ℕ, (k.val + 1 : ℕ) + (a k).val = m * m

def setToCheck : List ℕ := [11, 13, 15, 17, 19]

theorem good_numbers_in_set :
  ∀ n ∈ setToCheck, isGoodNumber n ↔ n ∈ [13, 15, 17, 19] :=
by
  intro n hn
  sorry

#check good_numbers_in_set

end NUMINAMATH_CALUDE_ERRORFEEDBACK_good_numbers_in_set_l1127_112734


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l1127_112728

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x^2 - x * Real.log x + 2

-- Define the derivative of f
noncomputable def f' (x : ℝ) : ℝ := 2 * x - Real.log x - 1

-- State the theorem
theorem tangent_line_through_origin :
  ∃ (m : ℝ), m > 0 ∧ 
  f m = (3 - Real.log 2) * m ∧
  f' m = 3 - Real.log 2 ∧
  ∀ x : ℝ, f x ≥ (3 - Real.log 2) * x := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_through_origin_l1127_112728


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1127_112797

-- Define the indicator function
noncomputable def indicator (x : ℝ) : ℝ := if x > 1 then 1 else 0

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := x + indicator x

-- Theorem statement
theorem min_value_of_f :
  ∃ (m : ℝ), m = 1 ∧ ∀ (x : ℝ), f x ≥ m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1127_112797


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l1127_112724

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := -1 / (x * Real.sin x)

-- State the theorem
theorem integral_proof (x : ℝ) (h : x ≠ 0) (h' : Real.sin x ≠ 0) :
  deriv f x = (x * Real.cos x + Real.sin x) / (x * Real.sin x)^2 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_proof_l1127_112724


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1127_112783

-- Define the triangle
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) :
  (t.a * Real.cos t.B + Real.sqrt 3 * t.b * Real.sin t.A = t.c) →
  (t.A = π / 6) ∧
  (t.a = 1 ∧ t.b * t.c = 2 - Real.sqrt 3 → t.b + t.c = Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l1127_112783


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1127_112701

/-- The eccentricity of a hyperbola with given parameters and asymptotes -/
theorem hyperbola_eccentricity (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h3 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (y = 2 * x ∨ y = -2 * x)) :
  Real.sqrt ((a^2 + b^2) / a^2) = Real.sqrt 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l1127_112701


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_path_length_on_cube_l1127_112764

/-- The path length of a dot on the center of a cube face after four quarter-turns -/
noncomputable def pathLength (cubeEdgeLength : ℝ) (numTurns : ℕ) : ℝ :=
  (numTurns : ℝ) * (Real.pi / 2) * (cubeEdgeLength / 2)

theorem dot_path_length_on_cube (cubeEdgeLength : ℝ) (numTurns : ℕ) :
  cubeEdgeLength = 2 → numTurns = 4 → pathLength cubeEdgeLength numTurns = 2 * Real.pi :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dot_path_length_on_cube_l1127_112764


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digit_products_10_to_2021_l1127_112767

def digit_product (n : ℕ) : ℕ := sorry

def sum_of_digit_products (start finish : ℕ) : ℕ :=
  (List.range (finish - start + 1)).map (fun i => digit_product (i + start))
    |> List.sum

theorem sum_of_digit_products_10_to_2021 :
  sum_of_digit_products 10 2021 = 184275 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_digit_products_10_to_2021_l1127_112767


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_AOB_l1127_112708

/-- Represents a point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Calculates the area of a triangle given two points in polar coordinates and the pole -/
noncomputable def triangleArea (a : PolarPoint) (b : PolarPoint) : ℝ :=
  (1/2) * |a.r| * |b.r| * Real.sin |a.θ - b.θ|

theorem triangle_area_AOB : 
  let a : PolarPoint := ⟨3, π/3⟩
  let b : PolarPoint := ⟨-4, 7*π/6⟩
  triangleArea a b = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_AOB_l1127_112708


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1127_112715

def U : Set ℕ := {1, 2, 3, 4, 5}

def A : Set ℕ := {x ∈ U | (x : ℤ) - 3 < 2 ∧ -((x : ℤ) - 3) < 2}

theorem complement_of_A : (U \ A) = {1, 5} := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_of_A_l1127_112715


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1127_112748

/-- A parabola with directrix y = -1 has the standard equation x² = 4y -/
theorem parabola_equation (C : Set (ℝ × ℝ)) 
  (h : ∀ (x y : ℝ), (x, y) ∈ C ↔ (y + 1)^2 = (x^2 + (y - 1)^2)) :
  ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 = 4*y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l1127_112748


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_mark_is_ninety_l1127_112776

noncomputable def average_of_four (a b c d : ℝ) : ℝ := (a + b + c + d) / 4

theorem fourth_mark_is_ninety 
  (avg : ℝ) 
  (mark1 mark2 mark3 : ℝ) 
  (h1 : avg = 60) 
  (h2 : mark1 = 30) 
  (h3 : mark2 = 55) 
  (h4 : mark3 = 65) 
  (h5 : ∃ mark4, average_of_four mark1 mark2 mark3 mark4 = avg) : 
  ∃ mark4, mark4 = 90 ∧ average_of_four mark1 mark2 mark3 mark4 = avg := by
  sorry

#check fourth_mark_is_ninety

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fourth_mark_is_ninety_l1127_112776


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sum_equals_pi_over_four_minus_half_l1127_112754

open MeasureTheory Interval Real

theorem integral_sum_equals_pi_over_four_minus_half :
  (∫ x in (0)..(π/2), (sin (x/2))^2) + (∫ x in (-1)..1, exp (abs x) * sin x) = π/4 - 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_sum_equals_pi_over_four_minus_half_l1127_112754


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1127_112732

/-- Represents a tetrahedron with vertices S, A, B, and C -/
structure Tetrahedron where
  SA : ℝ
  SB : ℝ
  SC : ℝ
  AB : ℝ
  AC : ℝ
  CB : ℝ

/-- The surface area of a sphere given its radius -/
noncomputable def sphereSurfaceArea (radius : ℝ) : ℝ := 4 * Real.pi * radius^2

/-- Theorem: The surface area of the circumscribed sphere of a tetrahedron
    with edge lengths SA = CB = √5, SB = AC = √10, and SC = AB = √13 is 14π -/
theorem circumscribed_sphere_surface_area
  (t : Tetrahedron)
  (h1 : t.SA = Real.sqrt 5)
  (h2 : t.CB = Real.sqrt 5)
  (h3 : t.SB = Real.sqrt 10)
  (h4 : t.AC = Real.sqrt 10)
  (h5 : t.SC = Real.sqrt 13)
  (h6 : t.AB = Real.sqrt 13) :
  sphereSurfaceArea (Real.sqrt 14 / 2) = 14 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumscribed_sphere_surface_area_l1127_112732


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1127_112751

theorem sufficient_but_not_necessary :
  (∃ x : ℝ, x > 0 ∧ (2 : ℝ)^(x-1) ≤ 1 ∧ 1/x < 1) ∧
  (∀ x : ℝ, x > 0 → (1/x ≥ 1 → (2 : ℝ)^(x-1) ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sufficient_but_not_necessary_l1127_112751


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisibility_l1127_112799

theorem smallest_n_divisibility : ∃! n : ℕ, 
  n > 0 ∧
  (∀ m : ℕ, m > 0 →
    (m^2 % 24 = 0 ∧ m^3 % 900 = 0 ∧ m^4 % 1024 = 0) → n ≤ m) ∧
  n^2 % 24 = 0 ∧ 
  n^3 % 900 = 0 ∧ 
  n^4 % 1024 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_divisibility_l1127_112799


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1127_112796

open Real Set

theorem unique_solution_condition (a : ℝ) : 
  (∃! x : ℝ, x ∈ Icc 0 π ∧ sin (2*x) * sin (4*x) - sin x * sin (3*x) = a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_condition_l1127_112796


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_stoppage_time_is_ten_l1127_112757

/-- Calculates the average stoppage time per hour for three trains given their speeds with and without stoppages. -/
noncomputable def average_stoppage_time (speed1_without speed1_with speed2_without speed2_with speed3_without speed3_with : ℝ) : ℝ :=
  let stoppage_time1 := (speed1_without - speed1_with) / speed1_without * 60
  let stoppage_time2 := (speed2_without - speed2_with) / speed2_without * 60
  let stoppage_time3 := (speed3_without - speed3_with) / speed3_without * 60
  (stoppage_time1 + stoppage_time2 + stoppage_time3) / 3

/-- Theorem stating that the average stoppage time for the given train speeds is 10 minutes. -/
theorem average_stoppage_time_is_ten :
  average_stoppage_time 48 40 54 45 60 50 = 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_stoppage_time_is_ten_l1127_112757


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1127_112760

theorem log_equation_solution (x : ℝ) (h : x > 0) :
  Real.log x^3 / Real.log 4 + Real.log (1/x) / Real.log 4 = 3 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_log_equation_solution_l1127_112760


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_vertices_tetrahedron_volume_l1127_112713

/-- A cube with side length 8 meters -/
def cube_side_length : ℚ := 8

/-- The volume of a cube given its side length -/
def cube_volume (s : ℚ) : ℚ := s^3

/-- The volume of a tetrahedron given its base area and height -/
def tetrahedron_volume (base_area height : ℚ) : ℚ := (1/3) * base_area * height

/-- The theorem stating the volume of the tetrahedron formed by alternately colored vertices of the cube -/
theorem blue_vertices_tetrahedron_volume :
  let total_volume := cube_volume cube_side_length
  let small_tetrahedron_volume := tetrahedron_volume (1/2 * cube_side_length^2) cube_side_length
  total_volume - 4 * small_tetrahedron_volume = 170 + 2/3 := by
  sorry

#eval cube_volume cube_side_length
#eval 4 * tetrahedron_volume (1/2 * cube_side_length^2) cube_side_length
#eval cube_volume cube_side_length - 4 * tetrahedron_volume (1/2 * cube_side_length^2) cube_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_vertices_tetrahedron_volume_l1127_112713


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_area_theorem_l1127_112741

/-- The area enclosed by a curve consisting of 12 congruent circular arcs, each of length π/2,
    with centers at the vertices of a regular octagon with side length 3 -/
noncomputable def enclosed_area : ℝ := 54 + 54 * Real.sqrt 2 + 3 * Real.pi

/-- The number of circular arcs in the curve -/
def num_arcs : ℕ := 12

/-- The length of each circular arc -/
noncomputable def arc_length : ℝ := Real.pi / 2

/-- The side length of the regular octagon -/
def octagon_side : ℝ := 3

/-- The area of a regular octagon with side length s -/
noncomputable def octagon_area (s : ℝ) : ℝ := 2 * (1 + Real.sqrt 2) * s^2

/-- Theorem stating that the enclosed area equals the sum of the octagon area and the areas of the circular sectors -/
theorem curve_area_theorem :
  enclosed_area = octagon_area octagon_side + num_arcs * (arc_length^2 / (4 * Real.pi)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_curve_area_theorem_l1127_112741


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_price_calculation_l1127_112768

/-- A pet store sells mice in pairs. This function represents the price for 2 mice. -/
def price_for_two : ℝ := sorry

/-- The total number of mice sold -/
def total_mice : ℕ := 7

/-- The total revenue from selling the mice -/
def total_revenue : ℝ := 18.69

/-- Theorem stating that if 7 mice were sold for $18.69, then the price for 2 mice is $5.34 -/
theorem mice_price_calculation (h : 3 * price_for_two + price_for_two / 2 = total_revenue) : 
  price_for_two = 5.34 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mice_price_calculation_l1127_112768


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l1127_112719

/-- The area of a pentagon with specific dimensions -/
theorem pentagon_area (s1 s2 s3 s4 s5 : ℝ) 
  (h1 : s1 = 18) (h2 : s2 = 25) (h3 : s3 = 30) (h4 : s4 = 28) (h5 : s5 = 25)
  (rect_width : ℝ) (rect_length : ℝ) (tri_base : ℝ) (tri_height : ℝ)
  (hw : rect_width = 25) (hl : rect_length = 28)
  (hb : tri_base = 18) (hh : tri_height = 25)
  (decomposition : s4 = rect_length ∧ s2 = rect_width ∧ s1 = tri_base ∧ s2 = tri_height) :
  rect_width * rect_length + (1/2) * tri_base * tri_height = 925 := by
  -- Substitute known values
  rw [hw, hl, hb, hh]
  -- Evaluate the expression
  norm_num
  -- QED
  
#check pentagon_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pentagon_area_l1127_112719


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1127_112782

noncomputable def f (x y : ℝ) : ℝ := 
  (13 * x^2 + 24 * x * y + 13 * y^2 + 16 * x + 14 * y + 68) / (9 - x^2 - 8 * x * y - 16 * y^2)^(5/2)

theorem min_value_of_f :
  ∀ x y : ℝ, 9 - x^2 - 8 * x * y - 16 * y^2 > 0 → f x y ≥ 7/27 ∧ ∃ x y : ℝ, f x y = 7/27 := by
  sorry

#check min_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_f_l1127_112782


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ab_fixed_point_no_concyclic_point_l1127_112798

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x^2 = 2py -/
structure Parabola where
  p : ℝ

/-- Represents a line in 2D space -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Given a parabola and a point on it, find the parameter p -/
noncomputable def find_p (m : Point) : ℝ :=
  m.x^2 / (2 * m.y)

/-- Check if a point lies on a parabola -/
def on_parabola (point : Point) (parabola : Parabola) : Prop :=
  point.x^2 = 2 * parabola.p * point.y

/-- Calculate the slope of a tangent line to a parabola at a given point -/
noncomputable def tangent_slope (point : Point) (parabola : Parabola) : ℝ :=
  point.x / (2 * parabola.p)

/-- Theorem: The line AB passes through a fixed point -/
theorem line_ab_fixed_point 
  (m : Point) 
  (parabola : Parabola) 
  (h_m_on_parabola : on_parabola m parabola)
  (h_m : m.x = 4 ∧ m.y = 4)
  (p : Point) -- moving point P
  (a b : Point) -- intersection points of tangents
  (h_slopes : tangent_slope a parabola * tangent_slope b parabola = -2)
  : ∃ (fixed : Point), fixed.x = 0 ∧ fixed.y = 2 ∧ 
    (∃ (line : Line), line.slope * fixed.x + line.intercept = fixed.y ∧
                      line.slope * a.x + line.intercept = a.y ∧
                      line.slope * b.x + line.intercept = b.y) := by
  sorry

/-- Theorem: No point P exists where A, C, P, and D are concyclic -/
theorem no_concyclic_point
  (m : Point)
  (parabola : Parabola)
  (h_m_on_parabola : on_parabola m parabola)
  (h_m : m.x = 4 ∧ m.y = 4)
  : ¬∃ (p : Point), ∃ (a b c d : Point),
    on_parabola a parabola ∧
    on_parabola b parabola ∧
    c.x = a.x ∧ c.y = -1 ∧
    d.x = b.x ∧ d.y = -1 ∧
    (a.x - c.x) * (p.x - d.x) = (a.y - c.y) * (p.y - d.y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_ab_fixed_point_no_concyclic_point_l1127_112798


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_60_l1127_112700

/-- The number of positive factors of 60 -/
def num_factors_60 : ℕ := 12

/-- Theorem stating that the number of positive factors of 60 is 12 -/
theorem factors_of_60 : 
  (Finset.filter (λ n ↦ 60 % n = 0) (Finset.range 61)).card = num_factors_60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factors_of_60_l1127_112700


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1127_112712

-- Define QuadraticEquation as a structure
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define HasSolution as a predicate
def HasSolution (q : QuadraticEquation) (x : ℝ) : Prop :=
  q.a * x^2 + q.b * x + q.c = 0

theorem negation_of_universal_proposition :
  ¬(∀ q : QuadraticEquation, ∃ x : ℝ, HasSolution q x) ↔
  (∃ q : QuadraticEquation, ∀ x : ℝ, ¬HasSolution q x) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negation_of_universal_proposition_l1127_112712


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coach_distance_lower_bound_l1127_112773

/-- Represents an athlete in the race -/
structure Athlete where
  speedAB : ℝ  -- Speed from A to B
  speedBA : ℝ  -- Speed from B to A

/-- The race scenario -/
structure RaceScenario where
  distanceAB : ℝ           -- Distance from A to B
  athletes : Fin 3 → Athlete  -- The three athletes

/-- Calculates the coach's position at any given time -/
noncomputable def coachPosition (scenario : RaceScenario) (t : ℝ) : ℝ :=
  sorry

/-- Calculates the total distance traveled by the coach -/
noncomputable def coachTotalDistance (scenario : RaceScenario) : ℝ :=
  sorry

/-- The main theorem to be proved -/
theorem coach_distance_lower_bound (scenario : RaceScenario) :
  scenario.distanceAB = 60 →
  (∀ i : Fin 3, (scenario.athletes i).speedAB > 0 ∧ (scenario.athletes i).speedBA > 0) →
  coachTotalDistance scenario ≥ 100 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coach_distance_lower_bound_l1127_112773


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_max_value_on_interval_existence_condition_l1127_112702

variable (a : ℝ)

def f (a : ℝ) (x : ℝ) : ℝ := a^2 * x^3 - 3*a*x^2 + 2
def g (a : ℝ) (x : ℝ) : ℝ := -3*a*x + 3

theorem tangent_line_at_one (h : a = 1) :
  ∃ (k m : ℝ), k * 1 + m = f a 1 ∧ k = (deriv (f a)) 1 ∧ k = -3 ∧ m = 3 := by
  sorry

theorem max_value_on_interval (h : a > 0) :
  ∃ (x_max : ℝ), x_max ∈ Set.Icc (-1 : ℝ) 1 ∧
    ∀ x ∈ Set.Icc (-1 : ℝ) 1, f a x ≤ f a x_max ∧ f a x_max = 2 := by
  sorry

theorem existence_condition (h : a > 0) :
  (∃ x_0 : ℝ, x_0 ∈ Set.Ioo (0 : ℝ) (1/2) ∧ f a x_0 > g a x_0) ↔ a > -3 + Real.sqrt 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_at_one_max_value_on_interval_existence_condition_l1127_112702


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_with_equal_perimeter_l1127_112711

/-- Given a square with area 16 and a regular octagon with equal perimeter to the square,
    the area of the octagon is 8 + 8√2. -/
theorem octagon_area_with_equal_perimeter (square_side : ℝ) (octagon_side : ℝ) :
  square_side ^ 2 = 16 →
  4 * square_side = 8 * octagon_side →
  2 * (1 + Real.sqrt 2) * octagon_side ^ 2 = 8 + 8 * Real.sqrt 2 := by
  intro h1 h2
  -- Proof steps would go here
  sorry

#check octagon_area_with_equal_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_octagon_area_with_equal_perimeter_l1127_112711


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_polynomial_count_l1127_112706

theorem negative_polynomial_count : 
  ∃ (S : Finset ℤ), S.card = 12 ∧ 
  ∀ x : ℤ, x ∈ S ↔ (x^4 - 63*x^2 + 62 : ℤ) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_negative_polynomial_count_l1127_112706


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_class_weighted_average_l1127_112774

theorem class_weighted_average (total_students : ℕ) (avg_age : ℚ)
  (group1_count : ℕ) (group1_avg : ℚ)
  (group2_count : ℕ) (group2_avg : ℚ)
  (group3_avg : ℚ)
  (h1 : total_students = 30)
  (h2 : avg_age = 17)
  (h3 : group1_count = 12)
  (h4 : group1_avg = 18)
  (h5 : group2_count = 8)
  (h6 : group2_avg = 15)
  (h7 : group3_avg = 20)
  (h8 : total_students = group1_count + group2_count + (total_students - group1_count - group2_count)) :
  let weighted_avg : ℚ := (group1_count * group1_avg + group2_count * group2_avg +
    (total_students - group1_count - group2_count) * group3_avg) / total_students
  ⌊weighted_avg * 100⌋ / 100 = 1787 / 100 := by
  sorry

#eval (⌊(1787 : ℚ) / 100 * 100⌋ : ℚ) / 100  -- To verify the rounding

end NUMINAMATH_CALUDE_ERRORFEEDBACK_class_weighted_average_l1127_112774


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_proof_l1127_112792

theorem larger_number_proof (a b : ℕ) : 
  (Nat.gcd a b = 23) →
  (Nat.lcm a b = 3036) →
  (∃ (x y : ℕ), x ≠ y ∧ x ∈ ({11, 12} : Set ℕ) ∧ y ∈ ({11, 12} : Set ℕ) ∧ Nat.lcm a b = 23 * x * y) →
  (max a b = 276) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_larger_number_proof_l1127_112792


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_intersections_l1127_112762

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line in 3D space
  -- (This is a simplified representation)
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- A cube in 3D space -/
structure Cube where
  -- Define properties of a cube
  -- (This is a simplified representation)
  center : ℝ × ℝ × ℝ
  side_length : ℝ

/-- The number of intersections between a line and the faces of a cube -/
def num_intersections (l : Line3D) (c : Cube) : ℕ :=
  sorry -- Implementation details omitted

/-- The theorem stating the possible number of intersections -/
theorem possible_intersections (l : Line3D) (c : Cube) :
  ∃ m : Fin 3, (↑m + 1) * 2 = num_intersections l c := by
  sorry

#check possible_intersections

end NUMINAMATH_CALUDE_ERRORFEEDBACK_possible_intersections_l1127_112762


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_savings_l1127_112735

theorem a_savings (income_ratio : ℚ) (expenditure_ratio : ℚ) (b_savings : ℕ) (b_income : ℕ) :
  income_ratio = 5/6 →
  expenditure_ratio = 3/4 →
  b_savings = 1600 →
  b_income = 7200 →
  ∃ (a_income a_expenditure : ℕ),
    a_income = (income_ratio * b_income).floor ∧
    a_expenditure = (expenditure_ratio * (b_income - b_savings)).floor ∧
    a_income - a_expenditure = 1800 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_savings_l1127_112735


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_division_l1127_112710

/-- Triangle ABC with vertices A = (0,2), B = (0,0), and C = (10,0) -/
def triangle_ABC : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p = (0, 2) ∨ p = (0, 0) ∨ p = (10, 0)}

/-- The area of a triangle given its base and height -/
noncomputable def triangle_area (base height : ℝ) : ℝ :=
  (1 / 2) * base * height

/-- The area of the region to the left of a vertical line x = a -/
noncomputable def left_area (a : ℝ) : ℝ :=
  triangle_area a 2

/-- The total area of triangle ABC -/
noncomputable def total_area : ℝ :=
  triangle_area 10 2

/-- Theorem: The vertical line x = 10/3 divides triangle ABC such that
    the area to the left of the line is one-third of the total area -/
theorem area_division :
  left_area (10/3) = (1/3) * total_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_division_l1127_112710


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l1127_112786

theorem find_c : ∃ c : ℝ, ∀ x : ℝ, x * (4 * x + 2) < c ↔ -3/2 < x ∧ x < 1 :=
by
  use 6
  intro x
  constructor
  · sorry  -- Proof for the forward direction
  · sorry  -- Proof for the reverse direction

#check find_c

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_c_l1127_112786


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XMY_l1127_112766

-- Define the triangle XYZ
structure Triangle where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  Z : ℝ × ℝ

-- Define point M on YZ
def M (t : Triangle) : ℝ × ℝ :=
  sorry -- The actual definition would go here

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry -- The actual definition would go here

-- Define the ratio of XM to MY
axiom ratio_XM_MY (t : Triangle) : 
  (distance t.X (M t)) / (distance (M t) t.Y) = 5 / 2

-- Define the area function for a triangle
def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  sorry -- The actual definition would go here

-- Define the area of triangle XYZ
axiom area_XYZ (t : Triangle) : area_triangle t.X t.Y t.Z = 35

-- State the theorem
theorem area_XMY (t : Triangle) : 
  area_triangle t.X (M t) t.Y = 10 := by
  sorry -- The proof would go here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_XMY_l1127_112766


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_seven_sixths_l1127_112743

noncomputable def f (x : ℝ) : ℝ := min (Real.sqrt x) (-x + 2)

theorem integral_f_equals_seven_sixths :
  ∫ x in (0 : ℝ)..(2 : ℝ), f x = 7/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_seven_sixths_l1127_112743


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_product_implies_no_even_integers_l1127_112794

theorem odd_product_implies_no_even_integers (a b c d e f : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  Odd (a * b * c * d * e * f) → 
  (¬ Even a ∧ ¬ Even b ∧ ¬ Even c ∧ ¬ Even d ∧ ¬ Even e ∧ ¬ Even f) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_product_implies_no_even_integers_l1127_112794


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l1127_112731

def rotate_x (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.2, p.2.1)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (p.1, -p.2.1, p.2.2)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (-p.1, p.2.1, p.2.2)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_yz (rotate_x (reflect_yz (reflect_xz (rotate_x p))))

theorem point_transformation :
  transform (2, 3, -4) = (2, -3, -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_transformation_l1127_112731


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_roots_l1127_112790

/-- The original function f(x) -/
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3) - 2 * (Real.cos (x - Real.pi / 6))^2 + 1

/-- The shifted function g(x) -/
noncomputable def g (x : ℝ) : ℝ := f (x + Real.pi / 6)

/-- Theorem stating the result of sin(2x₁ + 2x₂) -/
theorem sin_sum_of_roots (a : ℝ) (x₁ x₂ : ℝ) 
  (h₁ : g x₁ = a) (h₂ : g x₂ = a) 
  (h₃ : 0 ≤ x₁ ∧ x₁ ≤ Real.pi / 2) (h₄ : 0 ≤ x₂ ∧ x₂ ≤ Real.pi / 2) :
  Real.sin (2 * x₁ + 2 * x₂) = -3/5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_sum_of_roots_l1127_112790


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_l1127_112784

/-- A right pyramid with a square base -/
structure RightPyramid where
  base_side : ℝ
  slant_height : ℝ

/-- The area of a lateral face of a right pyramid -/
noncomputable def lateral_face_area (p : RightPyramid) : ℝ :=
  (1 / 2) * p.base_side * p.slant_height

theorem pyramid_base_side (p : RightPyramid) 
  (h1 : lateral_face_area p = 100)
  (h2 : p.slant_height = 20) : 
  p.base_side = 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_l1127_112784
