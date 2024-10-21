import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_available_implies_exists_not_available_and_not_all_l159_15950

-- Define the universe of books in the library
variable (Book : Type)

-- Define the property of being available for lending
variable (available_for_lending : Book → Prop)

-- Define the statement that all books are available for lending
def all_books_available (Book : Type) (available_for_lending : Book → Prop) : Prop :=
  ∀ b : Book, available_for_lending b

-- Theorem stating that if not all books are available, then there exists a book that is not available
-- and it's not the case that all books are available
theorem not_all_available_implies_exists_not_available_and_not_all
    (Book : Type) (available_for_lending : Book → Prop)
    (h : ¬all_books_available Book available_for_lending) :
    (∃ b : Book, ¬available_for_lending b) ∧ ¬(∀ b : Book, available_for_lending b) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_available_implies_exists_not_available_and_not_all_l159_15950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_2_30_l159_15972

noncomputable def minute_hand_angle (minutes : ℕ) : ℝ :=
  (360 / 60) * minutes

noncomputable def hour_hand_angle (hours minutes : ℕ) : ℝ :=
  (360 / 12) * hours + (360 / 12 / 60) * minutes

noncomputable def smaller_angle (α β : ℝ) : ℝ :=
  min (abs (α - β)) (360 - abs (α - β))

theorem clock_angle_at_2_30 :
  smaller_angle (hour_hand_angle 2 30) (minute_hand_angle 30) = 105 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_clock_angle_at_2_30_l159_15972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l159_15967

theorem triangle_trig_identity (A : ℝ) (h : Real.sin A + Real.cos A = 7/13) :
  Real.tan A = -12/5 ∧ 2 * Real.sin A * Real.cos A - Real.cos A ^ 2 = -145/169 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trig_identity_l159_15967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_theorem_l159_15938

noncomputable section

/-- The ellipse equation -/
def is_on_ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 4 = 1

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := (-Real.sqrt 5, 0)
def F2 : ℝ × ℝ := (Real.sqrt 5, 0)

/-- A point is on the ellipse -/
def P : ℝ × ℝ → Prop := λ p => is_on_ellipse p.1 p.2

/-- The triangle formed by P, F1, and F2 is right-angled -/
def is_right_triangle (p : ℝ × ℝ) : Prop :=
  (p.1 - F1.1)^2 + (p.2 - F1.2)^2 + (p.1 - F2.1)^2 + (p.2 - F2.2)^2 =
  max ((p.1 - F1.1)^2 + (p.2 - F1.2)^2) ((p.1 - F2.1)^2 + (p.2 - F2.2)^2)

/-- The distance from P to F1 is greater than the distance from P to F2 -/
def PF1_greater_PF2 (p : ℝ × ℝ) : Prop :=
  (p.1 - F1.1)^2 + (p.2 - F1.2)^2 > (p.1 - F2.1)^2 + (p.2 - F2.2)^2

/-- The ratio of PF1 to PF2 -/
def ratio (p : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - F1.1)^2 + (p.2 - F1.2)^2) / Real.sqrt ((p.1 - F2.1)^2 + (p.2 - F2.2)^2)

theorem ellipse_ratio_theorem :
  ∀ p : ℝ × ℝ, P p → is_right_triangle p → PF1_greater_PF2 p →
  ratio p = 2 ∨ ratio p = 7/2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_ratio_theorem_l159_15938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l159_15941

/-- Definition of the ⊗ operation for real numbers -/
noncomputable def otimes (a b c : ℝ) : ℝ := a / (b - c)

/-- The main theorem to prove -/
theorem otimes_calculation :
  otimes (otimes 1 2 3) (otimes 2 3 1) (otimes 3 1 2) = -1/4 := by
  -- Expand the definition of otimes
  unfold otimes
  -- Simplify the expression
  simp
  -- The rest of the proof would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_otimes_calculation_l159_15941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_equation_solution_l159_15919

theorem cubic_root_sum_equation_solution : ∃ (x : ℝ) (r s : ℤ),
  (x^(1/3 : ℝ) + (30 - x)^(1/3 : ℝ) = 3) ∧
  (x = r - Real.sqrt (s : ℝ)) ∧
  (∀ y : ℝ, (y^(1/3 : ℝ) + (30 - y)^(1/3 : ℝ) = 3) → y ≤ x) ∧
  (r + s = 96) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_equation_solution_l159_15919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_l159_15930

theorem max_books_borrowed (total_students : ℕ) (zero_books : ℕ) (one_book : ℕ) (two_books : ℕ) 
  (h1 : total_students = 35)
  (h2 : zero_books = 2)
  (h3 : one_book = 12)
  (h4 : two_books = 10)
  (h5 : (total_students : ℚ) * 2 = (zero_books * 0 + one_book * 1 + two_books * 2 + 
        (total_students - zero_books - one_book - two_books) * 3 + 
        (total_students * 2 - (zero_books * 0 + one_book * 1 + two_books * 2 + 
        (total_students - zero_books - one_book - two_books) * 3)))) :
  ∃ (max_books : ℕ), max_books = 8 ∧ 
    ∀ (student_books : ℕ), student_books ≤ max_books := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_books_borrowed_l159_15930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_mowing_money_l159_15926

/-- The amount of money Paul made from mowing lawns -/
def mowing_money : ℝ := sorry

/-- The amount of money Paul made from weed eating -/
def weed_eating_money : ℝ := sorry

/-- The number of weeks the money lasts -/
def weeks : ℝ := sorry

/-- The amount Paul spends per week -/
def spending_per_week : ℝ := sorry

theorem paul_mowing_money :
  mowing_money = weed_eating_money ∧
  weeks = 2 ∧
  spending_per_week = 3 ∧
  mowing_money + weed_eating_money = weeks * spending_per_week →
  mowing_money = 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paul_mowing_money_l159_15926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l159_15982

def equation (x : ℝ) : Prop :=
  2 * Real.cos (2 * x) * (Real.cos (2 * x) - Real.cos (1007 * Real.pi^2 / x)) = Real.cos (4 * x) - 1

def is_solution (x : ℝ) : Prop :=
  x > 0 ∧ equation x

theorem sum_of_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, is_solution x) ∧ (∀ x, is_solution x → x ∈ S) ∧ (S.sum id = 1080 * Real.pi) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_solutions_l159_15982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_l159_15908

theorem existence_of_n : ∃ n : ℕ, n > 0 ∧ ∀ p k : ℤ, 
  Prime p → p < 2008 → ¬(p ∣ k^2 + k + n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_n_l159_15908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_proof_l159_15998

def total_matches : ℕ := 5
def goals_fifth_match : ℕ := 3
def total_goals : ℕ := 11

theorem average_increase_proof :
  let old_average : ℚ := (total_goals - goals_fifth_match : ℚ) / (total_matches - 1 : ℚ)
  let new_average : ℚ := (total_goals : ℚ) / (total_matches : ℚ)
  new_average - old_average = 1/5 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_average_increase_proof_l159_15998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_all_special_numbers_satisfy_conditions_no_other_numbers_satisfy_conditions_l159_15997

def is_product_of_four_distinct_primes (n : ℕ) : Prop :=
  ∃ p q r s : ℕ, 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ Nat.Prime s ∧ 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s ∧
    n = p * q * r * s

def three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def special_numbers : List ℕ :=
  [210, 330, 390, 510, 570, 690, 870, 930, 462, 546, 714, 798, 966, 770, 910, 858]

theorem sum_of_special_numbers : 
  special_numbers.sum = 10494 := by
  sorry

theorem all_special_numbers_satisfy_conditions :
  ∀ n ∈ special_numbers, three_digit_number n ∧ is_product_of_four_distinct_primes n := by
  sorry

theorem no_other_numbers_satisfy_conditions :
  ∀ n, three_digit_number n ∧ is_product_of_four_distinct_primes n → n ∈ special_numbers := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_special_numbers_all_special_numbers_satisfy_conditions_no_other_numbers_satisfy_conditions_l159_15997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l159_15921

noncomputable def f (x : ℝ) : ℝ := (3 * x + 2) / (x + 1)

theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≥ 2 ∧ f x = y) ↔ y ∈ Set.Ici (8/3) ∩ Set.Iio 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l159_15921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_third_term_is_79_l159_15986

def jo_blair_sequence : ℕ → ℤ
  | 0 => 1
  | n + 1 => if n % 2 = 0 then jo_blair_sequence n + 2 else jo_blair_sequence n - 1

theorem fifty_third_term_is_79 : jo_blair_sequence 52 = 79 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_third_term_is_79_l159_15986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volumes_l159_15913

namespace CylinderVolumes

noncomputable def π : ℝ := Real.pi

noncomputable def V₁ : ℝ := π * 10^2 * 10
noncomputable def V₂ : ℝ := π * 5^2 * 10
noncomputable def V₃ : ℝ := π * 5^2 * 20

theorem cylinder_volumes :
  (V₂ < V₃ ∧ V₃ < V₁) ∧
  (250 * π < π * 5^2 * 15 ∧ π * 5^2 * 15 < 500 * π) ∧
  (500 * π < π * 8^2 * 10 ∧ π * 8^2 * 10 < 1000 * π) :=
by
  sorry

end CylinderVolumes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_volumes_l159_15913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l159_15937

open Matrix

theorem matrix_vector_computation 
  (M : Matrix (Fin 2) (Fin 2) ℝ) 
  (v w u : Fin 2 → ℝ) 
  (hv : M.vecMul v = ![2, -3])
  (hw : M.vecMul w = ![-1, 4])
  (hu : M.vecMul u = ![5, -2]) :
  M.vecMul (3 • v - 4 • w + 2 • u) = ![20, -29] := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_matrix_vector_computation_l159_15937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_multiples_of_four_l159_15939

/-- The sequence u_n defined recursively -/
def u (a : ℕ) : ℕ → ℕ
  | 0 => 2  -- Added case for 0
  | 1 => 2
  | 2 => a^2 + 2
  | n+3 => a * u a (n+1) - u a (n+2)

/-- The condition that a must satisfy -/
def valid_a (a : ℕ) : Prop :=
  ∃ k : ℕ, a = 4*k ∨ a = 4*k + 2

theorem no_multiples_of_four (a : ℕ) :
  (∀ n : ℕ, ¬ (4 ∣ u a n)) ↔ valid_a a :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_multiples_of_four_l159_15939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_are_irrational_l159_15932

theorem roots_are_irrational (m : ℝ) : 
  let eq := fun x : ℝ => x^2 - 5*m*x + 3*m^2 + 2
  let roots := {x : ℝ | eq x = 0}
  (∀ x y, x ∈ roots → y ∈ roots → x * y = 8) → 
  ∀ z, z ∈ roots → ¬ (∃ (p q : ℤ), z = ↑p / ↑q ∧ q ≠ 0) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_are_irrational_l159_15932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_property_l159_15954

/-- The area of a region bounded by three circular arcs -/
noncomputable def region_area (r : ℝ) (θ : ℝ) : ℝ :=
  3 * (θ / (2 * Real.pi) * Real.pi * r^2 - r^2 * Real.sqrt 3 / 2)

/-- Theorem stating the properties of the area of the specific region -/
theorem area_property :
  ∃ (a b c : ℝ),
    region_area 3 (Real.pi / 2) = a * Real.sqrt b + c * Real.pi ∧
    a + b + c = 3 / 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_property_l159_15954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trough_fill_time_l159_15931

/-- Represents the capacity of the trough in arbitrary units -/
noncomputable def C : ℝ := 1  -- We define C as a concrete value to avoid the 'variable' error

/-- Represents the time it takes to fill the trough using all pumps, considering the leak -/
noncomputable def fillTime : ℝ := 600 / 7

theorem trough_fill_time :
  let oldPumpRate : ℝ := C / 600
  let secondPumpRate : ℝ := C / 200
  let thirdPumpRate : ℝ := C / 400
  let fourthPumpRate : ℝ := C / 300
  let leakRate : ℝ := C / 1200
  let netFillRate : ℝ := oldPumpRate + secondPumpRate + thirdPumpRate + fourthPumpRate - leakRate
  C > 0 →
  C / netFillRate = fillTime :=
by
  sorry  -- We use 'sorry' to skip the proof for now

#check trough_fill_time  -- This line checks if the theorem is well-formed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trough_fill_time_l159_15931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_and_square_divisors_l159_15956

def N : ℕ := (Finset.range 17).prod (λ i => i + 1)

theorem largest_cube_and_square_divisors :
  (∃ (a : ℕ), a^3 ∣ N ∧ ∀ (b : ℕ), b^3 ∣ N → b ≤ a) ∧
  (∃ (c : ℕ), c^2 ∣ N ∧ ∀ (d : ℕ), d^2 ∣ N → d ≤ c) ∧
  1440^3 ∣ N ∧ 120960^2 ∣ N :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_and_square_divisors_l159_15956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_is_zero_l159_15940

/-- Projection matrix onto a 2D vector -/
noncomputable def projection_matrix (v : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm := Real.sqrt (v.1^2 + v.2^2)
  let cos_phi := v.1 / norm
  let sin_phi := v.2 / norm
  !![cos_phi^2, cos_phi * sin_phi;
     cos_phi * sin_phi, sin_phi^2]

/-- The vector we're projecting onto -/
def v : ℝ × ℝ := (3, 5)

/-- Theorem: The determinant of the projection matrix onto (3, 5) is 0 -/
theorem det_projection_matrix_is_zero :
  Matrix.det (projection_matrix v) = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_projection_matrix_is_zero_l159_15940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_survivor_quitters_probability_l159_15971

def total_contestants : ℕ := 18
def num_tribes : ℕ := 3
def contestants_per_tribe : ℕ := 6
def num_quitters : ℕ := 2

theorem survivor_quitters_probability : 
  (total_contestants = num_tribes * contestants_per_tribe) →
  (num_quitters = 2) →
  (Nat.choose total_contestants num_quitters > 0) →
  (∀ t : ℕ, t ≤ num_tribes → Nat.choose contestants_per_tribe num_quitters > 0) →
  (↑(num_tribes * Nat.choose contestants_per_tribe num_quitters) / 
   ↑(Nat.choose total_contestants num_quitters) : ℚ) = 5 / 17 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_survivor_quitters_probability_l159_15971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l159_15912

noncomputable def f (x : ℝ) := Real.sqrt 2 * Real.sin x + Real.sqrt 6 * Real.cos x

theorem problem_solution :
  (∃! a : ℝ, a ∈ Set.Icc 0 π ∧ f a = 2 ∧ a = 5 * π / 12) ∧
  (let g (x θ : ℝ) := f (2 * (x - θ))
   ∃ θ : ℝ, θ > 0 ∧
     (∀ x : ℝ, g x θ = g (3 * π / 2 - x) θ) ∧
     (∀ θ' : ℝ, θ' > 0 → (∀ x : ℝ, g x θ' = g (3 * π / 2 - x) θ') → θ ≤ θ') ∧
     θ = π / 6) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l159_15912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_principal_tripled_l159_15911

/-- Simple interest calculation function -/
noncomputable def simpleInterest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  (principal * rate * time) / 100

theorem total_interest_after_principal_tripled
  (P : ℝ) -- Initial principal
  (R : ℝ) -- Interest rate (in percentage per annum)
  (h1 : simpleInterest P R 10 = 1400) -- Condition: Simple interest after 10 years is 1400
  (h2 : P > 0) -- Assumption: Principal is positive
  (h3 : R > 0) -- Assumption: Interest rate is positive
  : simpleInterest P R 5 + simpleInterest (3 * P) R 5 = 1421 := by
  sorry

#check total_interest_after_principal_tripled

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_interest_after_principal_tripled_l159_15911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_min_area_is_four_min_area_line_l159_15942

/-- The line equation with parameter m -/
def line_equation (x y m : ℝ) : Prop :=
  (2 + m) * x + (1 - 2 * m) * y + 4 - 3 * m = 0

/-- The point that the line always passes through -/
def fixed_point : ℝ × ℝ := (-1, -2)

/-- Theorem: The line always passes through the fixed point for all m -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation (fixed_point.1) (fixed_point.2) m := by sorry

/-- The area of triangle AOB given the slope k -/
noncomputable def triangle_area (k : ℝ) : ℝ := 
  (1/2) * abs ((2/k - 1) * (k - 2))

/-- Theorem: The minimum area of triangle AOB is 4 -/
theorem min_area_is_four :
  ∃ k : ℝ, k < 0 ∧ 
  (∀ k' : ℝ, k' < 0 → triangle_area k ≤ triangle_area k') ∧
  triangle_area k = 4 := by sorry

/-- The equation of the line when the area is minimum -/
def min_area_line_equation (x y : ℝ) : Prop :=
  y + 2 * x + 4 = 0

/-- Theorem: The equation of the line when the area is minimum -/
theorem min_area_line :
  ∃ m : ℝ, (∀ x y : ℝ, line_equation x y m ↔ min_area_line_equation x y) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_passes_through_fixed_point_min_area_is_four_min_area_line_l159_15942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_percentage_conditions_l159_15962

theorem least_integer_with_percentage_conditions (N : ℕ) : 
  (∃ a : ℚ, N = Int.floor (0.78 * a)) ∧ 
  (∃ b : ℚ, N = Int.ceil (1.16 * b)) ∧ 
  (∀ m : ℕ, m < N → ¬((∃ x : ℚ, m = Int.floor (0.78 * x)) ∧ 
                      (∃ y : ℚ, m = Int.ceil (1.16 * y)))) →
  N % 1000 = 262 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_least_integer_with_percentage_conditions_l159_15962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l159_15903

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- The function f(x) = 1 + a / (2^x + 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  1 + a / (2^x + 1)

theorem odd_function_a_value :
  ∀ a : ℝ, IsOdd (f a) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_a_value_l159_15903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l159_15953

/-- The curve C in the Cartesian plane -/
def C (x y : ℝ) : Prop := y^2 = 4*x

/-- The line l in the Cartesian plane -/
def l (x y : ℝ) : Prop := y = -2*x + 4

/-- The distance between two points in the Cartesian plane -/
noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem intersection_distance :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C x₁ y₁ ∧ C x₂ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂ ∧
    distance x₁ y₁ x₂ y₂ = 3 * Real.sqrt 5 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_l159_15953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_is_line_radical_axis_perpendicular_to_centers_radical_axis_through_intersection_same_power_point_for_three_circles_l159_15973

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the radical axis of two circles
def radicalAxis (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.fst - c1.center.fst)^2 + (p.snd - c1.center.snd)^2 - c1.radius^2 
             = (p.fst - c2.center.fst)^2 + (p.snd - c2.center.snd)^2 - c2.radius^2}

-- Define a line in 2D space
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ  -- ax + by + c = 0

-- Define perpendicularity of two lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Theorem: The radical axis of two circles is a straight line
theorem radical_axis_is_line (c1 c2 : Circle) : 
  ∃ l : Line, ∀ p : ℝ × ℝ, p ∈ radicalAxis c1 c2 ↔ l.a * p.fst + l.b * p.snd + l.c = 0 :=
sorry

-- Theorem: The radical axis is perpendicular to the line joining the centers
theorem radical_axis_perpendicular_to_centers (c1 c2 : Circle) :
  ∃ l_rad : Line, ∃ l_centers : Line,
    (∀ p : ℝ × ℝ, p ∈ radicalAxis c1 c2 ↔ l_rad.a * p.fst + l_rad.b * p.snd + l_rad.c = 0) ∧
    (l_centers.a * (c2.center.fst - c1.center.fst) + l_centers.b * (c2.center.snd - c1.center.snd) = 0) ∧
    perpendicular l_rad l_centers :=
sorry

-- Define intersection of two circles
def circleIntersection (c1 c2 : Circle) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.fst - c1.center.fst)^2 + (p.snd - c1.center.snd)^2 = c1.radius^2 ∧
               (p.fst - c2.center.fst)^2 + (p.snd - c2.center.snd)^2 = c2.radius^2}

-- Theorem: If two circles intersect, their radical axis passes through the intersection points
theorem radical_axis_through_intersection (c1 c2 : Circle) :
  ∀ p : ℝ × ℝ, p ∈ circleIntersection c1 c2 → p ∈ radicalAxis c1 c2 :=
sorry

-- Theorem: For three circles, if a point has the same power with respect to all three,
-- it lies at the intersection of their pairwise radical axes
theorem same_power_point_for_three_circles (c1 c2 c3 : Circle) :
  ∀ p : ℝ × ℝ, 
    (p ∈ radicalAxis c1 c2 ∧ p ∈ radicalAxis c2 c3) → 
    p ∈ radicalAxis c1 c3 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radical_axis_is_line_radical_axis_perpendicular_to_centers_radical_axis_through_intersection_same_power_point_for_three_circles_l159_15973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_circle_from_three_points_l159_15991

-- Define a point in a 2D plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a circle
structure Circle where
  center : Point
  radius : ℝ

-- Define membership for a point in a circle
def Point.mem (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

instance : Membership Point Circle where
  mem := Point.mem

-- Three points not on the same line
def not_collinear (p1 p2 p3 : Point) : Prop :=
  (p2.x - p1.x) * (p3.y - p1.y) ≠ (p3.x - p1.x) * (p2.y - p1.y)

-- Theorem: Three points not on the same line determine exactly one circle
theorem unique_circle_from_three_points (p1 p2 p3 : Point) 
  (h : not_collinear p1 p2 p3) : 
  ∃! c : Circle, p1 ∈ c ∧ p2 ∈ c ∧ p3 ∈ c :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_circle_from_three_points_l159_15991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_star_polygon_exterior_angle_sum_l159_15925

/-- A star-shaped polygon formed from a convex polygon -/
structure StarPolygon where
  n : ℕ
  vertices : Fin n → ℝ × ℝ
  is_star_shaped : Bool
  h_n : n ≥ 3

/-- Sum of exterior angles function (placeholder) -/
def sum_exterior_angles (p : StarPolygon) : ℝ := sorry

/-- The sum of exterior angles of a star-shaped polygon is 360 degrees -/
theorem star_polygon_exterior_angle_sum (p : StarPolygon) :
  sum_exterior_angles p = 360 := by
  sorry

#check star_polygon_exterior_angle_sum

end NUMINAMATH_CALUDE_ERRORFEEDBACK_star_polygon_exterior_angle_sum_l159_15925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_money_is_1275_l159_15994

noncomputable def monthly_salary : ℝ := 6000
noncomputable def house_rental : ℝ := 640
noncomputable def food_expense : ℝ := 385
noncomputable def electric_water_bill : ℝ := monthly_salary / 4
noncomputable def insurance_cost : ℝ := monthly_salary / 5
noncomputable def tax_rate : ℝ := 0.10
noncomputable def transportation_rate : ℝ := 0.03
noncomputable def emergency_rate : ℝ := 0.02
noncomputable def student_loan : ℝ := 300
noncomputable def retirement_rate : ℝ := 0.05

noncomputable def total_expenses : ℝ := 
  house_rental + food_expense + electric_water_bill + insurance_cost + 
  (tax_rate * monthly_salary) + (transportation_rate * monthly_salary) + 
  (emergency_rate * monthly_salary) + student_loan + (retirement_rate * monthly_salary)

theorem remaining_money_is_1275 : 
  monthly_salary - total_expenses = 1275 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_money_is_1275_l159_15994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_no_inverse_mod_72_and_90_l159_15935

theorem smallest_no_inverse_mod_72_and_90 : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (∀ (x : ℕ), x < a → (∃ (y : ℕ), y < 72 ∧ x * y % 72 = 1) ∨ (∃ (z : ℕ), z < 90 ∧ x * z % 90 = 1)) ∧
  (∀ (y : ℕ), y < 72 → 6 * y % 72 ≠ 1) ∧
  (∀ (z : ℕ), z < 90 → 6 * z % 90 ≠ 1) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_no_inverse_mod_72_and_90_l159_15935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_distribution_theorem_l159_15977

inductive Grade
  | Two
  | Three
  | Four
  | Five

def gradeValue (g : Grade) : Nat :=
  match g with
  | Grade.Two => 2
  | Grade.Three => 3
  | Grade.Four => 4
  | Grade.Five => 5

def GradeCount := Grade → Nat

theorem grade_distribution_theorem (gc : GradeCount) 
  (h1 : (gc Grade.Two) + (gc Grade.Three) + (gc Grade.Four) + (gc Grade.Five) = 13)
  (h2 : ∃ (m : Nat), (gradeValue Grade.Two * gc Grade.Two + 
                      gradeValue Grade.Three * gc Grade.Three + 
                      gradeValue Grade.Four * gc Grade.Four + 
                      gradeValue Grade.Five * gc Grade.Five) = 13 * m) :
  ∃ (g : Grade), gc g ≤ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_grade_distribution_theorem_l159_15977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digital_roots_of_squares_digital_roots_of_cubes_l159_15907

-- Define digital root
def digitalRoot (n : ℤ) : ℤ :=
  if n % 9 = 0 then 9 else n % 9

-- Define perfect square
def isPerfectSquare (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m * m

-- Define perfect cube
def isPerfectCube (n : ℤ) : Prop :=
  ∃ m : ℤ, n = m * m * m

-- Theorem for digital roots of perfect squares
theorem digital_roots_of_squares :
  ∀ n : ℤ, isPerfectSquare n → digitalRoot n ∈ ({-1, 4, 7, 9} : Set ℤ) :=
by
  sorry

-- Theorem for digital roots of perfect cubes
theorem digital_roots_of_cubes :
  ∀ n : ℤ, isPerfectCube n → digitalRoot n ∈ ({-1, 8, 9} : Set ℤ) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digital_roots_of_squares_digital_roots_of_cubes_l159_15907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_special_set_l159_15917

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let special_number := (1 - 2 / (n : ℝ))^2
  let sum := special_number + (n - 1 : ℝ)
  sum / (n : ℝ) = 1 - 4 / (n : ℝ)^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_mean_of_special_set_l159_15917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_function_in_X_l159_15905

-- Define the set of functional expressions
inductive FunctionalExpr
| constant : ℝ → FunctionalExpr
| variable : ℕ → FunctionalExpr
| apply : (ℝ → ℝ) → FunctionalExpr → FunctionalExpr
| add : FunctionalExpr → FunctionalExpr → FunctionalExpr
| sub : FunctionalExpr → FunctionalExpr → FunctionalExpr
| mul : FunctionalExpr → FunctionalExpr → FunctionalExpr

-- Define the set X of functions from ℝ to ℤ
def X : Set (ℝ → ℝ) := {f | ∀ x, ∃ n : ℤ, f x = n}

-- Define evaluation function for FunctionalExpr
def eval (E : FunctionalExpr) (f : ℝ → ℝ) (vars : ℕ → ℝ) : ℝ :=
  match E with
  | FunctionalExpr.constant c => c
  | FunctionalExpr.variable n => vars n
  | FunctionalExpr.apply g e => g (eval e f vars)
  | FunctionalExpr.add e1 e2 => eval e1 f vars + eval e2 f vars
  | FunctionalExpr.sub e1 e2 => eval e1 f vars - eval e2 f vars
  | FunctionalExpr.mul e1 e2 => eval e1 f vars * eval e2 f vars

-- Define the set S of functions satisfying E = 0
def S (E : FunctionalExpr) : Set (ℝ → ℝ) :=
  {f | ∀ vars, eval E f vars = 0}

-- Statement to prove
theorem exists_unique_function_in_X :
  ∃ E : FunctionalExpr,
    (∃ f, f ∈ S E) ∧  -- S is nonempty
    (S E ⊆ X) ∧      -- S is a subset of X
    (∃! f, f ∈ S E)  -- S contains exactly one element
  := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_unique_function_in_X_l159_15905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l159_15948

-- Define the circle O
def circle_O : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 4}

-- Define the tangent line
def tangent_line : Set (ℝ × ℝ) := {p | p.1 - Real.sqrt 3 * p.2 = 4}

-- Define the symmetry line
def symmetry_line : Set (ℝ × ℝ) := {p | p.1 + 2 * p.2 = 0}

-- Define points A and B
def point_A : ℝ × ℝ := (-2, 0)
def point_B : ℝ × ℝ := (2, 0)

-- Define the internal point P
def point_P : Set (ℝ × ℝ) := {p | p ∈ circle_O ∧ p.1^2 + p.2^2 < 4}

-- Define the dot product of vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem circle_problem :
  ∀ P ∈ point_P,
  let PA := (P.1 - point_A.1, P.2 - point_A.2);
  let PB := (P.1 - point_B.1, P.2 - point_B.2);
  -2 ≤ dot_product PA PB ∧ dot_product PA PB < 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_problem_l159_15948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_border_length_l159_15992

-- Define π as 22/7
def π : ℚ := 22 / 7

-- Define the area of the circle
def circle_area : ℚ := 616

-- Define the extra length Mark wants to buy
def extra_length : ℚ := 3

-- Theorem statement
theorem border_length : 
  let r : ℚ := (circle_area / π).sqrt
  (2 * π * r + extra_length : ℚ) = 91 := by
  -- Proof steps would go here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_border_length_l159_15992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l159_15980

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Distance from a point to a line -/
noncomputable def distancePointToLine (x y : ℝ) (l : Line) : ℝ :=
  abs (l.slope * x - y + l.intercept) / Real.sqrt (l.slope^2 + 1)

/-- Theorem: Given a line passing through (1,2) and equidistant from (3,3) and (5,2),
    its equation is either x - 2y + 3 = 0 or x + 6y - 13 = 0 -/
theorem line_equation_theorem :
  ∀ l : Line,
    (l.slope * 1 + l.intercept = 2) →
    (distancePointToLine 3 3 l = distancePointToLine 5 2 l) →
    ((l.slope = 1/2 ∧ l.intercept = 3/2) ∨ (l.slope = -1/6 ∧ l.intercept = 13/6)) :=
by
  sorry

#check line_equation_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_theorem_l159_15980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l159_15945

-- Define the function f(x)
noncomputable def f (a b x : ℝ) : ℝ := a^x + b - 1

-- Define the conditions for the graph passing through quadrants
def passes_first_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x > 0, f x > 0

def passes_second_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x < 0, f x > 0

def passes_fourth_quadrant (f : ℝ → ℝ) : Prop :=
  ∃ x > 0, f x < 0

def not_passes_third_quadrant (f : ℝ → ℝ) : Prop :=
  ∀ x < 0, f x ≥ 0

-- Main theorem
theorem function_properties (a b : ℝ) (ha : a > 0) (ha_ne_1 : a ≠ 1) :
  passes_first_quadrant (f a b) →
  passes_second_quadrant (f a b) →
  passes_fourth_quadrant (f a b) →
  not_passes_third_quadrant (f a b) →
  (0 < a ∧ a < 1) ∧ (0 < b ∧ b < 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l159_15945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l159_15970

noncomputable section

structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

noncomputable def Ellipse.eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

theorem ellipse_eccentricity_range (e : Ellipse) 
  (h : ∃ (P : ℝ × ℝ), P.1^2 / e.a^2 + P.2^2 / e.b^2 = 1 ∧ 
    let F₁ := (-Real.sqrt (e.a^2 - e.b^2), 0)
    let F₂ := (Real.sqrt (e.a^2 - e.b^2), 0)
    (P.1 - F₁.1) * (P.1 - F₂.1) + (P.2 - F₁.2) * (P.2 - F₂.2) = 0) :
  Real.sqrt 2 / 2 ≤ e.eccentricity ∧ e.eccentricity < 1 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_range_l159_15970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l159_15952

open Real

theorem indefinite_integral_proof (x : ℝ) (h : x ≠ -1 ∧ x ≠ -2) :
  let f := λ x : ℝ ↦ (3*x^2 / 2) - 9*x + 22 * log (abs (x+1)) - log (abs (x+2))
  let g := λ x : ℝ ↦ (3*x^3 + 25) / (x^2 + 3*x + 2)
  deriv f x = g x :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_indefinite_integral_proof_l159_15952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_container_l159_15910

/-- The length of the sheet -/
def length : ℝ := 90

/-- The width of the sheet -/
def width : ℝ := 48

/-- The height of the container (which is the side length of the cut squares) -/
def container_height : ℝ := 10

/-- The volume of the container as a function of height -/
def volume (h : ℝ) : ℝ := (length - 2*h) * (width - 2*h) * h

theorem max_volume_container :
  (∀ h : ℝ, 0 < h → h < width/2 → h < length/2 → volume h ≤ volume container_height) ∧
  volume container_height = 19600 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_volume_container_l159_15910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_l159_15936

/-- The compound interest formula -/
noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℝ) (t : ℝ) : ℝ :=
  P * (1 + r/n)^(n*t)

/-- The problem statement -/
theorem initial_investment (P : ℝ) :
  compound_interest P 0.20 1 5 = 1000 →
  ∃ ε > 0, |P - 401.88| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_initial_investment_l159_15936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l159_15966

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def isAcute (t : Triangle) : Prop :=
  0 < t.A ∧ t.A < Real.pi/2 ∧
  0 < t.B ∧ t.B < Real.pi/2 ∧
  0 < t.C ∧ t.C < Real.pi/2

def satisfiesConditions (t : Triangle) : Prop :=
  isAcute t ∧
  Real.sqrt 3 * t.a = 2 * t.c * Real.sin t.A ∧
  t.c = Real.sqrt 7 ∧
  t.a * t.b = 6

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : satisfiesConditions t) : 
  t.C = Real.pi/3 ∧ t.a + t.b + t.c = 5 + Real.sqrt 7 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l159_15966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_X_equals_3_l159_15914

/-- The probability mass function for a binomial distribution -/
noncomputable def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

/-- The random variable X follows a binomial distribution B(6, 1/2) -/
noncomputable def X : ℕ → ℝ := binomial_pmf 6 (1/2)

/-- Theorem: The probability P(X=3) is equal to 5/16 -/
theorem prob_X_equals_3 : X 3 = 5/16 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_X_equals_3_l159_15914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_1_from_C_center_circle_Q_equation_no_perpendicular_bisector_l159_15963

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y + 4 = 0

-- Define point P
def P : ℝ × ℝ := (2, 0)

-- Part 1
theorem line_through_P_distance_1_from_C_center :
  ∃ (k : ℝ), (∀ x y : ℝ, y = k*(x - 2) → 
    (abs (3*k + 2 - 2*k) / Real.sqrt (k^2 + 1) = 1)) ∨
  (∀ x y : ℝ, x = 2 → 
    (abs (3*0 + 2 - 2*0) / Real.sqrt (0^2 + 1) = 1)) := by
  sorry

-- Part 2
theorem circle_Q_equation :
  ∃ (M N : ℝ × ℝ), C M.1 M.2 ∧ C N.1 N.2 ∧
  (∃ k : ℝ, (M.2 - P.2) = k*(M.1 - P.1) ∧ (N.2 - P.2) = k*(N.1 - P.1)) ∧
  Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 4 →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 ↔ 
    Real.sqrt ((x - M.1)^2 + (y - M.2)^2) +
    Real.sqrt ((x - N.1)^2 + (y - N.2)^2) = 4 := by
  sorry

-- Part 3
theorem no_perpendicular_bisector :
  ¬ ∃ a : ℝ, (∃ A B : ℝ × ℝ, 
    C A.1 A.2 ∧ C B.1 B.2 ∧
    A.2 = a*A.1 + 1 ∧ B.2 = a*B.1 + 1 ∧
    (∃ k : ℝ, (3 - 2) = k*(3 - P.1) ∧ (-2 - 0) = k*(-2 - P.2) ∧
      (B.2 - A.2) = -(1/k)*(B.1 - A.1))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_P_distance_1_from_C_center_circle_Q_equation_no_perpendicular_bisector_l159_15963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_color_l159_15904

/-- A color type with two possible values: blue and white -/
inductive Color : Type
| blue : Color
| white : Color

/-- The coloring function that assigns a color to each integer in M -/
def coloring (n : ℕ) : ℕ → Color := sorry

/-- The set M containing integers from 1 to n-1 -/
def M (n : ℕ) : Set ℕ := {i : ℕ | 1 ≤ i ∧ i < n}

theorem all_same_color (n k : ℕ) (h1 : k < n) (h2 : Nat.Coprime n k) :
  (∀ i ∈ M n, coloring n i = coloring n (n - i)) →
  (∀ i ∈ M n, i ≠ k → coloring n i = coloring n (Int.natAbs (i - k))) →
  ∃ c : Color, ∀ i ∈ M n, coloring n i = c :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_same_color_l159_15904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l159_15915

-- Define the triangle
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  0 < A ∧ A < Real.pi ∧ 0 < B ∧ B < Real.pi ∧ 0 < C ∧ C < Real.pi ∧
  A + B + C = Real.pi

-- State the theorem
theorem triangle_problem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_dot_product : c * a * Real.cos B = 3/2)
  (h_area : 1/2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4)
  (h_b : b = Real.sqrt 3) :
  (a + c = 2 * Real.sqrt 3) ∧ 
  (∀ x, 2 * Real.sin A - Real.sin C = x → -Real.sqrt 3 / 2 < x ∧ x < Real.sqrt 3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l159_15915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_beacons_l159_15951

-- Define the labyrinth graph
structure Labyrinth where
  rooms : Finset Nat
  corridors : Finset (Nat × Nat)

-- Define a beacon
structure Beacon where
  location : Nat

-- Define the distance function
noncomputable def distance (l : Labyrinth) (a b : Nat) : Nat :=
  sorry

-- Define the function to check if a set of beacons allows unambiguous location determination
def unambiguous_location (l : Labyrinth) (beacons : Finset Beacon) : Prop :=
  ∀ r1 r2 : Nat, r1 ∈ l.rooms → r2 ∈ l.rooms → r1 ≠ r2 →
    ∃ b : Beacon, b ∈ beacons ∧ distance l r1 b.location ≠ distance l r2 b.location

-- The main theorem
theorem min_beacons (l : Labyrinth) : 
  (∃ beacons : Finset Beacon, beacons.card = 3 ∧ unambiguous_location l beacons) ∧
  (∀ beacons : Finset Beacon, beacons.card < 3 → ¬(unambiguous_location l beacons)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_beacons_l159_15951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_zone_points_l159_15974

/-- A target with a bullseye and four concentric rings -/
structure Target where
  r : ℝ  -- radius of the bullseye
  bullseye_points : ℝ  -- points for hitting the bullseye

/-- The area of a circle with radius r -/
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

/-- The area of a ring with inner radius r1 and outer radius r2 -/
noncomputable def ring_area (r1 r2 : ℝ) : ℝ := circle_area r2 - circle_area r1

/-- The points for hitting a zone, given its area and the bullseye area and points -/
noncomputable def zone_points (zone_area bullseye_area bullseye_points : ℝ) : ℝ :=
  bullseye_points * bullseye_area / zone_area

theorem blue_zone_points (t : Target) :
  let bullseye_area := circle_area t.r
  let blue_zone_area := ring_area (3 * t.r) (4 * t.r)
  zone_points blue_zone_area bullseye_area t.bullseye_points = 45 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_blue_zone_points_l159_15974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l159_15927

/-- The perimeter of a triangle with sides of lengths 40, 50, and 70 is 160. -/
theorem triangle_perimeter (a b c : ℝ) : a = 40 ∧ b = 50 ∧ c = 70 → a + b + c = 160 := by
  intro h
  rw [h.1, h.2.1, h.2.2]
  norm_num

#check triangle_perimeter

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l159_15927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l159_15934

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (3 * x - x^2)

theorem domain_of_f : Set.Icc 0 3 = {x : ℝ | ∃ y, f x = y} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l159_15934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l159_15959

noncomputable def arithmetic_sequence (a : ℝ) (n : ℕ) : ℝ := n * a

noncomputable def sum_arithmetic_sequence (a : ℝ) (n : ℕ) : ℝ := (n + 1) * n * a / 2

noncomputable def A (a : ℝ) (n : ℕ) : ℝ := (2 / a) * (1 - 1 / (n + 1 : ℝ))

noncomputable def B (a : ℝ) (n : ℕ) : ℝ := (2 / a) * (1 - 1 / 2^n)

theorem arithmetic_sequence_properties (a : ℝ) (h : a ≠ 0) :
  (∀ n : ℕ, arithmetic_sequence a n = n * a) ∧
  (∀ n : ℕ, sum_arithmetic_sequence a n = (n + 1) * n * a / 2) ∧
  (1 / a)^2 = (1 / (a + a)) * (1 / (a + 3*a)) →
  (∀ n : ℕ, n ≥ 2 → (a > 0 → A a n < B a n) ∧ (a < 0 → A a n > B a n)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_properties_l159_15959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_A_and_value_l159_15949

-- Define the polynomial A
def A (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 4

-- Theorem statement
theorem polynomial_A_and_value :
  (∀ x : ℝ, A x - (x - 2)^2 = x * (x + 7)) ∧
  (∀ x : ℝ, (3 : ℝ)^(x + 1) = 1 → A x = 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_A_and_value_l159_15949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_product_l159_15968

theorem fraction_exponent_product : (1 / 3 : ℚ)^6 * (5 / 2 : ℚ)^4 = 625 / 11664 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_exponent_product_l159_15968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l159_15946

/-- An ellipse C with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- The eccentricity of an ellipse -/
noncomputable def eccentricity (e : Ellipse) : ℝ :=
  Real.sqrt (1 - (e.b / e.a)^2)

/-- A point on the xy-plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Predicate to check if a point lies on an ellipse -/
def on_ellipse (p : Point) (e : Ellipse) : Prop :=
  (p.x^2 / e.a^2) + (p.y^2 / e.b^2) = 1

theorem ellipse_equation (e : Ellipse) 
    (h_point : on_ellipse ⟨0, 4⟩ e)
    (h_ecc : eccentricity e = 3/5) :
    e.a = 5 ∧ e.b = 4 := by
  sorry

#check ellipse_equation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_equation_l159_15946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_afternoon_rate_is_eight_l159_15916

/-- Represents a wood stove --/
structure WoodStove where
  morning_rate : ℚ
  afternoon_rate : ℚ

/-- Represents the wood burning scenario --/
structure WoodBurningScenario where
  stove_a : WoodStove
  stove_b : WoodStove
  morning_duration : ℚ
  afternoon_duration : ℚ
  total_morning_burn : ℚ
  initial_wood : ℚ
  final_wood : ℚ

/-- Calculates the combined afternoon burning rate --/
def combined_afternoon_rate (scenario : WoodBurningScenario) : ℚ :=
  let total_burned := scenario.initial_wood - scenario.final_wood
  let afternoon_burned := total_burned - scenario.total_morning_burn
  afternoon_burned / scenario.afternoon_duration

/-- Theorem stating the combined afternoon burning rate is 8 bundles per hour --/
theorem afternoon_rate_is_eight (scenario : WoodBurningScenario) 
    (h1 : scenario.stove_a.morning_rate = 2)
    (h2 : scenario.stove_b.morning_rate = 1)
    (h3 : scenario.morning_duration = 4)
    (h4 : scenario.afternoon_duration = 4)
    (h5 : scenario.total_morning_burn = 12)
    (h6 : scenario.initial_wood = 50)
    (h7 : scenario.final_wood = 6) :
    combined_afternoon_rate scenario = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_afternoon_rate_is_eight_l159_15916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_properties_l159_15988

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * Real.sin x + Real.sqrt 2 * Real.cos x

theorem triangle_area_and_function_properties 
  (m : ℝ) 
  (h_m_pos : m > 0) 
  (h_max : ∀ x, f m x ≤ 2) :
  (∃ (A B C : ℝ) (a b c : ℝ),
    C = Real.pi / 3 ∧ 
    c = 3 ∧
    f m (A - Real.pi / 4) + f m (B - Real.pi / 4) = 4 * Real.sqrt 6 * Real.sin A * Real.sin B ∧
    (∀ x ∈ Set.Icc (Real.pi / 4 : ℝ) Real.pi, 
      ∀ y ∈ Set.Icc (Real.pi / 4 : ℝ) Real.pi, 
      x ≤ y → f m y ≤ f m x) ∧
    Real.sqrt 3 * (a * b * Real.sin C) / 2 = 3 * Real.sqrt 3 / 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_and_function_properties_l159_15988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_sum_of_sum_l159_15975

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value ≤ 999

/-- Check if a number starts with the digit 4 -/
def starts_with_four (n : ThreeDigitNumber) : Prop :=
  n.value / 100 = 4

/-- Get the digits of a three-digit number -/
def digits (n : ThreeDigitNumber) : Fin 3 → ℕ
| 0 => n.value / 100
| 1 => (n.value / 10) % 10
| 2 => n.value % 10

/-- Sum of digits of a three-digit number -/
def digit_sum (n : ThreeDigitNumber) : ℕ :=
  (digits n 0) + (digits n 1) + (digits n 2)

/-- All digits are different between two three-digit numbers -/
def all_digits_different (a b : ThreeDigitNumber) : Prop :=
  ∀ i j : Fin 3, digits a i ≠ digits b j

theorem smallest_digit_sum_of_sum (a b : ThreeDigitNumber) 
  (h1 : starts_with_four a)
  (h2 : starts_with_four b)
  (h3 : all_digits_different a b)
  (h4 : 100 ≤ a.value + b.value ∧ a.value + b.value ≤ 999) :
  ∃ (s : ThreeDigitNumber), s.value = a.value + b.value ∧ digit_sum s = 26 ∧
  ∀ (t : ThreeDigitNumber), t.value = a.value + b.value → digit_sum s ≤ digit_sum t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_digit_sum_of_sum_l159_15975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_ten_thirds_l159_15924

-- Define the curve and line functions
def curve (x : ℝ) : ℝ := x^2 + 1
def line (x : ℝ) : ℝ := -x + 3

-- Define the enclosed area
noncomputable def enclosed_area : ℝ :=
  ∫ x in Set.Icc 0 1, curve x - ∫ x in Set.Icc 0 1, line x

-- Theorem statement
theorem area_is_ten_thirds : enclosed_area = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_is_ten_thirds_l159_15924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l159_15995

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  h : 0 < a ∧ 0 < b ∧ 0 < c  -- Positive side lengths

-- Define the properties given in the problem
def satisfies_conditions (t : Triangle) : Prop :=
  t.b + t.c = 2 * Real.sin (t.C + Real.pi/6) ∧
  ∃ (d : Real), d = 2 * Real.sqrt 3 ∧ 
    d * Real.sin (t.A/2) = t.b * Real.sin (t.A/2) / Real.sin t.B

-- State the theorem
theorem triangle_property (t : Triangle) (h : satisfies_conditions t) : 
  t.A = Real.pi/3 ∧ 
  (∀ (t' : Triangle), satisfies_conditions t' → t.a * t.b * Real.sin t.C / 2 ≤ 4 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l159_15995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_four_l159_15983

theorem complex_power_four : 
  (3 * (Complex.cos (30 * π / 180) + Complex.I * Complex.sin (30 * π / 180)))^4 = 
  Complex.ofReal (-40.5) + Complex.I * Complex.ofReal (40.5 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_four_l159_15983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cube_packing_is_one_fourth_l159_15918

/-- The radius of a sphere in a specific packing arrangement within a cube. -/
noncomputable def sphere_radius_in_cube_packing : ℝ :=
  let cube_side_length : ℝ := 3
  let num_spheres : ℕ := 12
  let num_corner_spheres : ℕ := 4
  1 / 4

/-- Theorem stating that the radius of each sphere in the given packing arrangement is 1/4. -/
theorem sphere_radius_in_cube_packing_is_one_fourth :
  sphere_radius_in_cube_packing = 1 / 4 := by
  -- Unfold the definition of sphere_radius_in_cube_packing
  unfold sphere_radius_in_cube_packing
  -- The definition directly evaluates to 1/4, so we're done
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_radius_in_cube_packing_is_one_fourth_l159_15918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_factorial_plus_l159_15901

theorem prime_divisors_of_factorial_plus (n : ℕ) : 
  ∃ f : ℕ → ℕ, 
    (∀ k, k < n → Prime (f k) ∧ f k ∣ (Nat.factorial n + k)) ∧ 
    (∀ i j, i < n → j < n → i ≠ j → ¬(f i ∣ (Nat.factorial n + j))) :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_divisors_of_factorial_plus_l159_15901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l159_15928

/-- The domain of the function f, which is ℝ - {0, 1} -/
noncomputable def DomainF : Set ℝ := {x : ℝ | x ≠ 0 ∧ x ≠ 1}

/-- The function f as defined in the problem -/
noncomputable def f (x : ℝ) : ℝ := (1/2) * (x + 1 - 1/x - 1/(1-x))

/-- Theorem stating that f satisfies the given functional equation -/
theorem f_satisfies_equation :
  ∀ x ∈ DomainF, f x + f (1/(1-x)) = x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_satisfies_equation_l159_15928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_det_S_is_six_l159_15906

open Real Matrix

noncomputable def rotation_matrix (angle : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![Real.cos angle, -Real.sin angle; Real.sin angle, Real.cos angle]

def scaling_matrix (sx sy : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![sx, 0; 0, sy]

noncomputable def S : Matrix (Fin 2) (Fin 2) ℝ :=
  (scaling_matrix 2 3) * (rotation_matrix (Real.pi / 4))

theorem det_S_is_six :
  det S = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_det_S_is_six_l159_15906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_password_guessing_probabilities_l159_15920

/-- The number of digits in the password -/
def password_length : ℕ := 8

/-- The range of possible digits -/
def digit_range : Finset ℕ := Finset.range 10

/-- The set of even digits -/
def even_digits : Finset ℕ := digit_range.filter (fun n => n % 2 = 0)

/-- The probability of guessing the correct last digit in no more than 2 attempts -/
def prob_correct_guess : ℚ := 1/5

/-- The probability of guessing the correct last digit in no more than 2 attempts, given it's even -/
def prob_correct_guess_even : ℚ := 2/5

theorem password_guessing_probabilities :
  ((digit_range.card : ℚ)⁻¹ + 
    (1 - (digit_range.card : ℚ)⁻¹) * ((digit_range.card - 1 : ℕ) : ℚ)⁻¹ = prob_correct_guess) ∧
  ((even_digits.card : ℚ)⁻¹ + 
    (1 - (even_digits.card : ℚ)⁻¹) * ((even_digits.card - 1 : ℕ) : ℚ)⁻¹ = prob_correct_guess_even) :=
by sorry

#eval digit_range
#eval even_digits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_password_guessing_probabilities_l159_15920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circumscribed_sphere_area_l159_15987

/-- A right triangular prism with base triangle ABC and height h -/
structure RightTriangularPrism where
  AB : ℝ
  AC : ℝ
  h : ℝ

/-- The volume of a right triangular prism -/
noncomputable def volume (p : RightTriangularPrism) : ℝ :=
  1/2 * p.AB * p.AC * p.h

/-- The surface area of a sphere -/
noncomputable def sphereSurfaceArea (r : ℝ) : ℝ :=
  4 * Real.pi * r^2

/-- The theorem stating the minimum surface area of the circumscribed sphere -/
theorem min_circumscribed_sphere_area (p : RightTriangularPrism) 
  (hv : volume p = 3 * Real.sqrt 7)
  (hab : p.AB = 2)
  (hac : p.AC = 3) :
  ∃ (r : ℝ), sphereSurfaceArea r = 18 * Real.pi ∧ 
    ∀ (r' : ℝ), sphereSurfaceArea r' ≥ sphereSurfaceArea r := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circumscribed_sphere_area_l159_15987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seventeen_l159_15979

theorem divisibility_by_seventeen (a b c d : ℕ) 
  (h1 : a + b + c + d = 2023)
  (h2 : (2023 : ℕ) ∣ (a * b - c * d))
  (h3 : (2023 : ℕ) ∣ (a^2 + b^2 + c^2 + d^2))
  (h4 : ∀ x ∈ ({a, b, c, d} : Set ℕ), 7 ∣ x)
  (h5 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  ∀ x ∈ ({a, b, c, d} : Set ℕ), 17 ∣ x := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_by_seventeen_l159_15979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_problem_l159_15999

/-- The area of a sector with given arc length and diameter -/
noncomputable def sector_area (arc_length : ℝ) (diameter : ℝ) : ℝ :=
  (1/2) * arc_length * (diameter / 2)

/-- Theorem: The area of a sector with arc length 30 and diameter 16 is 120 -/
theorem sector_area_problem : sector_area 30 16 = 120 := by
  -- Unfold the definition of sector_area
  unfold sector_area
  -- Simplify the expression
  simp
  -- Perform the calculation
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sector_area_problem_l159_15999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_for_no_lattice_points_l159_15957

theorem max_b_for_no_lattice_points :
  let b := 51 / 149
  ∀ m : ℚ, 1/3 < m → m < b →
    ∀ x : ℤ, 0 < x → x ≤ 150 →
      ∀ y : ℤ, y ≠ (m * ↑x + 3).floor :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_b_for_no_lattice_points_l159_15957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_equal_expression_l159_15965

theorem unique_equal_expression (x : ℝ) (h : x > 0) :
  (2:ℝ)^x + (2:ℝ)^x = (2:ℝ)^(x+1) ∧
  (2:ℝ)^x + (2:ℝ)^x ≠ ((2:ℝ)^(x+1))^2 ∧
  (2:ℝ)^x + (2:ℝ)^x ≠ (2:ℝ)^x * (2:ℝ)^x ∧
  (2:ℝ)^x + (2:ℝ)^x ≠ (4:ℝ)^x :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_equal_expression_l159_15965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_migrating_male_percentage_l159_15933

/-- Represents the percentage of migrating geese that are male -/
noncomputable def percentage_migrating_male (total_geese : ℝ) (male_ratio : ℝ) (migration_rate_ratio : ℝ) : ℝ :=
  let male_geese := male_ratio * total_geese
  let female_geese := (1 - male_ratio) * total_geese
  let female_migration_rate := 1 -- Assume a base rate of 1 for female migration
  let male_migration_rate := migration_rate_ratio * female_migration_rate
  let migrating_male := male_migration_rate * male_geese
  let migrating_female := female_migration_rate * female_geese
  (migrating_male / (migrating_male + migrating_female)) * 100

/-- Theorem stating that given the conditions, 20% of migrating geese are male -/
theorem migrating_male_percentage :
  ∀ (total_geese : ℝ), total_geese > 0 →
  percentage_migrating_male total_geese 0.5 0.25 = 20 :=
by
  intro total_geese h
  -- The proof goes here
  sorry

-- This evaluation might not work due to the noncomputable nature of the function
-- #eval percentage_migrating_male 1000 0.5 0.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_migrating_male_percentage_l159_15933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l159_15989

/-- The radius of the inscribed circle in an isosceles triangle -/
theorem inscribed_circle_radius_isosceles_triangle (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : b < 2 * a) :
  let s := (2 * a + b) / 2
  let area := Real.sqrt (s * (s - a) * (s - a) * (s - b))
  area / s = 4 * Real.sqrt 65 / 13 → a = 9 ∧ b = 8 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_radius_isosceles_triangle_l159_15989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l159_15955

/-- Represents a grid with marked cells -/
structure Grid (n : ℕ) where
  marked : Finset (Fin n × Fin n)

/-- Checks if a 1x4 strip contains a marked cell -/
def strip_contains_marked (g : Grid 7) (start_row start_col : Fin 7) (is_horizontal : Bool) : Prop :=
  ∃ i : Fin 4, 
    let (row, col) := if is_horizontal then (start_row, ⟨start_col.val + i.val, by sorry⟩) 
                      else (⟨start_row.val + i.val, by sorry⟩, start_col)
    (row, col) ∈ g.marked

/-- All 1x4 strips contain a marked cell -/
def all_strips_marked (g : Grid 7) : Prop :=
  ∀ start_row start_col : Fin 7, ∀ is_horizontal : Bool,
    (if is_horizontal then start_col.val ≤ 3 else start_row.val ≤ 3) →
    strip_contains_marked g start_row start_col is_horizontal

/-- The main theorem -/
theorem min_marked_cells :
  ∃ (g : Grid 7), all_strips_marked g ∧ g.marked.card = 12 ∧
  ∀ (h : Grid 7), all_strips_marked h → h.marked.card ≥ 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_marked_cells_l159_15955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_and_equation_l159_15961

/-- A line passing through (3,2) and intersecting positive x and y axes -/
structure IntersectingLine where
  /-- The x-intercept of the line -/
  a : ℝ
  /-- The y-intercept of the line -/
  b : ℝ
  /-- The line passes through (3,2) -/
  passes_through_point : 3 / a + 2 / b = 1
  /-- The intercepts are positive -/
  positive_intercepts : a > 0 ∧ b > 0

/-- The area of triangle AOB formed by the line and the axes -/
noncomputable def triangle_area (l : IntersectingLine) : ℝ := (l.a * l.b) / 2

/-- The equation of the line in the form ax + by = c -/
noncomputable def line_equation (l : IntersectingLine) : ℝ × ℝ × ℝ := (l.b, l.a, l.a * l.b)

theorem min_area_and_equation (l : IntersectingLine) :
  triangle_area l ≥ 12 ∧
  (∃ l' : IntersectingLine, triangle_area l' = 12 ∧ line_equation l' = (4, 6, 24)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_area_and_equation_l159_15961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_correct_l159_15929

structure Subject where
  name : String
  required_percentage : ℚ
  marks_short : ℕ

def calculate_max_marks (s : Subject) : ℕ :=
  (((s.marks_short : ℚ) / (1 - s.required_percentage) * 100).ceil).toNat

def math : Subject := ⟨"Math", 95/100, 40⟩
def science : Subject := ⟨"Science", 92/100, 35⟩
def literature : Subject := ⟨"Literature", 90/100, 30⟩
def social_studies : Subject := ⟨"Social Studies", 88/100, 25⟩

theorem max_marks_correct :
  calculate_max_marks math = 800 ∧
  calculate_max_marks science = 438 ∧
  calculate_max_marks literature = 300 ∧
  calculate_max_marks social_studies = 209 := by
  sorry

#eval calculate_max_marks math
#eval calculate_max_marks science
#eval calculate_max_marks literature
#eval calculate_max_marks social_studies

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_marks_correct_l159_15929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l159_15902

/-- Represents the sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The arithmetic sequence property: S_n, S_m - S_n, S_k - S_m form an arithmetic sequence for k > m > n -/
axiom arithmetic_sequence_property (n m k : ℕ) (h : n < m ∧ m < k) :
  S m - S n = S k - S m

theorem arithmetic_sequence_sum :
  S 7 = 21 → S 17 = 34 → S 27 = 27 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_l159_15902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l159_15981

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + 2 * x + 3

theorem monotonic_increasing_f_implies_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) →
  a ∈ Set.Icc (-Real.sqrt 2) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotonic_increasing_f_implies_a_range_l159_15981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_theorem_l159_15909

-- Define the basic structures
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

-- Define the conditions
def insideAngleOutsideTriangle (P : Point) (T : Triangle) : Prop := sorry

def circumcenterOnRay (T : Triangle) (ray : Point → Point → Prop) : Prop := sorry

-- Helper functions (not proven, just declared)
noncomputable def circumcenter (T : Triangle) : Point := sorry
def onCircumcircle (P : Point) (T : Triangle) : Prop := sorry

-- Main theorem
theorem circumcenter_theorem (ABC : Triangle) (P : Point) 
  (h_inside : insideAngleOutsideTriangle P ABC) :
  (∃ (i j : Fin 3), i ≠ j ∧
    circumcenterOnRay (Triangle.mk P ABC.B ABC.C) (λ x y ↦ x = ABC.A ∨ y = ABC.A) ∧
    circumcenterOnRay (Triangle.mk ABC.C P ABC.A) (λ x y ↦ x = ABC.B ∨ y = ABC.B) ∧
    circumcenterOnRay (Triangle.mk ABC.A P ABC.B) (λ x y ↦ x = ABC.C ∨ y = ABC.C)) →
  (∀ (i : Fin 3), 
    circumcenterOnRay (Triangle.mk P ABC.B ABC.C) (λ x y ↦ x = ABC.A ∨ y = ABC.A) ∧
    circumcenterOnRay (Triangle.mk ABC.C P ABC.A) (λ x y ↦ x = ABC.B ∨ y = ABC.B) ∧
    circumcenterOnRay (Triangle.mk ABC.A P ABC.B) (λ x y ↦ x = ABC.C ∨ y = ABC.C)) ∧
  (∃ (circle : Point → Prop),
    circle (circumcenter (Triangle.mk ABC.B P ABC.C)) ∧
    circle (circumcenter (Triangle.mk ABC.C P ABC.A)) ∧
    circle (circumcenter (Triangle.mk ABC.A P ABC.B)) ∧
    (∀ (X : Point), onCircumcircle X ABC ↔ circle X)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumcenter_theorem_l159_15909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_l159_15984

noncomputable def F₁ : ℝ × ℝ := (0, 1)
noncomputable def F₂ : ℝ × ℝ := (4, 1)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  distance P F₁ + distance P F₂ = 6

noncomputable def center : ℝ × ℝ :=
  ((F₁.1 + F₂.1) / 2, (F₁.2 + F₂.2) / 2)

noncomputable def h : ℝ := center.1
noncomputable def k : ℝ := center.2

noncomputable def c : ℝ := distance F₁ center

noncomputable def a : ℝ := 3

noncomputable def b : ℝ := Real.sqrt (a^2 - c^2)

theorem ellipse_sum :
  h + k + a + b = 6 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_sum_l159_15984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_paid_594_l159_15964

/-- The total amount Michael paid for his clothing items with discounts applied --/
noncomputable def total_paid (suit_price shoes_price shirt_price tie_price : ℚ)
                (suit_discount shoes_discount shirt_tie_discount_percent : ℚ) : ℚ :=
  let suit_discounted := suit_price - suit_discount
  let shoes_discounted := shoes_price - shoes_discount
  let shirt_tie_total := shirt_price + tie_price
  let shirt_tie_discounted := shirt_tie_total * (1 - shirt_tie_discount_percent / 100)
  suit_discounted + shoes_discounted + shirt_tie_discounted

/-- Theorem stating that Michael paid $594 for his clothing items --/
theorem michael_paid_594 :
  total_paid 430 190 80 50 100 30 20 = 594 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_michael_paid_594_l159_15964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_intersection_l159_15978

-- Define the ellipse E
def E (x y : ℝ) : Prop := x^2/4 + y^2 = 1

-- Define point A as the right vertex of the ellipse
def A : ℝ × ℝ := (2, 0)

-- Define the chord BC passing through the origin
def chord_BC (x y : ℝ) : Prop :=
  E x y ∧ E (-x) (-y) ∧ (x, y) ≠ (0, 0)

-- Define points P and Q on the y-axis
def P : ℝ → ℝ × ℝ := λ y ↦ (0, y)
def Q : ℝ → ℝ × ℝ := λ y ↦ (0, y)

-- Define the circle Γ with diameter PQ
def Γ (center : ℝ × ℝ) (radius : ℝ) (x y : ℝ) : Prop :=
  (x - center.1)^2 + (y - center.2)^2 = radius^2

-- Define points M and N on the x-axis
def M : ℝ × ℝ := (-1, 0)
def N : ℝ × ℝ := (1, 0)

-- Define the locus of point T
def locus_T (x y : ℝ) : Prop := x^2 - y^2 = 1 ∧ x ≠ 1 ∧ x ≠ -1

theorem ellipse_chord_intersection :
  ∀ (x y : ℝ), chord_BC x y →
  (∃ (p q : ℝ), 
    (∃ (center : ℝ × ℝ) (radius : ℝ), Γ center radius M.1 M.2 ∧ Γ center radius N.1 N.2) ∧
    (∀ (t_x t_y : ℝ), locus_T t_x t_y ↔ 
      (∃ (k1 k2 : ℝ), k1 * (t_x + 1) = t_y ∧ k2 * (t_x - 1) = t_y))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_chord_intersection_l159_15978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_all_solutions_in_set_l159_15990

/-- A pair of integers (m, n) is a valid solution if both m^2 + 4n and n^2 + 4m are perfect squares. -/
def IsValidSolution (m n : ℤ) : Prop :=
  ∃ (k l : ℤ), m^2 + 4*n = k^2 ∧ n^2 + 4*m = l^2

/-- The set of all valid solutions (m, n) -/
def SolutionSet : Set (ℤ × ℤ) :=
  {p | ∃ (a : ℤ), 
    p = (0, a^2) ∨ 
    p = (-5, -6) ∨ 
    p = (-4, -4) ∨ 
    p = (a+1, -a)}

/-- Theorem stating that all valid solutions belong to the SolutionSet -/
theorem all_solutions_in_set :
  ∀ (m n : ℤ), IsValidSolution m n → (m, n) ∈ SolutionSet := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_all_solutions_in_set_l159_15990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_generating_set_characterization_l159_15958

structure Vector2D where
  x : Int
  y : Int

def GeneratingSet (S : Set Vector2D) : Prop :=
  ∀ v : Vector2D, ∃ (m n : Int) (a b : Vector2D), 
    a ∈ S ∧ b ∈ S ∧ v = Vector2D.mk (m * a.x + n * b.x) (m * a.y + n * b.y)

theorem generating_set_characterization (a b : Vector2D) :
  GeneratingSet {a, b} ↔ (a.x * b.y - a.y * b.x = 1 ∨ a.x * b.y - a.y * b.x = -1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_generating_set_characterization_l159_15958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l159_15922

theorem tan_theta_value (θ : Real) 
  (h1 : Real.sin (3 * Real.pi + θ) = 1/3)
  (h2 : π < θ ∧ θ < 3*π/2) : 
  Real.tan θ = -Real.sqrt 2 / 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l159_15922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_composite_probability_l159_15976

/-- The number of faces on a standard die -/
def diefaces : ℕ := 6

/-- The number of dice rolled -/
def numdice : ℕ := 4

/-- The set of prime numbers on a standard die -/
def primes_on_die : Finset ℕ := {2, 3, 5}

/-- The total number of possible outcomes when rolling 4 dice -/
def total_outcomes : ℕ := diefaces ^ numdice

/-- The number of ways to get a non-composite product -/
def non_composite_outcomes : ℕ := numdice * primes_on_die.card + 1

/-- The probability of getting a composite product when rolling 4 standard 6-sided dice -/
theorem dice_composite_probability :
  (1 : ℚ) - (non_composite_outcomes : ℚ) / total_outcomes = 1283 / 1296 := by
  sorry

#eval non_composite_outcomes
#eval total_outcomes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_composite_probability_l159_15976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_A_l159_15923

-- Define the labels for the faces of the cube
inductive FaceLabel
  | A | B | C | D | E | F
deriving Repr, DecidableEq

-- Define the structure of the net
structure Net where
  faces : List FaceLabel
  adjacent : FaceLabel → FaceLabel → Prop

-- Define the cube formed by folding the net
structure Cube where
  net : Net
  opposite : FaceLabel → FaceLabel

-- Define the specific net configuration
def specificNet : Net where
  faces := [FaceLabel.A, FaceLabel.B, FaceLabel.C, FaceLabel.D, FaceLabel.E, FaceLabel.F]
  adjacent := fun x y => 
    (x = FaceLabel.A ∧ y = FaceLabel.F) ∨
    (x = FaceLabel.A ∧ y = FaceLabel.B) ∨
    (x = FaceLabel.A ∧ y = FaceLabel.C) ∨
    (x = FaceLabel.A ∧ y = FaceLabel.E) ∨
    (y = FaceLabel.A ∧ x = FaceLabel.F) ∨
    (y = FaceLabel.A ∧ x = FaceLabel.B) ∨
    (y = FaceLabel.A ∧ x = FaceLabel.C) ∨
    (y = FaceLabel.A ∧ x = FaceLabel.E)

-- The theorem to prove
theorem opposite_face_of_A (c : Cube) (h : c.net = specificNet) : 
  c.opposite FaceLabel.A = FaceLabel.D := by
  sorry

-- Additional helper lemmas if needed
lemma not_adjacent_is_opposite (c : Cube) (x y : FaceLabel) :
  (¬ c.net.adjacent x y) → c.opposite x = y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_of_A_l159_15923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l159_15960

/-- Ellipse C with equation x^2/3 + y^2 = 1 -/
def ellipse_C (x y : ℝ) : Prop := x^2/3 + y^2 = 1

/-- Line l intersecting ellipse C -/
def line_l (k m : ℝ) (x y : ℝ) : Prop := y = k*x + m

/-- Points A and B are on ellipse C and line l -/
def points_AB (xA yA xB yB k m : ℝ) : Prop :=
  ellipse_C xA yA ∧ ellipse_C xB yB ∧ 
  line_l k m xA yA ∧ line_l k m xB yB

/-- Circle with AB as diameter passes through origin -/
def circle_through_origin (xA yA xB yB : ℝ) : Prop :=
  xA*xB + yA*yB = 0

/-- Distance from origin to line AB -/
noncomputable def distance_to_line (k m : ℝ) : ℝ :=
  abs m / Real.sqrt (k^2 + 1)

/-- Area of triangle OAB -/
noncomputable def triangle_area (xA yA xB yB : ℝ) : ℝ :=
  (1/2 : ℝ) * Real.sqrt ((xB - xA)^2 + (yB - yA)^2) * Real.sqrt 3 / 2

theorem ellipse_properties :
  ∀ (xA yA xB yB k m : ℝ),
  points_AB xA yA xB yB k m →
  circle_through_origin xA yA xB yB →
  (distance_to_line k m = Real.sqrt 3 / 2 ∧
   triangle_area xA yA xB yB ≤ Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_l159_15960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_ratio_l159_15969

/-- Given a cylinder of fixed volume V, the total surface area (including the two circular ends) 
    is minimized for a radius of r and height h. The ratio h/r is 2 if the height is twice the radius. -/
theorem cylinder_ratio (V : ℝ) (h r : ℝ) (h_pos : h > 0) (r_pos : r > 0) 
    (volume_eq : V = π * r^2 * h) (height_eq : h = 2*r) : h / r = 2 := by
  -- Proof goes here
  sorry

#check cylinder_ratio

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_ratio_l159_15969


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_i_value_l159_15985

/-- Represents a digit in the range [0, 9] -/
def Digit := Fin 10

/-- Represents the addition problem with letters -/
structure LetterAddition where
  n : Digit
  i : Digit
  t : Digit
  e : Digit
  s : Digit
  different : n ≠ i ∧ n ≠ t ∧ n ≠ e ∧ n ≠ s ∧ i ≠ t ∧ i ≠ e ∧ i ≠ s ∧ t ≠ e ∧ t ≠ s ∧ e ≠ s

/-- The addition problem is valid -/
def isValid (a : LetterAddition) : Prop :=
  (a.n.val * 100 + a.i.val * 10 + a.n.val) +
  (a.n.val * 100 + a.i.val * 10 + a.n.val) =
  a.t.val * 1000 + a.e.val * 100 + a.n.val * 10 + a.s.val

theorem unique_i_value (a : LetterAddition) (h1 : a.n = ⟨8, by norm_num⟩) (h2 : Even a.n.val) (h3 : isValid a) :
  a.i = ⟨5, by norm_num⟩ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_i_value_l159_15985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_club_costs_l159_15944

-- Define the cost functions
def f (x : ℝ) : ℝ := 5 * x

noncomputable def g (x : ℝ) : ℝ :=
  if x ≤ 30 then 90 else 2 * x + 30

-- Define the theorem
theorem table_tennis_club_costs (x : ℝ) (h : 15 ≤ x ∧ x ≤ 40) :
  (f x = 5 * x) ∧
  (x ≤ 30 → g x = 90) ∧
  (30 < x → g x = 2 * x + 30) ∧
  (15 ≤ x ∧ x < 18 → f x < g x) ∧
  (x = 18 → f x = g x) ∧
  (18 < x ∧ x ≤ 40 → f x > g x) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_tennis_club_costs_l159_15944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_fourth_six_given_three_sixes_l159_15943

/-- Represents a six-sided die -/
inductive Die
| Fair
| Biased

/-- Probability of rolling a six for a given die -/
def prob_six (d : Die) : ℚ :=
  match d with
  | Die.Fair => 1/6
  | Die.Biased => 3/4

/-- Probability of rolling three sixes in a row for a given die -/
def prob_three_sixes (d : Die) : ℚ :=
  (prob_six d) ^ 3

/-- The probability of choosing each die -/
def prob_choose_die : ℚ := 1/2

/-- The probability of the die being biased given three sixes were rolled -/
def prob_biased_given_three_sixes : ℚ :=
  (prob_three_sixes Die.Biased * prob_choose_die) /
  ((prob_three_sixes Die.Biased * prob_choose_die) + (prob_three_sixes Die.Fair * prob_choose_die))

/-- The main theorem: probability of rolling a six on the fourth roll given three sixes -/
theorem prob_fourth_six_given_three_sixes :
  (prob_six Die.Fair * (1 - prob_biased_given_three_sixes) + 
   prob_six Die.Biased * prob_biased_given_three_sixes) = 41/67 := by
  sorry

-- Remove the #eval statement as it's not necessary for building

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prob_fourth_six_given_three_sixes_l159_15943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l159_15996

/-- Represents the number of students sampled from a given population --/
def sampleSize (population : ℕ) (sampleProbability : ℚ) : ℕ :=
  (population : ℚ) * sampleProbability |>.floor.toNat

/-- Represents the total number of students sampled from all grades --/
def totalSampleSize (grade1 : ℕ) (grade2 : ℕ) (grade3 : ℕ) (sampleProbability : ℚ) : ℕ :=
  sampleSize grade1 sampleProbability +
  sampleSize grade2 sampleProbability +
  sampleSize grade3 sampleProbability

theorem stratified_sampling_theorem (grade1 grade2 grade3 : ℕ) (sampledGrade2 : ℕ) :
  grade1 = 1500 →
  grade2 = 1200 →
  grade3 = 1000 →
  sampledGrade2 = 60 →
  totalSampleSize grade1 grade2 grade3 ((sampledGrade2 : ℚ) / grade2) = 185 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_theorem_l159_15996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_calculation_l159_15993

theorem percentage_calculation : ∃! P : ℝ, 
  (0.47 * 1442 - P / 100 * 1412) + 66 = 6 ∧ 
  abs (P - 52.24) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_calculation_l159_15993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l159_15900

-- Define the points A, B, and C
def A : ℝ × ℝ := (-3, 0)
def B : ℝ × ℝ := (2, 1)
def C : ℝ × ℝ := (-2, 3)

-- Define the slope of line BC
noncomputable def slope_BC : ℝ := (C.2 - B.2) / (C.1 - B.1)

-- Define the slope of altitude AH
noncomputable def slope_AH : ℝ := -1 / slope_BC

-- Define the equation of altitude AH
def eq_AH (x y : ℝ) : Prop := 2 * x - y + 6 = 0

-- Define the equations of line l
def eq_l1 (x y : ℝ) : Prop := x - 2 * y = 0
def eq_l2 (x y : ℝ) : Prop := x - y - 1 = 0

theorem triangle_ABC_properties :
  eq_AH A.1 A.2 ∧
  (eq_l1 B.1 B.2 ∨ eq_l2 B.1 B.2) ∧
  (∀ x y : ℝ, eq_l1 x y → x = 0 → y = 0) ∧
  (∀ x y : ℝ, eq_l2 x y → ∃ a : ℝ, x = a ∧ y = -a ∧ a = 1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ABC_properties_l159_15900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l159_15947

/-- An ellipse with specific properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  c : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_equation : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 → (x, y) ∈ Set.range (λ t : ℝ × ℝ ↦ t)
  h_perpendicular : (-c) * a + (-b) * b = 0
  h_relation : a^2 = b^2 + c^2

/-- The eccentricity of an ellipse is (√5 - 1) / 2 -/
theorem ellipse_eccentricity (e : Ellipse) : e.c / e.a = (Real.sqrt 5 - 1) / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_eccentricity_l159_15947
